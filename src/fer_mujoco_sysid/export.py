"""Export a consumer model and prove both of its normal load paths.

These are deliberately different artifacts:

* :func:`export_consumer_model` writes the gravity-enabled full Panda source
  plus the accepted physical parameters. It verifies both an ordinary full
  ``MjModel.from_xml_path`` reload and the exact seven-DOF Hydrax planning
  derivation used by MPPI. The planning model retains gravity because its
  inverse-dynamics controls include gravity; the real LFC removes that term
  before commanding the gravity-compensated Franka.
* :func:`export_full_source_patched_model` performs only the parameter patch
  and whitelist reload check. It is a secondary transfer artifact, not proof
  that the consumer planning derivation matches the accepted fit.

Both paths fail closed on reload. The full-source patch also proves that
kinematics, actuator limits, geometry and naming outside the dynamic
whitelist did not move.
"""

from __future__ import annotations

import copy
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.fitting import body_full_inertia, set_hinge_damping
from fer_mujoco_sysid.io import sha256_file, write_json
from fer_mujoco_sysid.model import (
    HYDRAX_ARM_JOINT_NAMES,
    build_hydrax_arm_model,
    build_hydrax_arm_spec,
    resolve_model_paths,
)

FULL_SOURCE_PATCH_FORMAT = "fer-mujoco-sysid/identified-model@1"
CONSUMER_MODEL_FORMAT = "fer-mujoco-sysid/consumer-model@1"
XML_ROUNDTRIP_RTOL = 1e-5
XML_ROUNDTRIP_ATOL = 1e-8
BEHAVIOR_ROUNDTRIP_RTOL = 1e-4
BEHAVIOR_ROUNDTRIP_ATOL = 1e-8

#: The only compiled quantities an export is allowed to change.
EXPORTABLE_FIELDS = (
    "dof_frictionloss",
    "dof_damping",
    "dof_armature",
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
)

#: Everything else that must be bit-identical between nominal and identified.
#: Not exhaustive over the whole model, but it covers every quantity that
#: would change the robot rather than its dynamics estimate.
INVARIANT_FIELDS = (
    "body_pos",
    "body_quat",
    "jnt_range",
    "jnt_axis",
    "jnt_pos",
    "jnt_type",
    "actuator_forcerange",
    "actuator_ctrlrange",
    "actuator_trnid",
    "geom_size",
    "geom_pos",
)


class ExportError(RuntimeError):
    """An identified model does not match what it claims to be."""


@dataclass(frozen=True)
class BodyInertial:
    """Exact compiled inertial values for one body.

    ``inertia`` is the full tensor in the body frame, ordered
    ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``. This avoids the sign and degeneracy
    ambiguity of MuJoCo's compiled principal-axis quaternion.
    """

    body: str
    mass: float
    ipos: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]

    @classmethod
    def from_model(cls, model: mujoco.MjModel, body: str) -> BodyInertial:
        view = model.body(body)
        return cls(
            body=body,
            mass=float(view.mass[0]),
            ipos=tuple(float(value) for value in view.ipos),
            inertia=tuple(float(value) for value in body_full_inertia(model, body)),
        )

    @classmethod
    def from_mapping(cls, body: str, values: Mapping[str, object]) -> BodyInertial:
        try:
            mass = float(values["mass"])  # type: ignore[arg-type]
            ipos = tuple(float(value) for value in values["ipos"])  # type: ignore[union-attr]
            inertia = tuple(
                float(value)
                for value in values["inertia"]  # type: ignore[union-attr]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ExportError(f"{body}: malformed body inertial mapping") from error
        return cls(
            body=body,
            mass=mass,
            ipos=ipos,  # type: ignore[arg-type]
            inertia=inertia,  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mass": self.mass,
            "ipos": list(self.ipos),
            "inertia": list(self.inertia),
        }

    def validate(self) -> None:
        if not self.body:
            raise ExportError("body inertial entry has an empty body name")
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ExportError(f"{self.body}.mass must be finite and positive")
        if len(self.ipos) != 3 or not np.isfinite(self.ipos).all():
            raise ExportError(f"{self.body}.ipos must hold three finite values")
        if len(self.inertia) != 6 or not np.isfinite(self.inertia).all():
            raise ExportError(
                f"{self.body}.inertia must hold six finite body-frame values"
            )
        ixx, iyy, izz, ixy, ixz, iyz = self.inertia
        tensor = np.array(
            ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)),
            dtype=np.float64,
        )
        moments = np.linalg.eigvalsh(tensor)
        scale = max(float(np.abs(moments).max()), 1.0)
        tolerance = 1e-10 * scale
        if float(moments.min()) <= tolerance:
            raise ExportError(f"{self.body}.inertia is not positive definite")
        if float(moments[-1]) > float(moments[0] + moments[1]) + tolerance:
            raise ExportError(
                f"{self.body}.inertia violates the rigid-body triangle inequality"
            )


@dataclass(frozen=True)
class IdentifiedParameters:
    """Exact dynamic values approved for export."""

    frictionloss: tuple[float, ...] | None = None
    damping: tuple[float, ...] | None = None
    armature: tuple[float, ...] | None = None
    joints: tuple[str, ...] = HYDRAX_ARM_JOINT_NAMES
    body_inertials: tuple[BodyInertial, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> IdentifiedParameters:
        """Read the subset of a fit result that is exportable."""

        def column(name: str) -> tuple[float, ...] | None:
            raw = values.get(name)
            if raw is None:
                return None
            return tuple(float(value) for value in raw)  # type: ignore[union-attr]

        raw_bodies = values.get("body_inertials", {})
        if not isinstance(raw_bodies, Mapping):
            raise ExportError("body_inertials must map body names to values")
        bodies: list[BodyInertial] = []
        for body, raw in raw_bodies.items():
            if not isinstance(body, str) or not isinstance(raw, Mapping):
                raise ExportError(
                    "body_inertials must map body names to inertial mappings"
                )
            bodies.append(BodyInertial.from_mapping(body, raw))

        raw_joints = values.get("joints", HYDRAX_ARM_JOINT_NAMES)
        if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, str):
            raise ExportError("joints must be a sequence of joint names")
        return cls(
            frictionloss=column("frictionloss"),
            damping=column("damping"),
            armature=column("armature"),
            joints=tuple(str(name) for name in raw_joints),
            body_inertials=tuple(bodies),
        )

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        bodies: Sequence[str] = (),
        joints: Sequence[str] = HYDRAX_ARM_JOINT_NAMES,
        include_frictionloss: bool = True,
        include_damping: bool = True,
        include_armature: bool = True,
    ) -> IdentifiedParameters:
        """Capture exact approved values from the fitted compiled model."""
        joint_names = tuple(joints)

        def values(field: str) -> tuple[float, ...]:
            return tuple(
                float(getattr(model, field)[int(model.joint(name).dofadr[0])])
                for name in joint_names
            )

        result = cls(
            frictionloss=values("dof_frictionloss") if include_frictionloss else None,
            damping=values("dof_damping") if include_damping else None,
            armature=values("dof_armature") if include_armature else None,
            joints=joint_names,
            body_inertials=tuple(
                BodyInertial.from_model(model, body) for body in bodies
            ),
        )
        result.validate()
        return result

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            name: list(values)
            for name, values in (
                ("frictionloss", self.frictionloss),
                ("damping", self.damping),
                ("armature", self.armature),
            )
            if values is not None
        }
        if self.body_inertials:
            result["body_inertials"] = {
                body.body: body.as_dict() for body in self.body_inertials
            }
        return result

    def validate(self) -> None:
        for name, values in (
            ("frictionloss", self.frictionloss),
            ("damping", self.damping),
            ("armature", self.armature),
        ):
            if values is None:
                continue
            if len(values) != len(self.joints):
                raise ExportError(
                    f"{name} has {len(values)} values for {len(self.joints)} joints"
                )
            if not all(np.isfinite(values)):
                raise ExportError(f"{name} holds a non-finite value")
            if min(values) < 0.0:
                raise ExportError(
                    f"{name} holds a negative value; friction, damping and "
                    "armature are all non-negative by physics"
                )
        if len(set(self.joints)) != len(self.joints):
            raise ExportError("joint names must be unique")
        body_names = [body.body for body in self.body_inertials]
        if len(set(body_names)) != len(body_names):
            raise ExportError("body inertial names must be unique")
        for body in self.body_inertials:
            body.validate()


def apply_parameters(
    spec: mujoco.MjSpec, parameters: IdentifiedParameters
) -> mujoco.MjSpec:
    """Write the identified values onto a spec, in place."""
    parameters.validate()
    for index, name in enumerate(parameters.joints):
        joint = spec.joint(name)
        if parameters.frictionloss is not None:
            joint.frictionloss = float(parameters.frictionloss[index])
        if parameters.damping is not None:
            set_hinge_damping(joint, float(parameters.damping[index]))
        if parameters.armature is not None:
            joint.armature = float(parameters.armature[index])
    for values in parameters.body_inertials:
        body = spec.body(values.body)
        body.mass = values.mass
        body.ipos = np.asarray(values.ipos, dtype=np.float64)
        # Select the full-tensor representation. Setting iquat to NaN is the
        # public MjSpec convention used by mujoco.sysid for full inertias.
        body.inertia[:] = 0.0
        body.iquat[:] = np.nan
        body.fullinertia[:] = np.asarray(values.inertia, dtype=np.float64)
    return spec


@dataclass(frozen=True)
class ExportCheck:
    """What an exported model changed, and what it left alone."""

    source_path: Path
    exported_path: Path
    source_sha256: str
    changed: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    body_changes: dict[
        str, dict[str, tuple[float | tuple[float, ...], float | tuple[float, ...]]]
    ] = field(default_factory=dict)
    #: Worst absolute difference between a value the fit produced and the
    #: value that survives the round trip through XML. This is the entire
    #: cost of writing the model to disk.
    roundtrip_error: float = 0.0
    invariant_fields: tuple[str, ...] = INVARIANT_FIELDS

    def table(self) -> str:
        lines = [
            "| joint | field | nominal | identified | change |",
            "| --- | --- | --- | --- | --- |",
        ]
        for joint, fields in self.changed.items():
            for name, (before, after) in fields.items():
                lines.append(
                    f"| {joint} | {name} | {before:.4f} | {after:.4f} | "
                    f"{after - before:+.4f} |"
                )
        for body, fields in self.body_changes.items():
            for name, (before, after) in fields.items():
                if isinstance(before, tuple) and isinstance(after, tuple):
                    before_text = "[" + ", ".join(f"{v:.5g}" for v in before) + "]"
                    after_text = "[" + ", ".join(f"{v:.5g}" for v in after) + "]"
                    maximum_change = np.max(np.abs(np.subtract(after, before)))
                    change_text = f"max |Δ|={maximum_change:.3g}"
                else:
                    before_text = f"{float(before):.5g}"
                    after_text = f"{float(after):.5g}"
                    change_text = f"{float(after) - float(before):+.3g}"
                lines.append(
                    f"| {body} | {name} | {before_text} | {after_text} | "
                    f"{change_text} |"
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class ConsumerModelExportCheck(ExportCheck):
    """Evidence that both consumer load paths reproduce the accepted physics."""

    manifest_path: Path | None = None
    exported_sha256: str = ""
    projection: dict[str, object] = field(default_factory=dict)
    simulation_conventions: dict[str, object] = field(default_factory=dict)
    planning_compiled_error: float = 0.0
    planning_behavior_error: float = 0.0
    fit_convention_behavior_error: float = 0.0


def verify_export(
    source_path: str | Path,
    exported_path: str | Path,
    parameters: IdentifiedParameters,
) -> ExportCheck:
    """Reload both models and confirm the export changed only what it may.

    This is the gate the plan requires before an identified model reaches a
    consumer: the exported file must compile to the nominal model with the
    identified parameters substituted, and to nothing else. A mismatch here
    means the export path corrupted the model, which is worse than not
    exporting at all.
    """
    source_path = Path(source_path)
    exported_path = Path(exported_path)
    parameters.validate()
    nominal = mujoco.MjModel.from_xml_path(str(source_path))
    expected_spec = mujoco.MjSpec.from_file(str(source_path))
    apply_parameters(expected_spec, parameters)
    # Compare with the expected model after the same MuJoCo XML round trip.
    # MjSpec writes a finite number of significant digits; serializing the
    # reference here isolates that unavoidable precision from an actual edit
    # and lets the invariant gate below remain near machine precision.
    expected_spec.compile()
    expected_spec.meshdir = str((source_path.parent / expected_spec.meshdir).resolve())
    expected_spec.texturedir = str(
        (source_path.parent / expected_spec.texturedir).resolve()
    )
    expected = mujoco.MjModel.from_xml_string(spec_to_xml_exact_dynamics(expected_spec))
    identified = mujoco.MjModel.from_xml_path(str(exported_path))

    if (
        nominal.njnt != identified.njnt
        or nominal.nbody != identified.nbody
        or nominal.nv != identified.nv
        or nominal.nu != identified.nu
    ):
        raise ExportError(
            f"{exported_path} has a different structure than {source_path} "
            f"({identified.njnt} joints / {identified.nbody} bodies / "
            f"{identified.nv} DOFs / {identified.nu} actuators against "
            f"{nominal.njnt} / {nominal.nbody} / {nominal.nv} / {nominal.nu})"
        )

    # Check the dynamic arrays first to give a precise field name for both a
    # bad requested round trip and an unauthorized inertial edit.
    for name in (
        "dof_frictionloss",
        "dof_damping",
        "dof_armature",
        "body_mass",
        "body_ipos",
        "body_inertia",
    ):
        _assert_model_array_equal(
            name, getattr(expected, name), getattr(identified, name)
        )

    # Principal-axis quaternions have eigenvector sign ambiguity. Comparing
    # the reconstructed full tensor proves equivalent physics without
    # rejecting an otherwise identical serialization solely for q versus -q.
    for body_id in range(expected.nbody):
        body_name = mujoco.mj_id2name(expected, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_key: str | int = body_name if body_name is not None else body_id
        body_label = body_name if body_name is not None else f"body[{body_id}]"
        _assert_model_array_equal(
            f"body_full_inertia[{body_label}]",
            body_full_inertia(expected, body_key),
            body_full_inertia(identified, body_key),
        )

    # Exhaustive compiled-model equivalence is the strict invariant gate.
    # It is deliberately broader than INVARIANT_FIELDS: every public ndarray,
    # byte/string table and scalar exposed by MjModel is checked against the
    # expected source-plus-approved-edits model.
    _verify_complete_model(expected, identified, skip={"body_iquat"})

    changed: dict[str, dict[str, tuple[float, float]]] = {}
    roundtrip = 0.0
    requested = parameters.as_dict()
    for index, joint_name in enumerate(parameters.joints):
        joint = identified.joint(joint_name)
        nominal_joint = nominal.joint(joint_name)
        dof = int(joint.dofadr[0])
        nominal_dof = int(nominal_joint.dofadr[0])
        for field_name, compiled in (
            ("frictionloss", "dof_frictionloss"),
            ("damping", "dof_damping"),
            ("armature", "dof_armature"),
        ):
            if field_name not in requested:
                continue
            written = float(getattr(identified, compiled)[dof])
            wanted = float(requested[field_name][index])
            roundtrip = max(roundtrip, abs(written - wanted))
            if abs(written - wanted) > max(1e-6, 1e-5 * abs(wanted)):
                raise ExportError(
                    f"{joint_name}.{field_name} exported as {written} but the "
                    f"identification produced {wanted}"
                )
            before = float(getattr(nominal, compiled)[nominal_dof])
            changed.setdefault(joint_name, {})[field_name] = (before, written)

    body_changes: dict[
        str, dict[str, tuple[float | tuple[float, ...], float | tuple[float, ...]]]
    ] = {}
    for wanted in parameters.body_inertials:
        written = BodyInertial.from_model(identified, wanted.body)
        before = BodyInertial.from_model(nominal, wanted.body)
        for field_name in ("mass", "ipos", "inertia"):
            requested_value = getattr(wanted, field_name)
            written_value = getattr(written, field_name)
            requested_array = np.atleast_1d(
                np.asarray(requested_value, dtype=np.float64)
            )
            written_array = np.atleast_1d(np.asarray(written_value, dtype=np.float64))
            error = float(np.max(np.abs(written_array - requested_array)))
            roundtrip = max(roundtrip, error)
            if not np.allclose(written_array, requested_array, rtol=1e-5, atol=1e-8):
                raise ExportError(
                    f"{wanted.body}.{field_name} exported as {written_value} "
                    f"but the identification produced {requested_value}"
                )
            body_changes.setdefault(wanted.body, {})[field_name] = (
                getattr(before, field_name),
                written_value,
            )

    return ExportCheck(
        source_path=source_path,
        exported_path=exported_path,
        source_sha256=sha256_file(source_path),
        changed=changed,
        body_changes=body_changes,
        roundtrip_error=roundtrip,
    )


def _assert_model_array_equal(
    name: str,
    expected: object,
    actual: object,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    context: str = "export",
) -> float:
    """Compare one compiled array and return its worst absolute error."""
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        raise ExportError(
            f"{context} changed {name} shape from {expected_array.shape} "
            f"to {actual_array.shape}"
        )
    if np.issubdtype(expected_array.dtype, np.inexact):
        matches = np.allclose(
            expected_array,
            actual_array,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        finite = np.isfinite(expected_array) & np.isfinite(actual_array)
        differences = np.where(
            finite,
            np.abs(expected_array - actual_array),
            0.0,
        )
        maximum_error = float(np.max(differences, initial=0.0))
    else:
        matches = np.array_equal(expected_array, actual_array)
        maximum_error = 0.0
    if matches:
        return maximum_error
    if expected_array.size and np.issubdtype(expected_array.dtype, np.number):
        difference = np.abs(
            expected_array.astype(np.float64).ravel()
            - actual_array.astype(np.float64).ravel()
        )
        difference = np.nan_to_num(difference, nan=np.inf)
        worst = int(np.argmax(difference))
        detail = (
            f" (worst element {worst}: "
            f"{expected_array.ravel()[worst]} -> {actual_array.ravel()[worst]})"
        )
    else:
        detail = ""
    raise ExportError(
        f"{context} differs from its accepted reference in {name}{detail}"
    )


def _verify_public_container(
    expected: object,
    actual: object,
    *,
    prefix: str,
    rtol: float,
    atol: float,
    context: str,
) -> float:
    """Compare public scalar/array fields on nested MuJoCo structs."""
    maximum_error = 0.0
    for name in sorted(set(dir(expected)) & set(dir(actual))):
        if name.startswith("_"):
            continue
        try:
            before = getattr(expected, name)
            after = getattr(actual, name)
        except (AttributeError, RuntimeError):
            continue
        qualified = f"{prefix}.{name}"
        if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
            maximum_error = max(
                maximum_error,
                _assert_model_array_equal(
                    qualified,
                    before,
                    after,
                    rtol=rtol,
                    atol=atol,
                    context=context,
                ),
            )
        elif isinstance(
            before, (bytes, str, bool, int, float, np.generic)
        ) and isinstance(after, (bytes, str, bool, int, float, np.generic)):
            if isinstance(before, (float, np.floating)) or isinstance(
                after, (float, np.floating)
            ):
                error = abs(float(before) - float(after))
                maximum_error = max(maximum_error, error)
                if not np.isclose(float(before), float(after), rtol=rtol, atol=atol):
                    raise ExportError(
                        f"{context} differs from its accepted reference in "
                        f"{qualified}: {before} -> {after}"
                    )
            elif before != after:
                raise ExportError(
                    f"{context} differs from its accepted reference in "
                    f"{qualified}: {before!r} -> {after!r}"
                )
    return maximum_error


def _verify_complete_model(
    expected: mujoco.MjModel,
    actual: mujoco.MjModel,
    *,
    skip: set[str],
    rtol: float = 1e-12,
    atol: float = 1e-12,
    context: str = "export",
) -> float:
    """Compare every stable public compiled-model value exposed by MuJoCo."""
    maximum_error = 0.0
    for name in sorted(set(dir(expected)) & set(dir(actual))):
        if name.startswith("_") or name in skip:
            continue
        try:
            before = getattr(expected, name)
            after = getattr(actual, name)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
            maximum_error = max(
                maximum_error,
                _assert_model_array_equal(
                    name,
                    before,
                    after,
                    rtol=rtol,
                    atol=atol,
                    context=context,
                ),
            )
        elif isinstance(
            before, (bytes, str, bool, int, float, np.generic)
        ) and isinstance(after, (bytes, str, bool, int, float, np.generic)):
            if isinstance(before, (float, np.floating)) or isinstance(
                after, (float, np.floating)
            ):
                error = abs(float(before) - float(after))
                maximum_error = max(maximum_error, error)
                if not np.isclose(float(before), float(after), rtol=rtol, atol=atol):
                    raise ExportError(
                        f"{context} differs from its accepted reference in "
                        f"{name}: {before} -> {after}"
                    )
            elif before != after:
                raise ExportError(
                    f"{context} differs from its accepted reference in "
                    f"{name}: {before!r} -> {after!r}"
                )
    for nested_name in ("opt", "stat"):
        maximum_error = max(
            maximum_error,
            _verify_public_container(
                getattr(expected, nested_name),
                getattr(actual, nested_name),
                prefix=nested_name,
                rtol=rtol,
                atol=atol,
                context=context,
            ),
        )
    return maximum_error


def _verify_compiled_model_equivalence(
    expected: mujoco.MjModel,
    actual: mujoco.MjModel,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> float:
    """Verify structure, dynamics, options and full body inertias."""
    dimensions = ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "neq", "nsensor")
    mismatched = [
        name
        for name in dimensions
        if int(getattr(expected, name)) != int(getattr(actual, name))
    ]
    if mismatched:
        details = ", ".join(
            f"{name}={getattr(expected, name)}->{getattr(actual, name)}"
            for name in mismatched
        )
        raise ExportError(f"{context} changed model structure: {details}")

    maximum_error = 0.0
    for name in (
        "dof_frictionloss",
        "dof_damping",
        "dof_armature",
        "body_mass",
        "body_ipos",
        "body_inertia",
    ):
        maximum_error = max(
            maximum_error,
            _assert_model_array_equal(
                name,
                getattr(expected, name),
                getattr(actual, name),
                rtol=rtol,
                atol=atol,
                context=context,
            ),
        )
    for body_id in range(expected.nbody):
        body_name = mujoco.mj_id2name(expected, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_label = body_name if body_name is not None else f"body[{body_id}]"
        maximum_error = max(
            maximum_error,
            _assert_model_array_equal(
                f"body_full_inertia[{body_label}]",
                body_full_inertia(expected, body_id),
                body_full_inertia(actual, body_id),
                rtol=rtol,
                atol=atol,
                context=context,
            ),
        )
    return max(
        maximum_error,
        _verify_complete_model(
            expected,
            actual,
            skip={"body_iquat"},
            rtol=rtol,
            atol=atol,
            context=context,
        ),
    )


def _verify_parameters_match_model(
    model: mujoco.MjModel,
    parameters: IdentifiedParameters,
    *,
    rtol: float,
    atol: float,
    context: str,
) -> float:
    """Confirm a model carries every accepted dynamic value."""
    maximum_error = 0.0
    requested = parameters.as_dict()
    for index, joint_name in enumerate(parameters.joints):
        dof = int(model.joint(joint_name).dofadr[0])
        for field_name, compiled in (
            ("frictionloss", "dof_frictionloss"),
            ("damping", "dof_damping"),
            ("armature", "dof_armature"),
        ):
            if field_name not in requested:
                continue
            maximum_error = max(
                maximum_error,
                _assert_model_array_equal(
                    f"{joint_name}.{field_name}",
                    np.array((requested[field_name][index],)),
                    np.array((getattr(model, compiled)[dof],)),
                    rtol=rtol,
                    atol=atol,
                    context=context,
                ),
            )
    for wanted in parameters.body_inertials:
        actual = BodyInertial.from_model(model, wanted.body)
        for field_name in ("mass", "ipos", "inertia"):
            maximum_error = max(
                maximum_error,
                _assert_model_array_equal(
                    f"{wanted.body}.{field_name}",
                    np.atleast_1d(getattr(wanted, field_name)),
                    np.atleast_1d(getattr(actual, field_name)),
                    rtol=rtol,
                    atol=atol,
                    context=context,
                ),
            )
    return maximum_error


def _behavior_probe(model: mujoco.MjModel, *, steps: int = 8) -> NDArray[np.float64]:
    """Deterministic, contact-free torque rollout used for reload parity."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    if model.nv:
        data.qvel[:] = np.linspace(0.025, 0.075, model.nv)
    if model.nu:
        control = np.linspace(0.1, 0.7, model.nu)
        for index in range(model.nu):
            if bool(model.actuator_ctrllimited[index]):
                lower, upper = model.actuator_ctrlrange[index]
                control[index] = np.clip(control[index], lower, upper)
        data.ctrl[:] = control
    mujoco.mj_forward(model, data)

    snapshots: list[NDArray[np.float64]] = []
    for _ in range(steps):
        snapshots.append(
            np.concatenate(
                (
                    data.qpos.copy(),
                    data.qvel.copy(),
                    data.qacc.copy(),
                    data.qfrc_bias.copy(),
                    data.actuator_force.copy(),
                )
            )
        )
        mujoco.mj_step(model, data)
    return np.concatenate(snapshots)


def _verify_behavior_equivalence(
    expected: mujoco.MjModel,
    actual: mujoco.MjModel,
    *,
    context: str,
) -> float:
    """Compare a short torque rollout and return its worst absolute error."""
    for name in ("nq", "nv", "nu"):
        if int(getattr(expected, name)) != int(getattr(actual, name)):
            raise ExportError(
                f"{context} cannot compare behavior with different {name}: "
                f"{getattr(expected, name)} -> {getattr(actual, name)}"
            )
    wanted = _behavior_probe(expected)
    observed = _behavior_probe(actual)
    error = float(np.max(np.abs(wanted - observed), initial=0.0))
    if not np.allclose(
        wanted,
        observed,
        rtol=BEHAVIOR_ROUNDTRIP_RTOL,
        atol=BEHAVIOR_ROUNDTRIP_ATOL,
    ):
        worst = int(np.argmax(np.abs(wanted - observed)))
        raise ExportError(
            f"{context} behavior diverged at probe element {worst}: "
            f"{wanted[worst]} -> {observed[worst]}"
        )
    return error


def _object_names(model: mujoco.MjModel, kind: str, count: int) -> tuple[str, ...]:
    accessor = getattr(model, kind)
    return tuple(
        accessor(index).name or f"<unnamed-{kind}-{index}>" for index in range(count)
    )


def _model_dimensions(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: int(getattr(model, name))
        for name in ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "neq", "nsensor")
    }


def _disable_flag_names(flags: int) -> list[str]:
    return [
        name
        for name, member in mujoco.mjtDisableBit.__members__.items()
        if name != "mjNDISABLE" and int(member) != 0 and flags & int(member)
    ]


def _consumer_projection_manifest(
    source: mujoco.MjModel,
    planning: mujoco.MjModel,
) -> dict[str, object]:
    source_joints = _object_names(source, "joint", source.njnt)
    planning_joints = _object_names(planning, "joint", planning.njnt)
    source_actuators = _object_names(source, "actuator", source.nu)
    planning_actuators = _object_names(planning, "actuator", planning.nu)
    return {
        "derivation": "fer_mujoco_sysid.model.build_hydrax_arm_model",
        "source_dimensions": _model_dimensions(source),
        "planning_dimensions": _model_dimensions(planning),
        "planning_joint_names": list(planning_joints),
        "removed_source_joint_names": [
            name for name in source_joints if name not in set(planning_joints)
        ],
        "planning_actuator_names": list(planning_actuators),
        "removed_source_actuator_names": [
            name for name in source_actuators if name not in set(planning_actuators)
        ],
        "source_equality_count": int(source.neq),
        "planning_equality_count": int(planning.neq),
        "planning_sensor_names": list(
            _object_names(planning, "sensor", planning.nsensor)
        ),
    }


def _model_conventions(model: mujoco.MjModel) -> dict[str, object]:
    flags = int(model.opt.disableflags)
    contact_disabled = bool(flags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
    gravity = tuple(float(value) for value in model.opt.gravity)
    gravity_enabled = not np.allclose(gravity, 0.0, rtol=0.0, atol=1e-12)
    return {
        "gravity_m_s2": list(gravity),
        "gravity_enabled": gravity_enabled,
        "contacts_enabled": not contact_disabled,
        "contact_disabled_via_mjDSBL_CONTACT": contact_disabled,
        "integrator": {
            "name": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "value": int(model.opt.integrator),
        },
        "timestep_s": float(model.opt.timestep),
        "disableflags": flags,
        "disabled_features": _disable_flag_names(flags),
    }


def _relative_asset_dir(asset_dir: Path, destination: Path) -> str:
    """Re-express an asset directory relative to where the model is written."""
    return os.path.relpath(asset_dir, destination.parent.resolve())


def _repoint_assets(xml: str, *, meshdir: str, texturedir: str) -> str:
    """Rewrite the compiler's asset directories in emitted MJCF text.

    Fails loudly rather than silently leaving an absolute path behind: an
    exported model that only loads on the machine that produced it is worse
    than one that refuses to be written.
    """
    for attribute, value in (("meshdir", meshdir), ("texturedir", texturedir)):
        xml, count = re.subn(
            rf'{attribute}="[^"]*"', f'{attribute}="{value}"', xml, count=1
        )
        if count != 1:
            raise ExportError(
                f"could not repoint {attribute} in the exported model; it "
                "would keep an absolute path and load only on this machine"
            )
    return xml


def _numbers(values: Iterable[float]) -> str:
    """Serialize binary64 values without changing them on XML reload."""
    return " ".join(format(float(value), ".17g") for value in values)


def spec_to_xml_exact_dynamics(spec: mujoco.MjSpec) -> str:
    """Serialize a spec while retaining full precision for physical fields.

    MuJoCo's canonical writer intentionally emits compact decimal numbers.
    That is useful for readable MJCF, but it can change Panda inertials enough
    to defeat a numerical system-identification closure test. Start with the
    canonical document and restore all dynamics-bearing body and joint values
    from the authoritative spec at round-trip-safe binary64 precision.
    """
    root = ET.fromstring(spec.to_xml())
    joints = {joint.name: joint for joint in spec.joints}

    for element in root.findall(".//body"):
        name = element.get("name")
        if name is None:
            continue
        body = spec.body(name)
        element.set("pos", _numbers(body.pos))
        element.set("quat", _numbers(body.quat))

        inertial = element.find("inertial")
        if inertial is None:
            continue
        inertial.set("mass", format(float(body.mass), ".17g"))
        inertial.set("pos", _numbers(body.ipos))
        if np.isfinite(body.fullinertia[0]):
            inertial.set("fullinertia", _numbers(body.fullinertia))
            inertial.attrib.pop("diaginertia", None)
            inertial.attrib.pop("quat", None)
        else:
            inertial.set("diaginertia", _numbers(body.inertia))
            inertial.set("quat", _numbers(body.iquat))
            inertial.attrib.pop("fullinertia", None)

    for element in root.findall(".//joint"):
        name = element.get("name")
        joint = joints.get(name)
        if joint is None:
            continue
        element.set("pos", _numbers(joint.pos))
        element.set("axis", _numbers(joint.axis))
        element.set("range", _numbers(joint.range))
        element.set("armature", format(float(joint.armature), ".17g"))
        element.set("damping", format(float(joint.damping[0]), ".17g"))
        element.set("frictionloss", format(float(joint.frictionloss), ".17g"))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def export_identified_model(
    destination: str | Path,
    parameters: IdentifiedParameters,
    *,
    source_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ExportCheck:
    """Legacy wrapper that writes only a full-source parameter patch.

    This does not validate the Hydrax consumer planning derivation. New
    production callers must use :func:`export_consumer_model`.
    """
    if source_path is None:
        source_path = resolve_model_paths().require().hydrax
    source_path = Path(source_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(str(source_path))
    apply_parameters(spec, parameters)
    # Compiling resolves assets against the *source* directory, so it must
    # happen before the paths are rewritten. Compiling here rather than only
    # after writing means a spec that cannot build never reaches disk.
    spec.compile()

    # The written file lives elsewhere than the meshes it references, so its
    # asset directories have to be repointed. Both `compile` and `to_xml`
    # resolve them against the *source* directory, so the rewrite happens on
    # the emitted text: absolute while MuJoCo is reading, relative once
    # written. Relative because an absolute path would pin the exported model
    # to one machine; this way it survives the workspace being moved or
    # cloned. Consumer integration rewrites these into the consumer's own
    # asset tree; until then it keeps the file loadable where it is written.
    absolute_mesh = (source_path.parent / spec.meshdir).resolve()
    absolute_texture = (source_path.parent / spec.texturedir).resolve()
    spec.meshdir = str(absolute_mesh)
    spec.texturedir = str(absolute_texture)
    meshdir = _relative_asset_dir(absolute_mesh, destination)
    texturedir = _relative_asset_dir(absolute_texture, destination)
    xml = _repoint_assets(
        spec_to_xml_exact_dynamics(spec),
        meshdir=meshdir,
        texturedir=texturedir,
    )
    destination.write_text(xml, encoding="utf-8")

    check = verify_export(source_path, destination, parameters)
    if manifest_path is not None:
        write_json(
            manifest_path,
            {
                "format": FULL_SOURCE_PATCH_FORMAT,
                "artifact_kind": "full-source-parameter-patch",
                "consumer_ready": False,
                "validation_scope": (
                    "ordinary full-model reload and dynamic-field whitelist; "
                    "consumer planning derivation not validated"
                ),
                "source_model": {
                    "path": str(source_path),
                    "sha256": check.source_sha256,
                },
                "exported_model": {
                    "path": str(destination),
                    "sha256": sha256_file(destination),
                },
                "exported_fields": sorted(parameters.as_dict()),
                # Relative to the exported model, so the file travels with
                # the workspace rather than with one machine.
                "meshdir": meshdir,
                "parameters": parameters.as_dict(),
                "joints": list(parameters.joints),
            },
        )
    return check


def export_full_source_patched_model(
    destination: str | Path,
    parameters: IdentifiedParameters,
    *,
    source_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ExportCheck:
    """Explicit name for the secondary, non-consumer-validated patch."""
    return export_identified_model(
        destination,
        parameters,
        source_path=source_path,
        manifest_path=manifest_path,
    )


def _validate_accepted_fitted_model(
    accepted_fitted_model: mujoco.MjModel,
    parameters: IdentifiedParameters,
) -> None:
    parameters.validate()
    if not np.allclose(accepted_fitted_model.opt.gravity, 0.0, rtol=0.0, atol=1e-12):
        raise ExportError(
            "accepted fitted model must use the gravity-free Franka telemetry "
            "convention"
        )
    fit_contact_disabled = bool(
        int(accepted_fitted_model.opt.disableflags)
        & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    )
    if not fit_contact_disabled:
        raise ExportError("accepted fitted model must be contact-free")
    if int(accepted_fitted_model.opt.integrator) != int(
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ):
        raise ExportError("accepted fitted model must use mjINT_IMPLICITFAST")
    if (
        accepted_fitted_model.nq,
        accepted_fitted_model.nv,
        accepted_fitted_model.nu,
    ) != (7, 7, 7):
        raise ExportError(
            "accepted fitted model must be the seven-DOF Panda projection"
        )
    _verify_parameters_match_model(
        accepted_fitted_model,
        parameters,
        rtol=1e-9,
        atol=5e-12,
        context="accepted fitted model",
    )


def verify_consumer_model_export(
    accepted_fitted_model: mujoco.MjModel,
    exported_path: str | Path,
    parameters: IdentifiedParameters,
    *,
    source_path: str | Path,
    manifest_path: str | Path | None = None,
) -> ConsumerModelExportCheck:
    """Verify the full Panda reload and exact Hydrax planning derivation.

    The fitted reference is gravity-free because the recorded Franka effort
    is on top of internal gravity compensation. The delivered full Panda and
    Hydrax planning model remain gravity-enabled: MPPI inverse dynamics
    includes gravity, and the real LFC removes that contribution before
    commanding the robot.
    """
    source_path = Path(source_path)
    exported_path = Path(exported_path)
    _validate_accepted_fitted_model(accepted_fitted_model, parameters)

    full_check = verify_export(source_path, exported_path, parameters)
    source_model = mujoco.MjModel.from_xml_path(str(source_path))
    full_model = mujoco.MjModel.from_xml_path(str(exported_path))
    if np.allclose(source_model.opt.gravity, 0.0, rtol=0.0, atol=1e-12):
        raise ExportError("consumer source model must have gravity enabled")
    if np.allclose(full_model.opt.gravity, 0.0, rtol=0.0, atol=1e-12):
        raise ExportError("exported full consumer model lost gravity")

    expected_spec = build_hydrax_arm_spec(source_path)
    apply_parameters(expected_spec, parameters)
    expected_planning = expected_spec.compile()
    planning = build_hydrax_arm_model(exported_path)
    planning_error = _verify_compiled_model_equivalence(
        expected_planning,
        planning,
        rtol=XML_ROUNDTRIP_RTOL,
        atol=XML_ROUNDTRIP_ATOL,
        context="Hydrax planning derivation",
    )
    _verify_parameters_match_model(
        planning,
        parameters,
        rtol=XML_ROUNDTRIP_RTOL,
        atol=XML_ROUNDTRIP_ATOL,
        context="Hydrax planning model",
    )
    if np.allclose(planning.opt.gravity, 0.0, rtol=0.0, atol=1e-12):
        raise ExportError("Hydrax planning derivation must retain source gravity")

    planning_behavior_error = _verify_behavior_equivalence(
        expected_planning,
        planning,
        context="gravity-enabled Hydrax planning reload",
    )

    fit_convention_projection = copy.copy(planning)
    fit_convention_projection.opt.gravity[:] = accepted_fitted_model.opt.gravity
    fit_convention_behavior_error = _verify_behavior_equivalence(
        accepted_fitted_model,
        fit_convention_projection,
        context="gravity-normalized accepted-fit projection",
    )

    projection = _consumer_projection_manifest(source_model, planning)
    conventions = {
        "accepted_fit": _model_conventions(accepted_fitted_model),
        "full_consumer_model": _model_conventions(full_model),
        "hydrax_planning_model": _model_conventions(planning),
        "torque_semantics": {
            "mppi_inverse_dynamics_includes_gravity": True,
            "real_lfc_remove_gravity_compensation_effort": True,
            "simulation_remove_gravity_compensation_effort": False,
            "fit_effort_channel": (
                "commanded joint effort on top of Franka internal gravity compensation"
            ),
        },
    }
    return ConsumerModelExportCheck(
        source_path=full_check.source_path,
        exported_path=full_check.exported_path,
        source_sha256=full_check.source_sha256,
        changed=full_check.changed,
        body_changes=full_check.body_changes,
        roundtrip_error=full_check.roundtrip_error,
        invariant_fields=full_check.invariant_fields,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        exported_sha256=sha256_file(exported_path),
        projection=projection,
        simulation_conventions=conventions,
        planning_compiled_error=planning_error,
        planning_behavior_error=planning_behavior_error,
        fit_convention_behavior_error=fit_convention_behavior_error,
    )


def export_consumer_model(
    destination: str | Path,
    parameters: IdentifiedParameters,
    *,
    accepted_fitted_model: mujoco.MjModel,
    source_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ConsumerModelExportCheck:
    """Write the primary gravity-enabled consumer model and prove both paths."""
    if source_path is None:
        source_path = resolve_model_paths().require().hydrax
    source_path = Path(source_path)
    destination = Path(destination)
    if manifest_path is None:
        manifest_path = destination.with_suffix(".json")
    manifest_path = Path(manifest_path)

    _validate_accepted_fitted_model(accepted_fitted_model, parameters)
    export_full_source_patched_model(
        destination,
        parameters,
        source_path=source_path,
    )
    check = verify_consumer_model_export(
        accepted_fitted_model,
        destination,
        parameters,
        source_path=source_path,
        manifest_path=manifest_path,
    )
    write_json(
        manifest_path,
        {
            "format": CONSUMER_MODEL_FORMAT,
            "artifact_kind": "gravity-enabled-full-panda-consumer-model",
            "consumer_ready": True,
            "source_model": {
                "path": str(source_path),
                "sha256": check.source_sha256,
            },
            "consumer_model": {
                "path": str(destination),
                "sha256": check.exported_sha256,
            },
            "exported_fields": sorted(parameters.as_dict()),
            "parameters": parameters.as_dict(),
            "joints": list(parameters.joints),
            "projection": check.projection,
            "simulation_conventions": check.simulation_conventions,
            "verification": {
                "ordinary_full_mjmodel_reload": True,
                "hydrax_planning_derivation_reload": True,
                "planning_compiled_max_abs_error": (check.planning_compiled_error),
                "planning_behavior_max_abs_error": (check.planning_behavior_error),
                "gravity_normalized_fit_behavior_max_abs_error": (
                    check.fit_convention_behavior_error
                ),
                "parameter_xml_roundtrip_max_abs_error": (check.roundtrip_error),
            },
        },
    )
    return check


def rollout_comparison(
    model_paths: Sequence[str | Path],
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    timestep: float,
) -> list[NDArray[np.float64]]:  # pragma: no cover - diagnostic helper
    """Open-loop joint trajectories of several models under the same torque.

    Used to show that an exported file behaves like the model it was
    exported from: same input, same motion.
    """
    trajectories = []
    for path in model_paths:
        model = mujoco.MjModel.from_xml_path(str(path))
        model.opt.timestep = timestep
        data = mujoco.MjData(model)
        data.qpos[:7] = q_rad[0]
        data.qvel[:7] = dq_rad_s[0]
        states = np.zeros((len(tau_Nm), 7))
        for step in range(len(tau_Nm)):
            states[step] = data.qpos[:7]
            data.ctrl[:7] = tau_Nm[step]
            mujoco.mj_step(model, data)
        trajectories.append(states)
    return trajectories
