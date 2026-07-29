"""Write identified parameters back into a MuJoCo model, and prove it.

An identified model is only useful if it is the nominal model *plus the
parameters that were identified* and nothing else. This module therefore does
two things that are equally important: it applies a whitelist of physical
fields to the nominal MJCF, and it checks the written file against the
nominal one to confirm nothing outside that whitelist moved.

The whitelist is deliberately narrow. Joint friction, damping and armature
are what the friction and armature stages fit; link inertials join the list
once the inertial stage is released. Kinematics, joint ranges, actuator
limits, geometry and naming are never written — a change there is a different
robot, not a better estimate of this one.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.fitting import set_hinge_damping
from fer_mujoco_sysid.io import sha256_file, write_json
from fer_mujoco_sysid.model import HYDRAX_ARM_JOINT_NAMES, resolve_model_paths

#: The only compiled quantities an export is allowed to change.
EXPORTABLE_FIELDS = ("dof_frictionloss", "dof_damping", "dof_armature")

#: Everything else that must be bit-identical between nominal and identified.
#: Not exhaustive over the whole model, but it covers every quantity that
#: would change the robot rather than its dynamics estimate.
INVARIANT_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
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
class IdentifiedParameters:
    """Per-joint values to write, in :data:`HYDRAX_ARM_JOINT_NAMES` order."""

    frictionloss: tuple[float, ...] | None = None
    damping: tuple[float, ...] | None = None
    armature: tuple[float, ...] | None = None
    joints: tuple[str, ...] = HYDRAX_ARM_JOINT_NAMES

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> IdentifiedParameters:
        """Read the subset of a fit result that is exportable."""

        def column(name: str) -> tuple[float, ...] | None:
            raw = values.get(name)
            if raw is None:
                return None
            return tuple(float(value) for value in raw)  # type: ignore[union-attr]

        return cls(
            frictionloss=column("frictionloss"),
            damping=column("damping"),
            armature=column("armature"),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            name: list(values)
            for name, values in (
                ("frictionloss", self.frictionloss),
                ("damping", self.damping),
                ("armature", self.armature),
            )
            if values is not None
        }

    def validate(self) -> None:
        for name, values in self.as_dict().items():
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
    return spec


@dataclass(frozen=True)
class ExportCheck:
    """What an exported model changed, and what it left alone."""

    source_path: Path
    exported_path: Path
    source_sha256: str
    changed: dict[str, dict[str, float]] = field(default_factory=dict)
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
            for name, (before, after) in fields.items():  # type: ignore[misc]
                lines.append(
                    f"| {joint} | {name} | {before:.4f} | {after:.4f} | "
                    f"{after - before:+.4f} |"
                )
        return "\n".join(lines)


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
    nominal = mujoco.MjModel.from_xml_path(str(source_path))
    identified = mujoco.MjModel.from_xml_path(str(exported_path))

    if nominal.njnt != identified.njnt or nominal.nbody != identified.nbody:
        raise ExportError(
            f"{exported_path} has a different structure than {source_path} "
            f"({identified.njnt} joints / {identified.nbody} bodies against "
            f"{nominal.njnt} / {nominal.nbody})"
        )

    for name in INVARIANT_FIELDS:
        before = np.asarray(getattr(nominal, name), dtype=np.float64)
        after = np.asarray(getattr(identified, name), dtype=np.float64)
        # XML serialization keeps about seven significant digits; anything
        # beyond that is an edit, not a round-trip.
        if not np.allclose(before, after, rtol=1e-5, atol=1e-9):
            worst = int(np.argmax(np.abs(before.ravel() - after.ravel())))
            raise ExportError(
                f"export changed {name}, which is not exportable "
                f"(worst element {worst}: {before.ravel()[worst]} -> "
                f"{after.ravel()[worst]})"
            )

    changed: dict[str, dict[str, float]] = {}
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

    return ExportCheck(
        source_path=source_path,
        exported_path=exported_path,
        source_sha256=sha256_file(source_path),
        changed=changed,
        roundtrip_error=roundtrip,
    )


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


def export_identified_model(
    destination: str | Path,
    parameters: IdentifiedParameters,
    *,
    source_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ExportCheck:
    """Write the identified model next to a manifest, and verify it.

    Returns the verification, so a caller cannot obtain an exported model
    without the evidence that it is sound.
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
    xml = _repoint_assets(spec.to_xml(), meshdir=meshdir, texturedir=texturedir)
    destination.write_text(xml, encoding="utf-8")

    check = verify_export(source_path, destination, parameters)
    if manifest_path is not None:
        write_json(
            manifest_path,
            {
                "format": "fer-mujoco-sysid/identified-model@1",
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


def rollout_comparison(
    model_paths: Sequence[str | Path],
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    timestep: float,
) -> list[NDArray[np.float64]]:  # pragma: no cover - used by report.py
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
