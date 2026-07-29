"""Project-owned adapter around the public ``mujoco.sysid`` toolbox.

P2 of the implementation plan: the friction parameter group
(``frictionloss`` + ``damping``) and the in-memory entry points exercised by
the synthetic-recovery gates. Only public ``mujoco.sysid`` names are used.

Note on spec ownership: ``mujoco.sysid`` applies parameter modifiers to the
spec held by a ``ModelSequences`` in place on every residual evaluation. The
modifiers built here write absolute values, so repeated application is
idempotent, but a spec handed to :func:`measurement_sequences` must be treated
as owned by the fit from then on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mujoco import sysid
from numpy.typing import NDArray

from fer_mujoco_sysid.model import HYDRAX_ARM_JOINT_NAMES

# Synthetic-stage box bounds. The recorded 2026-07-22 pregrasp run proves
# breakaway torques of at least 1.5 Nm (joint 4), so the friction upper bound
# keeps a factor-two margin above the largest observed value. Bounds for the
# real campaign are fixed at the P7 fit review.
FRICTIONLOSS_BOUNDS_NM: tuple[float, float] = (0.0, 3.0)
DAMPING_BOUNDS_NM_S: tuple[float, float] = (0.0, 8.0)
# Reflected rotor inertia; the Menagerie nominal is 0.1 for every joint.
ARMATURE_BOUNDS_KG_M2: tuple[float, float] = (0.0, 2.0)

# Plan P2 thresholds: a parameter block is retained only if the scaled
# Jacobian conditioning ratio stays above this, and pairs correlated beyond
# the correlation limit are frozen, grouped, or reparameterized.
CONDITIONING_RATIO_MINIMUM: float = 1e-6
CORRELATION_FREEZE_LIMIT: float = 0.98

# Inertial convention: link0 is fixed to the world (parity-checked, never
# estimated). hand/left_finger/right_finger are rigid downstream of joint 7,
# so only the total tool composite is observable; its parameters live on
# link7 and the hand/finger inertials stay at nominal.
MOVING_LINK_BODIES: tuple[str, ...] = tuple(f"link{i}" for i in range(1, 8))
TOOL_COMPOSITE_BODIES: tuple[str, ...] = (
    "link7",
    "hand",
    "left_finger",
    "right_finger",
)


def set_hinge_damping(joint: mujoco.MjsJoint, value: float) -> None:
    """Write a hinge joint's damping into the per-DOF spec array.

    ``MjsJoint.damping`` holds one entry per potential DOF; a hinge uses only
    the first. ``frictionloss`` remains a scalar field.
    """
    damping = np.asarray(joint.damping, dtype=np.float64).copy()
    damping[0] = value
    joint.damping = damping


def friction_parameters(
    model: mujoco.MjModel,
    *,
    joints: Sequence[str] = HYDRAX_ARM_JOINT_NAMES,
    include_frictionloss: bool = True,
    include_damping: bool = True,
    frictionloss_bounds: tuple[float, float] = FRICTIONLOSS_BOUNDS_NM,
    damping_bounds: tuple[float, float] = DAMPING_BOUNDS_NM_S,
) -> sysid.ParameterDict:
    """Per-joint ``frictionloss``/``damping`` parameters, nominal = *model*.

    Fitting starts from the parameter ``value`` (initialized to the nominal);
    callers whose nominal sits on a bound (frictionloss is 0.0 in the nominal
    FER model) should call ``move_off_bounds()`` before optimizing.
    """
    if not include_frictionloss and not include_damping:
        raise ValueError("at least one of frictionloss/damping is required")

    parameters = sysid.ParameterDict()
    for joint_name in joints:
        dof = int(model.joint(joint_name).dofadr[0])
        if include_frictionloss:
            parameters.add(
                sysid.Parameter(
                    f"{joint_name}_frictionloss",
                    nominal=float(model.dof_frictionloss[dof]),
                    min_value=frictionloss_bounds[0],
                    max_value=frictionloss_bounds[1],
                    modifier=lambda spec, param, name=joint_name: setattr(
                        spec.joint(name), "frictionloss", float(param.value[0])
                    ),
                )
            )
        if include_damping:
            parameters.add(
                sysid.Parameter(
                    f"{joint_name}_damping",
                    nominal=float(model.dof_damping[dof]),
                    min_value=damping_bounds[0],
                    max_value=damping_bounds[1],
                    modifier=lambda spec, param, name=joint_name: set_hinge_damping(
                        spec.joint(name), float(param.value[0])
                    ),
                )
            )
    return parameters


def armature_parameters(
    model: mujoco.MjModel,
    *,
    joints: Sequence[str] = HYDRAX_ARM_JOINT_NAMES,
    bounds: tuple[float, float] = ARMATURE_BOUNDS_KG_M2,
) -> sysid.ParameterDict:
    """Per-joint ``armature`` parameters, nominal = compiled *model* values.

    Plan ordering rule: armature is fitted only after the friction stage, on
    dynamically exciting data (slow reversals barely accelerate, so armature
    is nearly invisible there).
    """
    parameters = sysid.ParameterDict()
    for joint_name in joints:
        dof = int(model.joint(joint_name).dofadr[0])
        parameters.add(
            sysid.Parameter(
                f"{joint_name}_armature",
                nominal=float(model.dof_armature[dof]),
                min_value=bounds[0],
                max_value=bounds[1],
                modifier=lambda spec, param, name=joint_name: setattr(
                    spec.joint(name), "armature", float(param.value[0])
                ),
            )
        )
    return parameters


def inertial_parameters(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    *,
    bodies: Sequence[str] | None = None,
    inertia_type: sysid.InertiaType = sysid.InertiaType.Pseudo,
) -> sysid.ParameterDict:
    """Explicit per-body inertia parameters for diagnostics and research.

    ``Pseudo`` (default) is the physically consistent parameterization —
    every candidate inertia is realizable by construction. Callers must name
    the bodies explicitly: silently constructing the seven-link, 70-scalar
    block would make an unobservable fit look like a supported production
    workflow. Prefer :func:`cad_prior_inertial_parameters` for released fits.

    *spec* is only read here. Parameters for ``link7`` represent the whole
    rigid tool composite (see ``TOOL_COMPOSITE_BODIES``); estimating
    hand/finger bodies alongside ``link7`` is structurally unidentifiable and
    must be rejected by a :func:`conditioning_report` check.
    """
    if bodies is None:
        raise ValueError(
            "the blind seven-link inertial block is not releasable; pass an "
            "explicit body subset for diagnostics, or use "
            "cad_prior_inertial_parameters() for a constrained fit"
        )
    if not bodies:
        raise ValueError("at least one explicit inertial body is required")
    parameters = sysid.ParameterDict()
    for body_name in bodies:
        modifier = None
        if inertia_type == sysid.InertiaType.Mass:
            # MuJoCo 3.10 workaround: the default Mass modifier assigns the
            # 1-element parameter array to the scalar MjsBody.mass setter.
            def modifier(spec, param, name=body_name):
                spec.body(name).mass = float(param.value[0])

        parameters.add(
            sysid.body_inertia_param(
                spec,
                model,
                body_name,
                inertia_type=inertia_type,
                modifier=modifier,
            )
        )
    return parameters


def body_full_inertia(
    model: mujoco.MjModel, body_name: str | int
) -> NDArray[np.float64]:
    """Return a body's full inertia tensor in its body frame.

    MuJoCo stores the compiled tensor as principal moments plus the
    ``body_iquat`` rotation. Export and CAD-prior corrections use the six
    unambiguous body-frame entries ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``;
    unlike an eigen-quaternion, this representation has no sign ambiguity.
    """
    body = model.body(body_name)
    rotation_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, np.asarray(body.iquat, dtype=np.float64))
    rotation = rotation_flat.reshape(3, 3)
    tensor = rotation @ np.diag(np.asarray(body.inertia, dtype=np.float64)) @ rotation.T
    tensor = 0.5 * (tensor + tensor.T)
    return np.array(
        (
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 2],
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BodyInertialCorrection:
    """A small, CAD-prior correction group for one explicitly chosen body.

    Each enabled component is scalar and tightly bounded around the compiled
    nominal model. ``inertia_scale`` scales the complete body-frame tensor,
    preserving its principal axes and triangle inequalities. This deliberately
    does not expose the ten unconstrained raw inertial coordinates.

    Selection is only a candidate declaration. Call
    :func:`select_observable_subset` on the actual protocol sequences before
    fitting; corrections that the data do not distinguish stay frozen at
    their CAD values.
    """

    body: str
    estimate_mass: bool = True
    com_axes: tuple[int, ...] = ()
    estimate_inertia_scale: bool = False
    mass_scale_bounds: tuple[float, float] = (0.7, 1.3)
    com_offset_bound_m: float = 0.02
    inertia_scale_bounds: tuple[float, float] = (0.7, 1.3)

    def validate(self) -> None:
        if not self.body:
            raise ValueError("body correction requires a body name")
        if not (self.estimate_mass or self.com_axes or self.estimate_inertia_scale):
            raise ValueError(
                f"{self.body}: enable mass, at least one COM axis, or inertia scale"
            )
        if len(set(self.com_axes)) != len(self.com_axes) or any(
            axis not in (0, 1, 2) for axis in self.com_axes
        ):
            raise ValueError(
                f"{self.body}: com_axes must be unique indices chosen from 0, 1, 2"
            )
        for label, bounds in (
            ("mass_scale_bounds", self.mass_scale_bounds),
            ("inertia_scale_bounds", self.inertia_scale_bounds),
        ):
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] <= 0.0
                or not bounds[0] < 1.0 < bounds[1]
            ):
                raise ValueError(
                    f"{self.body}: {label} must be finite, positive, and "
                    "strictly contain the CAD scale 1.0"
                )
        if not np.isfinite(self.com_offset_bound_m) or self.com_offset_bound_m <= 0.0:
            raise ValueError(
                f"{self.body}: com_offset_bound_m must be finite and positive"
            )


def cad_prior_inertial_parameters(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    corrections: Sequence[BodyInertialCorrection],
) -> sysid.ParameterDict:
    """Build a low-dimensional, selected-body correction block.

    Modifiers write absolute values derived from the supplied compiled CAD
    model, so repeated residual evaluations are idempotent. The *spec*
    argument is checked for the selected bodies but otherwise left untouched.
    """
    if not corrections:
        raise ValueError("at least one body correction is required")
    names = [correction.body for correction in corrections]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "duplicate body corrections are ambiguous: " + ", ".join(duplicates)
        )

    parameters = sysid.ParameterDict()
    axis_names = ("x", "y", "z")
    for correction in corrections:
        correction.validate()
        # Resolve both views now so a misspelled body fails before fitting.
        spec.body(correction.body)
        body = model.body(correction.body)
        nominal_mass = float(body.mass[0])
        nominal_com = np.asarray(body.ipos, dtype=np.float64).copy()
        nominal_inertia = body_full_inertia(model, correction.body)

        if correction.estimate_mass:
            parameters.add(
                sysid.Parameter(
                    f"{correction.body}_mass_scale",
                    nominal=1.0,
                    min_value=correction.mass_scale_bounds[0],
                    max_value=correction.mass_scale_bounds[1],
                    modifier=_body_mass_scale_modifier(correction.body, nominal_mass),
                )
            )

        for axis in correction.com_axes:
            parameters.add(
                sysid.Parameter(
                    f"{correction.body}_com_{axis_names[axis]}_offset_m",
                    nominal=0.0,
                    min_value=-correction.com_offset_bound_m,
                    max_value=correction.com_offset_bound_m,
                    modifier=_body_com_offset_modifier(
                        correction.body, nominal_com, axis
                    ),
                )
            )

        if correction.estimate_inertia_scale:
            parameters.add(
                sysid.Parameter(
                    f"{correction.body}_inertia_scale",
                    nominal=1.0,
                    min_value=correction.inertia_scale_bounds[0],
                    max_value=correction.inertia_scale_bounds[1],
                    modifier=_body_inertia_scale_modifier(
                        correction.body, nominal_inertia
                    ),
                )
            )
    return parameters


def _body_mass_scale_modifier(body_name: str, nominal_mass: float) -> Any:
    def modifier(target: mujoco.MjSpec, parameter: sysid.Parameter) -> None:
        target.body(body_name).mass = nominal_mass * float(parameter.value[0])

    return modifier


def _body_com_offset_modifier(
    body_name: str, nominal_com: NDArray[np.float64], axis: int
) -> Any:
    def modifier(target: mujoco.MjSpec, parameter: sysid.Parameter) -> None:
        _set_body_com_offset(
            target.body(body_name),
            nominal_com,
            axis,
            float(parameter.value[0]),
        )

    return modifier


def _body_inertia_scale_modifier(
    body_name: str, nominal_inertia: NDArray[np.float64]
) -> Any:
    def modifier(target: mujoco.MjSpec, parameter: sysid.Parameter) -> None:
        _set_body_full_inertia(
            target.body(body_name),
            nominal_inertia * float(parameter.value[0]),
        )

    return modifier


def _set_body_com_offset(
    body: mujoco.MjsBody,
    nominal_com: NDArray[np.float64],
    axis: int,
    offset_m: float,
) -> None:
    """Write one COM offset while preserving the other CAD coordinates."""
    ipos = np.asarray(body.ipos, dtype=np.float64).copy()
    ipos[axis] = nominal_com[axis] + offset_m
    body.ipos = ipos


def _set_body_full_inertia(
    body: mujoco.MjsBody, full_inertia: NDArray[np.float64]
) -> None:
    """Select MuJoCo's full-tensor input representation."""
    body.inertia[:] = 0.0
    body.iquat[:] = np.nan
    body.fullinertia[:] = np.asarray(full_inertia, dtype=np.float64)


def combine_parameters(*groups: sysid.ParameterDict) -> sysid.ParameterDict:
    """Deep-copy and combine parameter groups, rejecting duplicate names."""
    if not groups:
        raise ValueError("at least one parameter group is required")
    combined = sysid.ParameterDict()
    for group in groups:
        combined.update(group.copy())
    return combined


@dataclass(frozen=True)
class MeasuredRun:
    """One recorded playback: applied torque and measured joint states.

    ``control`` rows follow the model actuator order; ``measured`` rows follow
    the model sensor layout (``{joint}_pos`` then ``{joint}_vel`` for a spec
    built with ``joint_state_sensors=True``). Times are seconds on one clock.

    Row convention (mirrors ``mujoco.rollout``): ``measured[k]`` holds the
    pre-integration sensor state from which ``control[k]`` is applied. MuJoCo
    exposes that sensor row after the step and the toolbox stamps it at
    ``(k + 1) * dt``, although its q/dq values are the state at ``k * dt``.
    Controller feedback and same-row effort follow the same numeric ordering.
    Predicted and measured rows therefore align index-for-index.
    """

    label: str
    qpos0: NDArray[np.float64]
    qvel0: NDArray[np.float64]
    control_times: NDArray[np.float64]
    control: NDArray[np.float64]
    measured_times: NDArray[np.float64]
    measured: NDArray[np.float64]


def _joint_state_columns(
    model: mujoco.MjModel,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Map jointpos/jointvel sensor columns onto qpos/qvel addresses."""
    qpos_cols = np.full(model.nq, -1, dtype=np.intp)
    qvel_cols = np.full(model.nv, -1, dtype=np.intp)
    for index in range(model.nsensor):
        sensor = model.sensor(index)
        sensor_type = int(sensor.type[0])
        if sensor_type == int(mujoco.mjtSensor.mjSENS_JOINTPOS):
            joint = model.joint(int(sensor.objid[0]))
            qpos_cols[int(joint.qposadr[0])] = int(sensor.adr[0])
        elif sensor_type == int(mujoco.mjtSensor.mjSENS_JOINTVEL):
            joint = model.joint(int(sensor.objid[0]))
            qvel_cols[int(joint.dofadr[0])] = int(sensor.adr[0])
    if (qpos_cols < 0).any() or (qvel_cols < 0).any():
        raise ValueError(
            "windowing requires one jointpos and one jointvel sensor per "
            "joint (build the spec with joint_state_sensors=True)"
        )
    return qpos_cols, qvel_cols


def measurement_sequences(
    spec: mujoco.MjSpec,
    runs: Sequence[MeasuredRun],
    *,
    label: str = "fer_arm",
    window_s: float | None = None,
    max_windows_per_run: int | None = None,
    measurement_stride: int = 1,
) -> sysid.ModelSequences:
    """Bundle measured runs with the identification spec for fitting.

    With ``window_s`` every run is split into short rollout windows, each
    re-initialized from the measured joint state at the window boundary
    (plan rule: logged ``q``/``dq`` drive short-window forward-rollout
    residuals — long open-loop arm rollouts diverge and break the fit).
    Window times are re-zeroed because the toolbox rolls out from time 0.
    """
    if not runs:
        raise ValueError("at least one measured run is required")
    if max_windows_per_run is not None and max_windows_per_run < 1:
        raise ValueError("max_windows_per_run must be positive")
    if measurement_stride < 1:
        raise ValueError("measurement_stride must be positive")
    model = spec.compile()

    labels: list[str] = []
    initial_states: list[NDArray[np.float64]] = []
    control_series: list[sysid.TimeSeries] = []
    measured_series: list[sysid.TimeSeries] = []
    for run in runs:
        control = np.asarray(run.control, dtype=np.float64)
        measured = np.asarray(run.measured, dtype=np.float64)
        control_times = np.asarray(run.control_times, dtype=np.float64)
        measured_times = np.asarray(run.measured_times, dtype=np.float64)
        if control.ndim != 2 or control.shape[1] != model.nu:
            raise ValueError(
                f"run {run.label}: control shape {control.shape} does not "
                f"match ({len(control_times)}, nu={model.nu})"
            )
        if measured.ndim != 2 or measured.shape[1] != model.nsensordata:
            raise ValueError(
                f"run {run.label}: measured shape {measured.shape} does not "
                f"match ({len(measured_times)}, "
                f"nsensordata={model.nsensordata})"
            )

        if window_s is None:
            bounds = [(0, len(control_times))]
        else:
            steps = max(int(round(window_s / model.opt.timestep)), 2)
            all_bounds = [
                (start, min(start + steps, len(control_times)))
                for start in range(0, len(control_times), steps)
            ]
            bounds = _representative_bounds(all_bounds, maximum=max_windows_per_run)
            qpos_cols, qvel_cols = _joint_state_columns(model)

        for start, stop in bounds:
            if stop - start < 2:
                continue
            if window_s is None:
                qpos0 = np.asarray(run.qpos0, dtype=np.float64)
                qvel0 = np.asarray(run.qvel0, dtype=np.float64)
            else:
                # measured[start] is the pre-integration sensor state from
                # which control[start] is applied (see MeasuredRun).
                qpos0 = measured[start, qpos_cols]
                qvel0 = measured[start, qvel_cols]
            window_samples = stop - start
            # These runs have already been resampled onto the model grid.
            # Reconstruct local times from integer row indices instead of
            # subtracting large absolute timestamps: the latter can push the
            # endpoint above an exact timestep multiple and trigger an extra
            # interpolated control row inside mujoco.sysid.
            window_control_times = _resampling_safe_control_times(
                window_samples,
                float(model.opt.timestep),
            )
            measured_indices = np.arange(
                start,
                stop,
                measurement_stride,
                dtype=np.int64,
            )
            window_measured_times = (
                measured_indices - start + 1
            ) * float(model.opt.timestep)
            labels.append(f"{run.label}[{start}:{stop}]")
            initial_states.append(sysid.create_initial_state(model, qpos0, qvel0))
            control_series.append(
                sysid.TimeSeries(
                    window_control_times,
                    control[start:stop],
                )
            )
            measured_series.append(
                sysid.TimeSeries.from_names(
                    window_measured_times,
                    measured[start:stop:measurement_stride],
                    model,
                )
            )

    return sysid.ModelSequences(
        label,
        spec,
        labels,
        initial_states,
        control_series,
        measured_series,
    )


def _resampling_safe_control_times(
    samples: int,
    timestep: float,
) -> NDArray[np.float64]:
    """Return a uniform grid that the SysID resampler cannot lengthen.

    ``mujoco.sysid`` reconstructs a control grid with
    ``ceil(duration / timestep)``.  Re-zeroing timestamps by subtracting a
    large recording time can leave an endpoint a few ulps above the intended
    integer multiple.  The resampler then invents one extra row and linearly
    distorts every control in the window.

    Moving only the endpoint one representable float toward zero preserves the
    physical sample grid while making the intended row count unambiguous.
    ``linspace`` is used here because the toolbox uses the same construction;
    resampling therefore leaves the control values bit-for-bit unchanged.
    """
    if samples < 2:
        raise ValueError("a control window requires at least two samples")
    endpoint = np.nextafter((samples - 1) * timestep, 0.0)
    return np.linspace(0.0, endpoint, samples, dtype=np.float64)


def _representative_bounds(
    bounds: Sequence[tuple[int, int]],
    *,
    maximum: int | None,
) -> list[tuple[int, int]]:
    """Choose evenly distributed rollout windows without changing their rows."""
    available = list(bounds)
    if maximum is None or len(available) <= maximum:
        return available
    indices = np.floor(
        (np.arange(maximum, dtype=np.float64) + 0.5) * len(available) / maximum
    ).astype(int)
    return [available[int(index)] for index in indices]


@dataclass(frozen=True)
class FitResult:
    """Fitted parameter values with conditioning evidence.

    ``scaled_singular_values`` are the singular values of the final residual
    Jacobian with each column scaled by the parameter's box half-range, so
    parameters of different units are comparable. ``bound_proximity`` maps
    each non-frozen parameter to its componentwise distance to the nearest
    box bound as a fraction of the box range.
    """

    parameters: sysid.ParameterDict
    initial_parameters: sysid.ParameterDict
    values: dict[str, NDArray[np.float64]]
    confidence_halfwidths: dict[str, NDArray[np.float64]]
    parameter_covariance: NDArray[np.float64]
    parameter_correlations: NDArray[np.float64]
    scaled_singular_values: NDArray[np.float64]
    bound_proximity: dict[str, NDArray[np.float64]]
    initial_objective: float
    final_objective: float
    optimizer_result: Any

    @property
    def objective_reduction(self) -> float:
        """Fraction of the initial least-squares objective removed by the fit."""
        if self.initial_objective == 0.0:
            return 0.0
        return 1.0 - self.final_objective / self.initial_objective

    @property
    def conditioning_ratio(self) -> float:
        """``sigma_min / sigma_max`` of the scaled Jacobian (0.0 if unknown)."""
        return _conditioning_ratio(self.scaled_singular_values)

    @property
    def bound_hits(self) -> list[str]:
        """Parameters with any component within 0.1% of a box bound."""
        return [
            name
            for name, proximity in self.bound_proximity.items()
            if bool((proximity < 1e-3).any())
        ]


def _scaled_singular_values(
    jacobian: NDArray[np.float64], parameters: sysid.ParameterDict
) -> NDArray[np.float64]:
    """Singular values of the Jacobian with columns scaled by box half-range."""
    lower, upper = parameters.get_bounds()
    half_range = 0.5 * (upper - lower)
    return np.linalg.svd(jacobian * half_range[np.newaxis, :], compute_uv=False)


def _conditioning_ratio(singular_values: NDArray[np.float64]) -> float:
    if not singular_values.size:
        return 0.0
    largest = float(singular_values.max())
    if largest == 0.0:
        return 0.0
    return float(singular_values.min()) / largest


def _correlations(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    if not covariance.size:
        return np.empty((0, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        return np.where(scale > 0, covariance / scale, 0.0)


@dataclass(frozen=True)
class ConditioningReport:
    """Pre-fit identifiability evidence for a parameter block.

    Produced without optimizing, from one finite-difference Jacobian at the
    current parameter values — the plan's workflow is to freeze, group, or
    reparameterize an ill-conditioned block *before* fitting it.
    """

    scaled_singular_values: NDArray[np.float64]
    parameter_correlations: NDArray[np.float64]
    scaled_jacobian: NDArray[np.float64]
    component_names: tuple[str, ...]
    objective: float

    @property
    def conditioning_ratio(self) -> float:
        """``sigma_min / sigma_max`` of the scaled Jacobian (0.0 if unknown)."""
        return _conditioning_ratio(self.scaled_singular_values)


def conditioning_report(
    parameters: sysid.ParameterDict,
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences],
    *,
    diff_step: float | None = None,
) -> ConditioningReport:
    """Evaluate block identifiability at the current values, without fitting."""
    from mujoco import minimize

    if isinstance(sequences, sysid.ModelSequences):
        sequences = [sequences]
    residual_fn = sysid.build_residual_fn(models_sequences=list(sequences))
    probe = parameters.copy()

    def stacked_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        residuals, _, _ = residual_fn(x, probe)
        return np.concatenate(residuals)

    x0 = parameters.as_vector()
    lower, upper = parameters.get_bounds()
    eps = np.finfo(np.float64).eps ** 0.5 if diff_step is None else diff_step
    r0 = stacked_residual(x0).reshape(-1, 1)
    jacobian = np.asarray(
        minimize.jacobian_fd(
            residual=stacked_residual,
            x=x0.reshape(-1, 1),
            r=r0,
            eps=eps,
            n_res=0,
            bounds=[lower.reshape(-1, 1), upper.reshape(-1, 1)],
        )[0],
        dtype=np.float64,
    )
    covariance, _ = sysid.calculate_intervals([r0.ravel()], jacobian)
    return ConditioningReport(
        scaled_singular_values=_scaled_singular_values(jacobian, parameters),
        parameter_correlations=_correlations(covariance),
        scaled_jacobian=jacobian * (0.5 * (upper - lower))[np.newaxis, :],
        component_names=tuple(parameters.get_non_frozen_parameter_names()),
        objective=float(np.dot(r0.ravel(), r0.ravel())),
    )


def _worst_off_diagonal(correlations: NDArray[np.float64]) -> float:
    if correlations.ndim != 2 or correlations.shape[0] < 2:
        return 0.0
    off_diagonal = np.abs(correlations.copy())
    np.fill_diagonal(off_diagonal, 0.0)
    return float(np.nanmax(off_diagonal))


class IdentificationAcceptanceError(RuntimeError):
    """An identification block failed an observability or release gate."""


@dataclass(frozen=True)
class AcceptanceReport:
    """Machine-readable reasons why a parameter result may or may not ship."""

    problems: tuple[str, ...]
    conditioning_ratio: float
    worst_correlation: float
    objective_reduction: float | None = None
    bound_hits: tuple[str, ...] = ()
    maximum_multistart_spread: float | None = None

    @property
    def accepted(self) -> bool:
        return not self.problems

    def require(self, label: str = "identification") -> None:
        if self.problems:
            raise IdentificationAcceptanceError(
                f"{label} rejected: " + "; ".join(self.problems)
            )


def conditioning_acceptance(
    report: ConditioningReport,
    *,
    minimum_conditioning_ratio: float = CONDITIONING_RATIO_MINIMUM,
    correlation_limit: float = CORRELATION_FREEZE_LIMIT,
) -> AcceptanceReport:
    """Apply the production pre-fit release thresholds to a Jacobian report."""
    problems: list[str] = []
    ratio = report.conditioning_ratio
    worst_correlation = _worst_off_diagonal(report.parameter_correlations)
    if not np.isfinite(report.objective):
        problems.append("the nominal objective is not finite")
    if not report.scaled_singular_values.size:
        problems.append("the block has no active parameter sensitivity")
    elif not np.isfinite(report.scaled_singular_values).all():
        problems.append("the scaled Jacobian singular values are not finite")
    elif ratio < minimum_conditioning_ratio:
        problems.append(
            f"scaled Jacobian conditioning ratio {ratio:.3g} is below "
            f"{minimum_conditioning_ratio:.3g}"
        )
    if not np.isfinite(report.parameter_correlations).all():
        problems.append("parameter correlations are not finite")
    elif worst_correlation > correlation_limit:
        problems.append(
            f"worst parameter correlation {worst_correlation:.3g} exceeds "
            f"{correlation_limit:.3g}"
        )
    return AcceptanceReport(
        problems=tuple(problems),
        conditioning_ratio=ratio,
        worst_correlation=worst_correlation,
    )


def require_observable(
    parameters: sysid.ParameterDict,
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences],
    *,
    diff_step: float | None = None,
    minimum_conditioning_ratio: float = CONDITIONING_RATIO_MINIMUM,
    correlation_limit: float = CORRELATION_FREEZE_LIMIT,
) -> ConditioningReport:
    """Return pre-fit evidence, or fail closed when the block is confounded."""
    report = conditioning_report(parameters, sequences, diff_step=diff_step)
    conditioning_acceptance(
        report,
        minimum_conditioning_ratio=minimum_conditioning_ratio,
        correlation_limit=correlation_limit,
    ).require("parameter block")
    return report


@dataclass(frozen=True)
class ObservableSubset:
    """A deterministic scalar subset released by the local Jacobian.

    ``parameters`` is a deep copy of the requested block with rejected
    parameters frozen at their incoming values. ``sensitivity_ratios`` are
    bound-scaled column norms relative to the strongest requested column.
    """

    parameters: sysid.ParameterDict
    accepted_names: tuple[str, ...]
    rejected_names: tuple[str, ...]
    sensitivity_ratios: dict[str, float]
    conditioning_ratio: float
    worst_correlation: float

    def require_nonempty(self, label: str = "parameter block") -> None:
        if not self.accepted_names:
            raise IdentificationAcceptanceError(
                f"{label} rejected: no scalar correction is observable"
            )


def _subset_metrics(
    scaled_jacobian: NDArray[np.float64], indices: Sequence[int]
) -> tuple[float, float]:
    if not indices:
        return 0.0, 0.0
    selected = scaled_jacobian[:, indices]
    singular_values = np.linalg.svd(selected, compute_uv=False)
    ratio = _conditioning_ratio(singular_values)
    covariance = np.linalg.pinv(selected.T @ selected, rcond=1e-12)
    return ratio, _worst_off_diagonal(_correlations(covariance))


def select_observable_subset(
    parameters: sysid.ParameterDict,
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences],
    *,
    diff_step: float | None = None,
    minimum_conditioning_ratio: float = CONDITIONING_RATIO_MINIMUM,
    correlation_limit: float = CORRELATION_FREEZE_LIMIT,
    minimum_relative_sensitivity: float = 1e-6,
) -> ObservableSubset:
    """Freeze scalar corrections the supplied protocol cannot distinguish.

    The deterministic greedy pivot starts with the strongest bound-scaled
    Jacobian column, then repeatedly admits the candidate with the largest
    sensitivity orthogonal to the retained span. A candidate is released only
    if its independent sensitivity, the retained block's scaled SVD ratio,
    and its covariance correlation all pass the requested thresholds.

    This is a local observability decision at the current CAD prior, not a
    claim that the rejected physical coordinates were identified. Vector
    parameters are refused because MuJoCo freezes a whole ``Parameter`` at
    once; the CAD-prior and armature builders intentionally emit scalars.
    """
    if not 0.0 <= minimum_relative_sensitivity <= 1.0:
        raise ValueError("minimum_relative_sensitivity must lie in [0, 1]")
    active = [
        (name, parameter)
        for name, parameter in parameters.items()
        if not parameter.frozen
    ]
    non_scalar = [name for name, parameter in active if parameter.size != 1]
    if non_scalar:
        raise ValueError(
            "observable subset selection requires scalar parameters; split "
            "or reparameterize: " + ", ".join(non_scalar)
        )

    report = conditioning_report(parameters, sequences, diff_step=diff_step)
    names = report.component_names
    jacobian = report.scaled_jacobian
    if jacobian.shape[1] != len(names):
        raise RuntimeError("conditioning report lost its parameter-column mapping")

    norms = np.linalg.norm(jacobian, axis=0)
    strongest = float(norms.max()) if norms.size else 0.0
    ratios = {
        name: (float(norms[index]) / strongest if strongest > 0.0 else 0.0)
        for index, name in enumerate(names)
    }
    eligible = {
        index
        for index, norm in enumerate(norms)
        if strongest > 0.0
        and np.isfinite(norm)
        and float(norm) / strongest >= minimum_relative_sensitivity
    }
    accepted: list[int] = []

    while eligible:
        passing: list[tuple[float, float, int]] = []
        if accepted:
            basis, _ = np.linalg.qr(jacobian[:, accepted], mode="reduced")
        else:
            basis = np.empty((jacobian.shape[0], 0), dtype=np.float64)
        for index in sorted(eligible):
            column = jacobian[:, index]
            independent = column - basis @ (basis.T @ column)
            independent_ratio = float(np.linalg.norm(independent)) / strongest
            if independent_ratio < minimum_relative_sensitivity:
                continue
            ratio, correlation = _subset_metrics(jacobian, (*accepted, index))
            if ratio >= minimum_conditioning_ratio and correlation <= correlation_limit:
                # Higher orthogonal sensitivity wins; then higher raw
                # sensitivity; finally earlier input order for stable ties.
                passing.append((independent_ratio, float(norms[index]), -index))
        if not passing:
            break
        _, _, negative_index = max(passing)
        selected = -negative_index
        accepted.append(selected)
        eligible.remove(selected)

    accepted_names = tuple(names[index] for index in accepted)
    accepted_set = set(accepted_names)
    rejected_names = tuple(name for name in names if name not in accepted_set)
    released = parameters.copy()
    for name in rejected_names:
        released[name].frozen = True
    ratio, correlation = _subset_metrics(jacobian, accepted)
    return ObservableSubset(
        parameters=released,
        accepted_names=accepted_names,
        rejected_names=rejected_names,
        sensitivity_ratios=ratios,
        conditioning_ratio=ratio,
        worst_correlation=correlation,
    )


def _by_parameter(
    parameters: sysid.ParameterDict, vector: NDArray[np.float64] | None
) -> dict[str, NDArray[np.float64]]:
    """Split a non-frozen flat vector back into per-parameter arrays."""
    if vector is None:
        return {}
    result: dict[str, NDArray[np.float64]] = {}
    start = 0
    for name, param in parameters.items():
        if param.frozen:
            continue
        result[name] = np.asarray(vector[start : start + param.size])
        start += param.size
    return result


def _objective(residuals: Sequence[NDArray[np.float64]]) -> float:
    stacked = np.concatenate([np.asarray(r).ravel() for r in residuals])
    return float(np.dot(stacked, stacked))


def fit_parameters(
    parameters: sysid.ParameterDict,
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences],
    *,
    optimizer: str = "mujoco",
    max_iters: int = 100,
    verbose: bool = False,
) -> FitResult:
    """Fit *parameters* to measured *sequences* and report conditioning."""
    if isinstance(sequences, sysid.ModelSequences):
        sequences = [sequences]
    residual_fn = sysid.build_residual_fn(models_sequences=list(sequences))

    probe = parameters.copy()
    initial_residuals, _, _ = residual_fn(probe.as_vector(), probe)
    initial_objective = _objective(initial_residuals)

    opt_params, opt_result = sysid.optimize(
        initial_params=parameters,
        residual_fn=residual_fn,
        optimizer=optimizer,
        verbose=verbose,
        max_iters=max_iters,
    )

    probe = opt_params.copy()
    final_residuals, _, _ = residual_fn(probe.as_vector(), probe)
    final_objective = _objective(final_residuals)

    jacobian = getattr(opt_result, "jac", None)
    halfwidths: NDArray[np.float64] | None = None
    correlations = np.empty((0, 0))
    covariance = np.empty((0, 0))
    singular_values = np.empty(0)
    if jacobian is not None and np.asarray(jacobian).size:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        covariance, halfwidths = sysid.calculate_intervals(final_residuals, jacobian)
        correlations = _correlations(covariance)
        singular_values = _scaled_singular_values(jacobian, opt_params)

    lower, upper = opt_params.get_bounds()
    span = np.where(upper > lower, upper - lower, 1.0)
    fitted = opt_params.as_vector()
    proximity = np.minimum(fitted - lower, upper - fitted) / span

    return FitResult(
        parameters=opt_params,
        initial_parameters=parameters.copy(),
        values={name: param.value.copy() for name, param in opt_params.items()},
        confidence_halfwidths=_by_parameter(opt_params, halfwidths),
        parameter_covariance=covariance,
        parameter_correlations=correlations,
        scaled_singular_values=singular_values,
        bound_proximity=_by_parameter(opt_params, proximity),
        initial_objective=initial_objective,
        final_objective=final_objective,
        optimizer_result=opt_result,
    )


@dataclass(frozen=True)
class FitStage:
    """One stage of a staged fit: its parameters and the data that excites
    them (e.g. friction on slow reversals, armature on dynamic motion)."""

    parameters: sysid.ParameterDict
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences]


def fit_staged(
    stages: Sequence[FitStage],
    *,
    optimizer: str = "mujoco",
    max_iters: int = 100,
    verbose: bool = False,
) -> list[FitResult]:
    """Fit stages in order, persisting each stage's result for later stages.

    Implements the plan's friction-first ordering: after every stage its
    fitted values are written into the spec of every stage's sequences, so a
    later stage optimizes on top of the accepted earlier result. Stages must
    not share parameter fields.
    """
    all_sequences: list[sysid.ModelSequences] = []
    for stage in stages:
        if isinstance(stage.sequences, sysid.ModelSequences):
            all_sequences.append(stage.sequences)
        else:
            all_sequences.extend(stage.sequences)

    results: list[FitResult] = []
    for stage in stages:
        result = fit_parameters(
            stage.parameters,
            stage.sequences,
            optimizer=optimizer,
            max_iters=max_iters,
            verbose=verbose,
        )
        for sequences in all_sequences:
            sysid.apply_param_modifiers_spec(result.parameters, sequences.spec)
        results.append(result)
    return results


@dataclass(frozen=True)
class MultistartResult:
    """Independent fits from several starting points."""

    results: list[FitResult]

    @property
    def best(self) -> FitResult:
        return min(self.results, key=lambda result: result.final_objective)

    @property
    def value_spread(self) -> dict[str, float]:
        """Largest deviation from the best fit as a fraction of box span.

        Box-span normalization is meaningful for zero-centered corrections
        such as COM offsets. Dividing by the fitted value would turn harmless
        numerical disagreement around zero into an arbitrarily large spread.
        """
        best = self.best
        spread: dict[str, float] = {}
        for name, reference in best.values.items():
            parameter = best.parameters[name]
            lower, upper = parameter.get_bounds()
            span = upper - lower
            fallback = np.maximum(np.abs(np.asarray(reference).ravel()), 1e-12)
            scale = np.where(span > 0.0, span, fallback)
            deviations = (
                np.abs(np.asarray(result.values[name]).ravel() - reference.ravel())
                / scale
                for result in self.results
            )
            spread[name] = max(float(np.max(deviation)) for deviation in deviations)
        return spread


def fit_multistart(
    parameters: sysid.ParameterDict,
    sequences: sysid.ModelSequences | Sequence[sysid.ModelSequences],
    *,
    n_starts: int = 3,
    seed: int = 0,
    optimizer: str = "mujoco",
    max_iters: int = 100,
    verbose: bool = False,
) -> MultistartResult:
    """Fit from the given start plus uniformly sampled in-bounds starts.

    Sharing *sequences* across restarts is safe: every residual evaluation
    rewrites exactly the fitted spec fields, so no state leaks between fits.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    lower, upper = parameters.get_bounds()
    rng = np.random.default_rng(seed)

    results: list[FitResult] = []
    for start in range(n_starts):
        start_params = parameters.copy()
        if start > 0:
            start_params.update_from_vector(rng.uniform(lower, upper))
        results.append(
            fit_parameters(
                start_params,
                sequences,
                optimizer=optimizer,
                max_iters=max_iters,
                verbose=verbose,
            )
        )
    return MultistartResult(results=results)


def fit_acceptance(
    result: FitResult,
    *,
    multistart: MultistartResult | None = None,
    minimum_objective_reduction: float = 0.0,
    reject_bound_hits: bool = True,
    minimum_conditioning_ratio: float = CONDITIONING_RATIO_MINIMUM,
    correlation_limit: float = CORRELATION_FREEZE_LIMIT,
    maximum_multistart_spread: float = 0.05,
) -> AcceptanceReport:
    """Apply fail-closed post-fit release gates.

    This gate intentionally contains no sample-rate assumption. The Jacobian
    and objective are evaluated on the synchronized sequences supplied by the
    caller, whether those observations originated at 100 Hz or were decimated
    to it. Timestamp/skew validation must happen when those sequences are
    constructed.
    """
    problems: list[str] = []
    ratio = result.conditioning_ratio
    worst_correlation = _worst_off_diagonal(result.parameter_correlations)
    if not np.isfinite(result.initial_objective) or not np.isfinite(
        result.final_objective
    ):
        problems.append("fit objective is not finite")
    if result.final_objective > result.initial_objective:
        problems.append("fit made the objective worse")
    if result.objective_reduction < minimum_objective_reduction:
        problems.append(
            f"objective reduction {result.objective_reduction:.3g} is below "
            f"{minimum_objective_reduction:.3g}"
        )
    if not result.scaled_singular_values.size:
        problems.append("optimizer returned no Jacobian conditioning evidence")
    elif not np.isfinite(result.scaled_singular_values).all():
        problems.append("scaled Jacobian singular values are not finite")
    elif ratio < minimum_conditioning_ratio:
        problems.append(
            f"scaled Jacobian conditioning ratio {ratio:.3g} is below "
            f"{minimum_conditioning_ratio:.3g}"
        )
    if not np.isfinite(result.parameter_correlations).all():
        problems.append("parameter correlations are not finite")
    elif worst_correlation > correlation_limit:
        problems.append(
            f"worst parameter correlation {worst_correlation:.3g} exceeds "
            f"{correlation_limit:.3g}"
        )

    bound_hits = tuple(result.bound_hits)
    if reject_bound_hits and bound_hits:
        problems.append("parameters hit a box bound: " + ", ".join(bound_hits))

    observed_spread: float | None = None
    if multistart is not None:
        spreads = tuple(multistart.value_spread.values())
        observed_spread = max(spreads, default=0.0)
        if not np.isfinite(observed_spread):
            problems.append("multistart parameter spread is not finite")
        elif observed_spread > maximum_multistart_spread:
            problems.append(
                f"multistart spread {observed_spread:.3g} exceeds "
                f"{maximum_multistart_spread:.3g}"
            )

    return AcceptanceReport(
        problems=tuple(problems),
        conditioning_ratio=ratio,
        worst_correlation=worst_correlation,
        objective_reduction=result.objective_reduction,
        bound_hits=bound_hits,
        maximum_multistart_spread=observed_spread,
    )


def require_acceptable_fit(
    result: FitResult,
    *,
    multistart: MultistartResult | None = None,
    label: str = "fit",
    minimum_objective_reduction: float = 0.0,
    reject_bound_hits: bool = True,
    minimum_conditioning_ratio: float = CONDITIONING_RATIO_MINIMUM,
    correlation_limit: float = CORRELATION_FREEZE_LIMIT,
    maximum_multistart_spread: float = 0.05,
) -> AcceptanceReport:
    """Return release evidence, or raise when a fitted block may not ship."""
    report = fit_acceptance(
        result,
        multistart=multistart,
        minimum_objective_reduction=minimum_objective_reduction,
        reject_bound_hits=reject_bound_hits,
        minimum_conditioning_ratio=minimum_conditioning_ratio,
        correlation_limit=correlation_limit,
        maximum_multistart_spread=maximum_multistart_spread,
    )
    report.require(label)
    return report
