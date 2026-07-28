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
    bodies: Sequence[str] = MOVING_LINK_BODIES,
    inertia_type: sysid.InertiaType = sysid.InertiaType.Pseudo,
) -> sysid.ParameterDict:
    """Per-body inertia parameters, nominal = the spec's current inertials.

    ``Pseudo`` (default) is the physically consistent parameterization —
    every candidate inertia is realizable by construction. *spec* is only
    read here. Estimate ``MOVING_LINK_BODIES`` at most: parameters for
    ``link7`` represent the whole rigid tool composite (see
    ``TOOL_COMPOSITE_BODIES``); estimating hand/finger bodies alongside
    ``link7`` is structurally unidentifiable and must be rejected by a
    :func:`conditioning_report` check.
    """
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


@dataclass(frozen=True)
class MeasuredRun:
    """One recorded playback: applied torque and measured joint states.

    ``control`` rows follow the model actuator order; ``measured`` rows follow
    the model sensor layout (``{joint}_pos`` then ``{joint}_vel`` for a spec
    built with ``joint_state_sensors=True``). Times are seconds on one clock.

    Row convention (mirrors ``mujoco.rollout``): ``measured[k]`` holds the
    joint state after the first ``k`` control rows have been applied — the
    state at ``k * dt``, i.e. the sensor row MuJoCo emits after step ``k`` —
    stamped with the post-step time ``(k + 1) * dt``. Predicted rollouts carry
    the identical skew, so measured and predicted rows align index-for-index;
    do not "fix" the stamps on one side only.
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
            bounds = [
                (start, min(start + steps, len(control_times)))
                for start in range(0, len(control_times), steps)
            ]
            qpos_cols, qvel_cols = _joint_state_columns(model)

        for start, stop in bounds:
            if stop - start < 2:
                continue
            if window_s is None:
                qpos0 = np.asarray(run.qpos0, dtype=np.float64)
                qvel0 = np.asarray(run.qvel0, dtype=np.float64)
            else:
                # measured[start] is the state at start*dt (see MeasuredRun),
                # exactly the state from which control[start] is applied.
                qpos0 = measured[start, qpos_cols]
                qvel0 = measured[start, qvel_cols]
            origin = control_times[start]
            labels.append(f"{run.label}[{start}:{stop}]")
            initial_states.append(sysid.create_initial_state(model, qpos0, qvel0))
            control_series.append(
                sysid.TimeSeries(
                    control_times[start:stop] - origin, control[start:stop]
                )
            )
            measured_series.append(
                sysid.TimeSeries.from_names(
                    measured_times[start:stop] - origin,
                    measured[start:stop],
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
        objective=float(np.dot(r0.ravel(), r0.ravel())),
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
        """Largest relative deviation from the best fit, per parameter."""
        best = self.best
        spread: dict[str, float] = {}
        for name, reference in best.values.items():
            scale = max(float(np.abs(reference).max()), 1e-12)
            spread[name] = max(
                float(np.abs(result.values[name] - reference).max()) / scale
                for result in self.results
            )
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
