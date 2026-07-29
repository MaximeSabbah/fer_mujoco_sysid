"""Production fitting stages, kept separate from acquisition and reporting.

The order is deliberate:

1. friction on true constant-velocity cruise windows;
2. a jointly observable armature + CAD-prior body correction block on the
   dynamic protocol;
3. friction again with the accepted dynamic model held fixed.

The final refit prevents an incorrect nominal armature or link inertia from
being permanently absorbed into friction.  No model is exported here:
withheld reproduction in :mod:`fer_mujoco_sysid.validation` is the release
gate and belongs after fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid
from numpy.typing import NDArray

from fer_mujoco_sysid.classical import LinearFrictionFit, fit_friction
from fer_mujoco_sysid.diagnostics import ParameterQuality, parameter_quality
from fer_mujoco_sysid.export import IdentifiedParameters
from fer_mujoco_sysid.fitting import (
    AcceptanceReport,
    BodyInertialCorrection,
    FitResult,
    MeasuredRun,
    MultistartResult,
    ObservableSubset,
    armature_parameters,
    cad_prior_inertial_parameters,
    combine_parameters,
    fit_acceptance,
    fit_multistart,
    fit_parameters,
    friction_parameters,
    measurement_sequences,
    require_acceptable_fit,
    require_observable,
    select_observable_subset,
    set_hinge_damping,
)
from fer_mujoco_sysid.model import (
    HYDRAX_ARM_JOINT_NAMES,
    build_hydrax_arm_spec,
)
from fer_mujoco_sysid.protocol import FRICTION_FAMILY, INERTIAL_FAMILY
from fer_mujoco_sysid.selection import runs_from_mask

# MPPI's deployed horizon is 0.32 s (eight 40 ms planning steps). Fitting on
# the same horizon asks the optimizer to improve the behavior the consumer
# actually uses. Complete recordings remain available for reporting and every
# held-out window; only optimizer residual construction is bounded here.
FIT_WINDOW_S = 0.32
FIT_MEASUREMENT_STRIDE = 5
FRICTION_WINDOWS_PER_CRUISE = 1
INERTIAL_WINDOWS_PER_RECORDING = 8
DYNAMIC_SELECTION_CONDITIONING_MINIMUM = 1e-4
DYNAMIC_SELECTION_CORRELATION_LIMIT = 0.90
DYNAMIC_SELECTION_SENSITIVITY_MINIMUM = 1e-3
DYNAMIC_STD_SPAN_LIMIT_PERCENT = 20.0
DYNAMIC_POSTFIT_PASSES = 4
COUPLING_REFINEMENT_MAX_ROUNDS = 4
COUPLING_CONVERGENCE_FRACTION = 1e-5
MAXIMUM_METHOD_DISAGREEMENT_PERCENT = 20.0

# Every link is offered the same tightly bounded CAD correction vocabulary.
# The local Jacobian releases only scalar directions supported by the actual
# inertial recordings.  Rejected fields stay exactly nominal and are named in
# the result; this is how the pipeline improves all observable rigid-body
# dynamics without claiming that a serial arm reveals 70 independent raw
# inertia coordinates.
DEFAULT_BODY_CORRECTIONS = tuple(
    BodyInertialCorrection(
        body=f"link{index}",
        estimate_mass=True,
        com_axes=(0, 1, 2),
        estimate_inertia_scale=True,
    )
    for index in range(1, 8)
)


@dataclass(frozen=True)
class StageRecording:
    """One prepared recording with explicit scientific selections."""

    label: str
    protocol_id: str
    family: str
    role: str
    run: MeasuredRun
    ddq_rad_s2: NDArray[np.float64]
    classical_torque_Nm: NDArray[np.float64]
    classical_mask: NDArray[np.bool_]
    analysis_mask: NDArray[np.bool_]
    protocol_mask: NDArray[np.bool_]

    def validate(self) -> None:
        samples = len(self.run.control)
        if self.family not in (FRICTION_FAMILY, INERTIAL_FAMILY):
            raise ValueError(f"{self.protocol_id}: unknown family {self.family!r}")
        if self.role not in ("train", "holdout"):
            raise ValueError(f"{self.protocol_id}: unknown role {self.role!r}")
        if self.ddq_rad_s2.shape != (samples, 7):
            raise ValueError(
                f"{self.protocol_id}: acceleration shape "
                f"{self.ddq_rad_s2.shape}, expected {(samples, 7)}"
            )
        if self.classical_torque_Nm.shape != (samples, 7):
            raise ValueError(
                f"{self.protocol_id}: classical torque shape "
                f"{self.classical_torque_Nm.shape}, expected {(samples, 7)}"
            )
        for name, mask in (
            ("classical", self.classical_mask),
            ("analysis", self.analysis_mask),
            ("protocol", self.protocol_mask),
        ):
            if np.asarray(mask).shape != (samples,):
                raise ValueError(
                    f"{self.protocol_id}: {name} mask has shape "
                    f"{np.asarray(mask).shape}, expected {(samples,)}"
                )
        if np.any(self.analysis_mask & ~self.protocol_mask):
            raise ValueError(
                f"{self.protocol_id}: analysis mask extends outside the protocol"
            )
        if np.any(self.classical_mask & ~self.analysis_mask):
            raise ValueError(
                f"{self.protocol_id}: classical mask extends outside rollout analysis"
            )

    def analysis_runs(self) -> tuple[MeasuredRun, ...]:
        self.validate()
        return runs_from_mask(self.run, self.analysis_mask)


@dataclass(frozen=True)
class StageResult:
    """The accepted in-memory model and evidence from every fitted block."""

    nominal: mujoco.MjModel
    classical: mujoco.MjModel
    identified: mujoco.MjModel
    parameters: IdentifiedParameters
    linear: LinearFrictionFit
    first_friction: FitResult
    first_friction_acceptance: AcceptanceReport
    dynamic: FitResult | None
    dynamic_multistart: MultistartResult | None
    dynamic_acceptance: AcceptanceReport | None
    observable_dynamic: ObservableSubset | None
    friction_refit: FitResult
    friction_refit_acceptance: AcceptanceReport
    quality: ParameterQuality
    armature_quality: ParameterQuality | None
    method_disagreement_percent: float
    changed_bodies: tuple[str, ...]
    dynamic_precision_percent: dict[str, float]
    dynamic_frozen_postfit: tuple[str, ...]
    coupling_refinement_rounds: int
    coupling_max_change_fraction: float

    def summary(self) -> dict[str, object]:
        dynamic: dict[str, object] | None = None
        if self.dynamic is not None:
            dynamic = {
                "objective_reduction": self.dynamic.objective_reduction,
                "conditioning_ratio": self.dynamic.conditioning_ratio,
                "bound_hits": self.dynamic.bound_hits,
                "accepted_parameters": list(
                    self.observable_dynamic.accepted_names
                    if self.observable_dynamic is not None
                    else ()
                ),
                "rejected_parameters": list(
                    self.observable_dynamic.rejected_names
                    if self.observable_dynamic is not None
                    else ()
                ),
                "multistart_spread": (
                    self.dynamic_multistart.value_spread
                    if self.dynamic_multistart is not None
                    else {}
                ),
                "std_percent_of_bound_span": self.dynamic_precision_percent,
                "frozen_after_fit": list(self.dynamic_frozen_postfit),
                "coupling_refinement_rounds": self.coupling_refinement_rounds,
                "coupling_max_change_fraction": self.coupling_max_change_fraction,
            }
        return {
            "frictionloss": list(self.parameters.frictionloss or ()),
            "damping": list(self.parameters.damping or ()),
            "armature": list(self.parameters.armature or ()),
            "body_inertials": {
                body.body: body.as_dict() for body in self.parameters.body_inertials
            },
            "changed_bodies": list(self.changed_bodies),
            "method_disagreement_percent": self.method_disagreement_percent,
            "friction_first": _fit_summary(
                self.first_friction, self.first_friction_acceptance
            ),
            "dynamic": dynamic,
            "friction_refit": _fit_summary(
                self.friction_refit, self.friction_refit_acceptance
            ),
            "sigma_percent": self.quality.relative_percent.tolist(),
            "rejected_parameters": self.quality.rejected,
            "armature_sigma_percent": (
                self.armature_quality.relative_percent.tolist()
                if self.armature_quality is not None
                else None
            ),
        }


def _fit_summary(result: FitResult, acceptance: AcceptanceReport) -> dict[str, object]:
    return {
        "initial_objective": result.initial_objective,
        "final_objective": result.final_objective,
        "objective_reduction": result.objective_reduction,
        "conditioning_ratio": result.conditioning_ratio,
        "bound_hits": result.bound_hits,
        "acceptance": {
            "accepted": acceptance.accepted,
            "problems": list(acceptance.problems),
            "worst_correlation": acceptance.worst_correlation,
        },
    }


def fitting_spec(model_path: str | Path) -> mujoco.MjSpec:
    """Gravity-compensated torque-control projection used for rollouts."""
    spec = build_hydrax_arm_spec(model_path, joint_state_sensors=True)
    spec.option.gravity = [0.0, 0.0, 0.0]
    return spec


def _apply_joint_values(
    spec: mujoco.MjSpec,
    *,
    frictionloss: NDArray[np.float64] | None = None,
    damping: NDArray[np.float64] | None = None,
    armature: NDArray[np.float64] | None = None,
) -> None:
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        joint = spec.joint(name)
        if frictionloss is not None:
            joint.frictionloss = float(frictionloss[index])
        if damping is not None:
            set_hinge_damping(joint, float(damping[index]))
        if armature is not None:
            joint.armature = float(armature[index])


def _parameter_quality(result: FitResult) -> ParameterQuality:
    return parameter_quality(
        tuple(result.parameters.get_non_frozen_parameter_names()),
        result.parameters.as_vector(),
        result.parameter_covariance,
    )


def _friction_fit(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    runs: list[MeasuredRun],
    *,
    seed_frictionloss: NDArray[np.float64],
    seed_damping: NDArray[np.float64],
    max_iters: int,
    label: str,
) -> tuple[FitResult, AcceptanceReport]:
    sequences = measurement_sequences(
        spec,
        runs,
        window_s=FIT_WINDOW_S,
        max_windows_per_run=FRICTION_WINDOWS_PER_CRUISE,
        measurement_stride=FIT_MEASUREMENT_STRIDE,
    )
    parameters = friction_parameters(model).move_off_bounds()
    for index, joint in enumerate(HYDRAX_ARM_JOINT_NAMES):
        parameters[f"{joint}_frictionloss"].value[:] = seed_frictionloss[index]
        parameters[f"{joint}_damping"].value[:] = seed_damping[index]
    require_observable(parameters, sequences)
    result = fit_parameters(parameters, sequences, max_iters=max_iters)
    acceptance = require_acceptable_fit(result, label=label)
    return result, acceptance


def _result_joint_columns(
    result: FitResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frictionloss = np.array(
        [result.values[f"joint{index}_frictionloss"][0] for index in range(1, 8)]
    )
    damping = np.array(
        [result.values[f"joint{index}_damping"][0] for index in range(1, 8)]
    )
    return frictionloss, damping


def _dynamic_quality(result: FitResult) -> ParameterQuality | None:
    component_names = result.parameters.get_non_frozen_parameter_names()
    armature_indices = [
        index
        for index, name in enumerate(component_names)
        if name.endswith("_armature")
    ]
    if not armature_indices or not result.parameter_covariance.size:
        return None
    values = result.parameters.as_vector()[armature_indices]
    covariance = result.parameter_covariance[np.ix_(armature_indices, armature_indices)]
    names = tuple(component_names[index] for index in armature_indices)
    return parameter_quality(names, values, covariance)


def _dynamic_span_precision(result: FitResult) -> dict[str, float]:
    """Posterior standard deviation as a percentage of each parameter box."""
    names = tuple(result.parameters.get_non_frozen_parameter_names())
    if not names or result.parameter_covariance.shape != (len(names), len(names)):
        return {name: float("inf") for name in names}
    lower, upper = result.parameters.get_bounds()
    span = upper - lower
    std = np.sqrt(np.clip(np.diag(result.parameter_covariance), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = 100.0 * std / span
    return {name: float(value) for name, value in zip(names, percent, strict=True)}


def _dynamic_freeze_candidates(
    result: FitResult,
    multistart: MultistartResult,
    observable: ObservableSubset,
) -> tuple[str, ...]:
    """Choose unsupported scalar directions to return exactly to CAD."""
    active = tuple(result.parameters.get_non_frozen_parameter_names())
    active_set = set(active)
    freeze = {
        name
        for name, percent in _dynamic_span_precision(result).items()
        if not np.isfinite(percent) or percent > DYNAMIC_STD_SPAN_LIMIT_PERCENT
    }
    freeze.update(name for name in result.bound_hits if name in active_set)
    freeze.update(
        name
        for name, spread in multistart.value_spread.items()
        if not np.isfinite(spread) or spread > 0.05
    )

    correlations = np.asarray(result.parameter_correlations, dtype=np.float64)
    if correlations.shape == (len(active), len(active)):
        for first in range(len(active)):
            for second in range(first + 1, len(active)):
                value = correlations[first, second]
                if not np.isfinite(value) or abs(value) > 0.98:
                    pair = (active[first], active[second])
                    freeze.add(
                        min(
                            pair,
                            key=lambda name: (
                                observable.sensitivity_ratios.get(name, 0.0),
                                name,
                            ),
                        )
                    )
    return tuple(name for name in active if name in freeze)


def _freeze_at_nominal(
    parameters: sysid.ParameterDict,
    names: tuple[str, ...],
) -> sysid.ParameterDict:
    frozen = parameters.copy()
    for name in names:
        frozen[name].reset()
        frozen[name].frozen = True
    return frozen


def _maximum_parameter_change_fraction(
    previous: sysid.ParameterDict,
    current: sysid.ParameterDict,
) -> float:
    """Maximum coordinate change normalized by its declared box span."""
    previous_names = tuple(previous.get_non_frozen_parameter_names())
    current_names = tuple(current.get_non_frozen_parameter_names())
    if previous_names != current_names:
        raise ValueError("cannot compare parameter blocks with different active fields")
    if not current_names:
        return 0.0
    lower, upper = current.get_bounds()
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("active parameter bounds must have positive span")
    return float(
        np.max(np.abs(current.as_vector() - previous.as_vector()) / span)
    )


def fit_stages(
    model_path: str | Path,
    recordings: list[StageRecording],
    *,
    max_iters: int = 30,
    dynamic_starts: int = 3,
    body_corrections: tuple[BodyInertialCorrection, ...] = DEFAULT_BODY_CORRECTIONS,
) -> StageResult:
    """Fit all supported stages and return an unexported candidate model."""
    if max_iters < 1:
        raise ValueError("max_iters must be positive")
    if dynamic_starts < 1:
        raise ValueError("dynamic_starts must be positive")
    for recording in recordings:
        recording.validate()

    training = [record for record in recordings if record.role == "train"]
    friction_records = [
        record for record in training if record.family == FRICTION_FAMILY
    ]
    inertial_records = [
        record for record in training if record.family == INERTIAL_FAMILY
    ]
    if not friction_records:
        raise ValueError("a friction training recording is required")
    if not inertial_records:
        raise ValueError(
            "an inertial training recording is required for armature and "
            "link mass/COM/inertia identification"
        )

    nominal_spec = fitting_spec(model_path)
    nominal = nominal_spec.compile()
    friction_runs = [
        run for recording in friction_records for run in recording.analysis_runs()
    ]

    q = np.vstack(
        [record.run.measured[record.classical_mask, :7] for record in friction_records]
    )
    dq = np.vstack(
        [
            record.run.measured[record.classical_mask, 7:14]
            for record in friction_records
        ]
    )
    ddq = np.vstack(
        [record.ddq_rad_s2[record.classical_mask] for record in friction_records]
    )
    tau = np.vstack(
        [
            record.classical_torque_Nm[record.classical_mask]
            for record in friction_records
        ]
    )
    linear = fit_friction(nominal, q, dq, ddq, tau)
    if not np.isfinite(linear.condition_number).all() or bool(
        (linear.condition_number > 100.0).any()
    ):
        raise RuntimeError(
            "friction protocol is not sufficiently conditioned for release"
        )
    linear_rejected = [
        f"joint{joint + 1}"
        for joint in range(7)
        if max(
            linear.frictionloss_sigma_percent[joint],
            linear.damping_sigma_percent[joint],
        )
        > 20.0
    ]
    if linear_rejected:
        raise RuntimeError(
            "classical friction uncertainty rejected " + ", ".join(linear_rejected)
        )

    first_spec = fitting_spec(model_path)
    first_result, first_acceptance = _friction_fit(
        first_spec,
        first_spec.compile(),
        friction_runs,
        seed_frictionloss=linear.frictionloss,
        seed_damping=linear.damping,
        max_iters=max_iters,
        label="first friction fit",
    )
    frictionloss, damping = _result_joint_columns(first_result)
    _apply_joint_values(first_spec, frictionloss=frictionloss, damping=damping)

    dynamic_result: FitResult | None = None
    dynamic_multistart: MultistartResult | None = None
    dynamic_acceptance: AcceptanceReport | None = None
    observable: ObservableSubset | None = None
    changed_bodies: tuple[str, ...] = ()
    armature_quality: ParameterQuality | None = None
    dynamic_precision: dict[str, float] = {}
    dynamic_frozen_postfit: list[str] = []
    coupling_refinement_rounds = 0
    coupling_max_change_fraction = 0.0
    dynamic_spec = first_spec
    dynamic_sequences: sysid.ModelSequences | None = None
    requested: sysid.ParameterDict | None = None

    if inertial_records:
        inertial_runs = [
            run for recording in inertial_records for run in recording.analysis_runs()
        ]
        dynamic_sequences = measurement_sequences(
            dynamic_spec,
            inertial_runs,
            window_s=FIT_WINDOW_S,
            max_windows_per_run=INERTIAL_WINDOWS_PER_RECORDING,
            measurement_stride=FIT_MEASUREMENT_STRIDE,
        )
        dynamic_model = dynamic_spec.compile()
        groups = [armature_parameters(dynamic_model)]
        if body_corrections:
            groups.append(
                cad_prior_inertial_parameters(
                    dynamic_spec, dynamic_model, body_corrections
                )
            )
        requested = combine_parameters(*groups)
        observable = select_observable_subset(
            requested,
            dynamic_sequences,
            minimum_conditioning_ratio=DYNAMIC_SELECTION_CONDITIONING_MINIMUM,
            correlation_limit=DYNAMIC_SELECTION_CORRELATION_LIMIT,
            minimum_relative_sensitivity=(DYNAMIC_SELECTION_SENSITIVITY_MINIMUM),
        )
        observable.require_nonempty("dynamic armature/body block")
        active_parameters = observable.parameters
        for _ in range(DYNAMIC_POSTFIT_PASSES):
            dynamic_multistart = fit_multistart(
                active_parameters,
                dynamic_sequences,
                n_starts=dynamic_starts,
                max_iters=max_iters,
            )
            dynamic_result = dynamic_multistart.best
            freeze = _dynamic_freeze_candidates(
                dynamic_result,
                dynamic_multistart,
                observable,
            )
            if freeze:
                dynamic_frozen_postfit.extend(freeze)
                active_parameters = _freeze_at_nominal(active_parameters, freeze)
                if not active_parameters.get_non_frozen_parameter_names():
                    raise RuntimeError(
                        "dynamic armature/body fit left no precise parameter "
                        "direction after post-fit uncertainty checks"
                    )
                continue
            dynamic_acceptance = fit_acceptance(
                dynamic_result,
                multistart=dynamic_multistart,
            )
            if dynamic_acceptance.accepted:
                break
            # A block that remains globally ill-conditioned after pairwise
            # correlation checks loses its weakest remaining direction and is
            # refit. Objective/non-finite failures are not repairable this way.
            repairable = all(
                "conditioning ratio" in problem or "parameter correlation" in problem
                for problem in dynamic_acceptance.problems
            )
            if not repairable:
                dynamic_acceptance.require("dynamic armature/body fit")
            active = tuple(active_parameters.get_non_frozen_parameter_names())
            weakest = min(
                active,
                key=lambda name: (
                    observable.sensitivity_ratios.get(name, 0.0),
                    name,
                ),
            )
            dynamic_frozen_postfit.append(weakest)
            active_parameters = _freeze_at_nominal(active_parameters, (weakest,))
        else:
            raise RuntimeError(
                "dynamic armature/body fit did not stabilize after "
                f"{DYNAMIC_POSTFIT_PASSES} freeze/refit passes"
            )
        assert dynamic_result is not None
        assert dynamic_multistart is not None
        dynamic_acceptance = require_acceptable_fit(
            dynamic_result,
            multistart=dynamic_multistart,
            label="dynamic armature/body fit",
        )
        sysid.apply_param_modifiers_spec(dynamic_result.parameters, dynamic_spec)
        changed_bodies = tuple(
            correction.body
            for correction in body_corrections
            if any(
                name.startswith(f"{correction.body}_")
                for name in dynamic_result.parameters.get_non_frozen_parameter_names()
            )
        )

    # Refit friction against the now-corrected armature/body model.  The
    # dynamic modifiers remain on dynamic_spec; only friction is released.
    dynamic_model = dynamic_spec.compile()
    refit_result, refit_acceptance = _friction_fit(
        dynamic_spec,
        dynamic_model,
        friction_runs,
        seed_frictionloss=frictionloss,
        seed_damping=damping,
        max_iters=max_iters,
        label="friction refit",
    )
    frictionloss, damping = _result_joint_columns(refit_result)
    _apply_joint_values(dynamic_spec, frictionloss=frictionloss, damping=damping)

    # Friction and rigid-body dynamics affect both protocol families. A single
    # friction -> dynamics -> friction pass leaves the dynamic block fitted
    # against the deliberately imperfect first friction estimate. Alternate
    # the two *separate* family-specific solves until their shared model is a
    # fixed point. The expensive multi-start/observability work above is not
    # repeated: refinements start from its accepted basin and retain exactly
    # the same certified scalar coordinates.
    if dynamic_result is not None:
        assert dynamic_sequences is not None
        assert dynamic_multistart is not None
        assert observable is not None
        assert requested is not None
        for refinement in range(1, COUPLING_REFINEMENT_MAX_ROUNDS + 1):
            previous_dynamic = dynamic_result.parameters.copy()
            previous_friction = refit_result.parameters.copy()

            dynamic_result = fit_parameters(
                previous_dynamic.copy(),
                dynamic_sequences,
                max_iters=max_iters,
            )
            dynamic_acceptance = require_acceptable_fit(
                dynamic_result,
                multistart=dynamic_multistart,
                label=f"dynamic coupling refinement {refinement}",
            )
            sysid.apply_param_modifiers_spec(dynamic_result.parameters, dynamic_spec)

            dynamic_model = dynamic_spec.compile()
            refit_result, refit_acceptance = _friction_fit(
                dynamic_spec,
                dynamic_model,
                friction_runs,
                seed_frictionloss=frictionloss,
                seed_damping=damping,
                max_iters=max_iters,
                label=f"friction coupling refinement {refinement}",
            )
            frictionloss, damping = _result_joint_columns(refit_result)
            _apply_joint_values(
                dynamic_spec,
                frictionloss=frictionloss,
                damping=damping,
            )

            coupling_refinement_rounds = refinement
            coupling_max_change_fraction = max(
                _maximum_parameter_change_fraction(
                    previous_dynamic,
                    dynamic_result.parameters,
                ),
                _maximum_parameter_change_fraction(
                    previous_friction,
                    refit_result.parameters,
                ),
            )
            if coupling_max_change_fraction <= COUPLING_CONVERGENCE_FRACTION:
                break
        else:
            raise RuntimeError(
                "friction/dynamics alternating refinement did not converge: "
                f"last maximum parameter change was "
                f"{coupling_max_change_fraction:.3g} of its bound span"
            )

        dynamic_precision = _dynamic_span_precision(dynamic_result)
        active_names = tuple(dynamic_result.parameters.get_non_frozen_parameter_names())
        active_set = set(active_names)
        observable = ObservableSubset(
            parameters=dynamic_result.parameters,
            accepted_names=active_names,
            rejected_names=tuple(
                name
                for name in requested.get_non_frozen_parameter_names()
                if name not in active_set
            ),
            sensitivity_ratios=observable.sensitivity_ratios,
            conditioning_ratio=dynamic_result.conditioning_ratio,
            worst_correlation=dynamic_acceptance.worst_correlation,
        )
        armature_quality = _dynamic_quality(dynamic_result)
        if armature_quality is not None and armature_quality.rejected:
            raise RuntimeError(
                "armature uncertainty rejected " + ", ".join(armature_quality.rejected)
            )

    identified = dynamic_spec.compile()

    denominator = np.maximum(
        np.abs(np.concatenate([linear.frictionloss, linear.damping])), 1e-9
    )
    disagreement = (
        100.0
        * (
            np.concatenate([frictionloss, damping])
            - np.concatenate([linear.frictionloss, linear.damping])
        )
        / denominator
    )
    worst_disagreement = float(np.max(np.abs(disagreement)))
    if worst_disagreement > MAXIMUM_METHOD_DISAGREEMENT_PERCENT:
        raise RuntimeError(
            "friction model-class disagreement is "
            f"{worst_disagreement:.1f}% > "
            f"{MAXIMUM_METHOD_DISAGREEMENT_PERCENT:.1f}%; dynamic parameters "
            "must not absorb that mismatch"
        )

    quality = _parameter_quality(refit_result)
    if quality.rejected:
        raise RuntimeError(
            "friction refit uncertainty rejected " + ", ".join(quality.rejected)
        )

    classical_spec = fitting_spec(model_path)
    _apply_joint_values(
        classical_spec,
        frictionloss=linear.frictionloss,
        damping=linear.damping,
    )
    classical = classical_spec.compile()
    parameters = IdentifiedParameters.from_model(
        identified,
        bodies=changed_bodies,
        include_frictionloss=True,
        include_damping=True,
        include_armature=True,
    )
    return StageResult(
        nominal=nominal,
        classical=classical,
        identified=identified,
        parameters=parameters,
        linear=linear,
        first_friction=first_result,
        first_friction_acceptance=first_acceptance,
        dynamic=dynamic_result,
        dynamic_multistart=dynamic_multistart,
        dynamic_acceptance=dynamic_acceptance,
        observable_dynamic=observable,
        friction_refit=refit_result,
        friction_refit_acceptance=refit_acceptance,
        quality=quality,
        armature_quality=armature_quality,
        method_disagreement_percent=worst_disagreement,
        changed_bodies=changed_bodies,
        dynamic_precision_percent=dynamic_precision,
        dynamic_frozen_postfit=tuple(dict.fromkeys(dynamic_frozen_postfit)),
        coupling_refinement_rounds=coupling_refinement_rounds,
        coupling_max_change_fraction=coupling_max_change_fraction,
    )
