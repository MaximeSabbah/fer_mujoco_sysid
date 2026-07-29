"""Held-out simulator reproduction metrics and the release decision.

System identification is useful here only insofar as the resulting MuJoCo
model reproduces what the robot did under an input it did not see while
fitting.  Parameter proximity is deliberately not an acceptance criterion:
there is no parameter truth on hardware.  This module therefore owns the
MPPI-facing contract -- open-loop joint and end-effector error over several
rollout horizons, plus a fail-closed comparison with the nominal model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.fitting import MeasuredRun


@dataclass(frozen=True)
class RolloutMetrics:
    """Errors between one simulated model and one withheld recording."""

    q_rmse_rad: NDArray[np.float64]
    dq_rmse_rad_s: NDArray[np.float64]
    gripper_rmse_mm: float
    gripper_max_mm: float
    windows: int
    samples: int

    @property
    def worst_q_rmse_rad(self) -> float:
        return float(np.max(self.q_rmse_rad))

    @property
    def worst_dq_rmse_rad_s(self) -> float:
        return float(np.max(self.dq_rmse_rad_s))

    def as_dict(self) -> dict[str, object]:
        return {
            "q_rmse_rad": self.q_rmse_rad.tolist(),
            "dq_rmse_rad_s": self.dq_rmse_rad_s.tolist(),
            "worst_q_rmse_rad": self.worst_q_rmse_rad,
            "worst_dq_rmse_rad_s": self.worst_dq_rmse_rad_s,
            "gripper_rmse_mm": self.gripper_rmse_mm,
            "gripper_max_mm": self.gripper_max_mm,
            "windows": self.windows,
            "samples": self.samples,
        }


def _site_positions(
    model: mujoco.MjModel,
    q_rad: NDArray[np.float64],
    *,
    site_name: str,
) -> NDArray[np.float64]:
    site = int(model.site(site_name).id)
    data = mujoco.MjData(model)
    positions = np.empty((len(q_rad), 3), dtype=np.float64)
    for index, q in enumerate(q_rad):
        data.qpos[: len(q)] = q
        mujoco.mj_kinematics(model, data)
        positions[index] = data.site_xpos[site]
    return positions


def rollout_metrics(
    model: mujoco.MjModel,
    run: MeasuredRun,
    *,
    window_s: float,
    site_name: str = "gripper",
) -> RolloutMetrics:
    """Replay *run* open loop in reset windows and compare measured outputs.

    Resetting at every window asks the same question an MPPI rollout asks:
    starting from the state estimate now, how representative is this model
    over the controller horizon?  Both position and velocity are retained,
    because velocity error is often the first sign that an inertial fit is
    compensating one effect with another.
    """
    import mujoco.rollout

    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s must be finite and positive")
    if len(run.control) < 2:
        raise ValueError(f"{run.label} has too few samples for validation")

    steps = max(int(round(window_s / model.opt.timestep)), 2)
    data = mujoco.MjData(model)
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    q_errors: list[NDArray[np.float64]] = []
    dq_errors: list[NDArray[np.float64]] = []
    ee_errors: list[NDArray[np.float64]] = []

    complete_samples = (len(run.control) // steps) * steps
    for start in range(0, complete_samples, steps):
        stop = start + steps
        data.qpos[:7] = run.measured[start, :7]
        data.qvel[:7] = run.measured[start, 7:14]
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(
            model,
            data,
            state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
        )
        _, sensors = mujoco.rollout.rollout(model, data, state, run.control[start:stop])
        predicted = np.squeeze(sensors, axis=0)
        measured = run.measured[start:stop]
        q_errors.append(predicted[:, :7] - measured[:, :7])
        dq_errors.append(predicted[:, 7:14] - measured[:, 7:14])
        ee_errors.append(
            1e3
            * np.linalg.norm(
                _site_positions(model, predicted[:, :7], site_name=site_name)
                - _site_positions(model, measured[:, :7], site_name=site_name),
                axis=1,
            )
        )

    if not q_errors:
        raise ValueError(f"{run.label} has no complete validation window")
    q = np.vstack(q_errors)
    dq = np.vstack(dq_errors)
    ee = np.concatenate(ee_errors)
    return RolloutMetrics(
        q_rmse_rad=np.sqrt(np.mean(q**2, axis=0)),
        dq_rmse_rad_s=np.sqrt(np.mean(dq**2, axis=0)),
        gripper_rmse_mm=float(np.sqrt(np.mean(ee**2))),
        gripper_max_mm=float(np.max(ee)),
        windows=len(q_errors),
        samples=len(q),
    )


def evaluate_models(
    candidates: Mapping[str, mujoco.MjModel],
    run: MeasuredRun,
    *,
    horizons_s: Sequence[float],
) -> dict[str, dict[str, RolloutMetrics]]:
    """Evaluate every candidate on the same withheld input and state rows."""
    return {
        f"{horizon:g}s": {
            name: rollout_metrics(model, run, window_s=horizon)
            for name, model in candidates.items()
        }
        for horizon in horizons_s
    }


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Pre-registered release rules in relative and physical error units."""

    max_regression_fraction: float = 0.02
    minimum_improvement_fraction: float = 0.01
    q_absolute_tolerance_rad: float = 1e-5
    dq_absolute_tolerance_rad_s: float = 1e-4
    gripper_absolute_tolerance_mm: float = 0.05
    torque_absolute_tolerance_Nm: float = 1e-3
    q_rmse_ceiling_rad: float = 0.1
    dq_rmse_ceiling_rad_s: float = 0.5
    gripper_rmse_ceiling_mm: float = 25.0
    gripper_max_ceiling_mm: float = 75.0
    torque_rmse_ceiling_Nm: float = 5.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field.name} must be a finite nonnegative number"
                ) from error
            if not np.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{field.name} must be a finite nonnegative number")


@dataclass(frozen=True)
class ReproductionAcceptance:
    """Why a candidate model may or may not be released."""

    accepted: bool
    problems: tuple[str, ...]
    relative_scores: dict[str, float]

    def require(self) -> None:
        if not self.accepted:
            raise RuntimeError(
                "identified model failed held-out reproduction:\n  - "
                + "\n  - ".join(self.problems)
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "problems": list(self.problems),
            "relative_scores": self.relative_scores,
        }


def _metric_values(
    metrics: RolloutMetrics,
) -> tuple[tuple[str, float], ...]:
    return (
        *tuple(
            (f"q_rmse_rad[{joint}]", float(value))
            for joint, value in enumerate(metrics.q_rmse_rad)
        ),
        *tuple(
            (f"dq_rmse_rad_s[{joint}]", float(value))
            for joint, value in enumerate(metrics.dq_rmse_rad_s)
        ),
        ("gripper_rmse_mm", float(metrics.gripper_rmse_mm)),
        ("gripper_max_mm", float(metrics.gripper_max_mm)),
    )


def _metric_problem(
    metrics: RolloutMetrics,
    *,
    context: str,
) -> tuple[str, ...]:
    problems: list[str] = []
    for field_name in ("q_rmse_rad", "dq_rmse_rad_s"):
        values = np.asarray(getattr(metrics, field_name), dtype=np.float64)
        if values.shape != (7,):
            problems.append(
                f"{context} {field_name} has shape {values.shape}, expected (7,)"
            )
            continue
        for joint, value in enumerate(values):
            if not np.isfinite(value) or value < 0.0:
                problems.append(
                    f"{context} {field_name}[{joint}] must be finite and "
                    f"nonnegative, got {value!r}"
                )
    for field_name in ("gripper_rmse_mm", "gripper_max_mm"):
        value = float(getattr(metrics, field_name))
        if not np.isfinite(value) or value < 0.0:
            problems.append(
                f"{context} {field_name} must be finite and nonnegative, got {value!r}"
            )
    if metrics.windows <= 0 or metrics.samples <= 0:
        problems.append(
            f"{context} must contain at least one complete validation window"
        )
    return tuple(problems)


def _metric_limits(
    name: str,
    thresholds: AcceptanceThresholds,
) -> tuple[float, float]:
    if name.startswith("q_rmse_rad["):
        return (
            thresholds.q_absolute_tolerance_rad,
            thresholds.q_rmse_ceiling_rad,
        )
    if name.startswith("dq_rmse_rad_s["):
        return (
            thresholds.dq_absolute_tolerance_rad_s,
            thresholds.dq_rmse_ceiling_rad_s,
        )
    if name == "gripper_rmse_mm":
        return (
            thresholds.gripper_absolute_tolerance_mm,
            thresholds.gripper_rmse_ceiling_mm,
        )
    if name == "gripper_max_mm":
        return (
            thresholds.gripper_absolute_tolerance_mm,
            thresholds.gripper_max_ceiling_mm,
        )
    raise ValueError(f"unknown reproduction metric {name!r}")


def accept_reproduction(
    evaluations: Mapping[str, Mapping[str, Mapping[str, RolloutMetrics]]],
    *,
    required_families: Sequence[str],
    torque_rmse_Nm: Mapping[str, Mapping[str, float]] | None = None,
    baseline: str = "nominal",
    candidate: str = "identified",
    thresholds: AcceptanceThresholds | None = None,
) -> ReproductionAcceptance:
    """Fail closed unless the candidate reproduces every required holdout.

    A candidate may trade a tiny amount between metrics, so regressions within
    a small numerical/measurement tolerance are allowed.  Across all
    informative metrics for a family it must nevertheless show a real median
    improvement over nominal.  Missing families, horizons, or model entries
    are failures rather than silently skipped evidence.
    """
    thresholds = thresholds or AcceptanceThresholds()
    problems: list[str] = []
    relative_scores: dict[str, float] = {}

    for family in required_families:
        family_evaluations = evaluations.get(family)
        if not family_evaluations:
            problems.append(f"missing held-out {family} evaluation")
            continue
        ratios: list[float] = []
        for horizon, models in family_evaluations.items():
            if baseline not in models or candidate not in models:
                problems.append(
                    f"{family} {horizon} lacks {baseline}/{candidate} comparison"
                )
                continue
            base_metrics = models[baseline]
            candidate_metrics = models[candidate]
            base_problems = _metric_problem(
                base_metrics,
                context=f"{family} {horizon} {baseline}",
            )
            candidate_problems = _metric_problem(
                candidate_metrics,
                context=f"{family} {horizon} {candidate}",
            )
            problems.extend(base_problems)
            problems.extend(candidate_problems)
            if base_problems or candidate_problems:
                continue

            base_values = dict(_metric_values(base_metrics))
            candidate_values = dict(_metric_values(candidate_metrics))
            for name, base_value in base_values.items():
                value = candidate_values[name]
                tolerance, absolute_ceiling = _metric_limits(name, thresholds)
                relative_ceiling = (
                    base_value * (1.0 + thresholds.max_regression_fraction) + tolerance
                )
                if value > relative_ceiling:
                    problems.append(
                        f"{family} {horizon} {name} regressed "
                        f"{base_value:.6g} -> {value:.6g}"
                    )
                if value > absolute_ceiling:
                    problems.append(
                        f"{family} {horizon} {name} exceeds absolute fidelity "
                        f"ceiling {absolute_ceiling:.6g}: {value:.6g}"
                    )
                if base_value > tolerance:
                    ratios.append(value / base_value)

        if not ratios:
            problems.append(f"{family} holdout has no informative error metric")
        else:
            score = float(np.median(ratios))
            relative_scores[family] = score
            if score > 1.0 - thresholds.minimum_improvement_fraction:
                problems.append(
                    f"{family} median held-out error ratio is {score:.3f}; "
                    f"required <= "
                    f"{1.0 - thresholds.minimum_improvement_fraction:.3f}"
                )

        if torque_rmse_Nm is not None:
            torque = torque_rmse_Nm.get(family)
            if not torque or baseline not in torque or candidate not in torque:
                problems.append(f"missing held-out {family} torque comparison")
            else:
                base_torque = float(torque[baseline])
                candidate_torque = float(torque[candidate])
                base_valid = np.isfinite(base_torque) and base_torque >= 0.0
                candidate_valid = (
                    np.isfinite(candidate_torque) and candidate_torque >= 0.0
                )
                if not base_valid:
                    problems.append(
                        f"{family} {baseline} torque RMSE must be finite and "
                        f"nonnegative, got {base_torque!r}"
                    )
                if not candidate_valid:
                    problems.append(
                        f"{family} {candidate} torque RMSE must be finite and "
                        f"nonnegative, got {candidate_torque!r}"
                    )
                if candidate_valid and (
                    candidate_torque > thresholds.torque_rmse_ceiling_Nm
                ):
                    problems.append(
                        f"{family} torque RMSE exceeds absolute fidelity "
                        f"ceiling {thresholds.torque_rmse_ceiling_Nm:.6g}: "
                        f"{candidate_torque:.6g} Nm"
                    )
                relative_ceiling = (
                    base_torque * (1.0 + thresholds.max_regression_fraction)
                    + thresholds.torque_absolute_tolerance_Nm
                )
                if (
                    base_valid
                    and candidate_valid
                    and candidate_torque > relative_ceiling
                ):
                    problems.append(
                        f"{family} torque RMSE regressed "
                        f"{base_torque:.6g} -> {candidate_torque:.6g} Nm"
                    )

    return ReproductionAcceptance(
        accepted=not problems,
        problems=tuple(problems),
        relative_scores=relative_scores,
    )
