"""The exported-model release rule is based on withheld reproduction."""

from __future__ import annotations

from dataclasses import fields, replace

import mujoco
import mujoco.rollout
import numpy as np
import pytest

from fer_mujoco_sysid.fitting import MeasuredRun
from fer_mujoco_sysid.validation import (
    AcceptanceThresholds,
    RolloutMetrics,
    accept_reproduction,
    rollout_metrics,
)


def _metrics(scale: float) -> RolloutMetrics:
    return RolloutMetrics(
        q_rmse_rad=0.02 * scale * np.ones(7),
        dq_rmse_rad_s=0.1 * scale * np.ones(7),
        gripper_rmse_mm=10.0 * scale,
        gripper_max_mm=15.0 * scale,
        windows=4,
        samples=100,
    )


def test_every_fitted_family_requires_held_out_reproduction() -> None:
    evaluations = {
        "fer-friction": {
            "0.5s": {"nominal": _metrics(1.0), "identified": _metrics(0.5)}
        }
    }
    decision = accept_reproduction(
        evaluations,
        required_families=("fer-friction", "fer-inertial"),
    )

    assert not decision.accepted
    assert any("missing held-out fer-inertial" in item for item in decision.problems)
    with pytest.raises(RuntimeError, match="failed held-out reproduction"):
        decision.require()


def test_a_model_that_improves_every_holdout_is_releasable() -> None:
    evaluations = {
        family: {
            horizon: {
                "nominal": _metrics(1.0),
                "identified": _metrics(scale),
            }
            for horizon, scale in (("0.1s", 0.8), ("0.5s", 0.6), ("2s", 0.5))
        }
        for family in ("fer-friction", "fer-inertial")
    }
    torque = {
        "fer-friction": {"nominal": 2.0, "identified": 0.8},
        "fer-inertial": {"nominal": 3.0, "identified": 1.2},
    }

    decision = accept_reproduction(
        evaluations,
        required_families=("fer-friction", "fer-inertial"),
        torque_rmse_Nm=torque,
    )

    assert decision.accepted
    assert decision.problems == ()
    decision.require()


def test_a_single_material_regression_blocks_export() -> None:
    evaluations = {
        "fer-friction": {
            "0.5s": {
                "nominal": _metrics(1.0),
                "identified": RolloutMetrics(
                    q_rmse_rad=0.01 * np.ones(7),
                    dq_rmse_rad_s=0.05 * np.ones(7),
                    gripper_rmse_mm=12.0,
                    gripper_max_mm=15.0,
                    windows=4,
                    samples=100,
                ),
            }
        }
    }

    decision = accept_reproduction(
        evaluations,
        required_families=("fer-friction",),
    )

    assert not decision.accepted
    assert any("gripper_rmse_mm regressed" in item for item in decision.problems)


def test_non_worst_joint_regression_blocks_export() -> None:
    nominal = _metrics(1.0)
    nominal.q_rmse_rad[0] = 0.005
    identified = _metrics(0.5)
    identified.q_rmse_rad[0] = 0.007
    assert identified.worst_q_rmse_rad < nominal.worst_q_rmse_rad

    decision = accept_reproduction(
        {"fer-friction": {"0.5s": {"nominal": nominal, "identified": identified}}},
        required_families=("fer-friction",),
    )

    assert not decision.accepted
    assert any("q_rmse_rad[0] regressed" in item for item in decision.problems)


def test_gripper_max_regression_blocks_export() -> None:
    identified = replace(_metrics(0.5), gripper_max_mm=20.0)
    decision = accept_reproduction(
        {
            "fer-friction": {
                "0.5s": {
                    "nominal": _metrics(1.0),
                    "identified": identified,
                }
            }
        },
        required_families=("fer-friction",),
    )

    assert not decision.accepted
    assert any("gripper_max_mm regressed" in item for item in decision.problems)


def test_nonfinite_or_negative_metrics_fail_closed() -> None:
    nominal = replace(_metrics(1.0), gripper_rmse_mm=-1.0)
    identified = _metrics(0.5)
    identified.dq_rmse_rad_s[2] = np.nan

    decision = accept_reproduction(
        {"fer-friction": {"0.5s": {"nominal": nominal, "identified": identified}}},
        required_families=("fer-friction",),
        torque_rmse_Nm={"fer-friction": {"nominal": 2.0, "identified": np.nan}},
    )

    assert not decision.accepted
    assert any(
        "nominal gripper_rmse_mm must be finite and nonnegative" in problem
        for problem in decision.problems
    )
    assert any(
        "identified dq_rmse_rad_s[2] must be finite and nonnegative" in problem
        for problem in decision.problems
    )
    assert any(
        "identified torque RMSE must be finite and nonnegative" in problem
        for problem in decision.problems
    )


def test_absolute_fidelity_ceiling_blocks_an_improving_model() -> None:
    decision = accept_reproduction(
        {
            "fer-friction": {
                "0.5s": {
                    "nominal": _metrics(1.0),
                    "identified": _metrics(0.5),
                }
            }
        },
        required_families=("fer-friction",),
        thresholds=AcceptanceThresholds(q_rmse_ceiling_rad=0.005),
    )

    assert not decision.accepted
    assert any(
        "q_rmse_rad[0] exceeds absolute fidelity ceiling" in problem
        for problem in decision.problems
    )


def test_torque_absolute_fidelity_ceiling_is_independent_of_improvement() -> None:
    decision = accept_reproduction(
        {
            "fer-friction": {
                "0.5s": {
                    "nominal": _metrics(1.0),
                    "identified": _metrics(0.5),
                }
            }
        },
        required_families=("fer-friction",),
        torque_rmse_Nm={"fer-friction": {"nominal": 2.0, "identified": 0.8}},
        thresholds=AcceptanceThresholds(torque_rmse_ceiling_Nm=0.5),
    )

    assert not decision.accepted
    assert any(
        "torque RMSE exceeds absolute fidelity ceiling" in problem
        for problem in decision.problems
    )


@pytest.mark.parametrize("invalid", (-1.0, np.inf, np.nan))
def test_all_thresholds_are_finite_and_nonnegative(invalid: float) -> None:
    for field in fields(AcceptanceThresholds):
        with pytest.raises(ValueError, match=field.name):
            AcceptanceThresholds(**{field.name: invalid})


def test_rollout_ignores_incomplete_trailing_window(
    hydrax_model: mujoco.MjModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window_steps = 4
    total_samples = 2 * window_steps + 3
    rollout_lengths: list[int] = []

    def fake_rollout(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: np.ndarray,
        control: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del data, state
        rollout_lengths.append(len(control))
        states = np.empty((1, len(control), 0), dtype=np.float64)
        sensors = np.zeros((1, len(control), 14), dtype=np.float64)
        return states, sensors

    monkeypatch.setattr(mujoco.rollout, "rollout", fake_rollout)
    run = MeasuredRun(
        label="partial-window",
        qpos0=np.zeros(hydrax_model.nq),
        qvel0=np.zeros(hydrax_model.nv),
        control_times=np.arange(total_samples) * hydrax_model.opt.timestep,
        control=np.zeros((total_samples, hydrax_model.nu)),
        measured_times=np.arange(total_samples) * hydrax_model.opt.timestep,
        measured=np.zeros((total_samples, 14)),
    )

    metrics = rollout_metrics(
        hydrax_model,
        run,
        window_s=window_steps * hydrax_model.opt.timestep,
    )

    assert rollout_lengths == [window_steps, window_steps]
    assert metrics.windows == 2
    assert metrics.samples == 2 * window_steps
