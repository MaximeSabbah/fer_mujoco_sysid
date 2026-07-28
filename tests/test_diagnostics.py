"""Gates for the classical identification diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from fer_mujoco_sysid.diagnostics import (
    friction_regressor,
    friction_regressor_report,
    parameter_quality,
    predicted_torque,
    torque_residuals,
)
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec


def _velocity_profile(speeds: tuple[float, ...], samples: int = 400) -> np.ndarray:
    """Constant-velocity plateaus at +/- each speed, one column per joint."""
    blocks = [
        np.full(samples, sign * speed) for speed in speeds for sign in (1.0, -1.0)
    ]
    return np.tile(np.concatenate(blocks)[:, None], (1, 7))


def test_friction_regressor_columns_are_sign_and_velocity() -> None:
    dq = np.zeros((4, 7))
    dq[:, 0] = [0.5, -0.5, 0.0, 0.2]  # the zero sample is not "moving"
    regressor = friction_regressor(dq, 0)
    assert regressor.shape == (3, 2)
    assert np.array_equal(regressor[:, 0], [1.0, -1.0, 1.0])
    assert np.array_equal(regressor[:, 1], [0.5, -0.5, 0.2])


def test_several_speeds_are_well_conditioned() -> None:
    """Three speeds in both directions separate Coulomb from viscous."""
    report = friction_regressor_report(_velocity_profile((0.05, 0.15, 0.4)))
    assert report.parameter_names == ("frictionloss", "damping")
    assert report.worst_condition_number < 5.0
    assert report.is_well_excited()


def test_single_speed_is_ill_conditioned() -> None:
    """One speed makes sign(dq) and dq proportional: the classic failure."""
    report = friction_regressor_report(_velocity_profile((0.2,)))
    assert not report.is_well_excited()
    assert report.worst_condition_number > 1e6


def test_armature_needs_acceleration_to_be_identifiable() -> None:
    """With no acceleration the armature column is zero and cond blows up."""
    dq = _velocity_profile((0.05, 0.15, 0.4))
    constant_speed = np.zeros_like(dq)
    accelerating = np.tile(np.sin(np.linspace(0.0, 20.0, len(dq)))[:, None], (1, 7))

    without = friction_regressor_report(dq, constant_speed)
    with_acceleration = friction_regressor_report(dq, accelerating)

    assert not without.is_well_excited()
    assert with_acceleration.worst_condition_number < without.worst_condition_number


def test_parameter_quality_flags_uncertain_parameters() -> None:
    covariance = np.diag([0.01**2, 0.5**2])  # 1% and 50% relative
    quality = parameter_quality(("good", "bad"), np.array([1.0, 1.0]), covariance)
    assert quality.relative_percent[0] == pytest.approx(1.0)
    assert quality.relative_percent[1] == pytest.approx(50.0)
    assert quality.rejected == ["bad"]
    assert "FREEZE" in quality.table()


def test_parameter_quality_rejects_zero_valued_parameters() -> None:
    """A parameter identified as 0 has undefined relative error."""
    quality = parameter_quality(("zero",), np.array([0.0]), np.diag([0.1**2]))
    assert quality.rejected == ["zero"]


def test_predicted_torque_recovers_friction(model_paths: ModelPaths) -> None:
    """Inverse dynamics of a model with friction exceeds the frictionless one
    by exactly the friction torque — the quantity the torque plots show."""
    spec = build_hydrax_arm_spec(model_paths.hydrax)
    nominal = spec.compile()
    frictionloss = 0.8
    spec.joint("joint4").frictionloss = frictionloss
    with_friction = spec.compile()

    samples = 50
    q = np.tile(np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), (samples, 1))
    dq = np.zeros((samples, 7))
    dq[:, 3] = 0.2  # joint 4 moving at constant speed
    ddq = np.zeros((samples, 7))

    difference = predicted_torque(with_friction, q, dq, ddq) - predicted_torque(
        nominal, q, dq, ddq
    )
    assert difference[:, 3] == pytest.approx(frictionloss, abs=1e-6)
    assert np.abs(np.delete(difference, 3, axis=1)).max() < 1e-9


def test_torque_residual_report() -> None:
    measured = np.zeros((10, 7))
    predicted = np.zeros((10, 7))
    predicted[:, 2] = 0.5
    report = torque_residuals(measured, predicted)
    assert report.rmse_Nm[2] == pytest.approx(0.5)
    assert report.worst_rmse == pytest.approx(0.5)
    assert report.mean_Nm[2] == pytest.approx(-0.5)


def test_regressor_report_on_a_real_protocol(model_paths: ModelPaths) -> None:
    """The committed campaign must be well excited for friction."""
    from fer_mujoco_sysid.campaign import CAMPAIGN, CAMPAIGN_REVISION, repository_root
    from fer_mujoco_sysid.excitation import load_protocol_bundle

    _, arrays = load_protocol_bundle(
        repository_root() / "protocols" / CAMPAIGN[0].protocol_id / CAMPAIGN_REVISION
    )
    report = friction_regressor_report(arrays["dq_rad_s"])
    assert report.is_well_excited(limit=10.0), report.condition_number
    assert np.all(report.sample_count > 1000)
