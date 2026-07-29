"""Simulation-only numerical-closure diagnostics."""

from __future__ import annotations

import numpy as np

from fer_mujoco_sysid.diagnostics import predicted_torque
from fer_mujoco_sysid.simulation_truth import truth_model
from fer_mujoco_sysid.simulation_validation import _truth_consistent_acceleration


def test_truth_consistent_acceleration_closes_the_torque_equation(
    model_paths,
) -> None:
    model = truth_model(model_paths.hydrax, gravity=False)
    q = np.array(
        [
            [0.0, -0.7, 0.1, -2.1, 0.2, 1.5, 0.7],
            [0.1, -0.8, -0.1, -2.0, -0.2, 1.4, 0.8],
        ]
    )
    dq = np.array(
        [
            [0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2],
            [-0.1, 0.2, -0.2, 0.3, -0.3, 0.1, -0.2],
        ]
    )
    torque = np.array(
        [
            [1.0, -2.0, 1.5, -1.0, 0.5, -0.4, 0.3],
            [-1.2, 1.8, -1.4, 1.1, -0.6, 0.5, -0.2],
        ]
    )

    ddq = _truth_consistent_acceleration(model, q, dq, torque)

    np.testing.assert_allclose(
        predicted_torque(model, q, dq, ddq),
        torque,
        rtol=0.0,
        atol=2e-14,
    )
