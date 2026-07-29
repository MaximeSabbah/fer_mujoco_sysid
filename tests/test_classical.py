"""Focused gates for the direct friction regressor."""

from __future__ import annotations

import numpy as np
import pytest

from fer_mujoco_sysid.classical import newey_west_covariance


def test_hac_uncertainty_accounts_for_serially_correlated_telemetry() -> None:
    rng = np.random.default_rng(7)
    rows = 4000
    block = 200
    direction = np.repeat(
        np.tile(np.array([1.0, -1.0]), rows // (2 * block)), block
    )
    speed = np.repeat(
        np.tile(np.array([0.05, 0.15, 0.4, 0.15]), rows // (4 * block)),
        block,
    )
    design = np.column_stack([direction, direction * speed])

    innovations = rng.normal(scale=0.05, size=rows)
    residual = np.empty(rows)
    residual[0] = innovations[0]
    for index in range(1, rows):
        residual[index] = 0.85 * residual[index - 1] + innovations[index]

    row_independent = newey_west_covariance(design, residual, max_lags=0)
    robust = newey_west_covariance(design, residual, max_lags=10)
    assert np.all(np.diag(robust) > 2.0 * np.diag(row_independent))


def test_hac_rejects_invalid_lag_count() -> None:
    design = np.column_stack([np.ones(20), np.linspace(-1.0, 1.0, 20)])
    with pytest.raises(ValueError, match="max_lags"):
        newey_west_covariance(design, np.zeros(20), max_lags=20)
