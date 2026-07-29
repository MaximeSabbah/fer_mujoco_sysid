"""Control causality and campaign lineage at the recording/fitting boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fer_mujoco_sysid.preparation import (
    _causal_event_resample,
    _causal_resample,
    _thin_event_mask,
    validate_campaign_lineage,
)


def test_control_resampling_never_uses_a_future_sample() -> None:
    source_time = np.array([0.0, 0.1, 0.2])
    values = np.array([[1.0], [2.0], [3.0]])
    target_time = np.array([0.0, 0.05, 0.099, 0.1, 0.199, 0.2])

    result = _causal_resample(source_time, values, target_time)

    np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 1.0, 2.0, 2.0, 3.0])


def test_sparse_100hz_events_are_not_dilated_onto_the_model_grid() -> None:
    source_time = np.array([0.0, 0.01, 0.02])
    values = np.array([True, True, True])
    target_time = np.arange(0.0, 0.021, 0.002)

    result = _causal_event_resample(source_time, values, target_time)

    assert np.flatnonzero(result).tolist() == [0, 5, 10]


def test_simulation_diagnostics_are_thinned_to_the_real_100hz_rate() -> None:
    result = _thin_event_mask(np.ones(16, dtype=bool), step_s=0.002)
    assert np.flatnonzero(result).tolist() == [0, 5, 10, 15]


def test_empty_campaign_is_rejected() -> None:
    with pytest.raises(ValueError, match="no recordings"):
        validate_campaign_lineage([])


def _lineage_record(family: str, role: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        backend="mujoco",
        source_model_sha256="a" * 64,
        protocol_id=f"{family}-{role}-{index}",
        content_sha256=f"{index:064x}",
        family=family,
        role=role,
    )


def test_complete_campaign_requires_both_training_families() -> None:
    friction = [
        _lineage_record("fer-friction", "train", 1),
        _lineage_record("fer-friction", "holdout", 2),
    ]
    with pytest.raises(ValueError, match="training data for fer-inertial"):
        validate_campaign_lineage(friction)


def test_complete_campaign_requires_both_holdout_families() -> None:
    records = [
        _lineage_record("fer-friction", "train", 1),
        _lineage_record("fer-friction", "holdout", 2),
        _lineage_record("fer-inertial", "train", 3),
    ]
    with pytest.raises(ValueError, match="held-out reproduction for fer-inertial"):
        validate_campaign_lineage(records)


def test_complete_campaign_with_both_families_is_accepted() -> None:
    records = [
        _lineage_record("fer-friction", "train", 1),
        _lineage_record("fer-friction", "holdout", 2),
        _lineage_record("fer-inertial", "train", 3),
        _lineage_record("fer-inertial", "holdout", 4),
    ]
    validate_campaign_lineage(records)
