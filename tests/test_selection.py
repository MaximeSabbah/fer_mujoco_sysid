"""Analysis masks become real rollout boundaries, never fake transitions."""

from __future__ import annotations

import numpy as np
import pytest

from fer_mujoco_sysid.fitting import MeasuredRun
from fer_mujoco_sysid.selection import contiguous_regions, runs_from_mask


def _run(samples: int = 10) -> MeasuredRun:
    time = np.arange(samples, dtype=np.float64) * 0.01
    measured = np.column_stack(
        [
            np.repeat(np.arange(samples, dtype=np.float64)[:, None], 7, axis=1),
            np.ones((samples, 7)),
        ]
    )
    return MeasuredRun(
        label="recording",
        qpos0=measured[0, :7],
        qvel0=measured[0, 7:14],
        control_times=time,
        control=np.zeros((samples, 7)),
        measured_times=time + 0.01,
        measured=measured,
    )


def test_disjoint_masks_do_not_create_a_fake_rollout_transition() -> None:
    mask = np.array(
        [False, True, True, True, False, False, True, True, True, False]
    )

    selected = runs_from_mask(_run(), mask)

    assert len(selected) == 2
    assert [len(run.control) for run in selected] == [3, 3]
    np.testing.assert_allclose(selected[0].qpos0, 1.0)
    np.testing.assert_allclose(selected[1].qpos0, 6.0)
    np.testing.assert_allclose(selected[0].control_times[0], 0.0)
    np.testing.assert_allclose(selected[1].control_times[0], 0.0)


def test_short_mask_is_rejected_instead_of_silently_pooled() -> None:
    mask = np.array([False, True, False, False])
    assert contiguous_regions(mask, minimum_samples=2) == ()
    with pytest.raises(ValueError, match="no usable selected interval"):
        runs_from_mask(_run(4), mask)


def test_mask_and_run_lengths_must_agree() -> None:
    with pytest.raises(ValueError, match="mask has 3 rows"):
        runs_from_mask(_run(4), np.ones(3, dtype=bool))
