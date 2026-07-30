"""Focused tests for the staged friction/dynamics coordination."""

from __future__ import annotations

import numpy as np
import pytest

from fer_mujoco_sysid.fitting import friction_parameters
from fer_mujoco_sysid.stages import _maximum_parameter_change_fraction


def test_parameter_change_is_normalized_by_declared_bound_span(
    hydrax_model,
) -> None:
    previous = friction_parameters(
        hydrax_model,
        joints=("joint1",),
        include_damping=False,
    )
    current = previous.copy()
    current["joint1_frictionloss"].update_from_vector(np.array([0.3]))

    # Frictionloss is bounded on [0, 3] Nm.
    assert _maximum_parameter_change_fraction(previous, current) == pytest.approx(0.1)


def test_parameter_change_compares_only_coordinates_active_in_both(
    hydrax_model,
) -> None:
    """A freeze decision that differs between two solves must not abort the run.

    The alternating refinement re-decides which viscous slopes the data support
    on every round, so the two blocks it compares can differ by a coefficient.
    The change is then measured on what they share.
    """
    previous = friction_parameters(hydrax_model, joints=("joint1", "joint2"))
    current = previous.copy()
    current["joint1_frictionloss"].update_from_vector(np.array([0.6]))
    current["joint2_damping"].frozen = True

    assert _maximum_parameter_change_fraction(previous, current) == pytest.approx(0.2)
    # Symmetric: the frozen side may equally be the earlier block.
    assert _maximum_parameter_change_fraction(current, previous) == pytest.approx(0.2)
