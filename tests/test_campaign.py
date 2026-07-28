"""P3 slice 2 gates: the committed friction campaign."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.campaign import (
    CAMPAIGN,
    CAMPAIGN_REVISION,
    repository_root,
    tracking_run,
    verify_campaign,
)
from fer_mujoco_sysid.excitation import (
    generate_friction_protocol,
    load_protocol_bundle,
)
from fer_mujoco_sysid.fitting import (
    CONDITIONING_RATIO_MINIMUM,
    CORRELATION_FREEZE_LIMIT,
    conditioning_report,
    friction_parameters,
    measurement_sequences,
)
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec


@pytest.fixture(scope="module")
def nominal_model(model_paths: ModelPaths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True).compile()


def test_committed_bundles_validate() -> None:
    """Every campaign bundle loads, validates, and checksums cleanly."""
    protocols = repository_root() / "protocols"
    for spec in CAMPAIGN:
        manifest, arrays = load_protocol_bundle(
            protocols / spec.protocol_id / CAMPAIGN_REVISION
        )
        assert manifest["protocol_id"] == spec.protocol_id
        assert manifest["playback"]["command_interface"] == "joint_trajectory"
        assert len(arrays["q_rad"]) == len(arrays["time_s"])
        assert np.all(np.diff(arrays["time_s"]) > 0)


def test_campaign_regeneration_matches_committed() -> None:
    """Generator drift guard: regenerating every spec reproduces the
    committed content hashes exactly."""
    assert verify_campaign(repository_root() / "protocols") == []


@pytest.mark.slow
def test_committed_campaign_conditions_friction_block(
    model_paths: ModelPaths, nominal_model: mujoco.MjModel
) -> None:
    """Simulated playback of the canonical protocol identifies the full
    friction block (the numbers reported in docs/protocol_review/summary.md)."""
    compiled = generate_friction_protocol(CAMPAIGN[0], nominal_model)
    run = tracking_run(nominal_model, compiled)
    sequences = measurement_sequences(
        build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True),
        [run],
        window_s=0.5,
    )
    parameters = friction_parameters(nominal_model).move_off_bounds()
    report = conditioning_report(parameters, sequences)

    assert report.conditioning_ratio >= CONDITIONING_RATIO_MINIMUM
    correlations = report.parameter_correlations
    for joint in range(7):
        pair = abs(correlations[2 * joint, 2 * joint + 1])
        assert pair < CORRELATION_FREEZE_LIMIT, f"joint{joint + 1}: {pair:.3f}"
