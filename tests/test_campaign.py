"""P3 slice 2 gates: the committed friction campaign."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.campaign import (
    CAMPAIGN,
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
from fer_mujoco_sysid.io import sha256_file
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec
from fer_mujoco_sysid.protocol import (
    HOLDOUT_ROLE,
    TRAIN_ROLE,
    protocol_analysis_windows,
    protocol_family,
    protocol_role,
)


@pytest.fixture(scope="module")
def nominal_model(model_paths: ModelPaths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True).compile()


def test_committed_bundles_validate() -> None:
    """Every campaign bundle loads, validates, and checksums cleanly."""
    protocols = repository_root() / "protocols"
    for spec in CAMPAIGN:
        manifest, arrays = load_protocol_bundle(protocols / spec.protocol_id)
        assert manifest["protocol_id"] == spec.protocol_id
        assert protocol_family(manifest) == spec.family_id
        assert protocol_role(manifest) == spec.role
        expected_role = (
            HOLDOUT_ROLE if spec.protocol_id.endswith("-holdout") else TRAIN_ROLE
        )
        assert protocol_role(manifest) == expected_role
        assert protocol_analysis_windows(manifest)
        assert manifest["playback"]["command_interface"] == "joint_trajectory"
        assert len(arrays["q_rad"]) == len(arrays["time_s"])
        assert np.all(np.diff(arrays["time_s"]) > 0)


def test_campaign_regeneration_matches_committed() -> None:
    """Generator drift guard: regenerating every spec reproduces the
    committed content hashes exactly."""
    assert verify_campaign(repository_root() / "protocols") == []


def test_canonical_payloads_preserve_the_reviewed_motion() -> None:
    """Flattening protocol storage must not alter any trajectory archive."""
    protocols = repository_root() / "protocols"
    expected = {
        "fer-friction-a": (
            "a33e71274f8a900735ef0620052eca241819fa6028cb1855c13aa7dc169034fb"
        ),
        "fer-friction-b": (
            "53b7eac7469885c33cc8fd92c613759fdfecd69a2f7fb8e1003240ade5dbeb88"
        ),
        "fer-friction-holdout": (
            "f2c1f10e11effc50af1ee6f6e38474a9a44ed7bfe77e7355b4552ce3df13ea30"
        ),
        "fer-inertial-a": (
            "d0b703ca351d1bb0ac5086d01419b5246bc639f8533c757106fd51a5868bb7cc"
        ),
        "fer-inertial-b": (
            "9f348e2aa227d90442e48af2180e66662e9810e94ad17c912facc2458c298c19"
        ),
        "fer-inertial-holdout": (
            "0ad26ec39c6ced707097a6a6ce9f3b831eb8fc35a56bb1954a09354bf695cce8"
        ),
    }
    assert {spec.protocol_id for spec in CAMPAIGN} == set(expected)
    for protocol_id, digest in expected.items():
        assert sha256_file(protocols / protocol_id / "desired.npz") == digest


@pytest.mark.slow
def test_committed_campaign_conditions_friction_block(
    model_paths: ModelPaths, nominal_model: mujoco.MjModel
) -> None:
    """Simulated playback of the canonical protocol identifies the full
    friction block (the numbers reported in protocols/review/summary.md)."""
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
