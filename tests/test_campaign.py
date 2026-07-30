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
    """No trajectory archive changes without someone updating this ledger.

    Both families were re-reviewed on 2026-07-30. The friction family was
    rebuilt after the first hardware campaign showed 87% of its regressor
    samples coming from one cruise speed and one direction carrying twice the
    data of the reverse. The inertial amplitudes were raised in the same review
    because joint 1 was excited to 17% of its acceleration limit while the wrist
    sat at 80%, leaving the proximal link inertias unidentifiable.
    """
    protocols = repository_root() / "protocols"
    expected = {
        "fer-friction-a": (
            "ad579630337659a3ff988f4449fe84a8dae215f01f15d06b44fab2d242df2262"
        ),
        "fer-friction-b": (
            "ea6b65f6f09072f2d745ed753c26c93719086d467946d5818da408dd8f2c8bba"
        ),
        "fer-friction-holdout": (
            "7e43db19e1edf25bca47ef4779ee5212fe021c8ea4ab4c1351eb15036cafb279"
        ),
        "fer-inertial-a": (
            "aca8c1c17cfe3f4f141abe2fbdbc21c48c984e3026fedc306227e7ce8ecd9260"
        ),
        "fer-inertial-b": (
            "9902fb714d306442707a3757cb750380bfc90b3ed886470569179dc45587bbb2"
        ),
        "fer-inertial-holdout": (
            "4a71114ccb0ad03a41a019d8272999dceaf96e0e42b9f80d032750ce2b903703"
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
