"""The in-process backend writes the same bound contract as ROS conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fer_mujoco_sysid.campaign import (
    FRICTION_CAMPAIGN,
    repository_root,
)
from fer_mujoco_sysid.dataset import (
    BOUND_RECORDING_FORMAT,
    load_recording,
)
from fer_mujoco_sysid.model import ModelPaths
from fer_mujoco_sysid.preparation import prepare_recording
from fer_mujoco_sysid.protocol import (
    FRICTION_FAMILY,
    TRAIN_ROLE,
    alignment_from_start_time,
    friction_cruise_mask,
    load_protocol_bundle,
)
from fer_mujoco_sysid.simulate import (
    SYNTHETIC_MARKER_FORMAT,
    run,
)
from fer_mujoco_sysid.simulation_truth import write_truth_manifest
from fer_mujoco_sysid.stages import fitting_spec


def test_in_process_recording_is_fully_bound_without_trimming(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    spec = FRICTION_CAMPAIGN[0]
    model = fitting_spec(model_paths.hydrax).compile()
    [recording_root] = run(
        tmp_path,
        truth=model,
        protocols=(spec,),
        seed=17,
    )
    manifest, arrays = load_recording(recording_root)
    protocol, desired = load_protocol_bundle(
        repository_root() / "protocols" / spec.protocol_id
    )

    assert manifest["format"] == BOUND_RECORDING_FORMAT
    assert manifest["protocol"]["protocol_id"] == protocol["protocol_id"]
    assert manifest["protocol"]["family"] == FRICTION_FAMILY
    assert manifest["protocol"]["role"] == TRAIN_ROLE
    assert manifest["protocol"]["content_sha256"] == protocol["content_sha256"]
    assert manifest["protocol"]["segments"] == protocol["segments"]
    assert manifest["protocol"]["analysis_windows"] == protocol["analysis_windows"]
    assert manifest["source_model"] == protocol["source_model"]
    assert manifest["simulation"]["preprocessing_mode"] == "raw_engine_exact"
    assert manifest["simulation"]["position_noise_rad"] == 0.0
    assert manifest["simulation"]["velocity_noise_rad_s"] == 0.0

    expected_samples = int(
        round(float(desired["time_s"][-1]) / float(model.opt.timestep))
    )
    assert len(arrays["time_s"]) == expected_samples
    assert arrays["q_rad"].shape == (expected_samples, 7)
    assert np.all(arrays["protocol_segment_index"] >= 0)

    alignment = alignment_from_start_time(protocol, start_time_s=0.0)
    expected_analysis = friction_cruise_mask(
        protocol,
        desired,
        arrays["time_s"],
        alignment,
    )
    np.testing.assert_array_equal(
        arrays["analysis_eligible"],
        expected_analysis,
    )
    assert expected_analysis.any()
    assert not expected_analysis.all()

    assert manifest["protocol_timing"]["marker_format"] == SYNTHETIC_MARKER_FORMAT
    assert (
        manifest["torque_channels"]["tau_cmd_Nm"]["application_status"]
        == "not_asserted"
    )
    assert manifest["health"]["usable"]

    prepared = prepare_recording(
        recording_root,
        model,
        expected_source_model=model_paths.hydrax,
    )
    assert prepared.family == FRICTION_FAMILY
    assert prepared.role == TRAIN_ROLE
    assert prepared.protocol_content_sha256 == protocol["content_sha256"]
    assert prepared.preprocessing["kind"] == "none"


def test_in_process_recording_binds_a_shared_truth_and_is_immutable(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    truth_path = write_truth_manifest(
        tmp_path / "shared_truth.json",
        model_paths.hydrax,
    )
    output = tmp_path / "mujoco"
    [recording_root] = run(
        output,
        protocols=(FRICTION_CAMPAIGN[0],),
        simulation_truth_path=truth_path,
    )
    manifest, _ = load_recording(recording_root)
    assert manifest["simulation"]["truth"]["manifest"] == str(truth_path.resolve())

    with pytest.raises(FileExistsError, match="immutable"):
        run(
            output,
            protocols=(FRICTION_CAMPAIGN[0],),
            simulation_truth_path=truth_path,
        )
