"""Focused gates for protocol-bound ROS acquisition.

The converter helpers are intentionally ROS-free at import time, so timestamp,
marker, and telemetry semantics can be checked with small message-shaped
objects instead of requiring a running graph or a robot.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fer_mujoco_sysid.campaign import repository_root
from fer_mujoco_sysid.dataset import (
    BOUND_RECORDING_FORMAT,
    RecordingError,
    health_report,
    recording_analysis_mask,
    recording_rollout_mask,
    validate_recording,
)
from fer_mujoco_sysid.playback import (
    PlaybackError,
    approach_required,
    check_start_velocity,
)
from fer_mujoco_sysid.protocol import (
    DEFAULT_ANALYSIS_EDGE_GUARD_S,
    ROS_ARM_JOINT_NAMES,
    alignment_from_start_time,
    load_protocol_bundle,
    map_analysis_windows,
    map_protocol_segments,
)
from fer_mujoco_sysid.ros.convert import (
    CONTROLLER_STATE_TOPIC,
    CONTROLLER_STATE_TYPE,
    FRANKA_STATE_TOPIC,
    FRANKA_STATE_TYPE,
    PROTOCOL_EVENT_FORMAT,
    PROTOCOL_EVENT_TOPIC,
    PROTOCOL_EVENT_TYPE,
    ConversionError,
    _controller_series,
    _crop_controller,
    _franka_series,
    _mapped_intervals,
    _protocol_bounds,
    _row_interval_arrays,
    _simulation_truth_metadata,
    _transient_free_mask,
)
from fer_mujoco_sysid.simulation_truth import write_truth_manifest


def _bundle(protocol_id: str = "fer-friction-a"):
    root = repository_root() / "protocols" / protocol_id
    return load_protocol_bundle(root)


def _header(stamp_ns: int, *, frame_id: str = ""):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        ),
        frame_id=frame_id,
    )


def _event(stamp_ns: int, event: str, protocol, *, speed_scale: float = 1.0):
    payload = {
        "format": PROTOCOL_EVENT_FORMAT,
        "event": event,
        "protocol_id": protocol["protocol_id"],
        "content_sha256": protocol["content_sha256"],
        "speed_scale": speed_scale,
    }
    return SimpleNamespace(
        header=None,
        frame_id=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        stamp=_header(stamp_ns).stamp,
    )


def _controller(stamp_ns: int, value: float):
    vector = [value + joint for joint in range(7)]
    return SimpleNamespace(
        header=_header(stamp_ns),
        joint_names=list(ROS_ARM_JOINT_NAMES),
        output=SimpleNamespace(effort=vector),
        feedback=SimpleNamespace(positions=vector, velocities=vector),
        reference=SimpleNamespace(positions=vector),
    )


def _franka(stamp_ns: int, value: float):
    vector = [value] * 7
    return SimpleNamespace(
        header=_header(stamp_ns),
        measured_joint_state=SimpleNamespace(effort=vector),
        desired_joint_state=SimpleNamespace(effort=[value + 100.0] * 7),
        measured_joint_motor_state=SimpleNamespace(
            position=[value + 200.0] * 7,
            velocity=[value + 300.0] * 7,
        ),
    )


def test_simulation_truth_is_hash_and_source_bound(
    tmp_path: Path,
    model_paths,
) -> None:
    path = write_truth_manifest(tmp_path / "truth.json", model_paths.hydrax)
    protocol, _ = _bundle()
    metadata = _simulation_truth_metadata(
        path,
        expected_source_sha256=protocol["source_model"]["sha256"],
    )
    assert metadata["content_sha256"]
    assert metadata["source_model_sha256"] == protocol["source_model"]["sha256"]

    raw = json.loads(path.read_text())
    raw["parameters"]["frictionloss"][0] += 0.1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConversionError, match="content SHA-256 mismatch"):
        _simulation_truth_metadata(
            path,
            expected_source_sha256=protocol["source_model"]["sha256"],
        )


def test_markers_crop_the_approach_from_controller_rows() -> None:
    protocol, _ = _bundle()
    start_ns = 10_000_000_000
    end_ns = start_ns + 1_000_000_000
    messages = [
        (PROTOCOL_EVENT_TOPIC, _event(start_ns, "start", protocol), start_ns),
        (
            CONTROLLER_STATE_TOPIC,
            _controller(start_ns - 100_000_000, -1.0),
            start_ns - 100_000_000,
        ),
        (
            CONTROLLER_STATE_TOPIC,
            _controller(start_ns + 10_000_000, 1.0),
            start_ns + 10_000_000,
        ),
        (
            CONTROLLER_STATE_TOPIC,
            _controller(start_ns + 900_000_000, 2.0),
            start_ns + 900_000_000,
        ),
        (
            CONTROLLER_STATE_TOPIC,
            _controller(end_ns + 100_000_000, 3.0),
            end_ns + 100_000_000,
        ),
        (PROTOCOL_EVENT_TOPIC, _event(end_ns, "end", protocol), end_ns),
    ]
    types = {
        PROTOCOL_EVENT_TOPIC: PROTOCOL_EVENT_TYPE,
        CONTROLLER_STATE_TOPIC: CONTROLLER_STATE_TYPE,
    }

    bounds = _protocol_bounds(messages, types, protocol, 1.0)
    cropped = _crop_controller(_controller_series(messages, types), *bounds)

    np.testing.assert_array_equal(
        cropped[0], [start_ns + 10_000_000, start_ns + 900_000_000]
    )
    np.testing.assert_allclose(cropped[1][:, 0], [1.0, 2.0])


def test_missing_or_wrong_protocol_markers_fail_closed() -> None:
    protocol, _ = _bundle()
    start = _event(1_000_000_000, "start", protocol)
    types = {PROTOCOL_EVENT_TOPIC: PROTOCOL_EVENT_TYPE}
    with pytest.raises(ConversionError, match="end marker"):
        _protocol_bounds(
            [(PROTOCOL_EVENT_TOPIC, start, 1_000_000_000)],
            types,
            protocol,
            1.0,
        )

    wrong = _event(2_000_000_000, "end", protocol)
    payload = json.loads(wrong.frame_id)
    payload["content_sha256"] = "0" * 64
    wrong.frame_id = json.dumps(payload)
    with pytest.raises(ConversionError, match="content_sha256"):
        _protocol_bounds(
            [
                (PROTOCOL_EVENT_TOPIC, start, 1_000_000_000),
                (PROTOCOL_EVENT_TOPIC, wrong, 2_000_000_000),
            ],
            types,
            protocol,
            1.0,
        )


def test_mapped_recording_intervals_are_the_canonical_protocol_mapping() -> None:
    protocol, arrays = _bundle()
    speed_scale = 0.5
    segments, windows = _mapped_intervals(protocol, arrays, speed_scale)
    alignment = alignment_from_start_time(
        protocol, start_time_s=0.0, speed_scale=speed_scale
    )
    expected_segments = map_protocol_segments(protocol, arrays, alignment)
    expected_windows = map_analysis_windows(
        protocol,
        arrays,
        alignment,
        edge_guard_s=DEFAULT_ANALYSIS_EDGE_GUARD_S,
    )

    assert [(item["start_time_s"], item["end_time_s"]) for item in segments] == [
        (item.start_time_s, item.end_time_s) for item in expected_segments
    ]
    assert [(item["start_time_s"], item["end_time_s"]) for item in windows] == [
        (item.start_time_s, item.end_time_s) for item in expected_windows
    ]
    np.testing.assert_allclose(
        windows[0]["nominal_velocity_rad_s"],
        expected_windows[0].nominal_velocity_rad_s,
    )
    controller_time = np.arange(
        0.0, float(protocol["playback"]["duration_s"]) + 1e-9, 0.002
    )
    segment_index, eligible = _row_interval_arrays(controller_time, segments, windows)
    assert np.all(segment_index >= 0)
    expected_eligible = np.logical_or.reduce(
        [
            (controller_time >= interval.start_time_s)
            & (controller_time <= interval.end_time_s)
            for interval in expected_windows
        ]
    )
    np.testing.assert_array_equal(eligible, expected_eligible)


def test_franka_alignment_is_causal_and_preserves_native_100hz_stamps() -> None:
    messages = [
        (FRANKA_STATE_TOPIC, _franka(stamp, stamp / 1e6), stamp)
        for stamp in (0, 10_000_000, 20_000_000, 30_000_000)
    ]
    controller = np.asarray(
        [5_000_000, 11_000_000, 19_000_000, 21_000_000], dtype=np.int64
    )
    arrays, metadata = _franka_series(
        messages,
        {FRANKA_STATE_TOPIC: FRANKA_STATE_TYPE},
        controller,
        5_000_000,
        25_000_000,
    )

    np.testing.assert_array_equal(arrays["franka_time_ns"], [0, 10_000_000, 20_000_000])
    # Values encode the source stamp in milliseconds: every aligned value is
    # from the most recent past sample, never the closer future sample.
    np.testing.assert_allclose(arrays["tau_J_Nm"][:, 0], [0.0, 10.0, 10.0, 20.0])
    np.testing.assert_array_equal(
        arrays["franka_new_sample"], [True, True, False, True]
    )
    assert metadata["method"] == "causal_previous_sample_hold"
    assert metadata["expected_rate_hz"] == 100
    assert metadata["maximum_age_s"] == pytest.approx(0.009)


def test_stale_or_gapped_franka_telemetry_is_rejected() -> None:
    messages = [
        (FRANKA_STATE_TOPIC, _franka(stamp, 0.0), stamp)
        for stamp in (0, 40_000_000, 80_000_000)
    ]
    with pytest.raises(ConversionError, match="40.0 ms gap"):
        _franka_series(
            messages,
            {FRANKA_STATE_TOPIC: FRANKA_STATE_TYPE},
            np.asarray([5_000_000, 45_000_000], dtype=np.int64),
            5_000_000,
            75_000_000,
        )


def test_franka_rate_must_remain_the_proven_100hz() -> None:
    messages = [
        (FRANKA_STATE_TOPIC, _franka(stamp, 0.0), stamp)
        for stamp in (0, 20_000_000, 40_000_000, 60_000_000)
    ]
    with pytest.raises(ConversionError, match="50.0 Hz"):
        _franka_series(
            messages,
            {FRANKA_STATE_TOPIC: FRANKA_STATE_TYPE},
            np.asarray([5_000_000, 25_000_000, 45_000_000], dtype=np.int64),
            5_000_000,
            55_000_000,
        )


def test_telemetry_fit_mask_excludes_reversals_and_rate_limiter_requests() -> None:
    time_s = np.arange(0.0, 0.20, 0.01)
    dq = np.ones((len(time_s), 7))
    dq[10:] = -1.0
    torque = np.zeros((len(time_s), 7))
    torque[5:, 0] = 20.0  # 2000 Nm/s request: downstream limiter is unavoidable.
    mask, reversals, slew_transients = _transient_free_mask(time_s, dq, torque)

    assert reversals == 7
    assert slew_transients == 1
    assert not mask[3:9].any()
    assert not mask[7:13].any()
    assert mask[-1]


def _bound_sim_recording(*, controller_period_s: float = 0.01):
    protocol, protocol_arrays = _bundle()
    duration = float(protocol["playback"]["duration_s"])
    time_s = np.arange(0.0, duration + 1e-9, controller_period_s)
    zeros = np.zeros((len(time_s), 7))
    segments, windows = _mapped_intervals(protocol, protocol_arrays, 1.0)
    segment_index, analysis = _row_interval_arrays(time_s, segments, windows)
    arrays = {
        "time_s": time_s,
        "q_rad": zeros,
        "dq_rad_s": zeros,
        "tau_cmd_Nm": zeros,
        "protocol_segment_index": segment_index,
        "analysis_eligible": analysis,
    }
    start_ns = 10_000_000_000
    manifest = {
        "format": BOUND_RECORDING_FORMAT,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "mujoco_ros",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "family": protocol["family"],
            "role": protocol["role"],
            "content_sha256": protocol["content_sha256"],
            "duration_s": duration,
            "segments": protocol["segments"],
            "analysis_windows": protocol["analysis_windows"],
        },
        "source_model": protocol["source_model"],
        "protocol_timing": {
            "start_stamp_ns": start_ns,
            "end_stamp_ns": start_ns + int(duration * 1e9),
            "selected_first_stamp_ns": start_ns,
            "selected_last_stamp_ns": start_ns + int(duration * 1e9),
            "speed_scale": 1.0,
            "marker_topic": PROTOCOL_EVENT_TOPIC,
        },
        "mapped_segments": segments,
        "mapped_analysis_windows": windows,
        "torque_channels": {
            "tau_cmd_Nm": {
                "application_status": "not_asserted",
            }
        },
    }
    return manifest, arrays


def test_bound_recording_requires_lineage_and_persists_analysis_mask() -> None:
    manifest, arrays = _bound_sim_recording()
    validate_recording(manifest, arrays)
    np.testing.assert_array_equal(
        recording_analysis_mask(manifest, arrays), arrays["analysis_eligible"]
    )
    np.testing.assert_array_equal(
        recording_rollout_mask(manifest, arrays), arrays["analysis_eligible"]
    )

    broken = dict(manifest)
    broken["protocol"] = dict(manifest["protocol"])
    del broken["protocol"]["family"]
    with pytest.raises(RecordingError, match="missing.*family"):
        validate_recording(broken, arrays)


def test_real_rollout_mask_is_continuous_but_sparse_mask_requires_new_samples() -> None:
    manifest, arrays = _bound_sim_recording(controller_period_s=0.002)
    manifest["backend"] = "real"
    manifest["telemetry_alignment"] = {
        "method": "causal_previous_sample_hold",
        "expected_rate_hz": 100,
        "maximum_allowed_age_s": 0.035,
        "maximum_age_s": 0.010,
        "maximum_allowed_native_gap_s": 0.035,
        "maximum_native_gap_s": 0.010,
    }
    count = len(arrays["time_s"])
    zeros = np.zeros((count, 7))
    start_ns = manifest["protocol_timing"]["start_stamp_ns"]
    native_rows = np.arange(0, count, 5)
    native_zeros = np.zeros((len(native_rows), 7))
    arrays.update(
        {
            "tau_J_Nm": zeros,
            "tau_J_d_Nm": zeros,
            "theta_rad": zeros,
            "dtheta_rad_s": zeros,
            "franka_time_ns": start_ns + native_rows * 2_000_000,
            "tau_J_native_Nm": native_zeros,
            "tau_J_d_native_Nm": native_zeros,
            "theta_native_rad": native_zeros,
            "dtheta_native_rad_s": native_zeros,
            "franka_sample_age_s": (np.arange(count) % 5) * 0.002,
            "franka_new_sample": np.arange(count) % 5 == 0,
            "telemetry_transient_free": np.ones(count, dtype=np.bool_),
        }
    )
    eligible_rows = np.flatnonzero(arrays["analysis_eligible"])
    transient = int(eligible_rows[20])
    stale = int(eligible_rows[40])
    arrays["telemetry_transient_free"][transient] = False
    arrays["franka_sample_age_s"][stale] = 0.040
    expected_rollout = (
        arrays["analysis_eligible"]
        & arrays["telemetry_transient_free"]
        & (arrays["franka_sample_age_s"] <= 0.035)
    )
    arrays["telemetry_fit_eligible"] = expected_rollout & arrays["franka_new_sample"]

    validate_recording(manifest, arrays)
    np.testing.assert_array_equal(
        recording_rollout_mask(manifest, arrays, require_telemetry=True),
        expected_rollout,
    )
    np.testing.assert_array_equal(
        recording_analysis_mask(manifest, arrays, require_telemetry=True),
        expected_rollout & arrays["franka_new_sample"],
    )

    broken = dict(arrays)
    broken["telemetry_fit_eligible"] = arrays["telemetry_fit_eligible"].copy()
    broken["telemetry_fit_eligible"][stale] = True
    with pytest.raises(RecordingError, match="stale"):
        validate_recording(manifest, broken)


def test_health_reports_torque_slew_and_desired_telemetry_mismatch() -> None:
    time_s = np.arange(5, dtype=np.float64) * 0.01
    command = np.zeros((5, 7))
    command[2:] = 2.0
    arrays = {
        "time_s": time_s,
        "q_rad": np.zeros((5, 7)),
        "dq_rad_s": np.zeros((5, 7)),
        "tau_cmd_Nm": command,
        "tau_J_d_Nm": np.ones((5, 7)),
        "franka_new_sample": np.asarray([True, False, True, False, True]),
    }
    manifest = {
        "format": "fer-mujoco-sysid/recording@1",
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "real",
        "protocol": {"duration_s": 0.04},
        "torque_limit_Nm": [87.0] * 4 + [12.0] * 3,
    }
    report = health_report(manifest, arrays)
    assert report["torque_slew"]["maximum_Nm_s"][0] == pytest.approx(200.0)
    assert report["tau_cmd_minus_tau_J_d"]["samples"] == 3
    assert "not asserted" in report["tau_cmd_minus_tau_J_d"]["interpretation"]


def test_real_approach_is_opt_in_but_simulation_still_exercises_it() -> None:
    assert approach_required(
        real_robot=False,
        at_protocol_start=True,
        allow_real_approach=False,
    )
    assert not approach_required(
        real_robot=True,
        at_protocol_start=True,
        allow_real_approach=False,
    )
    assert approach_required(
        real_robot=True,
        at_protocol_start=False,
        allow_real_approach=True,
    )
    with pytest.raises(PlaybackError, match="disabled on hardware"):
        approach_required(
            real_robot=True,
            at_protocol_start=False,
            allow_real_approach=False,
        )


def test_real_start_gate_also_requires_the_arm_to_be_at_rest() -> None:
    protocol, _ = _bundle()
    stopped = check_start_velocity(np.zeros(7), protocol)
    assert stopped.satisfied

    moving = np.zeros(7)
    moving[3] = 0.05
    check = check_start_velocity(moving, protocol)
    assert not check.satisfied
    assert check.worst_joint == 3
    assert "fer_joint4" in check.describe()


def test_wrapper_rejects_an_existing_recording_before_launch(tmp_path: Path) -> None:
    existing = tmp_path / "fer-friction-a"
    existing.mkdir()
    result = subprocess.run(
        [
            str(repository_root() / "scripts" / "run-protocol"),
            str(repository_root() / "protocols" / "fer-friction-a"),
            "--record",
            str(tmp_path),
        ],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "recording destination already exists" in result.stderr


def _fake_wrapper_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    ros_log = tmp_path / "ros.log"

    uv = fake_bin / "uv"
    uv.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "${FER_TEST_UV_LOG:?}"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)

    ros2 = fake_bin / "ros2"
    ros2.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "${FER_TEST_ROS_LOG:?}"\n'
        "exit 23\n",
        encoding="utf-8",
    )
    ros2.chmod(0o755)

    ros_setup = tmp_path / "ros-setup.bash"
    ros_setup.write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FER_SYSID_ROS_SETUP": str(ros_setup),
            "FER_SYSID_WORKSPACE_SETUP": str(tmp_path / "no-workspace-setup.bash"),
            "FER_TEST_UV_LOG": str(uv_log),
            "FER_TEST_ROS_LOG": str(ros_log),
        }
    )
    return environment, uv_log, ros_log


@pytest.mark.parametrize("parameter_flag", ("--parameters", "--simulation-truth"))
def test_wrapper_resolves_record_and_parameter_paths_before_chdir(
    tmp_path: Path,
    parameter_flag: str,
) -> None:
    environment, uv_log, ros_log = _fake_wrapper_environment(tmp_path)
    parameter_path = tmp_path / "inputs" / "plant.json"
    parameter_path.parent.mkdir()
    parameter_path.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            str(repository_root() / "scripts" / "run-protocol"),
            str(repository_root() / "protocols" / "fer-friction-a"),
            "--record",
            "relative-recordings",
            parameter_flag,
            "inputs/plant.json",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    expected_parameter = str(parameter_path.resolve())
    expected_record = str(
        (
            tmp_path
            / "relative-recordings"
            / "fer-friction-a"
            / "raw"
        ).resolve()
    )
    assert f"--parameters {expected_parameter}" in uv_log.read_text()
    assert f"record:={expected_record}" in ros_log.read_text()


def test_wrapper_resolves_generated_truth_under_relative_record_root(
    tmp_path: Path,
) -> None:
    environment, uv_log, _ = _fake_wrapper_environment(tmp_path)
    result = subprocess.run(
        [
            str(repository_root() / "scripts" / "run-protocol"),
            str(repository_root() / "protocols" / "fer-friction-a"),
            "--record",
            "relative-recordings",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 23
    truth = str((tmp_path / "relative-recordings" / "simulation_truth.json").resolve())
    invocations = uv_log.read_text().splitlines()
    assert len(invocations) == 2
    assert f"fer_mujoco_sysid.simulation_truth {truth}" in invocations[0]
    assert f"--parameters {truth}" in invocations[1]


@pytest.mark.parametrize(
    "parameter_flag",
    ("--parameters", "--simulation-truth", "--friction"),
)
def test_wrapper_rejects_simulation_parameters_for_real_backend_without_side_effects(
    tmp_path: Path,
    parameter_flag: str,
) -> None:
    environment, uv_log, ros_log = _fake_wrapper_environment(tmp_path)
    result = subprocess.run(
        [
            str(repository_root() / "scripts" / "run-protocol"),
            str(repository_root() / "protocols" / "fer-friction-a"),
            "--backend",
            "real",
            "--record",
            "real-recordings",
            parameter_flag,
            "inputs/plant.json",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "they cannot be used with --backend real" in result.stderr
    assert not (tmp_path / "real-recordings").exists()
    assert not uv_log.exists()
    assert not ros_log.exists()


def test_launch_keeps_the_recorder_fail_closed_and_telemetry_at_100hz() -> None:
    root = repository_root()
    launch = (root / "launch" / "play_protocol.launch.py").read_text()
    config = (root / "config" / "fer_sysid_controllers.yaml").read_text()
    assert '"--start-paused"' in launch
    assert '"require_recorder": bool(record_dir)' in launch
    assert "PROTOCOL_EVENT_TOPIC" in launch
    assert "on_exit=Shutdown()" in launch
    assert "update_rate: 100" in config
