from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from fer_mujoco_sysid.artifacts.content import content_sha256
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import sha256_file
from fer_mujoco_sysid.artifacts.semantic import (
    validate_acquisition_run,
    validate_fit_result,
    validate_identified_parameters,
    validate_splits,
)
from fer_mujoco_sysid.artifacts.validation import FER_ARM_JOINT_ORDER

_ZERO_SHA256 = "0" * 64
_REVISION = "a" * 40


def _source_document() -> dict[str, Any]:
    return {
        "repository": "https://example.invalid/fer.git",
        "revision": _REVISION,
        "path": "models/fer.xml",
        "sha256": "b" * 64,
    }


def _payload() -> dict[str, Any]:
    return {
        "payload_id": "fer-hand",
        "mass_kg": 0.73,
        "center_of_mass_m": [0.0, 0.0, 0.03],
    }


def _file_reference(
    root: Path,
    relative: str = "metadata/config.json",
) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"fixture":true}\n')
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_reference(
    artifact_id: str,
    schema: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "schema": schema,
        "manifest_sha256": _ZERO_SHA256,
        "locator": {
            "kind": "relative",
            "path": f"artifacts/{artifact_id}.json",
        },
    }


def _stamp(manifest: dict[str, Any]) -> None:
    manifest["content_sha256"] = content_sha256(manifest, {})


def _state_signal(
    name: str,
    quantity: str,
    semantic_role: str,
    unit: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "quantity": quantity,
        "semantic_role": semantic_role,
        "unit": unit,
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "ros-clock",
        "source": {
            "kind": "ros_message",
            "topic": "/joint_states",
            "message_type": "sensor_msgs/msg/JointState",
            "field": {
                "joint_position": "position",
                "joint_velocity": "velocity",
                "joint_acceleration": "acceleration",
                "joint_temperature": "temperature",
                "joint_motor_current": "effort",
            }[quantity],
        },
        "sample_count": 100,
    }


def _effort_signal() -> dict[str, Any]:
    return {
        "name": "measured_effort",
        "quantity": "joint_effort",
        "semantic_role": "measured_link_effort",
        "unit": "N*m",
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "ros-clock",
        "source": {
            "kind": "ros_message",
            "topic": "/joint_states",
            "message_type": "sensor_msgs/msg/JointState",
            "field": "effort",
        },
        "sample_count": 100,
        "effort_semantics": {
            "stage": "sensor",
            "location": "link_side",
            "gravity": "included",
            "coriolis": "included",
            "friction_compensation": "unknown",
            "rate_limit_position": "not_applicable",
            "definition": "Measured generalized link-side effort.",
            "authority": _source_document(),
        },
    }


def _protocol_reference_signal(
    *,
    name: str,
    quantity: str,
    semantic_role: str,
    unit: str,
    field: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "quantity": quantity,
        "semantic_role": semantic_role,
        "unit": unit,
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "ros-clock",
        "source": {
            "kind": "ros_message",
            "topic": "/protocol_reference",
            "message_type": "fer_mujoco_sysid_msgs/msg/ProtocolReference",
            "field": field,
        },
        "sample_count": 100,
    }


def _acquisition(root: Path) -> dict[str, Any]:
    configuration = _file_reference(root)
    recording = _file_reference(root, "raw/run.mcap")
    return {
        "schema": "fer-mujoco-sysid/acquisition-run@1",
        "artifact_id": "acquisition-fixture",
        "started_at_utc": "2026-07-23T10:00:00Z",
        "ended_at_utc": "2026-07-23T10:01:00Z",
        "backend": "agimus_fer",
        "protocol": _artifact_reference(
            "protocol-fixture",
            "fer-mujoco-sysid/motion-protocol@1",
        ),
        "protocol_reference_signals": {
            "desired_position": "scheduled_position",
            "desired_velocity": "scheduled_velocity",
            "desired_acceleration": "scheduled_acceleration",
        },
        "protocol_timing": {
            "clock_id": "ros-clock",
            "protocol_start_timestamp_ns": "0",
            "source": {
                "kind": "ros_message",
                "topic": "/protocol_reference",
                "message_type": ("fer_mujoco_sysid_msgs/msg/ProtocolReference"),
                "message_index": 0,
                "stamp_fields": {
                    "sec": "header.stamp.sec",
                    "nanosec": "header.stamp.nanosec",
                },
            },
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "robot": {
            "platform": "FER",
            "robot_id": "fer-test",
            "end_effector_id": "fer-hand",
            "payload": _payload(),
        },
        "controller": {
            "name": "trajectory-controller",
            "type": "joint_trajectory",
            "update_rate_hz": 1000.0,
            "configuration": configuration,
        },
        "source_model": _source_document(),
        "clock_domains": [
            {
                "clock_id": "ros-clock",
                "domain": "ros_time",
                "epoch": "ROS clock epoch",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "message header",
            }
        ],
        "signals": [
            _state_signal(
                "measured_position",
                "joint_position",
                "measured_joint_position",
                "rad",
            ),
            _state_signal(
                "measured_velocity",
                "joint_velocity",
                "measured_joint_velocity",
                "rad/s",
            ),
            _effort_signal(),
            _protocol_reference_signal(
                name="scheduled_position",
                quantity="joint_position",
                semantic_role="desired_joint_position",
                unit="rad",
                field="q_rad",
            ),
            _protocol_reference_signal(
                name="scheduled_velocity",
                quantity="joint_velocity",
                semantic_role="desired_joint_velocity",
                unit="rad/s",
                field="dq_rad_s",
            ),
            _protocol_reference_signal(
                name="scheduled_acceleration",
                quantity="joint_acceleration",
                semantic_role="desired_joint_acceleration",
                unit="rad/s^2",
                field="ddq_rad_s2",
            ),
        ],
        "recording_files": [recording],
        "recorded_topics": [
            {
                "topic": "/joint_states",
                "message_type": "sensor_msgs/msg/JointState",
                "message_count": 100,
                "required_for_conversion": True,
                "clock_id": "ros-clock",
            },
            {
                "topic": "/protocol_reference",
                "message_type": ("fer_mujoco_sysid_msgs/msg/ProtocolReference"),
                "message_count": 100,
                "required_for_conversion": True,
                "clock_id": "ros-clock",
            },
        ],
        "software": [{"name": "rosbag2", "version": "0.30.0"}],
        "outcome": {"status": "completed", "reason_code": "completed_normally"},
    }


def _use_standalone_mujoco_sources(manifest: dict[str, Any]) -> None:
    manifest["backend"] = "standalone_mujoco"
    for signal, source in zip(
        manifest["signals"],
        [
            {"kind": "mujoco", "object_type": "state", "field": "qpos"},
            {"kind": "mujoco", "object_type": "state", "field": "qvel"},
            {"kind": "mujoco", "object_type": "actuator", "field": "force"},
            {
                "kind": "protocol_player",
                "protocol_artifact_id": manifest["protocol"]["artifact_id"],
                "field": "q_rad",
            },
            {
                "kind": "protocol_player",
                "protocol_artifact_id": manifest["protocol"]["artifact_id"],
                "field": "dq_rad_s",
            },
            {
                "kind": "protocol_player",
                "protocol_artifact_id": manifest["protocol"]["artifact_id"],
                "field": "ddq_rad_s2",
            },
        ],
        strict=True,
    ):
        signal["source"] = source
    manifest["protocol_timing"]["source"] = {
        "kind": "mujoco",
        "object_type": "data",
        "field": "time",
        "sample_index": 0,
        "unit": "s",
    }


def _component(
    name: str,
    *,
    value: float = 0.2,
    lower: float = 1e-6,
    upper: float = 1.0,
    bound_status: str = "none",
    status: str = "retained",
    export_policy: str = "mjcf",
    unit: str = "N*m",
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "nominal_value": 0.1,
        "initial_value": 0.1,
        "bounds": {"lower": lower, "upper": upper},
        "bound_status": bound_status,
        "identifiability_status": status,
        "export_policy": export_policy,
        "uncertainty": {
            "method": "trajectory_bootstrap",
            "confidence_level": 0.95,
            "lower": 0.05,
            "upper": 0.4,
        },
    }


def _conditioning() -> dict[str, Any]:
    return {
        "dimension": 1,
        "effective_rank": 1,
        "sigma_min_over_sigma_max": 0.8,
        "max_absolute_correlation": 0.1,
        "multistart_score_spread": 0.02,
    }


def _identified_parameters() -> dict[str, Any]:
    manifest = {
        "schema": "fer-mujoco-sysid/identified-parameters@1",
        "artifact_id": "parameters-fixture",
        "content_sha256": _ZERO_SHA256,
        "created_at": "2026-07-23T10:00:00Z",
        "source_model": _source_document(),
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "parameterization_id": "fer-dynamics-v1",
        "payload": _payload(),
        "model_parameters": [
            {
                "group_id": "friction.joint1",
                "family": "joint_frictionloss",
                "target": {"kind": "joint", "joint_name": "fer_joint1"},
                "stage_id": "friction",
                "optimization_transform": "log",
                "physical_constraint": "nonnegative",
                "retained": True,
                "components": [_component("frictionloss")],
                "conditioning": _conditioning(),
            }
        ],
        "nuisance_parameters": [
            {
                "group_id": "torque.scale",
                "family": "torque_scale",
                "target": {"kind": "global", "name": "torque_scale"},
                "stage_id": "nuisance",
                "optimization_transform": "identity",
                "physical_constraint": "positive",
                "retained": True,
                "components": [
                    {
                        **_component(
                            "scale",
                            value=1.0,
                            lower=0.5,
                            upper=1.5,
                            status="retained",
                            export_policy="report_only",
                            unit="1",
                        ),
                        "nominal_value": 1.0,
                        "initial_value": 1.0,
                        "uncertainty": {
                            "method": "trajectory_bootstrap",
                            "confidence_level": 0.95,
                            "lower": 0.9,
                            "upper": 1.1,
                        },
                    }
                ],
                "conditioning": _conditioning(),
            }
        ],
    }
    _stamp(manifest)
    return manifest


def _split_member(
    trajectory_id: str,
    lineage: str,
) -> dict[str, Any]:
    return {
        "trajectory": _artifact_reference(
            trajectory_id,
            "fer-mujoco-sysid/normalized-trajectory@1",
        ),
        "lineage_group_id": lineage,
        "protocol_artifact_id": "protocol-fixture",
        "source_run_artifact_id": f"run-{trajectory_id}",
    }


def _splits(root: Path) -> dict[str, Any]:
    configuration = _file_reference(root)
    metrics = _file_reference(root, "metadata/metrics.json")
    manifest = {
        "schema": "fer-mujoco-sysid/splits@1",
        "artifact_id": "splits-fixture",
        "content_sha256": _ZERO_SHA256,
        "dataset_id": "dataset-fixture",
        "dataset_version": "1.0.0",
        "created_at": "2026-07-23T10:00:00Z",
        "locked_at": "2026-07-23T11:00:00Z",
        "grouping_policy": {
            "group_key": "lineage_group_id",
            "description": "Keep source lineage in one active partition.",
            "configuration": configuration,
        },
        "held_out_policy": {
            "pre_registered_metrics": metrics,
            "unseal_condition": "Publish after final model selection.",
        },
        "partitions": {
            "fit": [_split_member("trajectory-fit", "lineage-fit")],
            "development": [],
            "held_out_test": [_split_member("trajectory-test", "lineage-test")],
            "diagnostic_only": [],
            "excluded": [
                {
                    "member": _split_member(
                        "trajectory-excluded",
                        "lineage-fit",
                    ),
                    "reason_code": "quality_failed",
                }
            ],
        },
    }
    _stamp(manifest)
    return manifest


def _fit_result(root: Path) -> dict[str, Any]:
    configuration = _file_reference(root)
    manifest = {
        "schema": "fer-mujoco-sysid/fit-result@1",
        "artifact_id": "fit-result-fixture",
        "content_sha256": _ZERO_SHA256,
        "created_at": "2026-07-23T12:00:00Z",
        "status": "completed",
        "toolchain": {
            "project": {"name": "fer-mujoco-sysid", "version": "0.1.0"},
            "python_version": "3.12.11",
            "mujoco_version": "3.10.0",
            "dependency_lock": configuration,
            "platform": {
                "operating_system": "Linux",
                "architecture": "x86_64",
                "cpu": "test-cpu",
                "thread_count": 4,
            },
        },
        "inputs": {
            "source_model": _source_document(),
            "dataset": _artifact_reference(
                "dataset-fixture",
                "fer-mujoco-sysid/dataset@1",
            ),
            "splits": _artifact_reference(
                "splits-fixture",
                "fer-mujoco-sysid/splits@1",
            ),
            "trajectories": [
                _artifact_reference(
                    "trajectory-fit",
                    "fer-mujoco-sysid/normalized-trajectory@1",
                )
            ],
        },
        "data_use": {
            "optimizer": "fit",
            "model_selection": ["fit", "development"],
            "final_evaluation": "held_out_test",
        },
        "forward_model_input": {
            "signal_name": "measured_effort",
            "semantic_role": "measured_link_effort",
            "selection_rationale": "Identified link-side generalized effort.",
            "transformation": {
                "scale": 1.0,
                "offset_Nm": 0.0,
                "time_shift_s": 0.0,
            },
        },
        "fit_configuration": {
            "parameters": configuration,
            "windowing": configuration,
            "residual": configuration,
            "pre_registered_metrics": configuration,
        },
        "optimizer_stages": [
            {
                "stage_id": "friction",
                "backend": "mujoco",
                "parameter_families": ["joint_frictionloss"],
                "seed": 7,
                "max_iterations": 10,
                "configuration": configuration,
                "status": "completed",
                "termination_reason": "converged",
                "initial_objective": 10.0,
                "final_objective": 1.0,
            }
        ],
        "reproduction": {
            "argv": ["fer-mujoco-sysid", "fit", "dataset-fixture"],
            "working_directory": "workspace",
        },
        "outputs": {
            "identified_parameters": _artifact_reference(
                "parameters-fixture",
                "fer-mujoco-sysid/identified-parameters@1",
            ),
            "metrics": configuration,
        },
    }
    _stamp(manifest)
    return manifest


def test_valid_acquisition_run(tmp_path: Path) -> None:
    validate_acquisition_run(_acquisition(tmp_path), root=tmp_path)


def test_valid_standalone_acquisition_uses_protocol_player_references(
    tmp_path: Path,
) -> None:
    manifest = _acquisition(tmp_path)
    _use_standalone_mujoco_sources(manifest)

    validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize("source_kind", ["ros_message", "protocol_player"])
def test_acquisition_rejects_wrong_scheduled_position_field(
    tmp_path: Path,
    source_kind: str,
) -> None:
    manifest = _acquisition(tmp_path)
    if source_kind == "protocol_player":
        _use_standalone_mujoco_sources(manifest)
        manifest["signals"][3]["source"]["field"] = "dq_rad_s"
    else:
        manifest["signals"][3]["source"]["field"] = "unrelated_payload"

    with pytest.raises(ArtifactValidationError, match="source field must be"):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("signal_index", "fake_field"),
    [
        (3, "desired_q"),
        (4, "desired_dq"),
        (5, "desired_ddq"),
    ],
)
def test_standalone_rejects_protocol_reference_from_mujoco_data(
    tmp_path: Path,
    signal_index: int,
    fake_field: str,
) -> None:
    manifest = _acquisition(tmp_path)
    _use_standalone_mujoco_sources(manifest)
    manifest["signals"][signal_index]["source"] = {
        "kind": "mujoco",
        "object_type": "data",
        "field": fake_field,
    }

    with pytest.raises(ArtifactValidationError, match="from the protocol player"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_standalone_rejects_scheduled_reference_from_another_protocol(
    tmp_path: Path,
) -> None:
    manifest = _acquisition(tmp_path)
    _use_standalone_mujoco_sources(manifest)
    manifest["signals"][3]["source"]["protocol_artifact_id"] = "other-protocol"

    with pytest.raises(ArtifactValidationError, match="different protocol artifact"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_standalone_checks_each_protocol_player_role_field(
    tmp_path: Path,
) -> None:
    manifest = _acquisition(tmp_path)
    _use_standalone_mujoco_sources(manifest)
    manifest["signals"][4]["source"]["field"] = "q_rad"

    with pytest.raises(ArtifactValidationError, match="source field must be 'dq_rad_s'"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_ros_rejects_protocol_velocity_from_controller_stream(
    tmp_path: Path,
) -> None:
    manifest = _acquisition(tmp_path)
    manifest["signals"][4]["source"] = {
        "kind": "ros_message",
        "topic": "/joint_states",
        "message_type": "sensor_msgs/msg/JointState",
        "field": "velocity",
    }

    with pytest.raises(ArtifactValidationError, match="desired_velocity.*must share"):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize("backend", ["ros_mujoco", "agimus_fer"])
def test_ros_backends_reject_direct_mujoco_signal_sources(
    tmp_path: Path,
    backend: str,
) -> None:
    manifest = _acquisition(tmp_path)
    manifest["backend"] = backend
    manifest["signals"][0]["source"] = {
        "kind": "mujoco",
        "object_type": "state",
        "field": "qpos",
    }

    with pytest.raises(ArtifactValidationError, match="permits signal source kinds"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_verifies_recursive_file_references(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    (tmp_path / manifest["controller"]["configuration"]["path"]).write_bytes(
        b"mutated\n"
    )

    with pytest.raises(ArtifactValidationError, match="size mismatch|SHA-256 mismatch"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_rejects_file_traversal(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    manifest["controller"]["configuration"]["path"] = "../outside.json"

    with pytest.raises(ArtifactValidationError, match="schema violation"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_rejects_inverted_timestamps(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    manifest["ended_at_utc"] = "2026-07-23T09:59:59Z"

    with pytest.raises(ArtifactValidationError, match="must not follow"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_completed_acquisition_rejects_zero_duration(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    manifest["ended_at_utc"] = manifest["started_at_utc"]

    with pytest.raises(ArtifactValidationError, match="positive duration"):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("collection", "field", "match"),
    [
        ("clock_domains", "clock_id", "clock domain ID"),
        ("signals", "name", "signal name"),
        ("recording_files", "path", "recording path"),
        ("recorded_topics", "topic", "recorded topic"),
    ],
)
def test_acquisition_rejects_duplicate_identities(
    tmp_path: Path,
    collection: str,
    field: str,
    match: str,
) -> None:
    manifest = _acquisition(tmp_path)
    duplicate = deepcopy(manifest[collection][0])
    if collection == "recording_files":
        manifest[collection].append(duplicate)
    elif collection == "recorded_topics":
        manifest[collection].append(duplicate)
    elif collection == "clock_domains":
        duplicate["domain"] = "steady_receive"
        manifest[collection].append(duplicate)
    else:
        duplicate["semantic_role"] = "desired_joint_position"
        manifest[collection].append(duplicate)
    assert manifest[collection][0][field] == duplicate[field]

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_rejects_unknown_signal_clock(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    manifest["signals"][0]["clock_id"] = "missing-clock"

    with pytest.raises(ArtifactValidationError, match="unknown clock"):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_rejects_unknown_protocol_timing_clock(
    tmp_path: Path,
) -> None:
    manifest = _acquisition(tmp_path)
    manifest["protocol_timing"]["clock_id"] = "missing-clock"

    with pytest.raises(ArtifactValidationError, match="unknown clock"):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize("fault", ["missing_topic", "wrong_type"])
def test_acquisition_rejects_ros_source_disagreement(
    tmp_path: Path,
    fault: str,
) -> None:
    manifest = _acquisition(tmp_path)
    if fault == "missing_topic":
        manifest["signals"][0]["source"]["topic"] = "/not_recorded"
        match = "not recorded"
    else:
        manifest["signals"][0]["source"]["message_type"] = "example/Wrong"
        match = "disagrees"

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("missing_topic", "not recorded"),
        ("wrong_type", "disagrees"),
        ("wrong_clock", "disagrees with recorded topic clock"),
        ("optional_topic", "must be required for conversion"),
        ("empty_topic", "no recorded protocol_timing source"),
    ],
)
def test_acquisition_rejects_protocol_timing_source_disagreement(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _acquisition(tmp_path)
    timing = manifest["protocol_timing"]
    topic = manifest["recorded_topics"][1]
    if fault == "missing_topic":
        timing["source"]["topic"] = "/not_recorded"
        for signal in manifest["signals"][3:6]:
            signal["source"]["topic"] = "/not_recorded"
    elif fault == "wrong_type":
        topic["message_type"] = "example/Wrong"
    elif fault == "wrong_clock":
        manifest["clock_domains"].append(
            {
                "clock_id": "receive-clock",
                "domain": "steady_receive",
                "epoch": "recorder start",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "steady clock",
            }
        )
        topic["clock_id"] = "receive-clock"
    elif fault == "optional_topic":
        topic["required_for_conversion"] = False
    else:
        topic["message_count"] = 0

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("unknown_signal", "names an unknown signal"),
        ("wrong_role", "must name the recorded desired_joint_position"),
        ("wrong_clock", "clock must equal protocol_timing.clock_id"),
        ("wrong_stream", "must share the protocol_timing topic"),
        ("missing_topic_clock", "disagrees with recorded topic clock"),
    ],
)
def test_acquisition_binds_protocol_timing_to_scheduled_reference(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _acquisition(tmp_path)
    timing = manifest["protocol_timing"]
    references = manifest["protocol_reference_signals"]
    if fault == "unknown_signal":
        references["desired_position"] = "not_recorded"
    elif fault == "wrong_role":
        references["desired_position"] = "measured_position"
    elif fault == "wrong_clock":
        manifest["clock_domains"].append(
            {
                "clock_id": "other-clock",
                "domain": "steady_receive",
                "epoch": "recorder start",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "steady clock",
            }
        )
        timing["clock_id"] = "other-clock"
    elif fault == "wrong_stream":
        timing["source"]["topic"] = "/joint_states"
    else:
        del manifest["recorded_topics"][1]["clock_id"]

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


def test_acquisition_rejects_incoherent_state_semantics(tmp_path: Path) -> None:
    manifest = _acquisition(tmp_path)
    signal = manifest["signals"][0]
    signal["quantity"] = "joint_velocity"
    signal["unit"] = "rad/s"
    signal["source"]["field"] = "velocity"

    with pytest.raises(ArtifactValidationError, match="requires quantity/unit"):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("topic_clock", "disagrees with recorded topic clock"),
        ("too_many_samples", "exceeds its recorded topic message_count"),
        ("joint_state_field", "must use sensor_msgs/msg/JointState.velocity"),
        ("empty_required_topic", "empty required topics"),
    ],
)
def test_acquisition_checks_recorded_topic_consistency(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _acquisition(tmp_path)
    if fault == "topic_clock":
        manifest["clock_domains"].append(
            {
                "clock_id": "receive-clock",
                "domain": "steady_receive",
                "epoch": "recorder start",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "steady clock",
            }
        )
        manifest["recorded_topics"][0]["clock_id"] = "receive-clock"
    elif fault == "too_many_samples":
        manifest["signals"][0]["sample_count"] = 101
    elif fault == "joint_state_field":
        manifest["signals"][1]["source"]["field"] = "position"
    else:
        manifest["recorded_topics"].append(
            {
                "topic": "/required_diagnostic",
                "message_type": "diagnostic_msgs/msg/DiagnosticArray",
                "message_count": 0,
                "required_for_conversion": True,
                "clock_id": "ros-clock",
            }
        )

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("position", "measured_joint_position"),
        ("velocity", "measured_joint_velocity"),
        ("effort", "joint-effort"),
        ("samples", "positive-sample measured"),
        ("scheduled_reference", "positive-sample protocol reference"),
    ],
)
def test_completed_acquisition_requires_core_positive_signals(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _acquisition(tmp_path)
    if fault == "position":
        manifest["signals"].pop(0)
    elif fault == "velocity":
        manifest["signals"].pop(1)
    elif fault == "effort":
        manifest["signals"].pop(2)
    elif fault == "scheduled_reference":
        manifest["signals"][3]["sample_count"] = 0
    else:
        manifest["signals"][0]["sample_count"] = 0

    with pytest.raises(ArtifactValidationError, match=match):
        validate_acquisition_run(manifest, root=tmp_path)


def test_valid_identified_parameters(tmp_path: Path) -> None:
    validate_identified_parameters(_identified_parameters(), root=tmp_path)


def test_identified_parameters_verifies_content_hash(tmp_path: Path) -> None:
    manifest = _identified_parameters()
    manifest["model_parameters"][0]["components"][0]["value"] = 0.3

    with pytest.raises(ArtifactValidationError, match="scientific content"):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_rejects_duplicate_group_ids(tmp_path: Path) -> None:
    manifest = _identified_parameters()
    duplicate = deepcopy(manifest["model_parameters"][0])
    duplicate["target"]["joint_name"] = "fer_joint2"
    manifest["model_parameters"].append(duplicate)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="parameter group ID"):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_rejects_duplicate_component_identity(
    tmp_path: Path,
) -> None:
    manifest = _identified_parameters()
    component = deepcopy(manifest["model_parameters"][0]["components"][0])
    manifest["model_parameters"][0]["components"].append(component)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="component identity"):
        validate_identified_parameters(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("inverted", "inverted bounds"),
        ("initial", "initial_value is outside"),
        ("value", "value is outside"),
    ],
)
def test_identified_parameters_enforces_bounds(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _identified_parameters()
    component = manifest["model_parameters"][0]["components"][0]
    if fault == "inverted":
        component["bounds"] = {"lower": 2.0, "upper": 1.0}
    elif fault == "initial":
        component["initial_value"] = -1.0
    else:
        component["value"] = 2.0
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_identified_parameters(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("value", "lower", "upper", "bound_status", "match"),
    [
        (1.0, 1.0, 1.0, "none", "must report bound_status 'both'"),
        (0.5, 0.0, 1.0, "both", "only for zero-width bounds"),
        (1.0, 0.0, 1.0, "lower", "value equals the distinct upper"),
        (0.0, 0.0, 1.0, "upper", "value equals the distinct lower"),
    ],
)
def test_identified_parameters_rejects_logically_impossible_bound_status(
    tmp_path: Path,
    value: float,
    lower: float,
    upper: float,
    bound_status: str,
    match: str,
) -> None:
    manifest = _identified_parameters()
    group = manifest["model_parameters"][0]
    group["optimization_transform"] = "identity"
    component = manifest["model_parameters"][0]["components"][0]
    component.update(
        {
            "value": value,
            "initial_value": lower,
            "bounds": {"lower": lower, "upper": upper},
            "bound_status": bound_status,
        }
    )
    if lower == upper:
        component["identifiability_status"] = "fixed"
        component["export_policy"] = "fixed"
        group["retained"] = False
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_identified_parameters(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("target_kind", "requires target kind 'joint'"),
        ("global_name", "requires global target name 'torque_scale'"),
        ("unit", "must use unit 'N\\*m'"),
        ("log_domain", "log transform with a nonpositive lower bound"),
    ],
)
def test_identified_parameters_checks_parameter_meaning(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _identified_parameters()
    model_group = manifest["model_parameters"][0]
    if fault == "target_kind":
        model_group["target"] = {"kind": "body", "body_name": "link1"}
    elif fault == "global_name":
        manifest["nuisance_parameters"][0]["target"]["name"] = "control_delay"
    elif fault == "unit":
        model_group["components"][0]["unit"] = "kg"
    else:
        model_group["components"][0]["bounds"]["lower"] = 0.0
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_identified_parameters(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("missing", "requires quantified uncertainty"),
        ("none", "requires quantified uncertainty"),
        ("inverted", "inverted uncertainty interval"),
    ],
)
def test_identified_parameters_checks_uncertainty(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _identified_parameters()
    component = manifest["model_parameters"][0]["components"][0]
    if fault == "missing":
        del component["uncertainty"]
    elif fault == "none":
        component["uncertainty"] = {"method": "none"}
    else:
        component["uncertainty"]["lower"] = 0.5
        component["uncertainty"]["upper"] = 0.4
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_requires_retained_conditioning(
    tmp_path: Path,
) -> None:
    manifest = _identified_parameters()
    del manifest["model_parameters"][0]["conditioning"]
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="requires conditioning"):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_rejects_rank_over_dimension(tmp_path: Path) -> None:
    manifest = _identified_parameters()
    manifest["model_parameters"][0]["conditioning"]["effective_rank"] = 2
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="greater than dimension"):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_rejects_more_retained_than_effective_rank(
    tmp_path: Path,
) -> None:
    manifest = _identified_parameters()
    manifest["model_parameters"][0]["conditioning"]["effective_rank"] = 0
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="effective rank supports"):
        validate_identified_parameters(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("retained_in_dropped_group", "non-retained group"),
        ("retained_model_report_only", "must use 'mjcf'"),
        ("weak_mjcf", "must use 'report_only'"),
        ("fixed_disagreement", "fixed status and export policy"),
        ("nuisance_mjcf", "must use 'report_only'"),
    ],
)
def test_identified_parameters_enforces_status_coherence(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _identified_parameters()
    model_group = manifest["model_parameters"][0]
    model_component = model_group["components"][0]
    nuisance_component = manifest["nuisance_parameters"][0]["components"][0]
    if fault == "retained_in_dropped_group":
        model_group["retained"] = False
    elif fault == "retained_model_report_only":
        model_component["export_policy"] = "report_only"
    elif fault == "weak_mjcf":
        model_component["identifiability_status"] = "weak"
    elif fault == "fixed_disagreement":
        model_component["identifiability_status"] = "fixed"
    else:
        nuisance_component["export_policy"] = "mjcf"
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_allows_partial_group_retention(tmp_path: Path) -> None:
    manifest = _identified_parameters()
    group = manifest["model_parameters"][0]
    group["components"].append(
        _component(
            "weak_alternative",
            status="weak",
            export_policy="report_only",
        )
    )
    group["conditioning"]["dimension"] = 2
    group["conditioning"]["effective_rank"] = 1
    _stamp(manifest)

    validate_identified_parameters(manifest, root=tmp_path)


def test_identified_parameters_rejects_duplicate_mjcf_target(tmp_path: Path) -> None:
    manifest = _identified_parameters()
    duplicate = deepcopy(manifest["model_parameters"][0])
    duplicate["group_id"] = "friction.joint1.alternative"
    manifest["model_parameters"].append(duplicate)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="same family/target/name"):
        validate_identified_parameters(manifest, root=tmp_path)


def test_valid_splits_allow_excluded_lineage_in_one_active_partition(
    tmp_path: Path,
) -> None:
    validate_splits(_splits(tmp_path), root=tmp_path)


def test_splits_reject_inverted_lock_time(tmp_path: Path) -> None:
    manifest = _splits(tmp_path)
    manifest["locked_at"] = "2026-07-23T09:00:00Z"
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="must not follow"):
        validate_splits(manifest, root=tmp_path)


def test_splits_reject_duplicate_trajectory_across_partitions(
    tmp_path: Path,
) -> None:
    manifest = _splits(tmp_path)
    duplicate = deepcopy(manifest["partitions"]["fit"][0])
    duplicate["lineage_group_id"] = "lineage-development"
    manifest["partitions"]["development"].append(duplicate)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="occurs in both"):
        validate_splits(manifest, root=tmp_path)


def test_splits_reject_lineage_across_active_partitions(tmp_path: Path) -> None:
    manifest = _splits(tmp_path)
    member = _split_member("trajectory-development", "lineage-fit")
    manifest["partitions"]["development"].append(member)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="multiple active partitions"):
        validate_splits(manifest, root=tmp_path)


def test_splits_rejects_one_source_run_with_multiple_lineages(
    tmp_path: Path,
) -> None:
    manifest = _splits(tmp_path)
    fit_member = manifest["partitions"]["fit"][0]
    development_member = _split_member(
        "trajectory-development",
        "lineage-development",
    )
    development_member["source_run_artifact_id"] = fit_member["source_run_artifact_id"]
    manifest["partitions"]["development"].append(development_member)
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="multiple lineages"):
        validate_splits(manifest, root=tmp_path)


def test_splits_other_exclusion_requires_details(tmp_path: Path) -> None:
    manifest = _splits(tmp_path)
    manifest["partitions"]["excluded"][0]["reason_code"] = "other"
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="details"):
        validate_splits(manifest, root=tmp_path)


def test_valid_completed_fit_result(tmp_path: Path) -> None:
    validate_fit_result(_fit_result(tmp_path), root=tmp_path)


def test_valid_failed_fit_result(tmp_path: Path) -> None:
    manifest = _fit_result(tmp_path)
    manifest["status"] = "failed"
    manifest["failure_reason"] = "held-out acceptance criterion failed"
    manifest["outputs"] = {}
    _stamp(manifest)

    validate_fit_result(manifest, root=tmp_path)


def test_failed_fit_may_record_optimizer_failure(tmp_path: Path) -> None:
    manifest = _fit_result(tmp_path)
    manifest["status"] = "failed"
    manifest["failure_reason"] = "optimizer diverged"
    manifest["optimizer_stages"][0]["status"] = "failed"
    del manifest["optimizer_stages"][0]["initial_objective"]
    del manifest["optimizer_stages"][0]["final_objective"]
    manifest["outputs"] = {}
    _stamp(manifest)

    validate_fit_result(manifest, root=tmp_path)


def test_fit_result_rejects_duplicate_stage_ids(tmp_path: Path) -> None:
    manifest = _fit_result(tmp_path)
    manifest["optimizer_stages"].append(deepcopy(manifest["optimizer_stages"][0]))
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match="optimizer stage ID"):
        validate_fit_result(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("completed_with_failed", "contains failed"),
        ("completed_without_completed", "at least one completed"),
        ("missing_objective", "requires both"),
    ],
)
def test_fit_result_enforces_stage_status_and_objectives(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    manifest = _fit_result(tmp_path)
    stage = manifest["optimizer_stages"][0]
    if fault == "completed_with_failed":
        stage["status"] = "failed"
    elif fault == "completed_without_completed":
        stage["status"] = "skipped"
    else:
        del stage["final_objective"]
    _stamp(manifest)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_fit_result(manifest, root=tmp_path)


def test_completed_optimizer_may_report_higher_objective(tmp_path: Path) -> None:
    manifest = _fit_result(tmp_path)
    manifest["optimizer_stages"][0]["final_objective"] = 11.0
    _stamp(manifest)

    validate_fit_result(manifest, root=tmp_path)
