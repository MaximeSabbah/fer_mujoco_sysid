"""Intrinsic semantic validation for portable metadata-only artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.validation import (
    FER_ARM_JOINT_ORDER,
    _schema_manifest,
    _validate_content_hash,
    _verify_file_references,
)

_ACTIVE_PARTITIONS = (
    "fit",
    "development",
    "held_out_test",
    "diagnostic_only",
)
_BACKEND_SIGNAL_SOURCE_KINDS = {
    "standalone_mujoco": frozenset({"mujoco", "file", "protocol_player"}),
    "ros_mujoco": frozenset({"ros_message"}),
    "agimus_fer": frozenset({"ros_message"}),
}
_PROTOCOL_REFERENCE_CONTRACT = {
    "desired_position": ("desired_joint_position", "q_rad"),
    "desired_velocity": ("desired_joint_velocity", "dq_rad_s"),
    "desired_acceleration": ("desired_joint_acceleration", "ddq_rad_s2"),
    "desired_effort_feedforward": (
        "desired_effort_feedforward",
        "tau_feedforward_Nm",
    ),
}
_PROTOCOL_PLAYER_FIELDS = {
    role: field for role, field in _PROTOCOL_REFERENCE_CONTRACT.values()
}
_STATE_ROLE_SEMANTICS = {
    "commanded_joint_position": ("joint_position", "rad"),
    "desired_joint_position": ("joint_position", "rad"),
    "measured_joint_position": ("joint_position", "rad"),
    "commanded_joint_velocity": ("joint_velocity", "rad/s"),
    "desired_joint_velocity": ("joint_velocity", "rad/s"),
    "measured_joint_velocity": ("joint_velocity", "rad/s"),
    "derived_joint_velocity": ("joint_velocity", "rad/s"),
    "desired_joint_acceleration": ("joint_acceleration", "rad/s^2"),
    "measured_joint_acceleration": ("joint_acceleration", "rad/s^2"),
    "derived_joint_acceleration": ("joint_acceleration", "rad/s^2"),
    "measured_joint_temperature": ("joint_temperature", "K"),
    "measured_joint_motor_current": ("joint_motor_current", "A"),
}
_PARAMETER_TARGET_KINDS = {
    "joint_frictionloss": "joint",
    "joint_damping": "joint",
    "joint_armature": "joint",
    "body_mass_com": "body",
    "body_inertia": "body",
    "terminal_composite_inertia": "rigid_composite",
    "sensor_bias": "signal",
    "control_delay": "global",
    "torque_scale": "global",
}
_PARAMETER_UNITS = {
    "joint_frictionloss": "N*m",
    "joint_damping": "N*m*s/rad",
    "joint_armature": "kg*m^2",
    "control_delay": "s",
    "torque_scale": "1",
}
_JOINT_STATE_FIELDS = {
    "joint_position": "position",
    "joint_velocity": "velocity",
    "joint_effort": "effort",
}


def _prepare(
    manifest: Mapping[str, Any],
    *,
    artifact: str,
    root: str | Path,
    schema_dir: str | Path | None,
) -> None:
    _schema_manifest(manifest, artifact, schema_dir)
    _verify_file_references(manifest, Path(root).resolve())
    if "content_sha256" in manifest:
        _validate_content_hash(manifest, {})


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{label} must be an RFC 3339 timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{label} is not a valid RFC 3339 timestamp: {value!r}"
        ) from exc


def _require_unique(values: list[Any], *, label: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ArtifactValidationError(f"duplicate {label}: {value!r}")
        seen.add(value)


def _require_canonical_signal_joints(
    signals: list[Mapping[str, Any]],
) -> None:
    expected = list(FER_ARM_JOINT_ORDER)
    for index, signal in enumerate(signals):
        if signal["joint_order"] != expected:
            raise ArtifactValidationError(
                f"signals[{index}].joint_order must be exactly {expected!r}"
            )


def validate_acquisition_run(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate intrinsic consistency of an acquisition-run manifest."""

    _prepare(
        manifest,
        artifact="acquisition-run",
        root=root,
        schema_dir=schema_dir,
    )
    if manifest["joint_order"] != list(FER_ARM_JOINT_ORDER):
        raise ArtifactValidationError(
            f"joint_order must be exactly {list(FER_ARM_JOINT_ORDER)!r}"
        )

    started = _timestamp(manifest["started_at_utc"], label="started_at_utc")
    ended = _timestamp(manifest["ended_at_utc"], label="ended_at_utc")
    if started > ended:
        raise ArtifactValidationError("started_at_utc must not follow ended_at_utc")
    if manifest["outcome"]["status"] == "completed" and started == ended:
        raise ArtifactValidationError(
            "completed acquisition must have a positive duration"
        )

    clocks = manifest["clock_domains"]
    signals = manifest["signals"]
    recordings = manifest["recording_files"]
    topics = manifest["recorded_topics"]
    _require_unique(
        [clock["clock_id"] for clock in clocks],
        label="clock domain ID",
    )
    _require_unique([signal["name"] for signal in signals], label="signal name")
    _require_unique(
        [recording["path"] for recording in recordings],
        label="recording path",
    )
    _require_unique([topic["topic"] for topic in topics], label="recorded topic")
    _require_canonical_signal_joints(signals)
    backend = manifest["backend"]
    allowed_source_kinds = _BACKEND_SIGNAL_SOURCE_KINDS[backend]
    protocol_timing = manifest["protocol_timing"]
    timing_source = protocol_timing["source"]
    invalid_source_kinds = sorted(
        {
            source["kind"]
            for source in [
                *(signal["source"] for signal in signals),
                timing_source,
            ]
            if source["kind"] not in allowed_source_kinds
        }
    )
    if invalid_source_kinds:
        raise ArtifactValidationError(
            f"backend {backend!r} permits signal source kinds "
            f"{sorted(allowed_source_kinds)!r}, got "
            f"{invalid_source_kinds!r}"
        )

    clock_ids = {clock["clock_id"] for clock in clocks}
    if protocol_timing["clock_id"] not in clock_ids:
        raise ArtifactValidationError(
            "protocol_timing.clock_id names an unknown clock: "
            f"{protocol_timing['clock_id']!r}"
        )
    signals_by_name = {signal["name"]: signal for signal in signals}
    protocol_reference_names = manifest["protocol_reference_signals"]
    _require_unique(
        list(protocol_reference_names.values()),
        label="protocol reference signal name",
    )
    for logical_name, reference_signal_name in protocol_reference_names.items():
        expected_role, expected_field = _PROTOCOL_REFERENCE_CONTRACT[logical_name]
        reference_signal = signals_by_name.get(reference_signal_name)
        if reference_signal is None:
            raise ArtifactValidationError(
                f"protocol_reference_signals.{logical_name} names an unknown "
                f"signal: {reference_signal_name!r}"
            )
        if reference_signal["semantic_role"] != expected_role:
            raise ArtifactValidationError(
                f"protocol_reference_signals.{logical_name} must name the "
                f"recorded {expected_role} signal"
            )
        if (
            manifest["outcome"]["status"] == "completed"
            and reference_signal["sample_count"] == 0
        ):
            raise ArtifactValidationError(
                "completed acquisition requires positive-sample protocol "
                f"reference signal {logical_name!r}"
            )
        if reference_signal["clock_id"] != protocol_timing["clock_id"]:
            raise ArtifactValidationError(
                f"protocol reference signal {logical_name!r} clock must equal "
                "protocol_timing.clock_id"
            )
        reference_source = reference_signal["source"]
        if backend == "standalone_mujoco":
            if reference_source["kind"] != "protocol_player":
                raise ArtifactValidationError(
                    f"standalone protocol reference {logical_name!r} must "
                    "originate from the protocol player"
                )
            if (
                reference_source["protocol_artifact_id"]
                != manifest["protocol"]["artifact_id"]
            ):
                raise ArtifactValidationError(
                    f"protocol reference {logical_name!r} source names a "
                    "different protocol artifact"
                )
        elif (
            reference_source["kind"] != "ros_message"
            or reference_source["topic"] != timing_source["topic"]
            or reference_source["message_type"] != timing_source["message_type"]
        ):
            raise ArtifactValidationError(
                f"ROS protocol reference {logical_name!r} must share the "
                "protocol_timing topic and message type"
            )
        if reference_source["field"] != expected_field:
            raise ArtifactValidationError(
                f"protocol reference {logical_name!r} source field must be "
                f"{expected_field!r}, got {reference_source['field']!r}"
            )
    for index, signal in enumerate(signals):
        if signal["clock_id"] not in clock_ids:
            raise ArtifactValidationError(
                f"signals[{index}].clock_id names an unknown clock: "
                f"{signal['clock_id']!r}"
            )
        expected_semantics = _STATE_ROLE_SEMANTICS.get(signal["semantic_role"])
        if expected_semantics is not None:
            actual_semantics = (signal["quantity"], signal["unit"])
            if actual_semantics != expected_semantics:
                raise ArtifactValidationError(
                    f"signals[{index}] role {signal['semantic_role']!r} requires "
                    f"quantity/unit {expected_semantics!r}, got "
                    f"{actual_semantics!r}"
                )
        source = signal["source"]
        if source["kind"] == "protocol_player":
            expected_field = _PROTOCOL_PLAYER_FIELDS.get(signal["semantic_role"])
            if expected_field is None:
                raise ArtifactValidationError(
                    f"signals[{index}] role {signal['semantic_role']!r} cannot "
                    "originate from the protocol player"
                )
            if source["field"] != expected_field:
                raise ArtifactValidationError(
                    f"signals[{index}] role {signal['semantic_role']!r} must "
                    f"use protocol-player field {expected_field!r}, got "
                    f"{source['field']!r}"
                )
            if source["protocol_artifact_id"] != manifest["protocol"]["artifact_id"]:
                raise ArtifactValidationError(
                    f"signals[{index}] protocol-player source names a different "
                    "protocol artifact"
                )
    for index, topic in enumerate(topics):
        topic_clock = topic.get("clock_id")
        if topic_clock is not None and topic_clock not in clock_ids:
            raise ArtifactValidationError(
                f"recorded_topics[{index}].clock_id names an unknown clock: "
                f"{topic_clock!r}"
            )

    topics_by_name = {topic["topic"]: topic for topic in topics}
    if timing_source["kind"] == "ros_message":
        recorded = topics_by_name.get(timing_source["topic"])
        if recorded is None:
            raise ArtifactValidationError(
                "protocol_timing ROS source topic is not recorded: "
                f"{timing_source['topic']!r}"
            )
        if recorded["message_type"] != timing_source["message_type"]:
            raise ArtifactValidationError(
                "protocol_timing ROS source message type "
                f"{timing_source['message_type']!r} disagrees with recorded "
                f"topic type {recorded['message_type']!r}"
            )
        if recorded.get("clock_id") != protocol_timing["clock_id"]:
            raise ArtifactValidationError(
                f"protocol_timing clock {protocol_timing['clock_id']!r} "
                "disagrees with recorded topic clock "
                f"{recorded.get('clock_id')!r}"
            )
        if not recorded["required_for_conversion"]:
            raise ArtifactValidationError(
                "protocol_timing ROS source topic must be required for conversion"
            )
        if (
            manifest["outcome"]["status"] == "completed"
            and recorded["message_count"] == 0
        ):
            raise ArtifactValidationError(
                "completed acquisition has no recorded protocol_timing source"
            )

    for index, signal in enumerate(signals):
        source = signal["source"]
        if source["kind"] != "ros_message":
            continue
        recorded = topics_by_name.get(source["topic"])
        if recorded is None:
            raise ArtifactValidationError(
                f"signals[{index}] ROS source topic is not recorded: "
                f"{source['topic']!r}"
            )
        if recorded["message_type"] != source["message_type"]:
            raise ArtifactValidationError(
                f"signals[{index}] ROS source message type "
                f"{source['message_type']!r} disagrees with recorded topic "
                f"type {recorded['message_type']!r}"
            )
        if (
            recorded.get("clock_id") is not None
            and recorded["clock_id"] != signal["clock_id"]
        ):
            raise ArtifactValidationError(
                f"signals[{index}] clock {signal['clock_id']!r} disagrees with "
                f"recorded topic clock {recorded['clock_id']!r}"
            )
        if signal["sample_count"] > recorded["message_count"]:
            raise ArtifactValidationError(
                f"signals[{index}].sample_count exceeds its recorded topic "
                f"message_count ({signal['sample_count']} > "
                f"{recorded['message_count']})"
            )
        expected_field = _JOINT_STATE_FIELDS.get(signal["quantity"])
        if source["message_type"] == "sensor_msgs/msg/JointState":
            if expected_field is None:
                raise ArtifactValidationError(
                    f"signals[{index}] quantity {signal['quantity']!r} cannot be "
                    "sourced from sensor_msgs/msg/JointState"
                )
            if source["field"] != expected_field:
                raise ArtifactValidationError(
                    f"signals[{index}] quantity {signal['quantity']!r} must use "
                    f"sensor_msgs/msg/JointState.{expected_field}, got "
                    f"{source['field']!r}"
                )

    if manifest["outcome"]["status"] == "completed":
        required_roles = {
            "measured_joint_position",
            "measured_joint_velocity",
        }
        positive_roles = {
            signal["semantic_role"] for signal in signals if signal["sample_count"] > 0
        }
        missing_roles = sorted(required_roles - positive_roles)
        if missing_roles:
            raise ArtifactValidationError(
                "completed acquisition is missing positive-sample measured signals: "
                + ", ".join(missing_roles)
            )
        if not any(
            signal["quantity"] == "joint_effort" and signal["sample_count"] > 0
            for signal in signals
        ):
            raise ArtifactValidationError(
                "completed acquisition requires at least one positive-sample "
                "joint-effort signal"
            )
        empty_required_topics = [
            topic["topic"]
            for topic in topics
            if topic["required_for_conversion"] and topic["message_count"] == 0
        ]
        if empty_required_topics:
            raise ArtifactValidationError(
                "completed acquisition has empty required topics: "
                + ", ".join(repr(topic) for topic in empty_required_topics)
            )
        empty_recordings = [
            recording["path"]
            for recording in recordings
            if recording["size_bytes"] == 0
        ]
        if empty_recordings:
            raise ArtifactValidationError(
                "completed acquisition has empty recording files: "
                + ", ".join(repr(path) for path in empty_recordings)
            )


def _target_identity(target: Mapping[str, Any]) -> tuple[str, str]:
    kind = target["kind"]
    identifier_field = {
        "joint": "joint_name",
        "body": "body_name",
        "rigid_composite": "composite_name",
        "signal": "signal_name",
        "global": "name",
    }[kind]
    return kind, target[identifier_field]


def validate_identified_parameters(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate intrinsic consistency of identified parameter groups."""

    _prepare(
        manifest,
        artifact="identified-parameters",
        root=root,
        schema_dir=schema_dir,
    )
    if manifest["joint_order"] != list(FER_ARM_JOINT_ORDER):
        raise ArtifactValidationError(
            f"joint_order must be exactly {list(FER_ARM_JOINT_ORDER)!r}"
        )

    groups = [
        *manifest["model_parameters"],
        *manifest["nuisance_parameters"],
    ]
    _require_unique([group["group_id"] for group in groups], label="parameter group ID")

    component_identities: set[tuple[str, str]] = set()
    mjcf_targets: set[tuple[str, str, str, str]] = set()
    nuisance_group_ids = {
        group["group_id"] for group in manifest["nuisance_parameters"]
    }
    for group in groups:
        group_id = group["group_id"]
        retained = group["retained"]
        family = group["family"]
        target_kind, target_name = _target_identity(group["target"])
        expected_target_kind = _PARAMETER_TARGET_KINDS[family]
        if target_kind != expected_target_kind:
            raise ArtifactValidationError(
                f"parameter family {family!r} requires target kind "
                f"{expected_target_kind!r}, got {target_kind!r}"
            )
        if family in {"control_delay", "torque_scale"} and target_name != family:
            raise ArtifactValidationError(
                f"parameter family {family!r} requires global target name "
                f"{family!r}, got {target_name!r}"
            )
        expected_unit = _PARAMETER_UNITS.get(family)
        conditioning = group.get("conditioning")
        if retained and conditioning is None:
            raise ArtifactValidationError(
                f"retained parameter group {group_id!r} requires conditioning"
            )
        if (
            conditioning is not None
            and conditioning["effective_rank"] > conditioning["dimension"]
        ):
            raise ArtifactValidationError(
                f"parameter group {group_id!r} has effective_rank greater "
                "than dimension"
            )

        retained_component_count = 0
        optimized_component_count = 0
        for component in group["components"]:
            component_name = component["name"]
            identity = (group_id, component_name)
            if identity in component_identities:
                raise ArtifactValidationError(
                    "duplicate parameter component identity: "
                    f"{group_id!r}/{component_name!r}"
                )
            component_identities.add(identity)

            lower = component["bounds"]["lower"]
            upper = component["bounds"]["upper"]
            initial = component["initial_value"]
            value = component["value"]
            if expected_unit is not None and component["unit"] != expected_unit:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} in family {family!r} must use "
                    f"unit {expected_unit!r}, got {component['unit']!r}"
                )
            if lower > upper:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} has inverted bounds"
                )
            if not lower <= initial <= upper:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} initial_value is outside bounds"
                )
            if not lower <= value <= upper:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} value is outside bounds"
                )
            transform = group["optimization_transform"]
            constraint = group["physical_constraint"]
            if transform == "log" and lower <= 0:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} uses a log transform with "
                    "a nonpositive lower bound"
                )
            if constraint == "nonnegative" and lower < 0:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} has a negative lower bound "
                    "under a nonnegative constraint"
                )
            if constraint == "positive" and lower <= 0:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} has a nonpositive lower bound "
                    "under a positive constraint"
                )

            bound_status = component["bound_status"]
            if lower == upper and bound_status != "both":
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} has zero-width bounds and "
                    "must report bound_status 'both'"
                )
            if lower != upper and bound_status == "both":
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} may report bound_status 'both' "
                    "only for zero-width bounds"
                )
            if bound_status == "lower" and value == upper:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} reports the lower bound while "
                    "its value equals the distinct upper bound"
                )
            if bound_status == "upper" and value == lower:
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} reports the upper bound while "
                    "its value equals the distinct lower bound"
                )

            status = component["identifiability_status"]
            export_policy = component["export_policy"]
            if status != "fixed":
                optimized_component_count += 1
            if lower == upper and status != "fixed":
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} has zero-width bounds and "
                    "must have fixed identifiability status"
                )
            if (status == "fixed") != (export_policy == "fixed"):
                raise ArtifactValidationError(
                    f"{group_id}.{component_name} fixed status and export policy "
                    "must agree"
                )
            uncertainty = component.get("uncertainty")
            if uncertainty is not None and uncertainty["method"] != "none":
                uncertainty_lower = uncertainty["lower"]
                uncertainty_upper = uncertainty["upper"]
                if uncertainty_lower > uncertainty_upper:
                    raise ArtifactValidationError(
                        f"{group_id}.{component_name} has an inverted uncertainty "
                        "interval"
                    )
            if status == "retained":
                if uncertainty is None or uncertainty["method"] == "none":
                    raise ArtifactValidationError(
                        f"retained component {group_id}.{component_name} requires "
                        "quantified uncertainty"
                    )
                retained_component_count += 1
                if not retained:
                    raise ArtifactValidationError(
                        f"non-retained group {group_id!r} contains retained "
                        f"component {component_name!r}"
                    )
                expected_export_policy = (
                    "report_only" if group_id in nuisance_group_ids else "mjcf"
                )
                if export_policy != expected_export_policy:
                    raise ArtifactValidationError(
                        f"retained component {group_id}.{component_name} must use "
                        f"{expected_export_policy!r} export policy"
                    )
            elif (
                status in {"weak", "unidentifiable"} and export_policy != "report_only"
            ):
                raise ArtifactValidationError(
                    f"{status} component {group_id}.{component_name} must use "
                    "'report_only' export policy"
                )

            if group_id in nuisance_group_ids and export_policy not in {
                "report_only",
                "fixed",
            }:
                raise ArtifactValidationError(
                    f"nuisance component {group_id}.{component_name} may only "
                    "use report_only or fixed export policy"
                )
            if export_policy == "mjcf":
                if not retained or status != "retained":
                    raise ArtifactValidationError(
                        f"MJCF component {group_id}.{component_name} must be retained"
                    )
                target = (
                    family,
                    target_kind,
                    target_name,
                    component_name,
                )
                if target in mjcf_targets:
                    raise ArtifactValidationError(
                        "multiple MJCF components target the same "
                        "family/target/name: "
                        f"{target!r}"
                    )
                mjcf_targets.add(target)

        if retained and retained_component_count == 0:
            raise ArtifactValidationError(
                f"retained parameter group {group_id!r} has no retained components"
            )
        if conditioning is not None:
            if conditioning["dimension"] != optimized_component_count:
                raise ArtifactValidationError(
                    f"parameter group {group_id!r} conditioning dimension must "
                    "equal its optimized component count"
                )
            if retained and conditioning["effective_rank"] < retained_component_count:
                raise ArtifactValidationError(
                    f"parameter group {group_id!r} retains more components than "
                    "its effective rank supports"
                )


def _split_member(entry: Mapping[str, Any], partition: str) -> Mapping[str, Any]:
    return entry["member"] if partition == "excluded" else entry


def validate_splits(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate intrinsic exclusivity and locking rules for dataset splits."""

    _prepare(manifest, artifact="splits", root=root, schema_dir=schema_dir)
    created = _timestamp(manifest["created_at"], label="created_at")
    locked = _timestamp(manifest["locked_at"], label="locked_at")
    if created > locked:
        raise ArtifactValidationError("created_at must not follow locked_at")

    trajectory_partition: dict[str, str] = {}
    lineage_active_partitions: dict[str, set[str]] = {}
    source_run_lineages: dict[str, str] = {}
    for partition, entries in manifest["partitions"].items():
        for entry in entries:
            member = _split_member(entry, partition)
            trajectory_id = member["trajectory"]["artifact_id"]
            previous = trajectory_partition.get(trajectory_id)
            if previous is not None:
                raise ArtifactValidationError(
                    f"trajectory {trajectory_id!r} occurs in both "
                    f"{previous!r} and {partition!r}"
                )
            trajectory_partition[trajectory_id] = partition
            source_run_id = member["source_run_artifact_id"]
            lineage_group_id = member["lineage_group_id"]
            previous_lineage = source_run_lineages.setdefault(
                source_run_id,
                lineage_group_id,
            )
            if previous_lineage != lineage_group_id:
                raise ArtifactValidationError(
                    f"source run {source_run_id!r} is assigned to multiple "
                    f"lineages: {previous_lineage!r} and {lineage_group_id!r}"
                )

            if partition in _ACTIVE_PARTITIONS:
                lineage_active_partitions.setdefault(
                    lineage_group_id,
                    set(),
                ).add(partition)
            if (
                partition == "excluded"
                and entry["reason_code"] == "other"
                and not entry.get("details")
            ):
                raise ArtifactValidationError(
                    f"excluded trajectory {trajectory_id!r} with reason 'other' "
                    "requires details"
                )

    for lineage_group_id, partitions in lineage_active_partitions.items():
        if len(partitions) > 1:
            raise ArtifactValidationError(
                f"lineage group {lineage_group_id!r} occurs in multiple active "
                f"partitions: {sorted(partitions)!r}"
            )


def validate_fit_result(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate intrinsic optimizer-stage consistency of a fit result."""

    _prepare(manifest, artifact="fit-result", root=root, schema_dir=schema_dir)
    stages = manifest["optimizer_stages"]
    _require_unique([stage["stage_id"] for stage in stages], label="optimizer stage ID")

    for stage in stages:
        if stage["status"] != "completed":
            continue
        if "initial_objective" not in stage or "final_objective" not in stage:
            raise ArtifactValidationError(
                f"completed optimizer stage {stage['stage_id']!r} requires both "
                "initial_objective and final_objective"
            )

    statuses = [stage["status"] for stage in stages]
    if manifest["status"] == "completed":
        invalid = [
            stage["stage_id"]
            for stage in stages
            if stage["status"] not in {"completed", "skipped"}
        ]
        if invalid:
            raise ArtifactValidationError(
                "completed fit contains failed optimizer stages: "
                + ", ".join(repr(stage_id) for stage_id in invalid)
            )
        if "completed" not in statuses:
            raise ArtifactValidationError(
                "completed fit requires at least one completed optimizer stage"
            )
