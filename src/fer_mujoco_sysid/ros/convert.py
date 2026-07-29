"""Convert one protocol-marked ROS bag into an immutable recording bundle.

Controller state remains on its native clock.  The protocol player's start
and end events remove the move-to-start from the derived dataset while the raw
bag retains it.  Real Franka telemetry remains at its proven 100 Hz rate with
exact integer timestamps; controller-clock views use only causal
previous-sample hold and carry their age explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fer_mujoco_sysid.dataset import (
    BACKENDS,
    BOUND_RECORDING_FORMAT,
    health_report,
    write_recording,
)
from fer_mujoco_sysid.protocol import (
    DEFAULT_ANALYSIS_EDGE_GUARD_S,
    PROTOCOL_FORMAT,
    ROS_ARM_JOINT_NAMES,
    alignment_from_start_time,
    load_protocol_bundle,
    map_analysis_windows,
    map_protocol_segments,
)

CONTROLLER_STATE_TYPE = "control_msgs/msg/JointTrajectoryControllerState"
FRANKA_STATE_TYPE = "agimus_franka_msgs/msg/AgimusFrankaRobotState"
PROTOCOL_EVENT_TYPE = "std_msgs/msg/Header"

CONTROLLER_STATE_TOPIC = "/fer_sysid_arm_controller/controller_state"
FRANKA_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
PROTOCOL_EVENT_TOPIC = "/fer_sysid/protocol_event"
PROTOCOL_EVENT_FORMAT = "fer-mujoco-sysid/protocol-event@1"

EXPECTED_FRANKA_RATE_HZ = 100
MINIMUM_FRANKA_RATE_HZ = 80.0
MAXIMUM_FRANKA_RATE_HZ = 120.0
MAXIMUM_FRANKA_AGE_S = 0.035
MAXIMUM_FRANKA_GAP_S = 0.035
MAXIMUM_DIAGNOSTIC_NEAREST_SKEW_S = 0.020
REVERSAL_GUARD_S = 0.030
# The deployed agimus_franka hardware plugin enables its torque command rate
# limiter and uses agimus_franka::kMaxTorqueRate = 1000 Nm/s on all joints.
# This is a detector for controller requests that necessarily encounter that
# downstream limiter, not a claim that controller output is applied torque.
HARDWARE_TORQUE_RATE_LIMIT_NM_S = 1000.0
TORQUE_SLEW_GUARD_S = 0.030
SIMULATION_TRUTH_FORMAT = "fer-mujoco-sysid/simulation-truth@1"


class ConversionError(RuntimeError):
    """A bag cannot be turned into a trustworthy recording."""


def _simulation_truth_metadata(
    path: str | Path,
    *,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Validate a lightweight truth manifest without importing MuJoCo in ROS."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(
            f"cannot read simulation truth {path}: {error}"
        ) from error
    if raw.get("format") != SIMULATION_TRUTH_FORMAT:
        raise ConversionError(
            f"{path}: expected {SIMULATION_TRUTH_FORMAT}, got {raw.get('format')!r}"
        )
    source = raw.get("source_model")
    if not isinstance(source, Mapping):
        raise ConversionError(f"{path}: simulation truth has no source_model")
    source_sha256 = str(source.get("sha256", ""))
    if source_sha256 != expected_source_sha256:
        raise ConversionError(
            f"{path}: truth source {source_sha256} does not match protocol "
            f"source {expected_source_sha256}"
        )
    if not isinstance(raw.get("parameters"), Mapping):
        raise ConversionError(f"{path}: simulation truth has no parameters")

    claimed = str(raw.get("content_sha256", ""))
    unsigned = {key: value for key, value in raw.items() if key != "content_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(
        b"fer-mujoco-sysid/simulation-truth@1\0" + canonical
    ).hexdigest()
    if claimed != actual:
        raise ConversionError(f"{path}: simulation truth content SHA-256 mismatch")
    return {
        "format": SIMULATION_TRUTH_FORMAT,
        "content_sha256": actual,
        "source_model_sha256": source_sha256,
    }


def _read_messages(bag: Path) -> list[tuple[str, Any, int]]:
    """Read every deserializable message as ``(topic, message, bag_stamp_ns)``."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    messages: list[tuple[str, Any, int]] = []
    while reader.has_next():
        topic, payload, stamp = reader.read_next()
        try:
            message = deserialize_message(payload, get_message(types[topic]))
        except (KeyError, ModuleNotFoundError):  # pragma: no cover - environment
            continue
        messages.append((topic, message, int(stamp)))
    return messages


def _topic_types(bag: Path) -> dict[str, str]:
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def _stamp_nanoseconds(header: Any) -> int:
    """Preserve a ROS header stamp exactly, without float epoch precision loss."""
    seconds = int(header.stamp.sec)
    nanoseconds = int(header.stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ConversionError(f"invalid ROS stamp sec={seconds}, nanosec={nanoseconds}")
    return seconds * 1_000_000_000 + nanoseconds


def _vector(values: Any, description: str) -> list[float]:
    vector = list(values)
    if len(vector) != 7 or not np.all(np.isfinite(vector)):
        raise ConversionError(f"{description} must contain seven finite values")
    return [float(value) for value in vector]


def _controller_series(
    messages: list[tuple[str, Any, int]], types: Mapping[str, str]
) -> tuple[np.ndarray, ...]:
    """Return absolute stamp ns, measured q/dq, output effort, and reference q."""
    rows: list[
        tuple[int, list[float], list[float], list[float], list[float] | None]
    ] = []
    order: list[str] | None = None
    for topic, message, _ in messages:
        if topic != CONTROLLER_STATE_TOPIC or types.get(topic) != CONTROLLER_STATE_TYPE:
            continue
        effort = list(message.output.effort)
        positions = list(message.feedback.positions)
        velocities = list(message.feedback.velocities)
        if not effort or not positions or not velocities:
            # State messages before the first controller command have no
            # output effort and cannot participate in identification.
            continue
        message_order = list(message.joint_names)
        if order is None:
            order = message_order
        elif message_order != order:
            raise ConversionError("controller joint order changed during the run")
        desired_values = list(message.reference.positions)
        desired = (
            _vector(desired_values, "controller reference.positions")
            if desired_values
            else None
        )
        rows.append(
            (
                _stamp_nanoseconds(message.header),
                _vector(positions, "controller feedback.positions"),
                _vector(velocities, "controller feedback.velocities"),
                _vector(effort, "controller output.effort"),
                desired,
            )
        )

    if not rows:
        raise ConversionError(
            f"the bag holds no usable messages on {CONTROLLER_STATE_TOPIC}"
        )
    if order != list(ROS_ARM_JOINT_NAMES):
        raise ConversionError(
            f"controller reports joints {order}, expected {list(ROS_ARM_JOINT_NAMES)}"
        )

    rows.sort(key=lambda row: row[0])
    unique: list[tuple[Any, ...]] = []
    for row in rows:
        if unique and row[0] == unique[-1][0]:
            unique[-1] = row
        else:
            unique.append(row)
    duplicates = len(rows) - len(unique)
    if duplicates:
        print(f"  dropped {duplicates} duplicate controller timestamps")
    rows = unique  # type: ignore[assignment]
    if len(rows) < 2:
        raise ConversionError("controller state has fewer than two unique timestamps")

    desired_rows = [row[4] for row in rows]
    desired = (
        np.asarray(desired_rows, dtype=np.float64)
        if all(row is not None for row in desired_rows)
        else None
    )
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
        np.asarray([row[3] for row in rows], dtype=np.float64),
        desired,
    )


def _event_payload(message: Any) -> Mapping[str, object]:
    try:
        payload = json.loads(str(message.frame_id))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConversionError("protocol event frame_id is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ConversionError("protocol event payload must be an object")
    if payload.get("format") != PROTOCOL_EVENT_FORMAT:
        raise ConversionError(
            f"unsupported protocol event format {payload.get('format')!r}"
        )
    return payload


def _protocol_bounds(
    messages: list[tuple[str, Any, int]],
    types: Mapping[str, str],
    protocol: Mapping[str, object],
    speed_scale: float,
) -> tuple[int, int]:
    """Return the unique start/end marker stamps matching the reviewed bundle."""
    expected = {
        "protocol_id": str(protocol.get("protocol_id", "")),
        "content_sha256": str(protocol.get("content_sha256", "")),
    }
    events: dict[str, list[int]] = {"start": [], "end": []}
    for topic, message, _ in messages:
        if topic != PROTOCOL_EVENT_TOPIC:
            continue
        if types.get(topic) != PROTOCOL_EVENT_TYPE:
            raise ConversionError(
                f"{PROTOCOL_EVENT_TOPIC} has type {types.get(topic)!r}, "
                f"expected {PROTOCOL_EVENT_TYPE}"
            )
        payload = _event_payload(message)
        for key, value in expected.items():
            if str(payload.get(key, "")) != value:
                raise ConversionError(
                    f"protocol event {key} does not match the reviewed bundle"
                )
        marker_scale = float(payload.get("speed_scale", np.nan))
        if not np.isfinite(marker_scale) or not np.isclose(
            marker_scale, speed_scale, rtol=0.0, atol=1e-12
        ):
            raise ConversionError(
                f"protocol event speed_scale {marker_scale!r} does not match "
                f"conversion scale {speed_scale:g}"
            )
        event = str(payload.get("event", ""))
        if event not in events:
            raise ConversionError(f"unknown protocol event {event!r}")
        events[event].append(_stamp_nanoseconds(message))

    for event, stamps in events.items():
        if len(stamps) != 1:
            raise ConversionError(
                f"expected exactly one matching protocol {event} marker on "
                f"{PROTOCOL_EVENT_TOPIC}, found {len(stamps)}"
            )
    start, end = events["start"][0], events["end"][0]
    if start >= end:
        raise ConversionError("protocol end marker is not after its start marker")
    return start, end


def _crop_controller(
    series: tuple[np.ndarray, ...], start_ns: int, end_ns: int
) -> tuple[np.ndarray, ...]:
    stamps = np.asarray(series[0], dtype=np.int64)
    selected = (stamps >= start_ns) & (stamps <= end_ns)
    if np.count_nonzero(selected) < 2:
        raise ConversionError(
            "fewer than two controller rows lie between protocol markers"
        )
    cropped: list[np.ndarray | None] = []
    for array in series:
        if array is None:
            cropped.append(None)
        else:
            cropped.append(np.asarray(array)[selected])
    return tuple(cropped)  # type: ignore[return-value]


def _mapped_intervals(
    protocol: Mapping[str, object],
    protocol_arrays: Mapping[str, np.ndarray],
    speed_scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Use the canonical protocol mapper so online/offline masks cannot drift."""
    alignment = alignment_from_start_time(
        protocol, start_time_s=0.0, speed_scale=speed_scale
    )
    segments = [
        asdict(interval)
        for interval in map_protocol_segments(
            protocol,
            protocol_arrays,
            alignment,
        )
    ]
    windows = [
        asdict(interval)
        for interval in map_analysis_windows(
            protocol,
            protocol_arrays,
            alignment,
            edge_guard_s=DEFAULT_ANALYSIS_EDGE_GUARD_S,
        )
    ]
    if not windows:
        raise ConversionError(
            "protocol has no explicit analysis_windows; regenerate it with the "
            f"current {PROTOCOL_FORMAT} compiler before collecting data"
        )
    return segments, windows


def _row_interval_arrays(
    time_s: np.ndarray,
    segments: list[dict[str, object]],
    windows: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    # Canonical mapped intervals end on their last compiled sample.  A faster
    # controller clock also observes interpolation rows between that sample
    # and the next segment's first sample.  Assign those rows to the segment
    # whose start was most recently reached so the marker-cropped protocol
    # remains contiguous; analysis eligibility still comes only from the
    # narrower, guarded canonical windows below.
    starts = np.asarray(
        [float(segment["start_time_s"]) for segment in segments],
        dtype=np.float64,
    )
    segment_index = (np.searchsorted(starts, time_s, side="right") - 1).astype(np.int16)
    segment_index[time_s < starts[0]] = -1

    eligible = np.zeros(len(time_s), dtype=np.bool_)
    for window in windows:
        eligible |= (time_s >= float(window["start_time_s"])) & (
            time_s <= float(window["end_time_s"])
        )
    return segment_index, eligible


def _transient_free_mask(
    time_s: np.ndarray,
    dq_rad_s: np.ndarray,
    tau_cmd_Nm: np.ndarray,
    reversal_guard_s: float = REVERSAL_GUARD_S,
    torque_slew_guard_s: float = TORQUE_SLEW_GUARD_S,
) -> tuple[np.ndarray, int, int]:
    """Exclude reversals and requests that exceed the hardware torque-rate limit."""
    mask = np.ones(len(time_s), dtype=np.bool_)
    reversal_times: list[float] = []
    velocity_epsilon = 1e-3
    for joint in range(dq_rad_s.shape[1]):
        active = np.flatnonzero(np.abs(dq_rad_s[:, joint]) >= velocity_epsilon)
        if len(active) < 2:
            continue
        signs = np.sign(dq_rad_s[active, joint])
        changes = np.flatnonzero(signs[1:] != signs[:-1])
        for change in changes:
            before = active[change]
            after = active[change + 1]
            reversal_times.append(float((time_s[before] + time_s[after]) / 2.0))
    for stamp in reversal_times:
        mask &= np.abs(time_s - stamp) > reversal_guard_s

    steps = np.diff(time_s)
    slew = np.abs(np.diff(tau_cmd_Nm, axis=0) / steps[:, None])
    slew_rows = np.flatnonzero(np.any(slew >= HARDWARE_TORQUE_RATE_LIMIT_NM_S, axis=1))
    for row in slew_rows:
        stamp = float((time_s[row] + time_s[row + 1]) / 2.0)
        mask &= np.abs(time_s - stamp) > torque_slew_guard_s
    return mask, len(reversal_times), int(len(slew_rows))


def _native_franka_series(
    messages: list[tuple[str, Any, int]], types: Mapping[str, str]
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    rows: list[tuple[int, list[float], list[float], list[float], list[float]]] = []
    for topic, message, _ in messages:
        if topic != FRANKA_STATE_TOPIC or types.get(topic) != FRANKA_STATE_TYPE:
            continue
        rows.append(
            (
                _stamp_nanoseconds(message.header),
                _vector(message.measured_joint_state.effort, "Franka tau_J"),
                _vector(message.desired_joint_state.effort, "Franka tau_J_d"),
                _vector(message.measured_joint_motor_state.position, "Franka theta"),
                _vector(message.measured_joint_motor_state.velocity, "Franka dtheta"),
            )
        )
    if not rows:
        return None
    rows.sort(key=lambda row: row[0])
    unique: list[tuple[Any, ...]] = []
    for row in rows:
        if unique and row[0] == unique[-1][0]:
            unique[-1] = row
        else:
            unique.append(row)
    if len(unique) < 2:
        raise ConversionError("Franka telemetry has fewer than two unique timestamps")
    stamps = np.asarray([row[0] for row in unique], dtype=np.int64)
    channels = {
        "tau_J_native_Nm": np.asarray([row[1] for row in unique], dtype=np.float64),
        "tau_J_d_native_Nm": np.asarray([row[2] for row in unique], dtype=np.float64),
        "theta_native_rad": np.asarray([row[3] for row in unique], dtype=np.float64),
        "dtheta_native_rad_s": np.asarray([row[4] for row in unique], dtype=np.float64),
    }
    return stamps, channels


def _nearest_skew_ns(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(source_ns, target_ns, side="left")
    lower = np.clip(upper - 1, 0, len(source_ns) - 1)
    upper = np.clip(upper, 0, len(source_ns) - 1)
    lower_skew = np.abs(target_ns - source_ns[lower])
    upper_skew = np.abs(source_ns[upper] - target_ns)
    return np.minimum(lower_skew, upper_skew)


def _franka_series(
    messages: list[tuple[str, Any, int]],
    types: Mapping[str, str],
    controller_stamp_ns: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Preserve native telemetry and causally align it onto controller rows."""
    native = _native_franka_series(messages, types)
    if native is None:
        raise ConversionError(
            f"real bag has no usable messages on {FRANKA_STATE_TOPIC}"
        )
    all_stamp_ns, all_channels = native
    predecessor = int(np.searchsorted(all_stamp_ns, start_ns, side="right") - 1)
    stop = int(np.searchsorted(all_stamp_ns, end_ns, side="right"))
    if predecessor < 0 or stop - predecessor < 2:
        raise ConversionError(
            "Franka telemetry does not bracket the marked protocol interval"
        )
    native_stamp_ns = all_stamp_ns[predecessor:stop]
    native_channels = {
        name: values[predecessor:stop] for name, values in all_channels.items()
    }

    causal_index = (
        np.searchsorted(native_stamp_ns, controller_stamp_ns, side="right") - 1
    )
    if np.any(causal_index < 0):
        raise ConversionError(
            "a controller row has no preceding Franka telemetry sample"
        )
    ages_s = (controller_stamp_ns - native_stamp_ns[causal_index]).astype(
        np.float64
    ) * 1e-9
    gaps_s = np.diff(native_stamp_ns).astype(np.float64) * 1e-9
    nearest_s = (
        _nearest_skew_ns(all_stamp_ns, controller_stamp_ns).astype(np.float64) * 1e-9
    )
    maximum_age = float(np.max(ages_s))
    maximum_gap = float(np.max(gaps_s))
    maximum_nearest = float(np.max(nearest_s))
    if maximum_age > MAXIMUM_FRANKA_AGE_S:
        raise ConversionError(
            f"Franka telemetry is stale by {maximum_age * 1e3:.1f} ms "
            f"(limit {MAXIMUM_FRANKA_AGE_S * 1e3:.1f} ms)"
        )
    if maximum_gap > MAXIMUM_FRANKA_GAP_S:
        raise ConversionError(
            f"Franka telemetry has a {maximum_gap * 1e3:.1f} ms gap "
            f"(limit {MAXIMUM_FRANKA_GAP_S * 1e3:.1f} ms)"
        )
    if maximum_nearest > MAXIMUM_DIAGNOSTIC_NEAREST_SKEW_S:
        raise ConversionError(
            f"controller/Franka clocks differ by up to {maximum_nearest * 1e3:.1f} ms"
        )

    new_sample = np.empty(len(causal_index), dtype=np.bool_)
    new_sample[0] = True
    new_sample[1:] = causal_index[1:] != causal_index[:-1]
    arrays: dict[str, np.ndarray] = {
        "franka_time_ns": native_stamp_ns.astype(np.int64, copy=False),
        "franka_sample_age_s": ages_s,
        "franka_new_sample": new_sample,
        **native_channels,
        "tau_J_Nm": native_channels["tau_J_native_Nm"][causal_index],
        "tau_J_d_Nm": native_channels["tau_J_d_native_Nm"][causal_index],
        "theta_rad": native_channels["theta_native_rad"][causal_index],
        "dtheta_rad_s": native_channels["dtheta_native_rad_s"][causal_index],
    }
    median_gap = float(np.median(gaps_s))
    observed_rate = 1.0 / median_gap
    if not MINIMUM_FRANKA_RATE_HZ <= observed_rate <= MAXIMUM_FRANKA_RATE_HZ:
        raise ConversionError(
            f"Franka telemetry median rate is {observed_rate:.1f} Hz; expected "
            f"the proven {EXPECTED_FRANKA_RATE_HZ} Hz broadcaster "
            f"({MINIMUM_FRANKA_RATE_HZ:.0f}–{MAXIMUM_FRANKA_RATE_HZ:.0f} Hz)"
        )
    metadata: dict[str, object] = {
        "method": "causal_previous_sample_hold",
        "control_channel_policy": (
            "never use a future telemetry sample; retain only rows where the "
            "native sample changes for telemetry-based fitting"
        ),
        "diagnostic_comparison_policy": "nearest sample only for skew diagnostics",
        "expected_rate_hz": EXPECTED_FRANKA_RATE_HZ,
        "observed_median_rate_hz": observed_rate,
        "minimum_allowed_median_rate_hz": MINIMUM_FRANKA_RATE_HZ,
        "maximum_allowed_median_rate_hz": MAXIMUM_FRANKA_RATE_HZ,
        "native_samples": int(len(native_stamp_ns)),
        "maximum_age_s": maximum_age,
        "mean_age_s": float(np.mean(ages_s)),
        "maximum_allowed_age_s": MAXIMUM_FRANKA_AGE_S,
        "maximum_native_gap_s": maximum_gap,
        "maximum_allowed_native_gap_s": MAXIMUM_FRANKA_GAP_S,
        "maximum_diagnostic_nearest_skew_s": maximum_nearest,
        "maximum_allowed_diagnostic_nearest_skew_s": (
            MAXIMUM_DIAGNOSTIC_NEAREST_SKEW_S
        ),
    }
    return arrays, metadata


def _torque_channels(real: bool) -> dict[str, object]:
    channels: dict[str, object] = {
        "tau_cmd_Nm": {
            "source_topic": CONTROLLER_STATE_TOPIC,
            "message_field": "output.effort",
            "meaning": "joint_trajectory_controller output effort request",
            "clock": "controller state header",
            "rate_limiting": "before_or_unknown_relative_to_hardware_limiting",
            "application_status": "not_asserted",
        }
    }
    if real:
        channels.update(
            {
                "tau_J_d_Nm": {
                    "source_topic": FRANKA_STATE_TOPIC,
                    "message_field": "desired_joint_state.effort",
                    "meaning": "hardware desired link-side joint effort telemetry",
                    "limiter_and_gravity_semantics": (
                        "not asserted beyond the upstream message definition"
                    ),
                    "fit_eligible_as_model_input": False,
                    "fit_exclusion_reason": (
                        "hardware limiter/gravity composition has not yet "
                        "been validated against the MuJoCo input convention"
                    ),
                    "native_array": "tau_J_d_native_Nm",
                    "aligned_array": "tau_J_d_Nm",
                    "aligned_resampling": "causal_previous_sample_hold",
                },
                "tau_J_Nm": {
                    "source_topic": FRANKA_STATE_TOPIC,
                    "message_field": "measured_joint_state.effort",
                    "meaning": "measured link-side joint effort diagnostic",
                    "native_array": "tau_J_native_Nm",
                    "aligned_array": "tau_J_Nm",
                    "aligned_resampling": "causal_previous_sample_hold",
                },
            }
        )
    return channels


def convert_bag(
    bag: str | Path,
    destination: str | Path,
    *,
    protocol_path: str | Path,
    speed_scale: float = 1.0,
    backend: str = "mujoco_ros",
    simulation_truth_path: str | Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Convert one raw bag, rejecting any unbound or stale acquisition."""
    bag = Path(bag)
    destination = Path(destination)
    if not bag.exists():
        raise ConversionError(f"no bag at {bag}")
    if destination.exists():
        raise ConversionError(f"recording destination already exists: {destination}")
    if backend not in BACKENDS:
        raise ConversionError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ConversionError("speed_scale must lie in (0, 1]")

    protocol, protocol_arrays = load_protocol_bundle(protocol_path)
    for key in ("family", "role", "analysis_windows"):
        if key not in protocol:
            raise ConversionError(
                f"protocol is missing {key!r}; regenerate it with the current "
                "compiler before collecting data"
            )

    simulation_metadata: dict[str, object] | None = None
    if simulation_truth_path is not None:
        if backend != "mujoco_ros":
            raise ConversionError(
                "simulation truth may only be attached to the mujoco_ros backend"
            )
        source = protocol.get("source_model")
        if not isinstance(source, Mapping):
            raise ConversionError("protocol source_model must be an object")
        simulation_metadata = {
            "preprocessing_mode": "raw_engine_exact",
            "position_noise_rad": 0.0,
            "velocity_noise_rad_s": 0.0,
            "plant_projection": "hydrax_contact_free_7dof",
            "truth": _simulation_truth_metadata(
                simulation_truth_path,
                expected_source_sha256=str(source.get("sha256", "")),
            ),
        }

    types = _topic_types(bag)
    messages = _read_messages(bag)
    start_ns, end_ns = _protocol_bounds(messages, types, protocol, speed_scale)
    nominal_duration = float(protocol["playback"]["duration_s"]) / speed_scale
    marker_duration = (end_ns - start_ns) * 1e-9
    duration_tolerance = max(0.25, 0.02 * nominal_duration)
    if abs(marker_duration - nominal_duration) > duration_tolerance:
        raise ConversionError(
            f"marked protocol duration {marker_duration:.3f} s does not match "
            f"the reviewed {nominal_duration:.3f} s (tolerance "
            f"{duration_tolerance:.3f} s)"
        )

    full_controller = _controller_series(messages, types)
    controller = _crop_controller(full_controller, start_ns, end_ns)
    controller_stamp_ns = np.asarray(controller[0], dtype=np.int64)
    time_s = (controller_stamp_ns - start_ns).astype(np.float64) * 1e-9
    mapped_segments, mapped_windows = _mapped_intervals(
        protocol, protocol_arrays, speed_scale
    )
    segment_index, analysis_eligible = _row_interval_arrays(
        time_s, mapped_segments, mapped_windows
    )
    transient_free, reversal_count, slew_transient_count = _transient_free_mask(
        time_s,
        np.asarray(controller[2], dtype=np.float64),
        np.asarray(controller[3], dtype=np.float64),
    )

    arrays: dict[str, np.ndarray] = {
        "time_s": time_s,
        "q_rad": np.asarray(controller[1], dtype=np.float64),
        "dq_rad_s": np.asarray(controller[2], dtype=np.float64),
        "tau_cmd_Nm": np.asarray(controller[3], dtype=np.float64),
        "protocol_segment_index": segment_index,
        "analysis_eligible": analysis_eligible,
    }
    if controller[4] is not None:
        arrays["q_desired_rad"] = np.asarray(controller[4], dtype=np.float64)

    telemetry_metadata: dict[str, object] | None = None
    if backend == "real":
        telemetry_arrays, telemetry_metadata = _franka_series(
            messages, types, controller_stamp_ns, start_ns, end_ns
        )
        arrays.update(telemetry_arrays)
        arrays["telemetry_transient_free"] = transient_free
        arrays["telemetry_fit_eligible"] = (
            analysis_eligible
            & transient_free
            & np.asarray(arrays["franka_new_sample"], dtype=np.bool_)
        )

    protocol_copy = {
        "protocol_id": protocol["protocol_id"],
        "family": protocol["family"],
        "role": protocol["role"],
        "content_sha256": protocol["content_sha256"],
        "duration_s": nominal_duration,
        "segments": protocol["segments"],
        "analysis_windows": protocol["analysis_windows"],
    }
    manifest: dict[str, object] = {
        "format": BOUND_RECORDING_FORMAT,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": backend,
        "speed_scale": speed_scale,
        "torque_source": (
            "controller output effort request; applied/post-limiter status "
            "is not asserted"
        ),
        "torque_channels": _torque_channels(backend == "real"),
        "raw_bag": str(bag),
        "protocol": protocol_copy,
        "protocol_timing": {
            "marker_format": PROTOCOL_EVENT_FORMAT,
            "marker_topic": PROTOCOL_EVENT_TOPIC,
            "start_stamp_ns": int(start_ns),
            "end_stamp_ns": int(end_ns),
            "selected_first_stamp_ns": int(controller_stamp_ns[0]),
            "selected_last_stamp_ns": int(controller_stamp_ns[-1]),
            "marked_duration_s": marker_duration,
            "speed_scale": speed_scale,
            "recording_time_origin": "protocol start marker",
        },
        "mapped_segments": mapped_segments,
        "mapped_analysis_windows": mapped_windows,
        "analysis_policy": {
            "edge_guard_s": DEFAULT_ANALYSIS_EDGE_GUARD_S,
            "protocol_rows_array": "analysis_eligible",
            "telemetry_rows_array": (
                "telemetry_fit_eligible" if backend == "real" else None
            ),
            "reversal_guard_s": REVERSAL_GUARD_S,
            "detected_joint_reversals": reversal_count,
            "hardware_torque_rate_limit_Nm_s": (HARDWARE_TORQUE_RATE_LIMIT_NM_S),
            "torque_slew_guard_s": TORQUE_SLEW_GUARD_S,
            "detected_rate_limit_request_transients": slew_transient_count,
            "segment_row_assignment": (
                "latest mapped segment start; interpolation rows before the "
                "next segment start remain with the preceding segment"
            ),
            "rate_limiter_policy": (
                "exclude controller requests at/above the deployed hardware "
                "rate limit; do not infer the unresolved limited waveform; "
                "use native tau_J_d samples and the telemetry fit mask"
            ),
        },
        "torque_limit_Nm": list(protocol["limits"]["torque_Nm"]),
        "torque_slew_warning_Nm_s": [HARDWARE_TORQUE_RATE_LIMIT_NM_S] * 7,
        "source_model": protocol["source_model"],
    }
    if telemetry_metadata is not None:
        manifest["telemetry_alignment"] = telemetry_metadata
    if simulation_metadata is not None:
        manifest["simulation"] = simulation_metadata

    written = write_recording(destination, manifest, arrays)
    return written, health_report(manifest, arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="recorded MCAP directory")
    parser.add_argument("destination", help="where to write the recording bundle")
    parser.add_argument("--protocol", required=True, help="protocol bundle played")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=BACKENDS, default="mujoco_ros")
    parser.add_argument(
        "--simulation-truth",
        default=None,
        help="simulation-only truth manifest used to generate the ROS plant",
    )
    arguments = parser.parse_args(argv)

    written, health = convert_bag(
        arguments.bag,
        arguments.destination,
        protocol_path=arguments.protocol,
        speed_scale=arguments.speed_scale,
        backend=arguments.backend,
        simulation_truth_path=arguments.simulation_truth,
    )
    print(written)
    print(json.dumps(health, indent=2))
    return 0 if health["usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
