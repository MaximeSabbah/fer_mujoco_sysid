"""Turn a recorded bag into a recording bundle the fit can consume.

Runs under ROS, because deserializing messages needs the message packages;
what it writes is plain NumPy, so the fitting side never needs ROS. That
split is the whole reason this file exists separately from the fit.

Everything comes from one topic: the trajectory controller's state message
carries the commanded effort (``output.effort``) *and* the measured joint
state (``feedback``) with a single timestamp. Taking both from one message
avoids interpolating one signal onto another's clock, which would smear the
torque against the velocity — the exact relationship friction identification
reads. Hardware telemetry is joined in separately, and is nearest-sample
aligned rather than interpolated, because it is diagnostic rather than fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fer_mujoco_sysid.dataset import health_report, write_recording
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES, load_protocol_bundle

CONTROLLER_STATE_TYPE = "control_msgs/msg/JointTrajectoryControllerState"
FRANKA_STATE_TYPE = "agimus_franka_msgs/msg/AgimusFrankaRobotState"


class ConversionError(RuntimeError):
    """A bag cannot be turned into a usable recording."""


def _read_messages(bag: Path) -> list[tuple[str, Any, int]]:
    """Every message in a bag, as (topic, deserialized message, timestamp ns)."""
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
        except (KeyError, ModuleNotFoundError):  # pragma: no cover - env dependent
            continue
        messages.append((topic, message, stamp))
    return messages


def _stamp_seconds(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _controller_series(
    messages: list[tuple[str, Any, int]], types: dict[str, str]
) -> tuple[np.ndarray, ...]:
    """Time, measured q/dq and commanded effort from the controller state."""
    rows: list[tuple[float, list[float], list[float], list[float]]] = []
    order: list[str] | None = None
    for topic, message, _ in messages:
        if types.get(topic) != CONTROLLER_STATE_TYPE:
            continue
        effort = list(message.output.effort)
        positions = list(message.feedback.positions)
        velocities = list(message.feedback.velocities)
        # What the controller was *aiming* at. Not used by the fit, but it is
        # the other half of every tracking question: the difference between
        # this and the feedback is what the arm failed to follow.
        desired = list(message.reference.positions)
        if not effort or not positions or not velocities:
            # The controller publishes state before it has a command; those
            # rows carry no torque and cannot be fitted from.
            continue
        if order is None:
            order = list(message.joint_names)
        rows.append(
            (_stamp_seconds(message.header), positions, velocities, effort, desired)
        )

    if not rows:
        raise ConversionError(
            "the bag holds no usable controller state. Was the controller "
            f"publishing {CONTROLLER_STATE_TYPE} while the protocol played?"
        )
    if order != list(ROS_ARM_JOINT_NAMES):
        raise ConversionError(
            f"controller reports joints {order}, expected {list(ROS_ARM_JOINT_NAMES)}"
        )

    rows.sort(key=lambda row: row[0])
    # Two messages can carry the same stamp — the controller publishes faster
    # than the clock it stamps with can resolve. A duplicate carries no new
    # time information, so the later one wins; keeping both would leave a
    # zero-length step that no resampling or differentiation can survive.
    unique: list[tuple] = []
    for row in rows:
        if unique and row[0] <= unique[-1][0]:
            unique[-1] = row
            continue
        unique.append(row)
    duplicates = len(rows) - len(unique)
    if duplicates:
        print(f"  dropped {duplicates} duplicate timestamps of {len(rows)} samples")
    rows = unique

    time_s = np.asarray([row[0] for row in rows], dtype=np.float64)
    desired = [row[4] for row in rows]
    return (
        time_s,
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
        np.asarray([row[3] for row in rows], dtype=np.float64),
        np.asarray(desired, dtype=np.float64)
        if all(len(row) == 7 for row in desired)
        else None,
    )


def _franka_series(
    messages: list[tuple[str, Any, int]], types: dict[str, str], time_s: np.ndarray
) -> dict[str, np.ndarray]:
    """Hardware telemetry, nearest-sample aligned onto the controller clock.

    Nearest-sample rather than interpolated on purpose: these channels are
    recorded for later analysis, and inventing values between samples would
    make them look better resolved than they are.
    """
    stamps: list[float] = []
    channels: dict[str, list[list[float]]] = {
        "tau_J_Nm": [],
        "tau_J_d_Nm": [],
        "theta_rad": [],
        "dtheta_rad_s": [],
    }
    for topic, message, _ in messages:
        if types.get(topic) != FRANKA_STATE_TYPE:
            continue
        stamps.append(_stamp_seconds(message.header))
        channels["tau_J_Nm"].append(list(message.measured_joint_state.effort))
        channels["tau_J_d_Nm"].append(list(message.desired_joint_state.effort))
        channels["theta_rad"].append(list(message.measured_joint_motor_state.position))
        channels["dtheta_rad_s"].append(
            list(message.measured_joint_motor_state.velocity)
        )
    if not stamps:
        return {}

    source = np.asarray(stamps, dtype=np.float64)
    order = np.argsort(source)
    nearest = np.clip(np.searchsorted(source[order], time_s), 0, len(order) - 1)
    out: dict[str, np.ndarray] = {}
    for name, values in channels.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 7:
            continue
        out[name] = array[order][nearest]
    return out


def convert_bag(
    bag: str | Path,
    destination: str | Path,
    *,
    protocol_path: str | Path,
    speed_scale: float = 1.0,
    backend: str = "mujoco_ros",
) -> tuple[Path, dict[str, object]]:
    """Convert one recorded bag into a recording bundle next to it."""
    bag = Path(bag)
    if not bag.exists():
        raise ConversionError(f"no bag at {bag}")

    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    del reader

    messages = _read_messages(bag)
    time_s, q_rad, dq_rad_s, tau_cmd, q_desired = _controller_series(messages, types)
    # Start the clock at the first sample: absolute ROS time carries no
    # meaning for a fit, and a shared origin makes runs comparable.
    time_s = time_s - time_s[0]

    arrays: dict[str, np.ndarray] = {
        "time_s": time_s,
        "q_rad": q_rad,
        "dq_rad_s": dq_rad_s,
        "tau_cmd_Nm": tau_cmd,
    }
    if q_desired is not None:
        arrays["q_desired_rad"] = q_desired
    arrays.update(_franka_series(messages, types, time_s))

    manifest_protocol, _ = load_protocol_bundle(protocol_path)
    manifest: dict[str, object] = {
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": backend,
        "speed_scale": speed_scale,
        "torque_source": "joint_trajectory_controller commanded effort",
        "raw_bag": str(bag),
        "protocol": {
            "protocol_id": manifest_protocol["protocol_id"],
            "revision": manifest_protocol["revision"],
            "content_sha256": manifest_protocol["content_sha256"],
            # What was actually asked of the robot, after time scaling.
            "duration_s": float(manifest_protocol["playback"]["duration_s"])
            / speed_scale,
        },
        "torque_limit_Nm": list(manifest_protocol["limits"]["torque_Nm"]),
        "source_model": manifest_protocol["source_model"],
    }

    written = write_recording(destination, manifest, arrays)
    health = health_report(manifest | {"protocol": manifest["protocol"]}, arrays)
    return written, health


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="recorded MCAP directory")
    parser.add_argument("destination", help="where to write the recording bundle")
    parser.add_argument("--protocol", required=True, help="protocol bundle played")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--backend", default="mujoco_ros")
    arguments = parser.parse_args(argv)

    written, health = convert_bag(
        arguments.bag,
        arguments.destination,
        protocol_path=arguments.protocol,
        speed_scale=arguments.speed_scale,
        backend=arguments.backend,
    )
    print(written)
    print(json.dumps(health, indent=2))
    return 0 if health["usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
