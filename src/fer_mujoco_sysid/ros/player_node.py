"""Play one compiled excitation protocol through a trajectory controller.

The node is deliberately linear and fail-closed: preflight, approach the
protocol start, verify arrival, play once, report. It owns no state machine
worth the name because there is nothing to recover from — anything unexpected
stops the arm and asks a human.

Every decision it makes lives in :mod:`fer_mujoco_sysid.playback`, which has
no ROS dependency and is covered by the ordinary test suite. What is left
here is the plumbing: parameters, service calls, action goals, and turning
failures into messages that say what to do about them.

Run through ``scripts/run-protocol``; the launch file wires the parameters.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from fer_mujoco_sysid.playback import (
    APPROACH_SPEED_RAD_S,
    START_TOLERANCE_RAD,
    PlaybackError,
    PlaybackPlan,
    approach_playback,
    check_claimed_interfaces,
    check_start_state,
    joint_order,
    protocol_playback,
    reorder_to,
    start_qpos,
)
from fer_mujoco_sysid.protocol import load_protocol_bundle

#: Robot modes in which it is safe to command a trajectory. Anything else —
#: guiding, reflex, user-stopped — means a human or the robot itself has taken
#: control, and commanding through that is how you get a surprise.
SAFE_ROBOT_MODES = {1, 2}  # ROBOT_MODE_IDLE, ROBOT_MODE_MOVE

#: How long to wait for the controller to acknowledge a cancel before giving
#: up and telling the operator to stop the arm themselves.
CANCEL_TIMEOUT_S = 5.0

ROBOT_MODE_NAMES = {
    0: "OTHER",
    1: "IDLE",
    2: "MOVE",
    3: "GUIDING",
    4: "REFLEX",
    5: "USER_STOPPED",
    6: "AUTOMATIC_ERROR_RECOVERY",
}


class ProtocolPlayer(Node):
    """Preflight, approach, and play a single protocol bundle."""

    def __init__(self) -> None:
        super().__init__("fer_sysid_player")

        self.declare_parameter("protocol_path", "")
        self.declare_parameter("controller_name", "fer_sysid_arm_controller")
        self.declare_parameter("controller_manager", "/controller_manager")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "robot_state_topic", "/franka_robot_state_broadcaster/robot_state"
        )
        self.declare_parameter("require_robot_state", False)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("approach_speed_rad_s", APPROACH_SPEED_RAD_S)
        self.declare_parameter("start_tolerance_rad", START_TOLERANCE_RAD)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("preflight_timeout_s", 30.0)

        self._joint_state: JointState | None = None
        self._robot_state = None
        self._goal_handle = None

        self.create_subscription(
            JointState,
            self._string("joint_states_topic"),
            self._on_joint_state,
            10,
        )
        if self._bool("require_robot_state"):
            self._subscribe_robot_state()

        controller = self._string("controller_name")
        self._list_controllers = self.create_client(
            ListControllers,
            f"{self._string('controller_manager')}/list_controllers",
        )
        self._action_name = f"{controller}/follow_joint_trajectory"
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, self._action_name
        )

    # -- parameters -------------------------------------------------------

    def _string(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    # -- subscriptions ----------------------------------------------------

    def _subscribe_robot_state(self) -> None:
        try:
            from agimus_franka_msgs.msg import AgimusFrankaRobotState
        except ImportError as error:  # pragma: no cover - hardware-only path
            raise PlaybackError(
                "require_robot_state is set but agimus_franka_msgs is not on the "
                "path. Source the Agimus workspace, or clear require_robot_state "
                "if this is not the real robot."
            ) from error
        self.create_subscription(
            AgimusFrankaRobotState,
            self._string("robot_state_topic"),
            self._on_robot_state,
            10,
        )

    def _on_joint_state(self, message: JointState) -> None:
        self._joint_state = message

    def _on_robot_state(self, message: object) -> None:
        self._robot_state = message

    def _wait_for(self, predicate, description: str, timeout_s: float) -> None:
        # Wall clock on purpose: under ``use_sim_time`` the ROS clock may not
        # be running yet, and a timeout that cannot expire is a hang, not a
        # timeout.
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                raise PlaybackError(
                    f"timed out after {timeout_s:.0f} s waiting for {description}"
                )
            self._sleep(0.05)

    def _sleep(self, seconds: float) -> None:
        # The executor spins in another thread; a plain sleep is enough and
        # keeps this sequence readable as the straight line that it is.
        threading.Event().wait(seconds)

    # -- preflight --------------------------------------------------------

    def measured_positions(self, joint_names) -> np.ndarray:
        message = self._joint_state
        if message is None:  # pragma: no cover - guarded by the wait above
            raise PlaybackError("no joint state received")
        return reorder_to(dict(zip(message.name, message.position)), joint_names)

    def check_controller(self, joint_names) -> None:
        name = self._string("controller_name")
        if not self._list_controllers.wait_for_service(timeout_sec=10.0):
            raise PlaybackError(
                f"{self._string('controller_manager')}/list_controllers did not "
                "appear. Is the controller manager running?"
            )
        response = self._call(self._list_controllers, ListControllers.Request())
        states = {controller.name: controller for controller in response.controller}
        if name not in states:
            raise PlaybackError(
                f"controller {name!r} is not loaded. Loaded: "
                + (", ".join(sorted(states)) or "none")
            )
        controller = states[name]
        if controller.state != "active":
            raise PlaybackError(
                f"controller {name!r} is {controller.state!r}, not 'active'."
            )
        check_claimed_interfaces(list(controller.claimed_interfaces), joint_names)
        self.get_logger().info(
            f"controller {name!r} active, commanding "
            f"{len(controller.claimed_interfaces)} effort interfaces"
        )

    def check_robot_state(self) -> None:
        if not self._bool("require_robot_state"):
            self.get_logger().info("robot state check skipped (simulation backend)")
            return
        self._wait_for(
            lambda: self._robot_state is not None,
            f"a message on {self._string('robot_state_topic')}",
            self._float("preflight_timeout_s"),
        )
        state = self._robot_state
        mode = int(state.robot_mode)
        if mode not in SAFE_ROBOT_MODES:
            raise PlaybackError(
                f"robot mode is {ROBOT_MODE_NAMES.get(mode, mode)}; "
                "release the user stop and clear any reflex before playing."
            )
        active = _active_errors(state.current_errors)
        if active:
            raise PlaybackError(
                "robot reports active errors: " + ", ".join(active) + ". "
                "Run error recovery first."
            )
        self.get_logger().info(
            f"robot mode {ROBOT_MODE_NAMES.get(mode, mode)}, no active errors"
        )

    def _call(self, client, request):
        future = client.call_async(request)
        deadline = time.monotonic() + 20.0
        while not future.done():
            if time.monotonic() > deadline:
                raise PlaybackError(f"service call to {client.srv_name} timed out")
            self._sleep(0.02)
        return future.result()

    # -- execution --------------------------------------------------------

    def send(self, plan: PlaybackPlan) -> None:
        """Send one plan as a single goal and block until it finishes."""
        if not self._trajectory_client.wait_for_server(timeout_sec=10.0):
            raise PlaybackError(f"action server {self._action_name} did not appear")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = _to_message(plan)
        self.get_logger().info(f"sending {plan.summary()}")

        send_future = self._trajectory_client.send_goal_async(goal)
        while not send_future.done():
            self._sleep(0.02)
        handle = send_future.result()
        if not handle.accepted:
            raise PlaybackError(
                f"{plan.label}: the controller rejected the trajectory. Its log "
                "says why — usually a joint name mismatch or a point in the past."
            )
        self._goal_handle = handle

        result_future = handle.get_result_async()
        while not result_future.done():
            self._sleep(0.05)
        self._goal_handle = None

        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise PlaybackError(
                f"{plan.label}: controller returned error {result.error_code} "
                f"({result.error_string or 'no detail'}). The arm has stopped "
                "where the failure happened."
            )
        self.get_logger().info(f"{plan.label}: completed")

    def cancel(self) -> None:
        """Cancel the running trajectory and wait for the controller to say so.

        Returning before the cancel is acknowledged would let the process exit
        — and the launch tear the controller manager down — while the arm is
        still executing. The controller decelerates on cancel; give it the
        chance to.
        """
        handle = self._goal_handle
        if handle is None:
            return
        self.get_logger().warn("cancelling the active trajectory")
        future = handle.cancel_goal_async()
        deadline = time.monotonic() + CANCEL_TIMEOUT_S
        while not future.done() and time.monotonic() < deadline:
            self._sleep(0.02)
        if not future.done():
            self.get_logger().error(
                "the controller did not acknowledge the cancel; stop the arm "
                "at the robot if it is still moving"
            )


def _active_errors(errors) -> list[str]:
    """Names of the boolean error flags that are set."""
    fields = errors.get_fields_and_field_types()
    return [
        name
        for name, kind in fields.items()
        if kind == "boolean" and bool(getattr(errors, name))
    ]


def _to_message(plan: PlaybackPlan) -> JointTrajectory:
    message = JointTrajectory()
    message.joint_names = list(plan.joint_names)
    points = []
    for index in range(plan.samples):
        point = JointTrajectoryPoint()
        point.positions = plan.q_rad[index].tolist()
        point.velocities = plan.dq_rad_s[index].tolist()
        if plan.ddq_rad_s2 is not None:
            point.accelerations = plan.ddq_rad_s2[index].tolist()
        # Split in integer nanoseconds. Rounding the fractional part on its
        # own can land on 1e9 ns, which is not a valid Duration.
        total_ns = int(round(float(plan.time_s[index]) * 1e9))
        point.time_from_start = Duration(
            sec=total_ns // 1_000_000_000, nanosec=total_ns % 1_000_000_000
        )
        points.append(point)
    message.points = points
    return message


def _run(node: ProtocolPlayer) -> None:
    path = node._string("protocol_path")
    if not path:
        raise PlaybackError("protocol_path is required")
    bundle = Path(path)
    manifest, arrays = load_protocol_bundle(bundle)
    names = joint_order(manifest)
    speed_scale = node._float("speed_scale")
    plan = protocol_playback(manifest, arrays, speed_scale=speed_scale)

    node.get_logger().info(
        f"protocol {manifest['protocol_id']} {manifest['revision']} "
        f"({str(manifest['content_sha256'])[:16]}...), "
        f"speed scale {speed_scale:g}, {plan.duration_s:.1f} s"
    )

    node.check_controller(names)
    node._wait_for(
        lambda: node._joint_state is not None,
        f"a message on {node._string('joint_states_topic')}",
        node._float("preflight_timeout_s"),
    )
    node.check_robot_state()

    measured = node.measured_positions(names)
    check = check_start_state(
        measured, manifest, tolerance_rad=node._float("start_tolerance_rad")
    )
    node.get_logger().info(f"before approach: {check.describe()}")

    if node._bool("dry_run"):
        node.get_logger().info(
            "dry run: preflight passed, no motion commanded. "
            f"Would approach over {np.max(np.abs(check.error_rad)):.3f} rad "
            f"and then play {plan.summary()}"
        )
        return

    # Always run the approach, even from a pose that already looks right: it
    # costs two seconds and it is the only thing that guarantees the arm is
    # settled and at rest where the protocol assumes it starts.
    node.send(
        approach_playback(
            measured,
            start_qpos(manifest),
            names,
            speed_rad_s=node._float("approach_speed_rad_s"),
        )
    )

    measured = node.measured_positions(names)
    check = check_start_state(
        measured, manifest, tolerance_rad=node._float("start_tolerance_rad")
    )
    if not check.satisfied:
        raise PlaybackError(
            "the arm did not reach the protocol start: "
            + check.describe()
            + ". Playing from here would run a different trajectory than the "
            "one that was validated."
        )
    node.get_logger().info(f"at start: {check.describe()}")

    node.send(plan)
    node.get_logger().info(f"protocol {plan.label} played to completion")


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = ProtocolPlayer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        _run(node)
        status = 0
    except PlaybackError as error:
        node.get_logger().error(str(error))
        node.cancel()
        status = 1
    except KeyboardInterrupt:
        node.get_logger().warn("interrupted")
        node.cancel()
        status = 130
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
