"""Bring up a FER — simulated or real — and play one excitation protocol.

``backend:=mujoco`` and ``backend:=real`` differ in exactly two places: which
hardware plugin the description loads, and whether the Franka telemetry
broadcaster is spawned. The controller configuration, the trajectory
controller, the player and every check it runs are identical, which is the
point: the rehearsal in simulation exercises the code that will move the real
arm.

Launched through ``scripts/run-protocol``; that wrapper sources the
environment and resolves protocol identifiers to bundle paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue

REPOSITORY = Path(__file__).resolve().parents[1]
CONTROLLERS_FILE = REPOSITORY / "config" / "fer_sysid_controllers.yaml"
ARM_CONTROLLER = "fer_sysid_arm_controller"
CONTROLLER_MANAGER = "/controller_manager"
SPAWNER_TIMEOUT = "30"

#: The arm controller publishes here; joint_state_publisher merges it with
#: the description's remaining joints into /joint_states.
ARM_JOINT_STATES_TOPIC = "/arm/joint_states"
FRANKA_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
FINGER_OPEN_M = 0.04


def _argument(context, name: str) -> str:
    return LaunchConfiguration(name).perform(context).strip()


def launch_setup(context, *args, **kwargs):
    del args, kwargs

    backend = _argument(context, "backend").lower()
    if backend not in {"mujoco", "real"}:
        raise RuntimeError(f"backend must be 'mujoco' or 'real', got {backend!r}")
    is_mujoco = backend == "mujoco"

    protocol = _argument(context, "protocol")
    if not protocol:
        raise RuntimeError("protocol:=<path to a bundle revision directory> is required")
    protocol_path = Path(protocol).resolve()
    if not (protocol_path / "protocol.json").is_file():
        raise RuntimeError(f"{protocol_path} does not hold a protocol bundle")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    actions: list = []
    if is_mujoco:
        # Generated ahead of the launch by scripts/run-protocol, which has the
        # MuJoCo-capable interpreter. Nothing inside a ROS process imports
        # MuJoCo — see fer_mujoco_sysid.protocol for why.
        scene = _argument(context, "mujoco_scene")
        if not scene or not Path(scene).is_file():
            raise RuntimeError(
                "backend:=mujoco needs mujoco_scene:=<path to a generated "
                "scene>. Launch through scripts/run-protocol, which generates "
                "it (python -m fer_mujoco_sysid.ros.scene)."
            )
        actions.append(LogInfo(msg=f"simulation scene: {scene}"))
        description_command = [
            FindExecutable(name="xacro"),
            " ",
            str(REPOSITORY / "urdf" / "fer_sysid_mujoco.urdf.xacro"),
            " mujoco_model:=",
            scene,
            " headless:=",
            LaunchConfiguration("headless"),
        ]
    else:
        robot_ip = _argument(context, "robot_ip")
        if not robot_ip:
            raise RuntimeError("backend:=real needs robot_ip:=<hostname or address>")
        description_command = [
            FindExecutable(name="xacro"),
            " ",
            str(REPOSITORY / "urdf" / "fer_sysid_real.urdf.xacro"),
            " robot_ip:=",
            robot_ip,
        ]

    robot_description = {
        "robot_description": ParameterValue(
            Command(description_command), value_type=str
        )
    }
    use_sim_time = {"use_sim_time": is_mujoco}

    actions.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, use_sim_time],
            on_exit=Shutdown(),
        )
    )

    if is_mujoco:
        control_node = Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            output="both",
            emulate_tty=True,
            parameters=[use_sim_time, ParameterFile(str(CONTROLLERS_FILE))],
            on_exit=Shutdown(),
        )
    else:
        control_node = Node(
            package="controller_manager",
            executable="ros2_control_node",
            output="screen",
            parameters=[ParameterFile(str(CONTROLLERS_FILE)), robot_description],
            on_exit=Shutdown(),
        )
    actions.append(control_node)

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(REPOSITORY / "config" / "protocol_review.rviz")],
            parameters=[use_sim_time],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="log",
        )
    )

    # The arm controller owns only the seven arm joints, so the broadcaster
    # publishes only those — and robot_state_publisher then has no transform
    # for the finger links, which is why they would be missing in RViz. Merge
    # in the rest of the description's joints the way the Agimus stack does:
    # the broadcaster keeps its own topic, and joint_state_publisher fills the
    # gaps to produce a complete /joint_states for TF and RViz. The player
    # still reads the broadcaster directly, at full rate and unpadded.
    actions.append(
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            output="log",
            parameters=[
                use_sim_time,
                {
                    "source_list": [ARM_JOINT_STATES_TOPIC],
                    "rate": 30,
                    # The gripper is not actuated during identification; show
                    # it where the plant actually holds it: open, out of the way.
                    "zeros.fer_finger_joint1": FINGER_OPEN_M,
                    "zeros.fer_finger_joint2": FINGER_OPEN_M,
                },
            ],
        )
    )

    broadcasters = ["joint_state_broadcaster"]
    if not is_mujoco:
        # Full FCI telemetry: only exists against real hardware, and slice B
        # records it. The player reads robot mode and error flags from it.
        broadcasters.append("franka_robot_state_broadcaster")
    broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            *broadcasters,
            "--controller-manager",
            CONTROLLER_MANAGER,
            "--controller-manager-timeout",
            SPAWNER_TIMEOUT,
            "--param-file",
            str(CONTROLLERS_FILE),
            "--controller-ros-args=--remap",
            f"--controller-ros-args=joint_states:={ARM_JOINT_STATES_TOPIC}",
        ],
        output="screen",
    )
    actions.append(broadcaster_spawner)

    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            ARM_CONTROLLER,
            "--controller-manager",
            CONTROLLER_MANAGER,
            "--controller-manager-timeout",
            SPAWNER_TIMEOUT,
            "--param-file",
            str(CONTROLLERS_FILE),
        ],
        output="screen",
    )
    # Ordered: the arm controller must not claim effort interfaces before the
    # state broadcaster is publishing, or the player's first check races it.
    actions.append(
        RegisterEventHandler(
            OnProcessExit(target_action=broadcaster_spawner, on_exit=[arm_spawner])
        )
    )

    # Recording is a plain `ros2 bag record` process: it subscribes like any
    # other node and never touches the control path. The raw bag is the
    # immutable record of what happened; conversion derives everything else
    # from it, so a dataset can always be rebuilt without re-running the robot.
    record_dir = _argument(context, "record")
    if record_dir:
        topics = [
            ARM_JOINT_STATES_TOPIC,
            f"/{ARM_CONTROLLER}/controller_state",
            "/dynamic_joint_states",
        ]
        if is_mujoco:
            topics.append("/clock")
        else:
            # Full FCI telemetry: tau_J, tau_J_d, motor-side theta/dtheta,
            # robot mode and error flags. Recorded even though the fit uses
            # the commanded effort, because re-running the robot to get a
            # channel we chose not to record is the expensive mistake.
            topics.append(FRANKA_STATE_TOPIC)
        actions.append(
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "record",
                    "--storage", "mcap",
                    "--output", str(Path(record_dir).resolve()),
                    *topics,
                ],
                output="log",
            )
        )

    player = Node(
        executable=sys.executable,
        arguments=["-m", "fer_mujoco_sysid.ros.player_node"],
        output="screen",
        emulate_tty=True,
        additional_env=environment,
        parameters=[
            use_sim_time,
            {
                "protocol_path": str(protocol_path),
                "controller_name": ARM_CONTROLLER,
                "controller_manager": CONTROLLER_MANAGER,
                "joint_states_topic": ARM_JOINT_STATES_TOPIC,
                "require_robot_state": not is_mujoco,
                "speed_scale": ParameterValue(
                    LaunchConfiguration("speed_scale"), value_type=float
                ),
                "approach_speed_rad_s": ParameterValue(
                    LaunchConfiguration("approach_speed_rad_s"), value_type=float
                ),
                "dry_run": ParameterValue(
                    LaunchConfiguration("dry_run"), value_type=bool
                ),
            },
        ],
        # The run is over when the protocol is: tear the stack down with it,
        # so a finished or failed run never leaves controllers holding the arm.
        on_exit=Shutdown(),
    )
    actions.append(
        RegisterEventHandler(OnProcessExit(target_action=arm_spawner, on_exit=[player]))
    )

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend",
                default_value="mujoco",
                description="'mujoco' for the simulated rehearsal, 'real' for the robot.",
            ),
            DeclareLaunchArgument(
                "protocol",
                description="Path to a protocol bundle revision directory.",
            ),
            DeclareLaunchArgument(
                "speed_scale",
                default_value="1.0",
                description="Replay the identical path over 1/scale times as long. "
                "Velocities scale linearly, accelerations quadratically. Must be <= 1.",
            ),
            DeclareLaunchArgument(
                "approach_speed_rad_s",
                default_value="0.2",
                description="Joint speed cap for the move to the protocol start.",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="Run every preflight check and stop before moving.",
            ),
            DeclareLaunchArgument(
                "robot_ip",
                default_value="",
                description="FCI hostname or address. Required for backend:=real.",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="MuJoCo viewer off/on. Simulation backend only.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Show the arm in RViz as well. Works on both backends: "
                "RViz draws the robot description from the measured joint states, "
                "so on hardware it shows the real arm.",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="",
                description="Directory to record the run into as an MCAP bag. "
                "Empty records nothing.",
            ),
            DeclareLaunchArgument(
                "mujoco_scene",
                default_value="",
                description="Generated simulation scene. Required for "
                "backend:=mujoco; scripts/run-protocol fills it in.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
