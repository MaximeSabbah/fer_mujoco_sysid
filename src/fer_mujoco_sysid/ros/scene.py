"""Build the MuJoCo plant the ROS simulation backend loads.

The plant is the project's **own** nominal model — the hydrax MJCF pinned by
`contracts/nominal_sources.toml`, which is the model this project exists to
identify — republished with the joint names the ROS stack uses and a start
keyframe. No other repository is involved: the identification stack simulates
the thing it identifies, and depends on nothing it does not already need for
fitting.

Three things are added that the source model does not carry, because they
belong to the experiment rather than to the robot:

* the arm joints renamed to the ROS convention (``fer_joint1..7``), which is
  how ``mujoco_ros2_control`` binds a MuJoCo joint to a ros2_control one;
* the table the robot is bolted to, at the same height every protocol's
  clearance check uses;
* a start keyframe, including the actuator commands that hold the arm still
  in the window between the simulator starting and the controller activating.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from fer_mujoco_sysid.excitation import _HOME_QPOS
from fer_mujoco_sysid.export import IdentifiedParameters, apply_parameters
from fer_mujoco_sysid.model import (
    HYDRAX_ARM_JOINT_NAMES,
    HYDRAX_FINGER_JOINT_NAMES,
    ROS_FINGER_JOINT_NAMES,
    resolve_model_paths,
)
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

#: Where the table sits relative to the robot base, matching the clearance
#: check applied to every protocol at generation time.
TABLE_Z_M = -0.02

#: Gripper opening in the start keyframe: fully open, out of the way.
FINGER_OPEN_M = 0.04

SCENE_NAME = "fer_sysid_plant"

#: Hydrax joint name -> ROS joint name. ``mujoco_ros2_control`` resolves a
#: ros2_control joint to the MuJoCo actuator driving the joint of the same
#: name, so these names are the binding between the two worlds.
JOINT_RENAMES = dict(
    zip(HYDRAX_ARM_JOINT_NAMES, ROS_ARM_JOINT_NAMES, strict=True)
) | dict(zip(HYDRAX_FINGER_JOINT_NAMES, ROS_FINGER_JOINT_NAMES, strict=True))


def build_plant_spec(
    model_path: str | Path,
    *,
    gravity: bool = False,
    friction: IdentifiedParameters | None = None,
) -> mujoco.MjSpec:
    """The nominal model under ROS joint names, ready to be a ros2_control plant.

    ``gravity`` defaults to **off**, which is what makes the simulated plant
    behave like the real one. Under a torque command the FER compensates its
    own weight: the effort a controller sends is the effort *on top of*
    gravity compensation. The deployment stack encodes the same fact from the
    other side (``remove_gravity_compensation_effort``: true on the robot,
    false in simulation). A gravity-enabled simulated plant makes the
    trajectory controller fight a gravity error the robot never has — worth
    36.9 mrad of standing error on joint 4 under these gains, which is not a
    rehearsal of anything real.
    """
    model_path = Path(model_path).resolve()
    spec = mujoco.MjSpec.from_file(str(model_path))
    # Asset paths are resolved relative to the source file; the scene is
    # written elsewhere, so make them absolute before it moves.
    spec.meshdir = str((model_path.parent / spec.meshdir).resolve())
    spec.texturedir = str((model_path.parent / spec.texturedir).resolve())
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]

    if friction is not None:
        # Applied under the *hydrax* joint names, before the rename below, so
        # the same IdentifiedParameters an export produces can be injected
        # here. This is what makes a simulated campaign a real test: the fit
        # reads only the recording and has to recover values it was never
        # given.
        apply_parameters(spec, friction)

    renamed = 0
    for joint in spec.joints:
        if joint.name in JOINT_RENAMES:
            joint.name = JOINT_RENAMES[joint.name]
            renamed += 1
    if renamed != len(JOINT_RENAMES):
        raise ValueError(
            f"{model_path} has {renamed} of the {len(JOINT_RENAMES)} joints "
            "this plant expects; the nominal model changed shape"
        )
    for actuator in spec.actuators:
        if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
            actuator.target = JOINT_RENAMES.get(actuator.target, actuator.target)
            # Name each actuator after the joint it drives, the convention
            # mujoco_ros2_control looks for first.
            actuator.name = actuator.target
    # Everything that refers to a joint by name has to follow the rename —
    # the fingers are coupled by an equality constraint.
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT:
            equality.name1 = JOINT_RENAMES.get(equality.name1, equality.name1)
            equality.name2 = JOINT_RENAMES.get(equality.name2, equality.name2)
    return spec


def _hold_command(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Actuator commands that hold ``qpos`` still.

    The keyframe's ``ctrl`` is what acts between the simulator starting and
    the trajectory controller activating. Left wrong, the arm visibly moves
    in that window and the controller then has to catch it, which looks like
    — and could mask — a real tracking failure.

    The arm motors are direct-drive with unit gear, so the command they need
    is the generalized force inverse dynamics asks for. The gripper actuator
    is a position servo, so its command is a position, not a force.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)

    command = np.zeros(model.nu, dtype=np.float64)
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name in ROS_ARM_JOINT_NAMES:
            command[actuator] = data.qfrc_inverse[model.jnt_dofadr[joint]]
        else:
            command[actuator] = qpos[model.jnt_qposadr[joint]]
    return command


def write_scene(
    destination: str | Path,
    *,
    model_path: str | Path | None = None,
    arm_qpos: np.ndarray | None = None,
    with_table: bool = True,
    gravity: bool = False,
    friction: IdentifiedParameters | None = None,
) -> Path:
    """Write the plant MJCF, and prove it compiles before anything loads it."""
    if model_path is None:
        model_path = resolve_model_paths().require().hydrax
    arm = np.asarray(_HOME_QPOS if arm_qpos is None else arm_qpos, dtype=np.float64)
    if arm.shape != (7,):
        raise ValueError(f"arm_qpos must hold 7 joint values, got {arm.shape}")

    spec = build_plant_spec(model_path, gravity=gravity, friction=friction)
    spec.modelname = SCENE_NAME
    if with_table:
        table = spec.worldbody.add_geom()
        table.name = "table_surface"
        table.type = mujoco.mjtGeom.mjGEOM_PLANE
        table.size = [3.0, 3.0, 0.05]
        table.pos = [0.0, 0.0, TABLE_Z_M]
        table.rgba = [0.55, 0.52, 0.48, 1.0]

    model = spec.compile()
    qpos = np.array(model.qpos0, dtype=np.float64)
    qpos[:7] = arm
    for name in ROS_FINGER_JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            qpos[model.jnt_qposadr[joint]] = FINGER_OPEN_M

    key = spec.add_key()
    key.name = "home"
    key.qpos = qpos
    key.ctrl = _hold_command(model, qpos)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(spec.to_xml(), encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(destination))
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="path of the plant MJCF to write")
    parser.add_argument("--model", default=None, help="nominal MJCF to republish")
    parser.add_argument(
        "--offset-rad",
        type=float,
        default=0.0,
        help="perturb every arm joint of the start keyframe, so the "
        "simulated run exercises the move to the protocol start",
    )
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument(
        "--friction",
        default=None,
        help="JSON file or inline JSON giving the plant hidden frictionloss / "
        "damping / armature. Simulation only: it makes a recorded campaign a "
        "real test of the fit, which never sees these values.",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="simulate with gravity on. Off by default, because the robot "
        "compensates its own gravity under torque commands",
    )
    arguments = parser.parse_args(argv)

    friction = None
    if arguments.friction:
        source = Path(arguments.friction)
        raw = source.read_text() if source.is_file() else arguments.friction
        friction = IdentifiedParameters.from_mapping(json.loads(raw))

    written = write_scene(
        arguments.destination,
        model_path=arguments.model,
        arm_qpos=np.asarray(_HOME_QPOS, dtype=np.float64) + arguments.offset_rad,
        with_table=not arguments.no_table,
        gravity=arguments.gravity,
        friction=friction,
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
