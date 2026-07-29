"""Gates for the generated ROS-simulation plant.

The plant is the project's own nominal model republished under ROS joint
names. These check that it is exactly that — same robot, same actuators, no
second description that could drift from the one being identified.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.excitation import _HOME_QPOS
from fer_mujoco_sysid.model import ModelPaths
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES
from fer_mujoco_sysid.ros.scene import FINGER_OPEN_M, TABLE_Z_M, write_scene


@pytest.fixture(scope="module")
def plant(tmp_path_factory, model_paths: ModelPaths) -> Path:
    return write_scene(tmp_path_factory.mktemp("plant") / "plant.xml")


def test_plant_carries_the_ros_joint_names(plant: Path) -> None:
    """These names are the binding between MuJoCo and ros2_control."""
    model = mujoco.MjModel.from_xml_path(str(plant))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    assert names[:7] == list(ROS_ARM_JOINT_NAMES)
    assert names[7:] == ["fer_finger_joint1", "fer_finger_joint2"]
    for joint in ROS_ARM_JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint) >= 0


def test_plant_is_the_nominal_model_and_nothing_else(
    plant: Path, model_paths: ModelPaths
) -> None:
    """Renamed and re-homed, but the same robot: no second description.

    Building the simulated plant from another repository's MJCF would make
    this project depend on that repository, and would let the plant drift
    from the model being identified.
    """
    wrapped = mujoco.MjModel.from_xml_path(str(plant))
    source = mujoco.MjModel.from_xml_path(str(model_paths.hydrax))

    assert wrapped.njnt == source.njnt
    assert wrapped.nu == source.nu
    assert wrapped.nbody == source.nbody
    # Serializing through XML keeps about seven significant digits, so the
    # comparison is relative rather than exact: 5e-6 kg on a 5 kg link. Tight
    # enough that any real edit to a physical parameter fails this.
    physical = 1e-5
    np.testing.assert_allclose(wrapped.body_mass, source.body_mass, rtol=physical)
    np.testing.assert_allclose(wrapped.body_inertia, source.body_inertia, rtol=physical)
    np.testing.assert_allclose(wrapped.body_ipos, source.body_ipos, atol=1e-9)
    np.testing.assert_allclose(wrapped.dof_armature, source.dof_armature, rtol=physical)
    np.testing.assert_allclose(wrapped.dof_damping, source.dof_damping, rtol=physical)
    np.testing.assert_allclose(
        wrapped.dof_frictionloss, source.dof_frictionloss, atol=1e-9
    )
    np.testing.assert_allclose(wrapped.jnt_range, source.jnt_range, rtol=physical)
    np.testing.assert_allclose(
        wrapped.actuator_forcerange, source.actuator_forcerange, rtol=physical
    )


def test_plant_is_gravity_free_by_default(plant: Path) -> None:
    """The robot compensates its own gravity under torque commands.

    Simulating with gravity on makes the trajectory controller stand against
    a weight the real arm never asks it to carry: with these gains that was
    36.9 mrad of steady-state error on joint 4, enough to fail the
    start-state check for a reason that does not exist on hardware. The
    deployment stack states the same convention from the other side
    (``remove_gravity_compensation_effort``: true on the robot).
    """
    model = mujoco.MjModel.from_xml_path(str(plant))
    np.testing.assert_allclose(model.opt.gravity, 0.0)


@pytest.mark.parametrize("gravity", [False, True])
def test_home_keyframe_holds_the_arm_before_control(
    tmp_path: Path, gravity: bool
) -> None:
    """The keyframe's ctrl acts until the controller activates; it must hold."""
    written = write_scene(tmp_path / "plant.xml", gravity=gravity)
    model = mujoco.MjModel.from_xml_path(str(written))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    drift = np.abs(data.qpos[:7] - np.asarray(_HOME_QPOS))
    assert np.max(drift) < 0.02, f"arm moved {np.max(drift):.4f} rad before control"


def test_home_keyframe_starts_at_the_protocol_pose(plant: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(plant))
    assert model.nkey == 1
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0) == "home"
    np.testing.assert_allclose(model.key_qpos[0][:7], _HOME_QPOS, atol=1e-6)
    np.testing.assert_allclose(model.key_qpos[0][7:], FINGER_OPEN_M, atol=1e-6)


def test_plant_adds_the_table_below_the_base(plant: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(plant))
    table = model.geom("table_surface")
    assert table.type == mujoco.mjtGeom.mjGEOM_PLANE
    assert table.pos[2] == pytest.approx(TABLE_Z_M)
