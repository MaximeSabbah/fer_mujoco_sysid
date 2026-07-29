"""Gates for the generated ROS system-identification plant.

The ROS backend must run the exact seven-axis projection used by fitting and
Hydrax, merely republished under ROS joint names. These tests prevent the
articulated gripper, a different integrator, or contact dynamics from creating
a second plant that cannot close numerically against identification.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.excitation import _HOME_QPOS
from fer_mujoco_sysid.export import BodyInertial, IdentifiedParameters
from fer_mujoco_sysid.model import ModelPaths
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES
from fer_mujoco_sysid.ros.scene import TABLE_Z_M, write_scene
from fer_mujoco_sysid.stages import fitting_spec


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
    assert names == list(ROS_ARM_JOINT_NAMES)
    assert (model.nq, model.nv, model.nu, model.neq) == (7, 7, 7, 0)
    for joint in ROS_ARM_JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint) >= 0
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fer_finger_joint1") == -1
    )


def test_plant_is_the_fitting_projection_and_nothing_else(
    plant: Path, model_paths: ModelPaths
) -> None:
    """Renamed and re-homed, but physically the same seven-axis model."""
    wrapped = mujoco.MjModel.from_xml_path(str(plant))
    source = fitting_spec(model_paths.hydrax).compile()

    assert wrapped.njnt == source.njnt
    assert wrapped.nu == source.nu
    assert wrapped.nbody == source.nbody
    assert wrapped.opt.timestep == source.opt.timestep
    assert wrapped.opt.integrator == source.opt.integrator
    assert wrapped.opt.disableflags == source.opt.disableflags
    np.testing.assert_array_equal(wrapped.opt.gravity, source.opt.gravity)
    # The scene writer restores full binary64 precision after MuJoCo's compact
    # XML serializer, so every compiled arm-dynamics value survives exactly.
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "jnt_range",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(wrapped, field), getattr(source, field))

    # Deleting the finger coordinates must not delete their physical bodies:
    # their inertia is rigidly reflected through the hand into joint 7.
    for body_name in ("hand", "left_finger", "right_finger"):
        wrapped_body = wrapped.body(body_name)
        source_body = source.body(body_name)
        assert wrapped_body.dofnum[0] == 0
        np.testing.assert_array_equal(wrapped_body.mass, source_body.mass)
        np.testing.assert_array_equal(wrapped_body.inertia, source_body.inertia)


def test_one_step_arm_dynamics_match_fitting_exactly(
    plant: Path, model_paths: ModelPaths
) -> None:
    """A ROS-system-ID step is the fitting model step, modulo ROS names."""
    wrapped = mujoco.MjModel.from_xml_path(str(plant))
    source = fitting_spec(model_paths.hydrax).compile()
    wrapped_data = mujoco.MjData(wrapped)
    source_data = mujoco.MjData(source)

    qpos = np.asarray(_HOME_QPOS, dtype=np.float64) + np.array(
        [0.08, -0.05, 0.04, -0.07, 0.06, -0.03, 0.05]
    )
    qvel = np.array([0.31, -0.27, 0.19, -0.23, 0.17, -0.13, 0.11])
    control = np.array([4.0, -3.0, 2.0, -2.5, 1.5, -1.0, 0.7])
    for data in (wrapped_data, source_data):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = control
        mujoco.mj_forward(wrapped if data is wrapped_data else source, data)

    # These are the compiled quantities that determine the next arm state.
    np.testing.assert_array_equal(wrapped_data.qM, source_data.qM)
    np.testing.assert_array_equal(wrapped_data.qfrc_bias, source_data.qfrc_bias)
    np.testing.assert_array_equal(wrapped_data.qacc, source_data.qacc)

    mujoco.mj_step(wrapped, wrapped_data)
    mujoco.mj_step(source, source_data)
    np.testing.assert_array_equal(wrapped_data.qpos, source_data.qpos)
    np.testing.assert_array_equal(wrapped_data.qvel, source_data.qvel)


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
    np.testing.assert_allclose(model.key_qpos[0], _HOME_QPOS, atol=1e-6)


def test_plant_adds_the_table_below_the_base(plant: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(plant))
    table = model.geom("table_surface")
    assert table.type == mujoco.mjtGeom.mjGEOM_PLANE
    assert table.pos[2] == pytest.approx(TABLE_Z_M)
    assert table.contype[0] == 0
    assert table.conaffinity[0] == 0


def test_hidden_dynamic_parameters_apply_before_ros_renaming(
    tmp_path: Path, model_paths: ModelPaths
) -> None:
    """The simulation truth API carries body and joint dynamics, not friction only."""
    nominal = fitting_spec(model_paths.hydrax).compile()
    nominal_link4 = BodyInertial.from_model(nominal, "link4")
    changed_link4 = BodyInertial(
        body="link4",
        mass=nominal_link4.mass * 1.01,
        ipos=nominal_link4.ipos,
        inertia=tuple(value * 1.01 for value in nominal_link4.inertia),
    )
    parameters = IdentifiedParameters(
        frictionloss=(0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17),
        damping=(0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97),
        armature=(0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087),
        body_inertials=(changed_link4,),
    )

    written = write_scene(tmp_path / "perturbed.xml", parameters=parameters)
    model = mujoco.MjModel.from_xml_path(str(written))
    for index, joint_name in enumerate(ROS_ARM_JOINT_NAMES):
        dof = int(model.joint(joint_name).dofadr[0])
        assert model.dof_frictionloss[dof] == pytest.approx(
            parameters.frictionloss[index]
        )
        assert model.dof_damping[dof] == pytest.approx(parameters.damping[index])
        assert model.dof_armature[dof] == pytest.approx(parameters.armature[index])
    written_link4 = BodyInertial.from_model(model, "link4")
    assert written_link4.mass == changed_link4.mass
    np.testing.assert_allclose(written_link4.inertia, changed_link4.inertia, rtol=1e-5)
