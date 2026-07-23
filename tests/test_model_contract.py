from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.model_contract import (
    ARM_JOINT_PAIRS,
    HYDRAX_FINGER_JOINT_NAMES,
    PHYSICAL_BODY_NAMES,
    ROS_FINGER_JOINT_NAMES,
    ModelPaths,
    build_hydrax_arm_model,
)


def _parent_name(model: mujoco.MjModel, body_id: int) -> str:
    return model.body(int(model.body_parentid[body_id])).name


@pytest.mark.parametrize("body_name", PHYSICAL_BODY_NAMES)
def test_physical_body_models_match(
    hydrax_model: mujoco.MjModel,
    ros_model: mujoco.MjModel,
    body_name: str,
) -> None:
    hydrax_id = hydrax_model.body(body_name).id
    ros_id = ros_model.body(body_name).id

    assert _parent_name(hydrax_model, hydrax_id) == _parent_name(ros_model, ros_id)
    for field in (
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "body_gravcomp",
    ):
        np.testing.assert_allclose(
            getattr(hydrax_model, field)[hydrax_id],
            getattr(ros_model, field)[ros_id],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{body_name}: {field}",
        )


@pytest.mark.parametrize(("hydrax_name", "ros_name"), ARM_JOINT_PAIRS)
def test_arm_joint_models_match(
    hydrax_model: mujoco.MjModel,
    ros_model: mujoco.MjModel,
    hydrax_name: str,
    ros_name: str,
) -> None:
    hydrax_joint = hydrax_model.joint(hydrax_name).id
    ros_joint = ros_model.joint(ros_name).id
    hydrax_dof = int(hydrax_model.jnt_dofadr[hydrax_joint])
    ros_dof = int(ros_model.jnt_dofadr[ros_joint])

    assert hydrax_model.jnt_type[hydrax_joint] == ros_model.jnt_type[ros_joint]
    for field in ("jnt_pos", "jnt_axis", "jnt_range"):
        np.testing.assert_allclose(
            getattr(hydrax_model, field)[hydrax_joint],
            getattr(ros_model, field)[ros_joint],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{hydrax_name}/{ros_name}: {field}",
        )
    for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_allclose(
            getattr(hydrax_model, field)[hydrax_dof],
            getattr(ros_model, field)[ros_dof],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{hydrax_name}/{ros_name}: {field}",
        )


@pytest.mark.parametrize(
    ("index", "hydrax_joint", "ros_joint"),
    (
        (index, hydrax_joint, ros_joint)
        for index, (hydrax_joint, ros_joint) in enumerate(ARM_JOINT_PAIRS)
    ),
)
def test_arm_actuator_limits_and_transmissions_match(
    hydrax_model: mujoco.MjModel,
    ros_model: mujoco.MjModel,
    index: int,
    hydrax_joint: str,
    ros_joint: str,
) -> None:
    hydrax_actuator = hydrax_model.actuator(f"motor{index + 1}").id
    ros_actuator = ros_model.actuator(ros_joint).id

    assert (
        hydrax_model.actuator_trnid[hydrax_actuator, 0]
        == hydrax_model.joint(hydrax_joint).id
    )
    assert ros_model.actuator_trnid[ros_actuator, 0] == ros_model.joint(ros_joint).id
    for field in (
        "actuator_ctrlrange",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
    ):
        np.testing.assert_allclose(
            getattr(hydrax_model, field)[hydrax_actuator],
            getattr(ros_model, field)[ros_actuator],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{hydrax_joint}/{ros_joint}: {field}",
        )
    # The wrappers deliberately enforce the same torque boundary differently:
    # Hydrax clips the direct control, while ros2_control also force-limits the
    # effort actuator.
    assert not hydrax_model.actuator_forcelimited[hydrax_actuator]
    assert ros_model.actuator_forcelimited[ros_actuator]
    np.testing.assert_allclose(
        ros_model.actuator_forcerange[ros_actuator],
        hydrax_model.actuator_ctrlrange[hydrax_actuator],
    )


def test_gripper_frame_is_identical(
    hydrax_model: mujoco.MjModel,
    ros_model: mujoco.MjModel,
) -> None:
    hydrax_site = hydrax_model.site("gripper").id
    ros_site = ros_model.site("gripper").id
    assert (
        hydrax_model.body(int(hydrax_model.site_bodyid[hydrax_site])).name
        == ros_model.body(int(ros_model.site_bodyid[ros_site])).name
        == "hand"
    )
    np.testing.assert_allclose(
        hydrax_model.site_pos[hydrax_site],
        ros_model.site_pos[ros_site],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hydrax_model.site_quat[hydrax_site],
        ros_model.site_quat[ros_site],
        rtol=1e-12,
        atol=1e-12,
    )


def _joint_addresses(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray([model.joint(name).id for name in joint_names])
    return model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]


def _arm_dynamics(
    model: mujoco.MjModel,
    arm_joint_names: tuple[str, ...],
    finger_joint_names: tuple[str, ...],
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    *,
    gripper_control: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    arm_qadr, arm_dadr = _joint_addresses(model, arm_joint_names)
    finger_qadr, finger_dadr = _joint_addresses(model, finger_joint_names)
    data.qpos[arm_qadr] = q
    data.qvel[arm_dadr] = v
    data.qpos[finger_qadr] = 0.04
    data.qvel[finger_dadr] = 0.0
    data.ctrl[:7] = tau
    data.ctrl[7] = gripper_control
    mujoco.mj_forward(model, data)
    return data.qfrc_bias[arm_dadr].copy(), data.qacc[arm_dadr].copy()


@pytest.mark.parametrize(
    ("q", "v", "tau"),
    (
        (
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            np.zeros(7),
            np.zeros(7),
        ),
        (
            np.array([0.2, -0.6, -0.3, -2.1, 0.4, 1.2, -0.2]),
            np.array([0.1, -0.2, 0.05, 0.08, -0.1, 0.03, 0.12]),
            np.array([2.0, -3.0, 1.0, 5.0, -0.5, 0.8, -0.2]),
        ),
    ),
)
def test_equivalent_arm_inputs_produce_matching_dynamics(
    hydrax_model: mujoco.MjModel,
    ros_model: mujoco.MjModel,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
) -> None:
    hydrax_bias, hydrax_qacc = _arm_dynamics(
        hydrax_model,
        tuple(pair[0] for pair in ARM_JOINT_PAIRS),
        HYDRAX_FINGER_JOINT_NAMES,
        q,
        v,
        tau,
        gripper_control=0.04,
    )
    ros_bias, ros_qacc = _arm_dynamics(
        ros_model,
        tuple(pair[1] for pair in ARM_JOINT_PAIRS),
        ROS_FINGER_JOINT_NAMES,
        q,
        v,
        tau,
        gripper_control=0.0,
    )
    np.testing.assert_allclose(hydrax_bias, ros_bias, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(hydrax_qacc, ros_qacc, rtol=1e-10, atol=1e-10)


def test_identification_projection_matches_hydrax_runtime_contract(
    model_paths: ModelPaths,
    hydrax_model: mujoco.MjModel,
) -> None:
    identification_model = build_hydrax_arm_model(
        model_paths.hydrax,
        timestep=0.001,
    )

    assert identification_model.nq == 7
    assert identification_model.nv == 7
    assert identification_model.nu == 7
    assert identification_model.opt.timestep == pytest.approx(0.001)
    assert (
        identification_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    )
    assert identification_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    assert tuple(identification_model.joint(i).name for i in range(7)) == tuple(
        pair[0] for pair in ARM_JOINT_PAIRS
    )
    assert identification_model.body("hand").subtreemass == pytest.approx(
        hydrax_model.body("hand").subtreemass
    )
    np.testing.assert_allclose(
        identification_model.site("gripper").pos,
        hydrax_model.site("gripper").pos,
    )
