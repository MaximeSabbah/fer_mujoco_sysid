"""Compatibility contract for the Hydrax and ROS FER MuJoCo models."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

HYDRAX_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ROS_ARM_JOINT_NAMES = tuple(f"fer_joint{i}" for i in range(1, 8))
ARM_JOINT_PAIRS = tuple(zip(HYDRAX_ARM_JOINT_NAMES, ROS_ARM_JOINT_NAMES, strict=True))

HYDRAX_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ROS_FINGER_JOINT_NAMES = ("fer_finger_joint1", "fer_finger_joint2")

PHYSICAL_BODY_NAMES = (
    *(f"link{i}" for i in range(8)),
    "hand",
    "left_finger",
    "right_finger",
)

HYDRAX_MODEL_ENV = "FER_SYSID_HYDRAX_MODEL"
ROS_MODEL_ENV = "FER_SYSID_SBMPC_ROS_MODEL"
WORKSPACE_ENV = "FER_SYSID_WORKSPACE_ROOT"


@dataclass(frozen=True)
class ModelPaths:
    """Locations of the two read-only source MJCFs."""

    hydrax: Path
    ros_overlay: Path

    def require(self) -> ModelPaths:
        """Return this contract after checking that both source files exist."""
        missing = [
            f"{label}: {path}"
            for label, path in (
                ("Hydrax model", self.hydrax),
                ("sbmpc_ros model", self.ros_overlay),
            )
            if not path.is_file()
        ]
        if missing:
            details = "\n".join(f"- {item}" for item in missing)
            raise FileNotFoundError(
                "FER source model(s) not found:\n"
                f"{details}\n"
                f"Set {HYDRAX_MODEL_ENV} and {ROS_MODEL_ENV}, or set "
                f"{WORKSPACE_ENV} to their common workspace."
            )
        return self


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_model_paths(workspace_root: str | Path | None = None) -> ModelPaths:
    """Resolve source models without importing or modifying their repositories."""
    if workspace_root is None:
        workspace_root = os.environ.get(WORKSPACE_ENV)
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[3]
    workspace = _resolved_path(workspace_root)

    hydrax = _resolved_path(
        os.environ.get(
            HYDRAX_MODEL_ENV,
            workspace / "hydrax" / "hydrax" / "models" / "panda" / "panda.xml",
        )
    )
    ros_overlay = _resolved_path(
        os.environ.get(
            ROS_MODEL_ENV,
            workspace
            / "sbmpc_ros"
            / "sbmpc_bringup"
            / "mujoco"
            / "fer_ros2_control.xml",
        )
    )
    return ModelPaths(hydrax=hydrax, ros_overlay=ros_overlay)


def build_hydrax_arm_model(
    model_path: str | Path,
    *,
    timestep: float | None = None,
) -> mujoco.MjModel:
    """Build the contact-free seven-axis model used for arm identification.

    The projection mirrors Hydrax's planning-model derivation: finger joints
    and their actuator/coupling are removed, but the hand and finger bodies
    remain so their rigidly attached inertia is preserved.
    """
    if timestep is not None and timestep <= 0.0:
        raise ValueError("timestep must be positive")

    spec = mujoco.MjSpec.from_file(str(_resolved_path(model_path)))
    for joint in list(spec.joints):
        if joint.name in HYDRAX_FINGER_JOINT_NAMES:
            spec.delete(joint)
    for actuator in list(spec.actuators):
        if actuator.name == "actuator8":
            spec.delete(actuator)
    for equality in list(spec.equalities):
        spec.delete(equality)

    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if timestep is not None:
        spec.option.timestep = timestep
    return spec.compile()


def load_ros_overlay_model(
    model_path: str | Path,
    *,
    mesh_directory: str | Path,
) -> mujoco.MjModel:
    """Compile the ROS wrapper using an explicit, non-legacy mesh directory.

    The checked-in ROS MJCF currently points at a deprecated absolute `sbmpc`
    asset path. Hydrax carries the same meshes. Replacing only `meshdir` in
    memory lets this standalone project verify the actual ROS robot XML without
    depending on or modifying that deprecated checkout.
    """
    model_path = _resolved_path(model_path)
    mesh_directory = _resolved_path(mesh_directory)
    if not mesh_directory.is_dir():
        raise FileNotFoundError(f"mesh directory not found: {mesh_directory}")

    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"ROS MJCF has no <compiler>: {model_path}")
    compiler.set("meshdir", str(mesh_directory))
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)
