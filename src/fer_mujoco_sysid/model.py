"""Compatibility contract for the Hydrax and ROS FER MuJoCo models."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

# Re-exported: the ROS-side joint order is defined with the bundle format,
# which must stay importable without MuJoCo.
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

HYDRAX_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
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


#: The vendored nominal model. Identification measures deviations *from* this
#: file, so it cannot live in a repository that also receives the identified
#: parameters: writing a fit into hydrax's panda.xml previously moved the very
#: baseline the fit is defined against, and every recorded campaign — which
#: binds the baseline's sha256 — stopped being loadable. It is a copy of
#: hydrax at the revision pinned in contracts/nominal_sources.toml, and it
#: changes only by explicit review.
VENDORED_NOMINAL_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "panda" / "panda.xml"
)


@dataclass(frozen=True)
class ModelPaths:
    """Locations of the nominal model and the optional deployment target.

    ``hydrax`` is the nominal model this project identifies, generates
    protocols against, and simulates; it defaults to the vendored copy so
    identification never depends on a sibling checkout. ``ros_overlay`` is a
    *deployment target* — a consumer's MJCF that the identified parameters
    will eventually be rendered into — and everything here works without it.
    Nothing in the identification or playback path may depend on a consumer
    repository being checked out.
    """

    hydrax: Path
    ros_overlay: Path

    def require(self) -> ModelPaths:
        """Return this contract after checking the nominal model exists."""
        if not self.hydrax.is_file():
            raise FileNotFoundError(
                f"FER nominal model not found: {self.hydrax}\n"
                f"Set {HYDRAX_MODEL_ENV}, or set {WORKSPACE_ENV} to the "
                "workspace holding the hydrax checkout."
            )
        return self

    @property
    def has_ros_overlay(self) -> bool:
        """Whether the optional deployment-target MJCF is available."""
        return self.ros_overlay.is_file()

    def require_ros_overlay(self) -> Path:
        """The deployment-target MJCF, for the checks that compare against it."""
        if not self.has_ros_overlay:
            raise FileNotFoundError(
                f"deployment-target model not found: {self.ros_overlay}\n"
                f"It is optional — identification and playback do not need "
                f"it. Set {ROS_MODEL_ENV} to compare against a consumer model."
            )
        return self.ros_overlay


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_model_paths(workspace_root: str | Path | None = None) -> ModelPaths:
    """Resolve source models without importing or modifying their repositories."""
    if workspace_root is None:
        workspace_root = os.environ.get(WORKSPACE_ENV)
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[3]
    workspace = _resolved_path(workspace_root)

    # The vendored copy is the default. A sibling hydrax checkout is a
    # deployment target, not the baseline: overriding this to point back at it
    # re-couples the fit to a file the fit's own output gets written into.
    hydrax = _resolved_path(
        os.environ.get(HYDRAX_MODEL_ENV, VENDORED_NOMINAL_MODEL)
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


def build_hydrax_arm_spec(
    model_path: str | Path,
    *,
    timestep: float | None = None,
    joint_state_sensors: bool = False,
) -> mujoco.MjSpec:
    """Configure the contact-free seven-axis spec used for arm identification.

    Same projection as :func:`build_hydrax_arm_model`, returned as an editable
    spec because the ``mujoco.sysid`` pipeline applies parameter modifiers to a
    spec before every compile. With ``joint_state_sensors`` the spec also gets
    one ``jointpos`` and one ``jointvel`` sensor per arm joint (named
    ``{joint}_pos``/``{joint}_vel``, all positions first), which defines the
    measured-signal layout of identification datasets.
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

    if joint_state_sensors:
        for sensor_type, suffix in (
            (mujoco.mjtSensor.mjSENS_JOINTPOS, "pos"),
            (mujoco.mjtSensor.mjSENS_JOINTVEL, "vel"),
        ):
            for joint_name in HYDRAX_ARM_JOINT_NAMES:
                sensor = spec.add_sensor()
                sensor.name = f"{joint_name}_{suffix}"
                sensor.type = sensor_type
                sensor.objtype = mujoco.mjtObj.mjOBJ_JOINT
                sensor.objname = joint_name
    return spec


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
    return build_hydrax_arm_spec(model_path, timestep=timestep).compile()


def load_ros_overlay_model(
    model_path: str | Path,
    *,
    mesh_directory: str | Path,
) -> mujoco.MjModel:
    """Compile an optional consumer MJCF using an explicit mesh directory.

    Only used by the compatibility checks that compare this project's nominal
    model against a deployment target, and only when such a checkout exists
    (see :meth:`ModelPaths.has_ros_overlay`). A consumer's MJCF may point its
    `meshdir` anywhere, including at a path that no longer exists; the nominal
    model carries the same meshes, so replacing `meshdir` in memory lets the
    comparison run without depending on or modifying that checkout.
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
