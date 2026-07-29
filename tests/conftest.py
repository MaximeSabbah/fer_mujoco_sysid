from __future__ import annotations

import mujoco
import pytest

from fer_mujoco_sysid.model import (
    ModelPaths,
    load_ros_overlay_model,
    resolve_model_paths,
)


@pytest.fixture(scope="session")
def model_paths() -> ModelPaths:
    return resolve_model_paths().require()


@pytest.fixture(scope="session")
def hydrax_model(model_paths: ModelPaths) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(model_paths.hydrax))


@pytest.fixture(scope="session")
def ros_model(model_paths: ModelPaths) -> mujoco.MjModel:
    """A consumer's deployment MJCF, when one is checked out.

    Optional on purpose: identification and playback never need a consumer
    repository, so its absence skips the compatibility checks rather than
    failing the suite.
    """
    if not model_paths.has_ros_overlay:
        pytest.skip(f"no deployment-target model at {model_paths.ros_overlay}")
    return load_ros_overlay_model(
        model_paths.ros_overlay,
        mesh_directory=model_paths.hydrax.parent / "assets",
    )
