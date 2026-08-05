from __future__ import annotations

import mujoco
import pytest

from fer_mujoco_sysid.model import (
    ModelPaths,
    resolve_model_paths,
)


@pytest.fixture(scope="session")
def model_paths() -> ModelPaths:
    return resolve_model_paths().require()


@pytest.fixture(scope="session")
def hydrax_model(model_paths: ModelPaths) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(model_paths.hydrax))


