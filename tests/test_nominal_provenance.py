from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from fer_mujoco_sysid.model import ModelPaths

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "contracts" / "nominal_sources.toml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_nominal_model_files_match_reviewed_sources(model_paths: ModelPaths) -> None:
    with CONTRACT_PATH.open("rb") as stream:
        contract = tomllib.load(stream)

    assert contract["contract_version"] == 1
    sources = contract["models"]
    assert _sha256(model_paths.hydrax) == sources["hydrax"]["sha256"]
    # The deployment target is optional: it belongs to a consumer repository
    # that this project must work without. Check it only when it is there.
    if model_paths.has_ros_overlay:
        assert _sha256(model_paths.ros_overlay) == sources["sbmpc_ros"]["sha256"]


def test_the_project_works_without_a_consumer_checkout(
    tmp_path: Path, monkeypatch, model_paths: ModelPaths
) -> None:
    """Identification and playback must not need a consumer repository.

    Pointed at a deployment-target path that does not exist, the contract
    still resolves, the nominal model still builds, and the ROS-simulation
    plant is still generated — because all of it comes from the nominal
    model this project owns.
    """
    import mujoco

    from fer_mujoco_sysid.model import ROS_MODEL_ENV, resolve_model_paths
    from fer_mujoco_sysid.ros.scene import write_scene

    monkeypatch.setenv(ROS_MODEL_ENV, str(tmp_path / "absent" / "consumer.xml"))
    paths = resolve_model_paths().require()
    assert not paths.has_ros_overlay

    mujoco.MjModel.from_xml_path(str(paths.hydrax))
    plant = write_scene(tmp_path / "plant.xml", model_path=paths.hydrax)
    model = mujoco.MjModel.from_xml_path(str(plant))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fer_joint1") >= 0
