from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from fer_mujoco_sysid.model_contract import ModelPaths

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
    assert _sha256(model_paths.ros_overlay) == sources["sbmpc_ros"]["sha256"]
