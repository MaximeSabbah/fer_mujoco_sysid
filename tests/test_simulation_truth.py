from __future__ import annotations

from pathlib import Path

import numpy as np

from fer_mujoco_sysid.export import IdentifiedParameters
from fer_mujoco_sysid.io import sha256_file
from fer_mujoco_sysid.model import ModelPaths
from fer_mujoco_sysid.simulation_truth import (
    SIMULATION_TRUTH_FORMAT,
    load_truth_manifest,
    parameters_from_truth_manifest,
    truth_manifest,
    truth_model,
    truth_model_from_manifest,
    truth_parameters,
    write_truth_manifest,
)


def test_truth_manifest_is_deterministic_and_bound_to_source(
    model_paths: ModelPaths,
) -> None:
    first = truth_manifest(model_paths.hydrax)
    second = truth_manifest(model_paths.hydrax)
    assert first == second
    assert first["format"] == SIMULATION_TRUTH_FORMAT
    assert first["source_model"]["sha256"] == sha256_file(model_paths.hydrax)
    assert len(first["content_sha256"]) == 64


def test_truth_manifest_round_trips_parameters(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    path = write_truth_manifest(tmp_path / "simulation_truth.json", model_paths.hydrax)
    loaded = parameters_from_truth_manifest(path)
    assert loaded.as_dict() == truth_parameters(model_paths.hydrax).as_dict()
    assert load_truth_manifest(path, source_path=model_paths.hydrax)[
        "content_sha256"
    ] == truth_manifest(model_paths.hydrax)["content_sha256"]
    np.testing.assert_allclose(
        truth_model_from_manifest(path, model_paths.hydrax).dof_armature,
        truth_model(model_paths.hydrax).dof_armature,
        rtol=0.0,
        atol=0.0,
    )


def test_truth_model_is_the_exact_parameter_projection(
    model_paths: ModelPaths,
) -> None:
    model = truth_model(model_paths.hydrax)
    expected = IdentifiedParameters.from_model(
        model,
        bodies=("link4", "link5", "link6", "link7"),
    )
    actual = truth_parameters(model_paths.hydrax)
    np.testing.assert_allclose(expected.frictionloss, actual.frictionloss, atol=0.0)
    np.testing.assert_allclose(expected.damping, actual.damping, atol=0.0)
    np.testing.assert_allclose(expected.armature, actual.armature, atol=0.0)
    for compiled, requested in zip(
        expected.body_inertials,
        actual.body_inertials,
        strict=True,
    ):
        assert compiled.body == requested.body
        np.testing.assert_allclose(compiled.mass, requested.mass, rtol=1e-12)
        np.testing.assert_allclose(compiled.ipos, requested.ipos, atol=1e-12)
        # MuJoCo diagonalizes the requested full tensor internally. Rebuilding
        # the tensor from its principal axes is the measured representation
        # floor, not a physical-model mismatch.
        np.testing.assert_allclose(
            compiled.inertia,
            requested.inertia,
            rtol=5e-7,
            atol=3e-9,
        )
    np.testing.assert_allclose(model.opt.gravity, 0.0, atol=0.0)
