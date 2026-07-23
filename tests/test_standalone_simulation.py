from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

import fer_mujoco_sysid.simulation.standalone as standalone_module
from fer_mujoco_sysid.artifacts import (
    ArtifactValidationError,
    content_sha256,
    sha256_file,
    write_json,
)
from fer_mujoco_sysid.model_contract import ModelPaths, build_hydrax_arm_model
from fer_mujoco_sysid.simulation import run_open_loop_effort_protocol

_PERIOD_NS = 2_000_000
_START_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _file_reference(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _array_reference(
    file_reference: dict[str, Any],
    key: str,
    array: np.ndarray,
    unit: str,
) -> dict[str, Any]:
    return {
        "file": dict(file_reference),
        "key": key,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "unit": unit,
    }


def _refresh_bundle(
    root: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    np.savez(root / "desired.npz", **arrays)
    archive = _file_reference(root, root / "desired.npz")
    for reference in manifest["arrays"].values():
        reference["file"] = dict(archive)
        reference["shape"] = list(arrays[reference["key"]].shape)
        reference["dtype"] = arrays[reference["key"]].dtype.str
    manifest["content_sha256"] = content_sha256(manifest, arrays)


def _protocol_bundle(
    root: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root.mkdir()
    sample_count = 12
    time_ns = np.arange(sample_count, dtype="<i8") * _PERIOD_NS
    q = np.repeat(_START_Q[None, :], sample_count, axis=0).astype("<f8")
    dq = np.zeros((sample_count, 7), dtype="<f8")
    ddq = np.zeros((sample_count, 7), dtype="<f8")
    tau = np.zeros((sample_count, 7), dtype="<f8")
    tau[:, 0] = np.arange(1, sample_count + 1)
    tau[:, 3] = -0.5
    arrays = {
        "time_from_start_ns": time_ns,
        "q_rad": q,
        "dq_rad_s": dq,
        "ddq_rad_s2": ddq,
        "tau_feedforward_Nm": tau,
    }

    write_json(root / "generator.json", {"kind": "deterministic-test"})
    write_json(root / "constraints.json", {"kind": "test-limits"})
    generator = _file_reference(root, root / "generator.json")
    constraints = _file_reference(root, root / "constraints.json")
    placeholder = {"path": "desired.npz", "sha256": "0" * 64, "size_bytes": 0}
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/motion-protocol@1",
        "artifact_id": "standalone-effort-fixture",
        "content_sha256": "0" * 64,
        "title": "Standalone effort fixture",
        "description": "Short deterministic effort sequence.",
        "family_id": "standalone-fixture",
        "created_at": "2026-07-23T00:00:00Z",
        "generator": {
            "software": {"name": "pytest", "version": "1.0"},
            "seed": 7,
            "configuration": generator,
        },
        "joint_order": [f"fer_joint{index}" for index in range(1, 8)],
        "command_interface": "joint_effort",
        "sample_period_ns": _PERIOD_NS,
        "arrays": {
            "time_from_start": _array_reference(
                placeholder,
                "time_from_start_ns",
                time_ns,
                "ns",
            ),
            "desired_position": _array_reference(
                placeholder,
                "q_rad",
                q,
                "rad",
            ),
            "desired_velocity": _array_reference(
                placeholder,
                "dq_rad_s",
                dq,
                "rad/s",
            ),
            "desired_acceleration": _array_reference(
                placeholder,
                "ddq_rad_s2",
                ddq,
                "rad/s^2",
            ),
            "desired_effort_feedforward": _array_reference(
                placeholder,
                "tau_feedforward_Nm",
                tau,
                "N*m",
            ),
        },
        "segments": [
            {
                "segment_id": "excitation",
                "kind": "excitation",
                "start_index": 0,
                "end_index_exclusive": sample_count,
                "analysis_eligible": True,
            }
        ],
        "start_state": {
            "position_rad": q[0].tolist(),
            "velocity_rad_s": dq[0].tolist(),
        },
        "end_state": {
            "position_rad": q[-1].tolist(),
            "velocity_rad_s": dq[-1].tolist(),
        },
        "context": {
            "source_model": {
                "repository": "https://example.invalid/hydrax.git",
                "revision": "a" * 40,
                "path": "hydrax/models/panda/panda.xml",
                "sha256": sha256_file(model_path),
            },
            "end_effector_id": "fer-hand",
            "payload": {
                "payload_id": "fer-hand",
                "mass_kg": 0.73,
                "center_of_mass_m": [0.0, 0.0, 0.03],
            },
        },
        "constraint_profile": constraints,
    }
    _refresh_bundle(root, manifest, arrays)
    return manifest, arrays


def _run(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    root: Path,
    model_path: Path,
    *,
    command_semantics: str = "zero_order_hold",
):
    return run_open_loop_effort_protocol(
        manifest,
        arrays,
        protocol_root=root,
        model_source_path=model_path,
        command_semantics=command_semantics,
    )


def test_repository_projection_rollout_is_deterministic_and_aligned(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)

    first = _run(manifest, arrays, root, model_paths.hydrax)
    second = _run(manifest, arrays, root, model_paths.hydrax)

    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(first, field),
            getattr(second, field),
        )
    expected_time = np.arange(arrays["time_from_start_ns"].shape[0], dtype="<f8") * (
        _PERIOD_NS * 1e-9
    )
    np.testing.assert_array_equal(first.state_time_s, expected_time)
    np.testing.assert_array_equal(
        first.control_time_s,
        first.state_time_s[:-1],
    )
    np.testing.assert_array_equal(
        first.desired_effort_feedforward_Nm,
        arrays["tau_feedforward_Nm"][:-1],
    )
    np.testing.assert_array_equal(
        first.simulated_actuator_effort_Nm,
        arrays["tau_feedforward_Nm"][:-1],
    )
    np.testing.assert_array_equal(first.q_rad[0], _START_Q)
    np.testing.assert_array_equal(first.dq_rad_s[0], np.zeros(7))


def test_rollout_is_copy_owned_read_only_and_does_not_mutate_inputs(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    manifest_before = deepcopy(manifest)
    arrays_before = {key: value.copy() for key, value in arrays.items()}
    source_digest = sha256_file(model_paths.hydrax)

    result = _run(manifest, arrays, root, model_paths.hydrax)

    assert manifest == manifest_before
    for key, expected in arrays_before.items():
        np.testing.assert_array_equal(arrays[key], expected)
    assert sha256_file(model_paths.hydrax) == source_digest

    for field in result.__dataclass_fields__:
        value = getattr(result, field)
        assert value.flags.c_contiguous
        assert not value.flags.writeable
        for source in arrays.values():
            assert not np.shares_memory(value, source)
    with pytest.raises(ValueError, match="read-only"):
        result.q_rad[0, 0] = 1.0


def test_compiles_identification_projection_at_protocol_timestep(
    tmp_path: Path,
    model_paths: ModelPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    period_ns = 1_000_000
    arrays["time_from_start_ns"] = (
        np.arange(arrays["time_from_start_ns"].shape[0], dtype="<i8") * period_ns
    )
    manifest["sample_period_ns"] = period_ns
    _refresh_bundle(root, manifest, arrays)
    observed_timestep: list[float | None] = []
    original_builder = standalone_module.build_hydrax_arm_model

    def recording_builder(
        path: str | Path,
        *,
        timestep: float | None = None,
    ) -> mujoco.MjModel:
        observed_timestep.append(timestep)
        return original_builder(path, timestep=timestep)

    monkeypatch.setattr(
        standalone_module,
        "build_hydrax_arm_model",
        recording_builder,
    )

    result = _run(manifest, arrays, root, model_paths.hydrax)

    assert observed_timestep == [period_ns * 1e-9]
    np.testing.assert_array_equal(
        result.state_time_s,
        np.arange(arrays["time_from_start_ns"].shape[0], dtype="<f8")
        * (period_ns * 1e-9),
    )


def test_rejects_missing_effort_feedforward(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    del manifest["arrays"]["desired_effort_feedforward"]
    del arrays["tau_feedforward_Nm"]
    _refresh_bundle(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match="require.*effort_feedforward"):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_rejects_unsupported_interface_and_command_semantics(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    first = tmp_path / "interface"
    manifest, arrays = _protocol_bundle(first, model_paths.hydrax)
    manifest["command_interface"] = "joint_trajectory"
    manifest["content_sha256"] = content_sha256(manifest, arrays)
    with pytest.raises(ArtifactValidationError, match="only joint_effort"):
        _run(manifest, arrays, first, model_paths.hydrax)

    second = tmp_path / "semantics"
    manifest, arrays = _protocol_bundle(second, model_paths.hydrax)
    with pytest.raises(ArtifactValidationError, match="zero_order_hold"):
        _run(
            manifest,
            arrays,
            second,
            model_paths.hydrax,
            command_semantics="linear",
        )


def test_rejects_command_shape_mismatch(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    arrays["tau_feedforward_Nm"] = arrays["tau_feedforward_Nm"][:-1].copy()
    _refresh_bundle(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match=r"shape \(N, 7\)"):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_rejects_source_hash_mismatch(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    manifest["context"]["source_model"]["sha256"] = "f" * 64
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    with pytest.raises(ArtifactValidationError, match="source-model SHA-256 mismatch"):
        _run(manifest, arrays, root, model_paths.hydrax)


@pytest.mark.parametrize(
    "fault", ["full_model", "contact", "integrator", "mapping", "gear"]
)
def test_rejects_model_contract_inconsistencies(
    tmp_path: Path,
    model_paths: ModelPaths,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    if fault == "full_model":
        model = mujoco.MjModel.from_xml_path(str(model_paths.hydrax))
        match = "exactly seven"
    else:
        model = build_hydrax_arm_model(model_paths.hydrax)
        if fault == "contact":
            model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            match = "contacts must be disabled"
        elif fault == "integrator":
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            match = "integrator"
        elif fault == "mapping":
            model.actuator_trnid[0, 0] = model.joint("joint2").id
            match = "exactly one direct joint actuator"
        else:
            model.actuator_gear[0, 0] = 2.0
            match = "unit scalar gear"

    monkeypatch.setattr(
        standalone_module,
        "_compile_source_model_at_timestep",
        lambda source_path, *, timestep_s: model,
    )
    with pytest.raises(ArtifactValidationError, match=match):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_rejects_initial_state_outside_model_limits(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    arrays["q_rad"][:, 0] = 4.0
    manifest["start_state"]["position_rad"][0] = 4.0
    manifest["end_state"]["position_rad"][0] = 4.0
    _refresh_bundle(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match="outside model limits"):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_rejects_nonfinite_rollout(
    tmp_path: Path,
    model_paths: ModelPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    original_step = standalone_module.mujoco.mj_step

    def nonfinite_step(step_model: mujoco.MjModel, data: mujoco.MjData) -> None:
        original_step(step_model, data)
        data.qvel[0] = np.nan

    monkeypatch.setattr(standalone_module.mujoco, "mj_step", nonfinite_step)

    with pytest.raises(ArtifactValidationError, match="non-finite"):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_long_rollout_returns_exact_canonical_grid(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    sample_count = 12_001
    model = build_hydrax_arm_model(
        model_paths.hydrax,
        timestep=_PERIOD_NS * 1e-9,
    )
    data = mujoco.MjData(model)
    data.qpos[:] = _START_Q
    mujoco.mj_forward(model, data)
    holding_effort = data.qfrc_bias.copy()
    arrays = {
        "time_from_start_ns": (np.arange(sample_count, dtype="<i8") * _PERIOD_NS),
        "q_rad": np.repeat(_START_Q[None, :], sample_count, axis=0).astype("<f8"),
        "dq_rad_s": np.zeros((sample_count, 7), dtype="<f8"),
        "ddq_rad_s2": np.zeros((sample_count, 7), dtype="<f8"),
        "tau_feedforward_Nm": np.repeat(
            holding_effort[None, :],
            sample_count,
            axis=0,
        ).astype("<f8"),
    }
    manifest["end_state"] = {
        "position_rad": arrays["q_rad"][-1].tolist(),
        "velocity_rad_s": arrays["dq_rad_s"][-1].tolist(),
    }
    manifest["segments"][0]["end_index_exclusive"] = sample_count
    _refresh_bundle(root, manifest, arrays)

    result = _run(manifest, arrays, root, model_paths.hydrax)
    expected = np.arange(sample_count, dtype="<f8") * (_PERIOD_NS * 1e-9)

    np.testing.assert_array_equal(result.state_time_s, expected)
    np.testing.assert_array_equal(result.control_time_s, expected[:-1])


def test_rejects_process_global_mujoco_callback(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)

    def control_callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model, data

    mujoco.set_mjcb_control(control_callback)
    try:
        with pytest.raises(ArtifactValidationError, match="mjcb_control"):
            _run(manifest, arrays, root, model_paths.hydrax)
    finally:
        mujoco.set_mjcb_control(None)


def test_rejects_effort_that_would_be_saturated(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    arrays["tau_feedforward_Nm"][0, 4] = 12.5
    _refresh_bundle(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match="above.*control limit"):
        _run(manifest, arrays, root, model_paths.hydrax)


def test_terminal_effort_knot_has_no_physical_interval(
    tmp_path: Path,
    model_paths: ModelPaths,
) -> None:
    root = tmp_path / "protocol"
    manifest, arrays = _protocol_bundle(root, model_paths.hydrax)
    baseline = _run(manifest, arrays, root, model_paths.hydrax)
    arrays["tau_feedforward_Nm"][-1] = 1_000.0
    _refresh_bundle(root, manifest, arrays)

    changed_terminal = _run(manifest, arrays, root, model_paths.hydrax)

    for field in baseline.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(baseline, field),
            getattr(changed_terminal, field),
        )
