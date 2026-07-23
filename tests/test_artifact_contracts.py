from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import mujoco.sysid as sysid
import numpy as np
import pytest

from fer_mujoco_sysid.artifacts import (
    FER_ARM_JOINT_ORDER,
    ArtifactValidationError,
    content_sha256,
    load_json,
    load_numeric_npz,
    load_schema,
    schema_directory,
    sha256_file,
    validate_motion_protocol,
    validate_normalized_trajectory,
    verify_checksum_manifest,
    write_json,
)

_PERIOD_NS = 10_000_000
_ZERO_SHA256 = "0" * 64
_SOURCE_REVISION = "a" * 40
_SCHEMA_ARTIFACTS = {
    "acquisition-run",
    "common",
    "dataset",
    "fit-result",
    "identified-parameters",
    "motion-protocol",
    "normalized-trajectory",
    "seal",
    "splits",
}


def _source_document(path: str = "models/fer.xml") -> dict[str, Any]:
    return {
        "repository": "https://example.invalid/fer-model.git",
        "revision": _SOURCE_REVISION,
        "path": path,
        "sha256": "b" * 64,
    }


def _write_auxiliary_json(
    root: Path,
    relative_path: str,
    content: dict[str, Any],
) -> dict[str, Any]:
    path = root / relative_path
    write_json(path, content)
    return _file_reference(root, path)


def _file_reference(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _array_reference(
    file_reference: dict[str, Any],
    *,
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


def _write_checksum_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    )
    (root / "checksums.sha256").write_text(text, encoding="utf-8")


def _write_motion_archive(
    root: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    refresh_content_hash: bool = True,
) -> None:
    archive = root / "desired.npz"
    np.savez(archive, **arrays)
    file_reference = _file_reference(root, archive)
    for reference in manifest["arrays"].values():
        reference["file"] = dict(file_reference)
    if refresh_content_hash:
        manifest["content_sha256"] = content_sha256(manifest, arrays)


def _motion_bundle(
    root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root.mkdir(parents=True)
    knot_count = 4
    time_ns = np.arange(knot_count, dtype="<i8") * _PERIOD_NS
    q_rad = np.arange(knot_count * 7, dtype="<f8").reshape(knot_count, 7) * 1e-3
    dq_rad_s = np.zeros((knot_count, 7), dtype="<f8")
    ddq_rad_s2 = np.zeros((knot_count, 7), dtype="<f8")
    arrays = {
        "time_from_start_ns": time_ns,
        "q_rad": q_rad,
        "dq_rad_s": dq_rad_s,
        "ddq_rad_s2": ddq_rad_s2,
    }

    generator_configuration = _write_auxiliary_json(
        root,
        "generator.json",
        {"kind": "fixture", "period_ns": _PERIOD_NS},
    )
    constraint_profile = _write_auxiliary_json(
        root,
        "constraints.json",
        {"kind": "fixture", "joint_count": 7},
    )

    placeholder_file = {
        "path": "desired.npz",
        "sha256": _ZERO_SHA256,
        "size_bytes": 0,
    }
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/motion-protocol@1",
        "artifact_id": "motion-fixture",
        "content_sha256": "c" * 64,
        "title": "Test motion",
        "description": "Small deterministic motion used by contract tests.",
        "family_id": "fixture-family",
        "created_at": "2026-07-23T00:00:00Z",
        "generator": {
            "software": {
                "name": "fer-mujoco-sysid-test-generator",
                "version": "1.0.0",
            },
            "seed": 7,
            "configuration": generator_configuration,
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "command_interface": "joint_trajectory",
        "sample_period_ns": _PERIOD_NS,
        "arrays": {
            "time_from_start": _array_reference(
                placeholder_file,
                key="time_from_start_ns",
                array=time_ns,
                unit="ns",
            ),
            "desired_position": _array_reference(
                placeholder_file,
                key="q_rad",
                array=q_rad,
                unit="rad",
            ),
            "desired_velocity": _array_reference(
                placeholder_file,
                key="dq_rad_s",
                array=dq_rad_s,
                unit="rad/s",
            ),
            "desired_acceleration": _array_reference(
                placeholder_file,
                key="ddq_rad_s2",
                array=ddq_rad_s2,
                unit="rad/s^2",
            ),
        },
        "segments": [
            {
                "segment_id": "excitation",
                "kind": "excitation",
                "start_index": 0,
                "end_index_exclusive": knot_count,
                "analysis_eligible": True,
            }
        ],
        "start_state": {
            "position_rad": q_rad[0].tolist(),
            "velocity_rad_s": dq_rad_s[0].tolist(),
        },
        "end_state": {
            "position_rad": q_rad[-1].tolist(),
            "velocity_rad_s": dq_rad_s[-1].tolist(),
        },
        "context": {
            "source_model": _source_document(),
            "end_effector_id": "fer_hand",
            "payload": {
                "payload_id": "empty-hand",
                "mass_kg": 0.0,
                "center_of_mass_m": [0.0, 0.0, 0.0],
            },
        },
        "constraint_profile": constraint_profile,
    }
    _write_motion_archive(root, manifest, arrays)
    write_json(root / "protocol.json", manifest)
    _write_checksum_manifest(root)
    return manifest, arrays


def _artifact_reference(
    *,
    artifact_id: str,
    schema: str,
    path: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "schema": schema,
        "manifest_sha256": "d" * 64,
        "locator": {"kind": "relative", "path": path},
    }


def _source_binding(
    *,
    raw_signal_name: str,
    sample_count: int,
    end_timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "raw_signal_name": raw_signal_name,
        "source_clock_id": "canonical",
        "start_message_index": 0,
        "end_message_index_exclusive": sample_count,
        "start_timestamp_ns": "0",
        "end_timestamp_ns": str(end_timestamp_ns),
    }


def _normalized_signal(
    *,
    descriptor: dict[str, Any],
    reference: dict[str, Any],
    time_base: str,
    sample_count: int,
    end_timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "descriptor": descriptor,
        "array": reference,
        "time_base": time_base,
        "source_binding": _source_binding(
            raw_signal_name=reference["key"],
            sample_count=sample_count,
            end_timestamp_ns=end_timestamp_ns,
        ),
        "resampling": {"method": "none", "applied_time_shift_s": 0.0},
        "numeric_transform": {"scale": 1.0, "offset": 0.0},
    }


def _joint_state_descriptor(
    *,
    name: str,
    quantity: str,
    semantic_role: str,
    unit: str,
    sample_count: int,
    field: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "quantity": quantity,
        "semantic_role": semantic_role,
        "unit": unit,
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "canonical",
        "source": {
            "kind": "mujoco",
            "object_type": "state",
            "field": field,
        },
        "sample_count": sample_count,
    }


def _effort_descriptor(sample_count: int) -> dict[str, Any]:
    return {
        "name": "controller_effort_request",
        "quantity": "joint_effort",
        "semantic_role": "controller_effort_command",
        "unit": "N*m",
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "canonical",
        "source": {
            "kind": "mujoco",
            "object_type": "actuator",
            "field": "ctrl",
        },
        "sample_count": sample_count,
        "effort_semantics": {
            "stage": "controller",
            "location": "generalized_coordinate",
            "gravity": "unknown",
            "coriolis": "unknown",
            "friction_compensation": "unknown",
            "rate_limit_position": "before",
            "definition": "Controller effort request before the simulated plant.",
            "authority": _source_document("interfaces/controller_effort.md"),
        },
    }


def _write_normalized_archive(
    root: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    refresh_content_hash: bool = True,
) -> None:
    archive = root / "signals.npz"
    np.savez(archive, **arrays)
    file_reference = _file_reference(root, archive)
    manifest["state_time"]["file"] = dict(file_reference)
    manifest["control_time"]["file"] = dict(file_reference)
    for signal in manifest["signals"]:
        signal["array"]["file"] = dict(file_reference)
    valid_mask = manifest["quality"].get("valid_sample_mask")
    if valid_mask is not None:
        valid_mask["file"] = dict(file_reference)
    if refresh_content_hash:
        manifest["content_sha256"] = content_sha256(manifest, arrays)


def _normalized_bundle(
    root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root.mkdir(parents=True)
    transitions = 3
    state_time_s = np.arange(transitions + 1, dtype="<f8") * (_PERIOD_NS * 1e-9)
    control_time_s = state_time_s[:-1].copy()
    q_rad = (
        np.arange((transitions + 1) * 7, dtype="<f8").reshape(transitions + 1, 7) * 1e-3
    )
    dq_rad_s = np.zeros((transitions + 1, 7), dtype="<f8")
    tau_controller_request_nm = np.full((transitions, 7), 0.05, dtype="<f8")
    arrays = {
        "state_time_s": state_time_s,
        "control_time_s": control_time_s,
        "q_rad": q_rad,
        "dq_rad_s": dq_rad_s,
        "tau_controller_request_Nm": tau_controller_request_nm,
    }

    converter_configuration = _write_auxiliary_json(
        root,
        "converter.json",
        {"kind": "fixture", "interpolation": "none"},
    )
    quality_report = _write_auxiliary_json(
        root,
        "quality.json",
        {"status": "pass", "invalid_samples": 0},
    )
    placeholder_file = {
        "path": "signals.npz",
        "sha256": _ZERO_SHA256,
        "size_bytes": 0,
    }
    q_reference = _array_reference(
        placeholder_file,
        key="q_rad",
        array=q_rad,
        unit="rad",
    )
    dq_reference = _array_reference(
        placeholder_file,
        key="dq_rad_s",
        array=dq_rad_s,
        unit="rad/s",
    )
    effort_reference = _array_reference(
        placeholder_file,
        key="tau_controller_request_Nm",
        array=tau_controller_request_nm,
        unit="N*m",
    )
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/normalized-trajectory@1",
        "artifact_id": "normalized-fixture",
        "content_sha256": "e" * 64,
        "created_at": "2026-07-23T00:00:00Z",
        "source_run": _artifact_reference(
            artifact_id="run-fixture",
            schema="fer-mujoco-sysid/acquisition-run@1",
            path="runs/run-fixture/run.json",
        ),
        "protocol": _artifact_reference(
            artifact_id="motion-fixture",
            schema="fer-mujoco-sysid/motion-protocol@1",
            path="protocols/motion-fixture/protocol.json",
        ),
        "protocol_segment_id": "excitation",
        "lineage_group_id": "fixture-lineage",
        "converter": {
            "software": {
                "name": "fer-mujoco-sysid-test-converter",
                "version": "1.0.0",
            },
            "configuration": converter_configuration,
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "canonical_clock": {
            "clock_id": "canonical",
            "domain": "simulation_clock",
            "epoch": "trajectory start",
            "tick_unit": "ns",
            "resolution_ns": 1,
            "timestamp_source": "state_time_s",
        },
        "clock_alignments": [],
        "sample_grid": {"kind": "uniform", "period_ns": _PERIOD_NS},
        "state_time": _array_reference(
            placeholder_file,
            key="state_time_s",
            array=state_time_s,
            unit="s",
        ),
        "control_time": _array_reference(
            placeholder_file,
            key="control_time_s",
            array=control_time_s,
            unit="s",
        ),
        "initial_state": {
            "position_rad": q_rad[0].tolist(),
            "velocity_rad_s": dq_rad_s[0].tolist(),
        },
        "signals": [
            _normalized_signal(
                descriptor=_joint_state_descriptor(
                    name="measured_joint_position",
                    quantity="joint_position",
                    semantic_role="measured_joint_position",
                    unit="rad",
                    sample_count=transitions + 1,
                    field="qpos",
                ),
                reference=q_reference,
                time_base="state",
                sample_count=transitions + 1,
                end_timestamp_ns=transitions * _PERIOD_NS,
            ),
            _normalized_signal(
                descriptor=_joint_state_descriptor(
                    name="measured_joint_velocity",
                    quantity="joint_velocity",
                    semantic_role="measured_joint_velocity",
                    unit="rad/s",
                    sample_count=transitions + 1,
                    field="qvel",
                ),
                reference=dq_reference,
                time_base="state",
                sample_count=transitions + 1,
                end_timestamp_ns=transitions * _PERIOD_NS,
            ),
            _normalized_signal(
                descriptor=_effort_descriptor(transitions),
                reference=effort_reference,
                time_base="control",
                sample_count=transitions,
                end_timestamp_ns=(transitions - 1) * _PERIOD_NS,
            ),
        ],
        "quality": {"status": "pass", "report": quality_report},
    }
    _write_normalized_archive(root, manifest, arrays)
    write_json(root / "trajectory.json", manifest)
    _write_checksum_manifest(root)
    return manifest, arrays


def test_all_schemas_are_packaged_and_metaschema_valid() -> None:
    root = schema_directory()
    packaged = {
        path.name.removesuffix("-v1.schema.json")
        for path in root.iterdir()
        if path.is_file() and path.name.endswith("-v1.schema.json")
    }

    assert packaged == _SCHEMA_ARTIFACTS
    for artifact in sorted(packaged):
        schema = load_schema(artifact)
        assert schema["$schema"].startswith(
            "https://json-schema.org/draft/2020-12/schema"
        )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_constants(
    tmp_path: Path,
    token: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(f'{{"value": {token}}}\n', encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="non-finite JSON number"):
        load_json(path)


def test_strict_json_writer_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="non-finite JSON number"):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})


def test_strict_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="duplicate JSON object key"):
        load_json(path)


def test_schema_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    manifest, arrays = _motion_bundle(tmp_path / "motion")
    manifest["unexpected_field"] = "must not be ignored"

    with pytest.raises(
        ArtifactValidationError,
        match=r"motion-protocol@1 schema violation.*unexpected_field",
    ):
        validate_motion_protocol(manifest, arrays, root=tmp_path / "motion")


def test_valid_motion_bundle_loads_and_validates(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    expected_manifest, expected_arrays = _motion_bundle(root)

    manifest = load_json(root / "protocol.json")
    arrays = load_numeric_npz(root / "desired.npz")

    assert manifest == expected_manifest
    assert isinstance(manifest, dict)
    validate_motion_protocol(manifest, arrays, root=root)
    verify_checksum_manifest(root)
    for key, expected in expected_arrays.items():
        np.testing.assert_array_equal(arrays[key], expected)


def test_valid_normalized_bundle_loads_and_validates(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    expected_manifest, expected_arrays = _normalized_bundle(root)

    manifest = load_json(root / "trajectory.json")
    arrays = load_numeric_npz(root / "signals.npz")

    assert manifest == expected_manifest
    assert isinstance(manifest, dict)
    validate_normalized_trajectory(manifest, arrays, root=root)
    verify_checksum_manifest(root)
    for key, expected in expected_arrays.items():
        np.testing.assert_array_equal(arrays[key], expected)


def test_motion_rejects_wrong_joint_order_actionably(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    manifest["joint_order"][0], manifest["joint_order"][1] = (
        manifest["joint_order"][1],
        manifest["joint_order"][0],
    )

    with pytest.raises(ArtifactValidationError, match="joint_order"):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_wrong_unit_actionably(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    manifest["arrays"]["desired_velocity"]["unit"] = "rad"

    with pytest.raises(
        ArtifactValidationError,
        match=r"arrays\.desired_velocity\.unit",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


def test_effort_protocol_requires_compiled_effort_array(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    manifest["command_interface"] = "joint_effort"

    with pytest.raises(
        ArtifactValidationError,
        match=r"joint_effort.*desired_effort_feedforward",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_uncontained_archive_path(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    for reference in manifest["arrays"].values():
        reference["file"]["path"] = "../desired.npz"

    with pytest.raises(
        ArtifactValidationError,
        match=r"(schema violation.*path|escapes its artifact root)",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_wrong_archive_hash(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    for reference in manifest["arrays"].values():
        reference["file"]["sha256"] = "f" * 64

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_stale_scientific_content_hash(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    arrays["q_rad"][1, 0] += 0.01
    _write_motion_archive(
        root,
        manifest,
        arrays,
        refresh_content_hash=False,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="scientific content SHA-256 mismatch",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_archive_dtype_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = dict(arrays)
    arrays["q_rad"] = arrays["q_rad"].astype("<f4")
    _write_motion_archive(
        root,
        manifest,
        arrays,
        refresh_content_hash=False,
    )

    with pytest.raises(
        ArtifactValidationError,
        match=r"q_rad.*canonical little-endian dtype|dtype mismatch for q_rad",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_archive_shape_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = dict(arrays)
    arrays["q_rad"] = arrays["q_rad"][:, :6].copy()
    _write_motion_archive(
        root,
        manifest,
        arrays,
        refresh_content_hash=False,
    )

    with pytest.raises(ArtifactValidationError, match="shape mismatch for q_rad"):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_rejects_nonfinite_numeric_samples(tmp_path: Path) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    arrays["q_rad"][1, 0] = np.nan
    _write_motion_archive(
        root,
        manifest,
        arrays,
        refresh_content_hash=False,
    )

    with pytest.raises(
        ArtifactValidationError,
        match=r"q_rad.*contains NaN or infinity",
    ):
        validate_motion_protocol(manifest, arrays, root=root)


@pytest.mark.parametrize(
    ("time_ns", "message"),
    [
        (
            np.array([0, _PERIOD_NS, _PERIOD_NS, 3 * _PERIOD_NS], dtype="<i8"),
            "increase strictly",
        ),
        (
            np.array(
                [0, 2 * _PERIOD_NS, _PERIOD_NS, 3 * _PERIOD_NS],
                dtype="<i8",
            ),
            "increase strictly",
        ),
        (
            np.array(
                [0, _PERIOD_NS, 2 * _PERIOD_NS + 1, 3 * _PERIOD_NS],
                dtype="<i8",
            ),
            "declared exact sample period",
        ),
    ],
    ids=["duplicate", "nonmonotonic", "off-grid"],
)
def test_motion_rejects_invalid_time_grid(
    tmp_path: Path,
    time_ns: np.ndarray,
    message: str,
) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = dict(arrays)
    arrays["time_from_start_ns"] = time_ns
    _write_motion_archive(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match=message):
        validate_motion_protocol(manifest, arrays, root=root)


def test_motion_timestamp_validation_is_safe_from_int64_overflow(
    tmp_path: Path,
) -> None:
    root = tmp_path / "motion"
    manifest, arrays = _motion_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    arrays["time_from_start_ns"] = np.array(
        [0, np.iinfo(np.int64).max, -2, np.iinfo(np.int64).max - 3],
        dtype="<i8",
    )
    manifest["sample_period_ns"] = int(np.iinfo(np.int64).max)
    _write_motion_archive(root, manifest, arrays)

    with pytest.raises(ArtifactValidationError, match="increase strictly"):
        validate_motion_protocol(manifest, arrays, root=root)


def test_numeric_loader_rejects_object_arrays_without_pickle(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.npz"
    np.savez(archive, unsafe=np.array([{"value": 1}], dtype=object))

    with pytest.raises(
        ArtifactValidationError,
        match=r"unsafe array 'unsafe'|Object arrays cannot be loaded",
    ):
        load_numeric_npz(archive)


def test_normalized_rejects_state_control_count_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    arrays["state_time_s"] = arrays["state_time_s"][:-1].copy()
    manifest["state_time"]["shape"] = list(arrays["state_time_s"].shape)
    _write_normalized_archive(root, manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match=r"M\+1/M convention",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_m_plus_one_effort_samples(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    effort_key = "tau_controller_request_Nm"
    arrays[effort_key] = np.vstack([arrays[effort_key], arrays[effort_key][-1]])
    effort_signal = next(
        signal for signal in manifest["signals"] if signal["array"]["key"] == effort_key
    )
    effort_signal["array"]["shape"] = list(arrays[effort_key].shape)
    effort_signal["descriptor"]["sample_count"] = arrays[effort_key].shape[0]
    _write_normalized_archive(root, manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match=r"signal array.*shape \(3, 7\).*control time base",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_missing_noncanonical_clock_alignment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    manifest["signals"][0]["source_binding"]["source_clock_id"] = "robot"
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match=r"clock alignments.*missing=\['robot'\]",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_accepts_covered_noncanonical_clock_alignment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    manifest["signals"][0]["source_binding"]["source_clock_id"] = "robot"
    manifest["clock_alignments"] = [
        {
            "source_clock_id": "robot",
            "target_clock_id": "canonical",
            "method": "affine",
            "scale": 1.0,
            "offset_s": 0.0,
            "rms_residual_s": 0.0,
            "max_residual_s": 0.0,
            "valid_source_start_ns": "0",
            "valid_source_end_ns": str(3 * _PERIOD_NS),
            "extrapolation_allowed": False,
            "evidence": "Shared simulation clock sampled at both boundaries.",
        }
    ]
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_signal_on_wrong_time_base(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    manifest["signals"][0]["time_base"] = "control"
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match=r"q_rad.*control time base|signal array 'q_rad'.*control time base",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_unrepresentable_period_as_artifact_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    manifest["sample_grid"]["period_ns"] = 10**1000
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    with pytest.raises(ArtifactValidationError, match=r"sample_grid\.period_ns"):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_noncausal_control_effort_resampling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    effort_signal = next(
        signal
        for signal in manifest["signals"]
        if signal["descriptor"]["quantity"] == "joint_effort"
    )
    effort_signal["resampling"]["method"] = "linear"
    manifest["content_sha256"] = content_sha256(manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match=r"control-effort signal.*zero_order_hold",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_normalized_rejects_floating_point_off_grid_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, arrays = _normalized_bundle(root)
    arrays = {key: value.copy() for key, value in arrays.items()}
    arrays["state_time_s"][2] += 1e-12
    arrays["control_time_s"] = arrays["state_time_s"][:-1].copy()
    _write_normalized_archive(root, manifest, arrays)

    with pytest.raises(
        ArtifactValidationError,
        match="declared exact fixed grid",
    ):
        validate_normalized_trajectory(manifest, arrays, root=root)


def test_checksum_verification_detects_corruption(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"original payload")
    expected = sha256_file(payload)
    (tmp_path / "checksums.sha256").write_text(
        f"{expected}  payload.bin\n",
        encoding="utf-8",
    )
    verify_checksum_manifest(tmp_path)

    payload.write_bytes(b"corrupted payload")

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        verify_checksum_manifest(tmp_path)


def _identification_only_spec() -> mujoco.MjSpec:
    child = ""
    axes = ("1 0 0", "0 1 0", "0 0 1")
    for index in range(7, 0, -1):
        joint = FER_ARM_JOINT_ORDER[index - 1]
        child = (
            f'<body name="link{index}" pos="0 0 0.1">'
            f'<joint name="{joint}" type="hinge" axis="{axes[(index - 1) % 3]}"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 0.08" '
            'size="0.01" mass="0.1"/>'
            f"{child}</body>"
        )
    actuators = "".join(
        f'<motor name="{joint}_actuator" joint="{joint}"/>'
        for joint in FER_ARM_JOINT_ORDER
    )
    position_sensors = "".join(
        f'<jointpos name="{joint}_q" joint="{joint}"/>' for joint in FER_ARM_JOINT_ORDER
    )
    velocity_sensors = "".join(
        f'<jointvel name="{joint}_dq" joint="{joint}"/>'
        for joint in FER_ARM_JOINT_ORDER
    )
    xml = (
        '<mujoco model="artifact-adapter-fixture">'
        f'<option timestep="{_PERIOD_NS * 1e-9}" gravity="0 0 0"/>'
        f"<worldbody>{child}</worldbody>"
        f"<actuator>{actuators}</actuator>"
        f"<sensor>{position_sensors}{velocity_sensors}</sensor>"
        "</mujoco>"
    )
    return mujoco.MjSpec.from_string(xml)


def test_valid_normalized_arrays_are_public_sysid_api_compatible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "normalized"
    manifest, expected_arrays = _normalized_bundle(root)
    arrays = load_numeric_npz(root / "signals.npz")
    validate_normalized_trajectory(manifest, arrays, root=root)
    for key, expected in expected_arrays.items():
        np.testing.assert_array_equal(arrays[key], expected)

    spec = _identification_only_spec()
    model = spec.compile()
    assert model.nq == model.nv == model.nu == 7
    assert model.nsensordata == 14

    state_time = arrays["state_time_s"]
    interval_effort = arrays["tau_controller_request_Nm"]
    transitions = interval_effort.shape[0]

    # Portable data contains only physical intervals. MuJoCo's interpolation
    # contract needs a terminal sample, so the adapter duplicates it in memory.
    padded_effort = np.vstack([interval_effort, interval_effort[-1]])
    control = sysid.TimeSeries.from_control_names(
        times=state_time,
        data=padded_effort,
        model=model,
        names=[f"{joint}_actuator" for joint in FER_ARM_JOINT_ORDER],
    )

    sensor_names = [
        *(f"{joint}_q" for joint in FER_ARM_JOINT_ORDER),
        *(f"{joint}_dq" for joint in FER_ARM_JOINT_ORDER),
    ]
    sensor_values = np.hstack(
        [
            arrays["q_rad"][1:],
            arrays["dq_rad_s"][1:],
        ]
    )
    sensordata = sysid.TimeSeries.from_names(
        times=state_time[1:],
        data=sensor_values,
        model=model,
        names=sensor_names,
    )

    data = mujoco.MjData(model)
    data.time = float(state_time[0])
    data.qpos[:] = arrays["q_rad"][0]
    data.qvel[:] = arrays["dq_rad_s"][0]
    state_kind = mujoco.mjtState.mjSTATE_FULLPHYSICS
    initial_state = np.empty(mujoco.mj_stateSize(model, state_kind))
    mujoco.mj_getState(model, data, initial_state, state_kind)

    sequences = sysid.ModelSequences(
        name="normalized-fixture",
        spec=spec,
        sequence_name="excitation",
        initial_state=initial_state,
        control=control,
        sensordata=sensordata,
    )

    assert interval_effort.shape == (transitions, 7)
    assert control.times.shape == (transitions + 1,)
    assert control.data.shape == (transitions + 1, 7)
    assert sensordata.times.shape == (transitions,)
    assert sensordata.data.shape == (transitions, 14)
    np.testing.assert_array_equal(control.times, state_time)
    np.testing.assert_array_equal(control.data[-1], interval_effort[-1])
    np.testing.assert_array_equal(sensordata.times, state_time[1:])
    assert len(sequences.control) == len(sequences.sensordata) == 1
    assert sequences.measured_rollout[0].check_compatible() is None
