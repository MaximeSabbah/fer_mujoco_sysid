from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fer_mujoco_sysid import cli
from fer_mujoco_sysid.artifacts.bundle import (
    CHECKSUM_MANIFEST_NAME,
    finalize_bundle,
)
from fer_mujoco_sysid.artifacts.catalog import (
    LocalArtifactResolver,
    validate_dataset_bundle,
    validate_fit_against_dataset,
)
from fer_mujoco_sysid.artifacts.content import content_sha256
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import sha256_file, write_json
from fer_mujoco_sysid.artifacts.validation import FER_ARM_JOINT_ORDER

_PERIOD_NS = 10_000_000
_ZERO_SHA = "0" * 64
_CREATED = "2026-07-23T00:00:00Z"
_CLOCK_VARIANTS = frozenset(
    {
        "affine_clock",
        "affine_clock_residual",
        "affine_clock_timing_outside_validity",
        "piecewise_clock",
    }
)


def _source(path: str = "models/fer.xml") -> dict[str, Any]:
    return {
        "repository": "https://example.invalid/fer.git",
        "revision": "a" * 40,
        "path": path,
        "sha256": "b" * 64,
    }


def _payload() -> dict[str, Any]:
    return {
        "payload_id": "empty-hand",
        "mass_kg": 0.0,
        "center_of_mass_m": [0.0, 0.0, 0.0],
    }


def _file(root: Path, relative: str, value: Any = None) -> dict[str, Any]:
    path = root / relative
    write_json(path, {"fixture": True} if value is None else value)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _array(
    file_reference: dict[str, Any],
    key: str,
    value: np.ndarray,
    unit: str,
) -> dict[str, Any]:
    return {
        "file": dict(file_reference),
        "key": key,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "unit": unit,
    }


def _artifact_reference(
    root: Path,
    manifest_path: Path,
    *,
    include_bundle: bool = True,
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    reference = {
        "artifact_id": manifest["artifact_id"],
        "schema": manifest["schema"],
        "manifest_sha256": sha256_file(manifest_path),
        "locator": {
            "kind": "relative",
            "path": manifest_path.relative_to(root).as_posix(),
        },
    }
    if include_bundle:
        reference["bundle_sha256"] = sha256_file(
            manifest_path.parent / CHECKSUM_MANIFEST_NAME
        )
    return reference


def _read_manifest(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _motion_values(
    motion_variant: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    count = 6 if motion_variant in {"two_segments", "fit_embedded"} else 3
    arrays = {
        "time_from_start_ns": (np.arange(count, dtype="<i8") * _PERIOD_NS),
        "q_rad": np.zeros((count, 7), dtype="<f8"),
        "dq_rad_s": np.zeros((count, 7), dtype="<f8"),
        "ddq_rad_s2": np.zeros((count, 7), dtype="<f8"),
    }
    if motion_variant == "heldout":
        arrays["q_rad"][:, 0] = np.asarray([0.0, 0.01, 0.0], dtype="<f8")
        arrays["dq_rad_s"][:, 0] = np.asarray([1.0, -1.0, 1.0], dtype="<f8")
    elif motion_variant == "two_segments":
        arrays["q_rad"][:, 0] = np.asarray(
            [0.0, 0.01, 0.0, 0.2, 0.21, 0.2],
            dtype="<f8",
        )
        arrays["dq_rad_s"][:, 0] = np.asarray(
            [1.0, -1.0, 1.0, 2.0, -2.0, 2.0],
            dtype="<f8",
        )
    elif motion_variant == "fit_embedded":
        arrays["q_rad"][:, 0] = np.asarray(
            [0.3, 0.31, 0.3, 0.0, 0.0, 0.0],
            dtype="<f8",
        )
        arrays["dq_rad_s"][:, 0] = np.asarray(
            [3.0, -3.0, 3.0, 0.0, 0.0, 0.0],
            dtype="<f8",
        )
    elif motion_variant in {"effort_fit", "effort_heldout"}:
        arrays["tau_feedforward_Nm"] = np.zeros((count, 7), dtype="<f8")
        arrays["tau_feedforward_Nm"][:-1, 0] = np.asarray(
            [0.1, 0.2],
            dtype="<f8",
        )
        arrays["tau_feedforward_Nm"][-1, 0] = (
            100.0 if motion_variant == "effort_fit" else -100.0
        )
    elif motion_variant == "negative_zero":
        arrays["q_rad"][1, 0] = -0.0
    segments = (
        [
            {
                "segment_id": "excitation-a",
                "kind": "excitation",
                "start_index": 0,
                "end_index_exclusive": 3,
                "analysis_eligible": True,
            },
            {
                "segment_id": "excitation-b",
                "kind": "excitation",
                "start_index": 3,
                "end_index_exclusive": 6,
                "analysis_eligible": True,
            },
        ]
        if motion_variant in {"two_segments", "fit_embedded"}
        else [
            {
                "segment_id": "excitation",
                "kind": "excitation",
                "start_index": 0,
                "end_index_exclusive": count,
                "analysis_eligible": True,
            }
        ]
    )
    return arrays, segments


def _motion_bundle(
    root: Path,
    *,
    protocol_id: str,
    analysis_eligible: bool,
    motion_variant: str,
) -> Path:
    root.mkdir(parents=True)
    arrays, segments = _motion_values(motion_variant)
    if not analysis_eligible:
        segments[0]["analysis_eligible"] = False
    archive = root / "motion.npz"
    np.savez(archive, **arrays)
    archive_ref = {
        "path": archive.name,
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/motion-protocol@1",
        "artifact_id": protocol_id,
        "content_sha256": _ZERO_SHA,
        "title": "Fixture protocol",
        "description": "A compact deterministic protocol.",
        "family_id": "fixture-family",
        "created_at": _CREATED,
        "generator": {
            "software": {"name": "fixture-generator", "version": "1"},
            "seed": 1,
            "configuration": _file(root, "generator.json"),
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "command_interface": (
            "joint_effort" if "tau_feedforward_Nm" in arrays else "joint_trajectory"
        ),
        "sample_period_ns": _PERIOD_NS,
        "arrays": {
            "time_from_start": _array(
                archive_ref,
                "time_from_start_ns",
                arrays["time_from_start_ns"],
                "ns",
            ),
            "desired_position": _array(
                archive_ref,
                "q_rad",
                arrays["q_rad"],
                "rad",
            ),
            "desired_velocity": _array(
                archive_ref,
                "dq_rad_s",
                arrays["dq_rad_s"],
                "rad/s",
            ),
            "desired_acceleration": _array(
                archive_ref,
                "ddq_rad_s2",
                arrays["ddq_rad_s2"],
                "rad/s^2",
            ),
        },
        "segments": segments,
        "start_state": {
            "position_rad": arrays["q_rad"][0].tolist(),
            "velocity_rad_s": arrays["dq_rad_s"][0].tolist(),
        },
        "end_state": {
            "position_rad": arrays["q_rad"][-1].tolist(),
            "velocity_rad_s": arrays["dq_rad_s"][-1].tolist(),
        },
        "context": {
            "source_model": _source(),
            "end_effector_id": "fer-hand",
            "payload": _payload(),
        },
        "constraint_profile": _file(root, "constraints.json"),
    }
    if "tau_feedforward_Nm" in arrays:
        manifest["arrays"]["desired_effort_feedforward"] = _array(
            archive_ref,
            "tau_feedforward_Nm",
            arrays["tau_feedforward_Nm"],
            "N*m",
        )
    manifest["content_sha256"] = content_sha256(manifest, arrays)
    path = root / "protocol.json"
    write_json(path, manifest)
    finalize_bundle(root)
    return path


def _effort_semantics() -> dict[str, Any]:
    return {
        "stage": "simulator",
        "location": "generalized_coordinate",
        "gravity": "included",
        "coriolis": "included",
        "friction_compensation": "included",
        "rate_limit_position": "not_applicable",
        "definition": "Simulated actuator effort at the generalized coordinate.",
        "authority": _source("interfaces/effort.md"),
    }


def _desired_effort_semantics() -> dict[str, Any]:
    return {
        "stage": "controller",
        "location": "generalized_coordinate",
        "gravity": "unknown",
        "coriolis": "unknown",
        "friction_compensation": "unknown",
        "rate_limit_position": "before",
        "definition": "Compiled effort feedforward before feedback and limits.",
        "authority": _source("interfaces/effort-feedforward.md"),
    }


def _descriptor(
    *,
    name: str,
    quantity: str,
    role: str,
    unit: str,
    source: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    descriptor = {
        "name": name,
        "quantity": quantity,
        "semantic_role": role,
        "unit": unit,
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "positive_direction": "same_as_joint_coordinate",
        "clock_id": "canonical",
        "source": source,
        "sample_count": count,
    }
    if quantity == "joint_effort":
        descriptor["effort_semantics"] = _effort_semantics()
    return descriptor


def _raw_descriptors(
    *,
    protocol_artifact_id: str,
    desired_sample_count: int = 3,
    include_desired_effort: bool = False,
) -> list[dict[str, Any]]:
    descriptors = [
        _descriptor(
            name="raw_position",
            quantity="joint_position",
            role="measured_joint_position",
            unit="rad",
            source={"kind": "mujoco", "object_type": "state", "field": "qpos"},
            count=3,
        ),
        _descriptor(
            name="raw_velocity",
            quantity="joint_velocity",
            role="measured_joint_velocity",
            unit="rad/s",
            source={"kind": "mujoco", "object_type": "state", "field": "qvel"},
            count=3,
        ),
        _descriptor(
            name="raw_effort",
            quantity="joint_effort",
            role="simulated_actuator_effort",
            unit="N*m",
            source={
                "kind": "mujoco",
                "object_type": "actuator",
                "field": "force",
            },
            count=2,
        ),
        _descriptor(
            name="raw_desired_position",
            quantity="joint_position",
            role="desired_joint_position",
            unit="rad",
            source={
                "kind": "protocol_player",
                "protocol_artifact_id": protocol_artifact_id,
                "field": "q_rad",
            },
            count=desired_sample_count,
        ),
        _descriptor(
            name="raw_desired_velocity",
            quantity="joint_velocity",
            role="desired_joint_velocity",
            unit="rad/s",
            source={
                "kind": "protocol_player",
                "protocol_artifact_id": protocol_artifact_id,
                "field": "dq_rad_s",
            },
            count=desired_sample_count,
        ),
        _descriptor(
            name="raw_desired_acceleration",
            quantity="joint_acceleration",
            role="desired_joint_acceleration",
            unit="rad/s^2",
            source={
                "kind": "protocol_player",
                "protocol_artifact_id": protocol_artifact_id,
                "field": "ddq_rad_s2",
            },
            count=desired_sample_count,
        ),
    ]
    if include_desired_effort:
        desired_effort = _descriptor(
            name="raw_desired_effort_feedforward",
            quantity="joint_effort",
            role="desired_effort_feedforward",
            unit="N*m",
            source={
                "kind": "protocol_player",
                "protocol_artifact_id": protocol_artifact_id,
                "field": "tau_feedforward_Nm",
            },
            count=desired_sample_count - 1,
        )
        desired_effort["effort_semantics"] = _desired_effort_semantics()
        descriptors.append(desired_effort)
    return descriptors


def _run_bundle(
    catalog_root: Path,
    root: Path,
    *,
    suffix: str,
    protocol_reference: dict[str, Any],
    completed: bool,
    desired_sample_count: int,
    include_desired_effort: bool,
    fault: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    manifest = {
        "schema": "fer-mujoco-sysid/acquisition-run@1",
        "artifact_id": f"run-{suffix}",
        "started_at_utc": "2026-07-23T00:00:00Z",
        "ended_at_utc": "2026-07-23T00:00:01Z",
        "backend": "standalone_mujoco",
        "protocol": deepcopy(protocol_reference),
        "protocol_reference_signals": {
            "desired_position": "raw_desired_position",
            "desired_velocity": "raw_desired_velocity",
            "desired_acceleration": "raw_desired_acceleration",
        },
        "protocol_timing": {
            "clock_id": "canonical",
            "protocol_start_timestamp_ns": "0",
            "source": {
                "kind": "mujoco",
                "object_type": "data",
                "field": "time",
                "sample_index": 0,
                "unit": "s",
            },
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "robot": {
            "platform": "FER",
            "robot_id": "fer-sim",
            "end_effector_id": "fer-hand",
            "payload": _payload(),
        },
        "controller": {
            "name": "fixture-player",
            "type": "joint_trajectory",
            "update_rate_hz": 100.0,
            "configuration": _file(root, "controller.json"),
        },
        "source_model": _source(),
        "clock_domains": [
            {
                "clock_id": "canonical",
                "domain": "simulation_clock",
                "epoch": "run start",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "MuJoCo time",
            }
        ],
        "signals": _raw_descriptors(
            protocol_artifact_id=str(protocol_reference["artifact_id"]),
            desired_sample_count=desired_sample_count,
            include_desired_effort=include_desired_effort,
        ),
        "recording_files": [_file(root, "raw.json")],
        "recorded_topics": [],
        "software": [{"name": "mujoco", "version": "3.10.0"}],
        "outcome": {
            "status": "completed" if completed else "simulation_error",
            "reason_code": "completed_normally" if completed else "plant_error",
        },
    }
    if include_desired_effort:
        manifest["protocol_reference_signals"]["desired_effort_feedforward"] = (
            "raw_desired_effort_feedforward"
        )
    if fault in _CLOCK_VARIANTS:
        manifest["clock_domains"][0]["clock_id"] = "source-clock"
        manifest["protocol_timing"]["clock_id"] = "source-clock"
        manifest["protocol_timing"]["protocol_start_timestamp_ns"] = "1000000000"
        for signal in manifest["signals"]:
            signal["clock_id"] = "source-clock"
    if fault == "protocol_timing_reference_binding":
        alternate_reference = deepcopy(manifest["signals"][3])
        alternate_reference["name"] = "raw_alternate_desired_position"
        manifest["signals"].append(alternate_reference)
    if fault == "protocol_reference_source":
        manifest["signals"][4]["source"] = {
            "kind": "mujoco",
            "object_type": "data",
            "field": "desired_dq",
        }
    if fault == "end_effector":
        manifest["robot"]["end_effector_id"] = "different-hand"
    elif fault == "payload":
        manifest["robot"]["payload"]["mass_kg"] = 1.0
    elif fault == "protocol_validation":
        manifest["protocol_validation"] = {
            "artifact_id": f"protocol-validation-{suffix}",
            "schema": "fer-mujoco-sysid/protocol-validation@1",
            "manifest_sha256": _ZERO_SHA,
            "locator": {
                "kind": "relative",
                "path": f"protocol-validation-{suffix}/validation.json",
            },
        }
    path = root / "run.json"
    write_json(path, manifest)
    finalize_bundle(root)
    assert path.is_relative_to(catalog_root)
    return path


def _normalized_signal(
    descriptor: dict[str, Any],
    array_reference: dict[str, Any],
    *,
    raw_name: str,
    time_base: str,
    raw_count: int,
    raw_start: int = 0,
    timestamp_start: int | None = None,
) -> dict[str, Any]:
    raw_end = raw_start + raw_count
    timestamp_start = raw_start if timestamp_start is None else timestamp_start
    timestamp_end = timestamp_start + raw_count
    return {
        "descriptor": descriptor,
        "array": array_reference,
        "time_base": time_base,
        "source_binding": {
            "raw_signal_name": raw_name,
            "source_clock_id": "canonical",
            "start_message_index": raw_start,
            "end_message_index_exclusive": raw_end,
            "start_timestamp_ns": str(timestamp_start * _PERIOD_NS),
            "end_timestamp_ns": str((timestamp_end - 1) * _PERIOD_NS),
        },
        "resampling": {"method": "none", "applied_time_shift_s": 0.0},
        "numeric_transform": {"scale": 1.0, "offset": 0.0},
    }


def _trajectory_bundle(
    root: Path,
    *,
    suffix: str,
    run_reference: dict[str, Any],
    protocol_reference: dict[str, Any],
    protocol_desired: Mapping[str, np.ndarray],
    declared_protocol_interval: tuple[int, int],
    reference_source_interval: tuple[int, int],
    default_protocol_segment_id: str,
    quality: str,
    raw_fault: str | None,
) -> Path:
    root.mkdir(parents=True)
    source_start, source_end = reference_source_interval
    state_time = np.arange(3, dtype="<f8") * (_PERIOD_NS * 1e-9)
    arrays = {
        "state_time_s": state_time,
        "control_time_s": state_time[:-1].copy(),
        "q_rad": np.zeros((3, 7), dtype="<f8"),
        "dq_rad_s": np.zeros((3, 7), dtype="<f8"),
        "tau_simulated_Nm": np.ones((2, 7), dtype="<f8"),
        "desired_q_rad": np.array(
            protocol_desired["q_rad"][source_start:source_end],
            dtype="<f8",
            copy=True,
        ),
        "desired_dq_rad_s": np.array(
            protocol_desired["dq_rad_s"][source_start:source_end],
            dtype="<f8",
            copy=True,
        ),
        "desired_ddq_rad_s2": np.array(
            protocol_desired["ddq_rad_s2"][source_start:source_end],
            dtype="<f8",
            copy=True,
        ),
    }
    if "tau_feedforward_Nm" in protocol_desired:
        arrays["desired_tau_feedforward_Nm"] = np.array(
            protocol_desired["tau_feedforward_Nm"][source_start : source_end - 1],
            dtype="<f8",
            copy=True,
        )
    if raw_fault == "protocol_velocity_mismatch":
        arrays["desired_dq_rad_s"][0, 0] += 0.5
    archive = root / "signals.npz"
    np.savez(archive, **arrays)
    archive_ref = {
        "path": archive.name,
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
    }
    raw = _raw_descriptors(
        protocol_artifact_id=str(protocol_reference["artifact_id"]),
        desired_sample_count=protocol_desired["q_rad"].shape[0],
        include_desired_effort="tau_feedforward_Nm" in protocol_desired,
    )
    q_descriptor = {**deepcopy(raw[0]), "name": "measured_position"}
    dq_descriptor = {**deepcopy(raw[1]), "name": "measured_velocity"}
    effort_descriptor = {**deepcopy(raw[2]), "name": "simulated_effort"}
    desired_descriptor = {
        **deepcopy(raw[3]),
        "name": "desired_position_reference",
        "sample_count": 3,
    }
    desired_velocity_descriptor = {
        **deepcopy(raw[4]),
        "name": "desired_velocity_reference",
        "sample_count": 3,
    }
    desired_acceleration_descriptor = {
        **deepcopy(raw[5]),
        "name": "desired_acceleration_reference",
        "sample_count": 3,
    }
    desired_effort_descriptor = (
        {
            **deepcopy(raw[6]),
            "name": "desired_effort_feedforward_reference",
            "sample_count": 2,
        }
        if "tau_feedforward_Nm" in protocol_desired
        else None
    )
    if raw_fault == "source":
        effort_descriptor["source"] = {
            "kind": "mujoco",
            "object_type": "data",
            "field": "qfrc_actuator",
        }
    elif raw_fault == "effort_semantics":
        effort_descriptor["effort_semantics"]["gravity"] = "excluded"
    reference_timestamp_start = (
        3 if raw_fault == "protocol_anchor_shift" else source_start
    )
    measurement_timestamp_start = (
        3
        if raw_fault in {"measured_window_mismatch", "protocol_anchor_shift"}
        else source_start
    )
    signals = [
        _normalized_signal(
            q_descriptor,
            _array(archive_ref, "q_rad", arrays["q_rad"], "rad"),
            raw_name="raw_position",
            time_base="state",
            raw_count=3,
            timestamp_start=measurement_timestamp_start,
        ),
        _normalized_signal(
            dq_descriptor,
            _array(archive_ref, "dq_rad_s", arrays["dq_rad_s"], "rad/s"),
            raw_name="raw_velocity",
            time_base="state",
            raw_count=3,
            timestamp_start=measurement_timestamp_start,
        ),
        _normalized_signal(
            effort_descriptor,
            _array(
                archive_ref,
                "tau_simulated_Nm",
                arrays["tau_simulated_Nm"],
                "N*m",
            ),
            raw_name="raw_effort",
            time_base="control",
            raw_count=(
                3
                if raw_fault == "bounds"
                else 1
                if raw_fault == "none_cardinality"
                else 2
            ),
            timestamp_start=measurement_timestamp_start,
        ),
        _normalized_signal(
            desired_descriptor,
            _array(
                archive_ref,
                "desired_q_rad",
                arrays["desired_q_rad"],
                "rad",
            ),
            raw_name="raw_desired_position",
            time_base="state",
            raw_count=3,
            raw_start=source_start,
            timestamp_start=reference_timestamp_start,
        ),
        _normalized_signal(
            desired_velocity_descriptor,
            _array(
                archive_ref,
                "desired_dq_rad_s",
                arrays["desired_dq_rad_s"],
                "rad/s",
            ),
            raw_name="raw_desired_velocity",
            time_base="state",
            raw_count=3,
            raw_start=source_start,
            timestamp_start=reference_timestamp_start,
        ),
        _normalized_signal(
            desired_acceleration_descriptor,
            _array(
                archive_ref,
                "desired_ddq_rad_s2",
                arrays["desired_ddq_rad_s2"],
                "rad/s^2",
            ),
            raw_name="raw_desired_acceleration",
            time_base="state",
            raw_count=3,
            raw_start=source_start,
            timestamp_start=reference_timestamp_start,
        ),
    ]
    if raw_fault == "protocol_timing_reference_binding":
        signals[3]["source_binding"]["raw_signal_name"] = (
            "raw_alternate_desired_position"
        )
    if raw_fault == "none_shift":
        signals[2]["resampling"]["applied_time_shift_s"] = 0.001
    if desired_effort_descriptor is not None:
        signals.append(
            _normalized_signal(
                desired_effort_descriptor,
                _array(
                    archive_ref,
                    "desired_tau_feedforward_Nm",
                    arrays["desired_tau_feedforward_Nm"],
                    "N*m",
                ),
                raw_name="raw_desired_effort_feedforward",
                time_base="control",
                raw_count=2,
                raw_start=source_start,
                timestamp_start=reference_timestamp_start,
            )
        )
    if raw_fault == "protocol_anchor_duration":
        for signal in signals:
            binding = signal["source_binding"]
            binding["start_timestamp_ns"] = "0"
            binding["end_timestamp_ns"] = (
                "1000000000" if signal["time_base"] == "state" else "990000000"
            )
    clock_alignments: list[dict[str, Any]] = []
    if raw_fault in _CLOCK_VARIANTS:
        timestamp_offset_ns = (
            1_006_000_000 if raw_fault == "affine_clock_residual" else 1_000_000_000
        )
        for signal in signals:
            binding = signal["source_binding"]
            binding["source_clock_id"] = "source-clock"
            binding["start_timestamp_ns"] = str(
                int(binding["start_timestamp_ns"]) + timestamp_offset_ns
            )
            binding["end_timestamp_ns"] = str(
                int(binding["end_timestamp_ns"]) + timestamp_offset_ns
            )
        selected_timestamps = [
            int(signal["source_binding"][field])
            for signal in signals
            for field in ("start_timestamp_ns", "end_timestamp_ns")
        ]
        valid_start_ns = min(selected_timestamps)
        if raw_fault != "affine_clock_timing_outside_validity":
            valid_start_ns = min(valid_start_ns, 1_000_000_000)
        alignment: dict[str, Any] = {
            "source_clock_id": "source-clock",
            "target_clock_id": "canonical",
            "method": "affine",
            "scale": 1.0,
            "offset_s": -1.0,
            "rms_residual_s": (0.0005 if raw_fault == "affine_clock_residual" else 0.0),
            "max_residual_s": (0.001 if raw_fault == "affine_clock_residual" else 0.0),
            "valid_source_start_ns": str(valid_start_ns),
            "valid_source_end_ns": str(max(selected_timestamps)),
            "extrapolation_allowed": False,
            "evidence": "fixture clock synchronization",
        }
        if raw_fault == "piecewise_clock":
            alignment.pop("scale")
            alignment.pop("offset_s")
            alignment["method"] = "piecewise_affine"
            alignment["mapping"] = _file(root, "clock-mapping.json")
        clock_alignments.append(alignment)
    protocol_interval = {
        "start_index": declared_protocol_interval[0],
        "end_index_exclusive": declared_protocol_interval[1],
    }
    if raw_fault == "protocol_interval_bounds":
        protocol_interval["end_index_exclusive"] = 4
    elif raw_fault == "protocol_interval_duration":
        protocol_interval["end_index_exclusive"] = 2
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/normalized-trajectory@1",
        "artifact_id": f"trajectory-{suffix}",
        "content_sha256": _ZERO_SHA,
        "created_at": _CREATED,
        "source_run": deepcopy(run_reference),
        "protocol": deepcopy(protocol_reference),
        "protocol_segment_id": default_protocol_segment_id,
        "protocol_sample_interval": protocol_interval,
        "protocol_reference_signals": {
            "desired_position": "desired_position_reference",
            "desired_velocity": "desired_velocity_reference",
            "desired_acceleration": "desired_acceleration_reference",
        },
        "lineage_group_id": f"lineage-{suffix}",
        "converter": {
            "software": {"name": "fixture-converter", "version": "1"},
            "configuration": _file(root, "converter.json"),
        },
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "canonical_clock": {
            "clock_id": "canonical",
            "domain": "simulation_clock",
            "epoch": "trajectory start",
            "tick_unit": "ns",
            "resolution_ns": 1,
            "timestamp_source": "normalized grid",
        },
        "clock_alignments": clock_alignments,
        "sample_grid": {"kind": "uniform", "period_ns": _PERIOD_NS},
        "state_time": _array(
            archive_ref,
            "state_time_s",
            arrays["state_time_s"],
            "s",
        ),
        "control_time": _array(
            archive_ref,
            "control_time_s",
            arrays["control_time_s"],
            "s",
        ),
        "initial_state": {
            "position_rad": arrays["q_rad"][0].tolist(),
            "velocity_rad_s": arrays["dq_rad_s"][0].tolist(),
        },
        "signals": signals,
        "quality": {
            "status": quality,
            "report": _file(root, "quality.json", {"status": quality}),
        },
    }
    if desired_effort_descriptor is not None:
        manifest["protocol_reference_signals"]["desired_effort_feedforward"] = (
            "desired_effort_feedforward_reference"
        )
    manifest["content_sha256"] = content_sha256(manifest, arrays)
    path = root / "trajectory.json"
    write_json(path, manifest)
    finalize_bundle(root)
    return path


def _split_member(
    reference: dict[str, Any],
    suffix: str,
    *,
    protocol_id: str,
) -> dict[str, Any]:
    return {
        "trajectory": deepcopy(reference),
        "lineage_group_id": f"lineage-{suffix}",
        "protocol_artifact_id": protocol_id,
        "source_run_artifact_id": f"run-{suffix}",
    }


def _dataset_bundle(
    root: Path,
    *,
    fault: str | None = None,
    seal_root: bool = True,
    catalog_root: Path | None = None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root.mkdir(parents=True)
    catalog = root if catalog_root is None else catalog_root
    catalog.mkdir(parents=True, exist_ok=True)
    protocol_refs: dict[str, dict[str, Any]] = {}
    protocol_variants: dict[str, str] = {}
    protocol_arrays: dict[str, dict[str, np.ndarray]] = {}
    references: dict[str, dict[str, Any]] = {}
    for protocol_suffix in ("fit", "heldout"):
        motion_variant = (
            "two_segments"
            if fault
            in {
                "protocol_interval_relabel",
                "affine_clock_timing_outside_validity",
            }
            and protocol_suffix == "fit"
            else "fit_embedded"
            if (
                fault == "protocol_embedded_interval_leakage"
                and protocol_suffix == "heldout"
            )
            else f"effort_{protocol_suffix}"
            if fault == "protocol_terminal_effort_difference"
            else "negative_zero"
            if fault == "protocol_signed_zero_duplicate"
            and protocol_suffix == "heldout"
            else "fit"
            if fault == "protocol_duplicate_content"
            else protocol_suffix
        )
        protocol_path = _motion_bundle(
            root / f"protocol-{protocol_suffix}",
            protocol_id=f"protocol-{protocol_suffix}",
            analysis_eligible=not (fault == "analysis" and protocol_suffix == "fit"),
            motion_variant=motion_variant,
        )
        protocol_ref = _artifact_reference(catalog, protocol_path)
        protocol_refs[protocol_suffix] = protocol_ref
        protocol_variants[protocol_suffix] = motion_variant
        protocol_arrays[protocol_suffix] = _motion_values(motion_variant)[0]
        references[f"protocol-{protocol_suffix}"] = protocol_ref

    trajectory_refs: dict[str, dict[str, Any]] = {}
    trajectory_protocol_ids: dict[str, str] = {}
    for suffix in ("fit", "heldout", "diagnostic"):
        protocol_suffix = "heldout" if suffix == "heldout" else "fit"
        if fault == "protocol_leakage" and suffix == "heldout":
            protocol_suffix = "fit"
        protocol_ref = protocol_refs[protocol_suffix]
        trajectory_protocol_ids[suffix] = str(protocol_ref["artifact_id"])
        completed = not (fault == "incomplete" and suffix == "heldout")
        run_path = _run_bundle(
            catalog,
            root / f"run-{suffix}",
            suffix=suffix,
            protocol_reference=protocol_ref,
            completed=completed,
            desired_sample_count=protocol_arrays[protocol_suffix]["q_rad"].shape[0],
            include_desired_effort=(
                "tau_feedforward_Nm" in protocol_arrays[protocol_suffix]
            ),
            fault=(
                fault
                if suffix == "fit"
                and fault
                in {
                    "end_effector",
                    "payload",
                    "protocol_validation",
                    "protocol_timing_reference_binding",
                    "protocol_reference_source",
                    *_CLOCK_VARIANTS,
                }
                else None
            ),
        )
        run_ref = _artifact_reference(catalog, run_path)
        references[f"run-{suffix}"] = run_ref
        quality = "warning" if fault == "quality" and suffix == "heldout" else "pass"
        raw_fault = (
            fault
            if suffix == "fit"
            and fault
            in {
                "source",
                "bounds",
                "effort_semantics",
                "none_cardinality",
                "none_shift",
                "protocol_interval_bounds",
                "protocol_interval_duration",
                "protocol_interval_relabel",
                "protocol_velocity_mismatch",
                "measured_window_mismatch",
                "protocol_anchor_shift",
                "protocol_anchor_duration",
                "protocol_timing_reference_binding",
                *_CLOCK_VARIANTS,
            }
            else None
        )
        uses_second_segment = (
            fault
            in {
                "protocol_interval_relabel",
                "affine_clock_timing_outside_validity",
            }
            and suffix == "fit"
        ) or protocol_variants[protocol_suffix] == "fit_embedded"
        declared_interval = (3, 6) if uses_second_segment else (0, 3)
        reference_source_interval = (
            (0, 3)
            if fault == "protocol_interval_relabel" and suffix == "fit"
            else declared_interval
        )
        trajectory_path = _trajectory_bundle(
            root / f"trajectory-{suffix}",
            suffix=suffix,
            run_reference=run_ref,
            protocol_reference=protocol_ref,
            protocol_desired=protocol_arrays[protocol_suffix],
            declared_protocol_interval=declared_interval,
            reference_source_interval=reference_source_interval,
            default_protocol_segment_id=(
                "excitation-b"
                if uses_second_segment
                else "excitation-a"
                if protocol_variants[protocol_suffix]
                in {"two_segments", "fit_embedded"}
                else "excitation"
            ),
            quality=quality,
            raw_fault=raw_fault,
        )
        trajectory_ref = _artifact_reference(catalog, trajectory_path)
        trajectory_refs[suffix] = trajectory_ref
        references[f"trajectory-{suffix}"] = trajectory_ref

    policy_file = _file(root, "split-policy.json")
    metric_file = _file(root, "metrics.json")
    partitions = {
        "fit": [
            _split_member(
                trajectory_refs["fit"],
                "fit",
                protocol_id=trajectory_protocol_ids["fit"],
            )
        ],
        "development": [],
        "held_out_test": [
            _split_member(
                trajectory_refs["heldout"],
                "heldout",
                protocol_id=trajectory_protocol_ids["heldout"],
            )
        ],
        "diagnostic_only": [
            _split_member(
                trajectory_refs["diagnostic"],
                "diagnostic",
                protocol_id=trajectory_protocol_ids["diagnostic"],
            )
        ],
        "excluded": [],
    }
    if fault == "lineage":
        partitions["fit"][0]["lineage_group_id"] = "lineage-wrong"
    if fault == "uncovered":
        partitions["diagnostic_only"] = []
    splits: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/splits@1",
        "artifact_id": "splits-main",
        "content_sha256": _ZERO_SHA,
        "dataset_id": "dataset-main",
        "dataset_version": "1.0.0",
        "created_at": _CREATED,
        "locked_at": "2026-07-23T00:01:00Z",
        "grouping_policy": {
            "group_key": "lineage_group_id",
            "description": "Keep source lineage in one active partition.",
            "configuration": policy_file,
        },
        "held_out_policy": {
            "pre_registered_metrics": metric_file,
            "unseal_condition": "After final model selection.",
        },
        "partitions": partitions,
    }
    splits["content_sha256"] = content_sha256(splits, {})
    splits_path = root / "splits.json"
    write_json(splits_path, splits)
    splits_ref = _artifact_reference(catalog, splits_path, include_bundle=False)
    references["splits"] = splits_ref

    license_ref = _file(root, "LICENSE.dataset.json")
    dataset: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/dataset@1",
        "artifact_id": "dataset-main",
        "content_sha256": _ZERO_SHA,
        "dataset_id": "dataset-main",
        "version": "1.0.0",
        "title": "Catalog fixture",
        "description": "A compact sealed cross-artifact fixture.",
        "created_at": _CREATED,
        "source_kind": "real_robot" if fault == "source_kind" else "simulation",
        "joint_order": list(FER_ARM_JOINT_ORDER),
        "authors": [{"name": "Fixture Author"}],
        "license": {"spdx": "CC0-1.0", "text": license_ref},
        "citation": {"preferred_text": "Fixture Author (2026)."},
        "protocols": [deepcopy(protocol_refs[suffix]) for suffix in ("fit", "heldout")],
        "runs": [
            deepcopy(references[f"run-{suffix}"])
            for suffix in ("fit", "heldout", "diagnostic")
        ],
        "trajectories": [
            deepcopy(trajectory_refs[suffix])
            for suffix in ("fit", "heldout", "diagnostic")
        ],
        "splits": deepcopy(splits_ref),
        "assets": [],
    }
    if fault == "duplicate_locator":
        dataset["runs"][1]["locator"] = deepcopy(dataset["runs"][0]["locator"])
    elif fault == "duplicate_id":
        dataset["trajectories"][1]["artifact_id"] = dataset["trajectories"][0][
            "artifact_id"
        ]
    elif fault == "supersedes":
        dataset["supersedes"] = {
            "artifact_id": "dataset-previous",
            "schema": "fer-mujoco-sysid/dataset@1",
            "manifest_sha256": _ZERO_SHA,
            "locator": {
                "kind": "relative",
                "path": "previous/dataset.json",
            },
        }
    dataset["content_sha256"] = content_sha256(dataset, {})
    write_json(root / "dataset.json", dataset)
    if seal_root:
        finalize_bundle(root)
    return root, references


def _fit_manifest(
    root: Path,
    dataset_root: Path,
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root.mkdir()
    config = _file(root, "fit-config.json")
    dataset_ref = _artifact_reference(
        dataset_root,
        dataset_root / "dataset.json",
    )
    trajectory_refs = [
        deepcopy(references["trajectory-fit"]),
        deepcopy(references["trajectory-heldout"]),
    ]
    manifest: dict[str, Any] = {
        "schema": "fer-mujoco-sysid/fit-result@1",
        "artifact_id": "fit-main",
        "content_sha256": _ZERO_SHA,
        "created_at": _CREATED,
        "status": "completed",
        "toolchain": {
            "project": {"name": "fer-mujoco-sysid", "version": "0.1.0"},
            "python_version": "3.12",
            "mujoco_version": "3.10.0",
            "dependency_lock": config,
            "platform": {
                "operating_system": "Linux",
                "architecture": "x86_64",
                "cpu": "fixture",
                "thread_count": 1,
            },
        },
        "inputs": {
            "source_model": _source(),
            "dataset": dataset_ref,
            "splits": deepcopy(references["splits"]),
            "trajectories": trajectory_refs,
        },
        "data_use": {
            "optimizer": "fit",
            "model_selection": ["fit", "development"],
            "final_evaluation": "held_out_test",
        },
        "forward_model_input": {
            "signal_name": "simulated_effort",
            "semantic_role": "simulated_actuator_effort",
            "selection_rationale": "Use the known simulator actuator effort.",
            "transformation": {
                "scale": 1.0,
                "offset_Nm": 0.0,
                "time_shift_s": 0.0,
            },
        },
        "fit_configuration": {
            "parameters": config,
            "windowing": config,
            "residual": config,
            "pre_registered_metrics": config,
        },
        "optimizer_stages": [
            {
                "stage_id": "friction",
                "backend": "mujoco",
                "parameter_families": ["joint_frictionloss"],
                "seed": 1,
                "max_iterations": 2,
                "configuration": config,
                "status": "completed",
                "termination_reason": "converged",
                "initial_objective": 2.0,
                "final_objective": 1.0,
            }
        ],
        "reproduction": {
            "argv": ["fer-mujoco-sysid", "fit"],
            "working_directory": "workspace",
        },
        "outputs": {
            "identified_parameters": {
                "artifact_id": "parameters-main",
                "schema": "fer-mujoco-sysid/identified-parameters@1",
                "manifest_sha256": "c" * 64,
                "locator": {
                    "kind": "relative",
                    "path": "parameters/parameters.json",
                },
            },
            "metrics": config,
        },
    }
    manifest["content_sha256"] = content_sha256(manifest, {})
    return manifest


def _restamp(manifest: dict[str, Any]) -> None:
    manifest["content_sha256"] = content_sha256(manifest, {})


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _mutate_trajectory(
    dataset: Any,
    trajectory_id: str,
    mutate: Any,
) -> Any:
    record = dataset.trajectories[trajectory_id]
    manifest = _thaw(record.manifest)
    mutate(manifest)
    trajectories = dict(dataset.trajectories)
    trajectories[trajectory_id] = replace(
        record,
        manifest=MappingProxyType(manifest),
    )
    return replace(
        dataset,
        trajectories=MappingProxyType(trajectories),
    )


def test_valid_dataset_closure_and_fit(tmp_path: Path) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")

    dataset = validate_dataset_bundle(root)
    assert set(dataset.protocols) == {"protocol-fit", "protocol-heldout"}
    assert set(dataset.runs) == {"run-fit", "run-heldout", "run-diagnostic"}
    assert dataset.partition_by_trajectory == {
        "trajectory-fit": "fit",
        "trajectory-heldout": "held_out_test",
        "trajectory-diagnostic": "diagnostic_only",
    }
    with pytest.raises(TypeError):
        dataset.runs["other"] = dataset.runs["run-fit"]  # type: ignore[index]
    with pytest.raises(TypeError):
        dataset.runs["run-fit"].manifest["signals"][0]["sample_count"] = 1

    fit_root = tmp_path / "fit"
    fit = _fit_manifest(fit_root, root, references)
    validate_fit_against_dataset(fit, root=fit_root, dataset=dataset)


@pytest.mark.parametrize("clock_variant", ["affine_clock", "affine_clock_residual"])
def test_dataset_accepts_supported_clock_alignment(
    tmp_path: Path,
    clock_variant: str,
) -> None:
    root, _ = _dataset_bundle(
        tmp_path / "dataset",
        fault=clock_variant,
    )

    validate_dataset_bundle(root)


def test_resolved_sealed_fit_is_accepted_without_mutability_leak(
    tmp_path: Path,
) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")
    dataset = validate_dataset_bundle(root)
    fit_root = root / "fit-result"
    fit = _fit_manifest(fit_root, root, references)
    fit_path = fit_root / "result.json"
    write_json(fit_path, fit)
    finalize_bundle(fit_root)

    fit_reference = _artifact_reference(root, fit_path)
    resolved_fit = LocalArtifactResolver(root).resolve(
        fit_reference,
        expected_schema="fer-mujoco-sysid/fit-result@1",
    )

    validate_fit_against_dataset(
        resolved_fit,
        root=resolved_fit.bundle_root,
        dataset=dataset,
    )
    with pytest.raises(TypeError):
        resolved_fit.manifest["status"] = "failed"  # type: ignore[index]


def test_resolved_fit_rejects_a_caller_selected_bundle_root(
    tmp_path: Path,
) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")
    dataset = validate_dataset_bundle(root)
    fit_root = root / "fit-result"
    fit = _fit_manifest(fit_root, root, references)
    fit_path = fit_root / "result.json"
    write_json(fit_path, fit)
    finalize_bundle(fit_root)
    resolved_fit = LocalArtifactResolver(root).resolve(
        _artifact_reference(root, fit_path),
        expected_schema="fer-mujoco-sysid/fit-result@1",
    )

    with pytest.raises(ArtifactValidationError, match="sealed artifact bundle root"):
        validate_fit_against_dataset(
            resolved_fit,
            root=root,
            dataset=dataset,
        )


def test_fit_rejects_nonidentity_structured_input_transformation(
    tmp_path: Path,
) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")
    dataset = validate_dataset_bundle(root)
    fit_root = tmp_path / "fit"
    fit = _fit_manifest(fit_root, root, references)
    fit["forward_model_input"]["transformation"]["scale"] = 2.0
    _restamp(fit)

    with pytest.raises(ArtifactValidationError, match="schema violation"):
        validate_fit_against_dataset(fit, root=fit_root, dataset=dataset)


def test_parent_catalog_root_is_supported(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    dataset_root, _ = _dataset_bundle(
        catalog_root / "dataset",
        catalog_root=catalog_root,
    )

    dataset = validate_dataset_bundle(
        dataset_root,
        catalog_root=catalog_root,
    )

    assert dataset.catalog_root == catalog_root.resolve()


def test_dataset_root_must_be_inside_explicit_catalog_root(
    tmp_path: Path,
) -> None:
    dataset_root, _ = _dataset_bundle(tmp_path / "dataset")
    other_catalog = tmp_path / "other-catalog"
    other_catalog.mkdir()

    with pytest.raises(ArtifactValidationError, match="must be contained"):
        validate_dataset_bundle(dataset_root, catalog_root=other_catalog)


def test_cli_validates_a_real_sealed_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _dataset_bundle(tmp_path / "dataset")

    assert cli.main(["validate-dataset", str(root)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "valid sealed dataset" in captured.out
    assert "protocols (2): protocol-fit, protocol-heldout" in captured.out


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("source_kind", "source_kind"),
        ("duplicate_locator", "duplicate dataset artifact locator"),
        ("duplicate_id", "duplicate dataset artifact ID"),
        ("lineage", "split metadata"),
        ("uncovered", "split coverage"),
        ("analysis", "analysis eligible"),
        ("incomplete", "did not complete"),
        ("quality", "quality status"),
        ("source", "field 'source'"),
        ("effort_semantics", "effort_semantics do not exactly match"),
        ("bounds", "outside sample_count"),
        ("none_cardinality", "raw cardinality"),
        ("none_shift", "nonzero applied time shift"),
        ("protocol_interval_bounds", "lies outside segment"),
        ("protocol_interval_duration", "does not match declared protocol"),
        ("protocol_interval_relabel", "raw bounds"),
        ("protocol_velocity_mismatch", "desired_velocity.*does not match"),
        ("measured_window_mismatch", "source timestamps do not align"),
        ("protocol_anchor_shift", "anchor timestamps do not match"),
        ("protocol_anchor_duration", "anchor timestamps do not match"),
        (
            "protocol_timing_reference_binding",
            "must bind the source run signal named by protocol_reference_signals",
        ),
        ("protocol_reference_source", "desired_velocity.*protocol player"),
        (
            "affine_clock_timing_outside_validity",
            "lies outside its alignment validity interval",
        ),
        ("piecewise_clock", "uses piecewise clock alignment"),
        ("protocol_leakage", "same executable command fingerprint"),
        ("protocol_duplicate_content", "same executable command fingerprint"),
        (
            "protocol_embedded_interval_leakage",
            "same executable command fingerprint",
        ),
        (
            "protocol_terminal_effort_difference",
            "same executable command fingerprint",
        ),
        (
            "protocol_signed_zero_duplicate",
            "same executable command fingerprint",
        ),
        ("end_effector", "end effector does not match protocol"),
        ("payload", "payload does not match protocol"),
        ("protocol_validation", "protocol_validation is not supported"),
        ("supersedes", "dataset.supersedes is not supported"),
    ],
)
def test_dataset_cross_artifact_failures_are_actionable(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    root, _ = _dataset_bundle(tmp_path / "dataset", fault=fault)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_dataset_bundle(root)


def test_unsealed_dataset_requires_explicit_opt_out(tmp_path: Path) -> None:
    root, _ = _dataset_bundle(tmp_path / "dataset", seal_root=False)

    with pytest.raises(ArtifactValidationError, match="missing regular control file"):
        validate_dataset_bundle(root)
    validate_dataset_bundle(root, require_sealed=False)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("manifest", "manifest SHA-256 mismatch"),
        ("bundle", "bundle SHA-256 mismatch"),
        ("external", "external artifact locators are unsupported"),
        ("escape", "escapes its artifact root"),
    ],
)
def test_local_resolver_fails_closed(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")
    reference = deepcopy(references["trajectory-fit"])
    if fault == "manifest":
        reference["manifest_sha256"] = "f" * 64
    elif fault == "bundle":
        reference["bundle_sha256"] = "f" * 64
    elif fault == "external":
        reference["locator"] = {
            "kind": "external",
            "uri": "https://example.invalid/trajectory.json",
        }
    else:
        reference["locator"]["path"] = "../trajectory.json"

    resolver = LocalArtifactResolver(root)
    with pytest.raises(ArtifactValidationError, match=match):
        resolver.resolve(reference)


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("reference_set", "must exactly equal"),
        ("reference_hash", "manifest_sha256"),
        ("missing_signal", "exactly once"),
        ("wrong_role", "has role"),
        ("wrong_time", "control time base"),
        ("unknown", "unknown semantics"),
        ("before_limit", "before rate limiting"),
        ("role_stage", "requires stage/location"),
        ("motor_side", "requires stage/location"),
        ("estimated_external", "requires stage/location"),
        ("incoherent", "incoherent"),
        ("scale", "identity numeric transform"),
        ("offset", "identity numeric transform"),
        ("interpolation", "causal none/zero_order_hold"),
        ("time_shift", "zero applied time shift"),
        ("source_model", "canonical source model"),
    ],
)
def test_fit_effort_selection_failures_are_actionable(
    tmp_path: Path,
    fault: str,
    match: str,
) -> None:
    root, references = _dataset_bundle(tmp_path / "dataset")
    dataset = validate_dataset_bundle(root)
    fit_root = tmp_path / "fit"
    fit = _fit_manifest(fit_root, root, references)

    if fault == "reference_set":
        fit["inputs"]["trajectories"].pop()
        _restamp(fit)
    elif fault == "reference_hash":
        fit["inputs"]["trajectories"][0]["manifest_sha256"] = "f" * 64
        _restamp(fit)
    elif fault == "source_model":
        fit["inputs"]["source_model"]["revision"] = "d" * 40
        fit["inputs"]["source_model"]["sha256"] = "e" * 64
        _restamp(fit)
    else:

        def mutate(manifest: dict[str, Any]) -> None:
            selected = [
                signal
                for signal in manifest["signals"]
                if signal["descriptor"]["name"] == "simulated_effort"
            ][0]
            descriptor = selected["descriptor"]
            semantics = descriptor["effort_semantics"]
            if fault == "missing_signal":
                descriptor["name"] = "renamed_effort"
            elif fault == "wrong_role":
                descriptor["semantic_role"] = "measured_link_effort"
            elif fault == "wrong_time":
                selected["time_base"] = "state"
            elif fault == "unknown":
                semantics["gravity"] = "unknown"
            elif fault == "before_limit":
                semantics["stage"] = "controller"
                semantics["rate_limit_position"] = "before"
            elif fault == "role_stage":
                descriptor["semantic_role"] = "measured_link_effort"
            elif fault == "motor_side":
                semantics["location"] = "motor_side"
            elif fault == "estimated_external":
                semantics["location"] = "estimated_external"
            elif fault == "scale":
                selected["numeric_transform"]["scale"] = 2.0
            elif fault == "offset":
                selected["numeric_transform"]["offset"] = 0.5
            elif fault == "interpolation":
                selected["resampling"]["method"] = "linear"
            elif fault == "time_shift":
                selected["resampling"]["applied_time_shift_s"] = 0.001
            elif fault == "incoherent":
                semantics["gravity"] = "excluded"
            else:  # pragma: no cover - parameterization owns this branch
                raise AssertionError(f"unhandled fault: {fault}")

        dataset = _mutate_trajectory(dataset, "trajectory-fit", mutate)
        if fault == "role_stage":
            fit["forward_model_input"]["semantic_role"] = "measured_link_effort"
            _restamp(fit)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_fit_against_dataset(fit, root=fit_root, dataset=dataset)
