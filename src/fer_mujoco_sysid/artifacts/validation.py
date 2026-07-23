"""Semantic validation beyond what JSON Schema can express."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np

from fer_mujoco_sysid.artifacts.content import content_sha256
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import (
    JsonValue,
    load_numeric_npz,
    resolve_contained_path,
    sha256_file,
)
from fer_mujoco_sysid.artifacts.schemas import validate_schema

FER_ARM_JOINT_ORDER = tuple(f"fer_joint{i}" for i in range(1, 8))

_MOTION_KEYS = {
    "time_from_start": "time_from_start_ns",
    "desired_position": "q_rad",
    "desired_velocity": "dq_rad_s",
    "desired_acceleration": "ddq_rad_s2",
    "desired_effort_feedforward": "tau_feedforward_Nm",
}
_CANONICAL_DTYPES = {"<f8", "<i8", "<i4", "<u8", "<u4", "|b1"}
_STATE_QUANTITY_UNITS = {
    "joint_position": "rad",
    "joint_velocity": "rad/s",
    "joint_acceleration": "rad/s^2",
    "joint_temperature": "K",
    "joint_motor_current": "A",
}


def _error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(message)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    return value


def _require_canonical_joints(manifest: Mapping[str, Any]) -> None:
    joints = manifest.get("joint_order")
    if not isinstance(joints, list) or tuple(joints) != FER_ARM_JOINT_ORDER:
        raise _error(
            f"joint_order must be exactly {list(FER_ARM_JOINT_ORDER)!r}, got {joints!r}"
        )


def _iter_file_references(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
        for child in value.values():
            yield from _iter_file_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_file_references(child)


def _verify_file_references(
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    expected_by_path: dict[str, tuple[str, int]] = {}
    for reference in _iter_file_references(manifest):
        relative = reference.get("path")
        expected_hash = reference.get("sha256")
        expected_size = reference.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not isinstance(expected_size, int)
        ):
            raise _error("file reference has invalid path/hash/size fields")

        expected = (expected_hash, expected_size)
        previous = expected_by_path.setdefault(relative, expected)
        if previous != expected:
            raise _error(f"conflicting file references for {relative!r}")
        if relative in verified:
            continue

        path = resolve_contained_path(root, relative, label="file reference")
        if not path.is_file():
            raise _error(f"referenced artifact file does not exist: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise _error(
                f"size mismatch for {relative}: expected {expected_size}, "
                f"got {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise _error(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        verified[relative] = path
    return verified


def _validate_primitive_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        if not isinstance(name, str) or not name:
            raise _error(f"invalid array name: {name!r}")
        if not isinstance(array, np.ndarray):
            raise _error(f"array {name!r} is not a numpy.ndarray")
        if array.dtype.str not in _CANONICAL_DTYPES:
            raise _error(
                f"array {name!r} must use a canonical little-endian dtype, "
                f"got {array.dtype.str}"
            )
        if array.dtype.itemsize > 1:
            byteorder = array.dtype.byteorder
            if byteorder == ">" or (byteorder == "=" and sys.byteorder != "little"):
                raise _error(f"array {name!r} is not little-endian")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise _error(f"array {name!r} contains NaN or infinity")
        if not array.flags.c_contiguous:
            raise _error(f"array {name!r} must be C-contiguous")


def _numeric_references(
    manifest: Mapping[str, Any],
    *,
    artifact: str,
) -> list[Mapping[str, Any]]:
    if artifact == "motion-protocol":
        arrays = _require_mapping(manifest.get("arrays"), label="arrays")
        return [
            _require_mapping(reference, label=f"arrays.{name}")
            for name, reference in arrays.items()
        ]

    references = [
        _require_mapping(manifest.get("state_time"), label="state_time"),
        _require_mapping(manifest.get("control_time"), label="control_time"),
    ]
    signals = manifest.get("signals")
    if not isinstance(signals, list):
        raise _error("signals must be an array")
    for index, signal in enumerate(signals):
        signal_mapping = _require_mapping(signal, label=f"signals[{index}]")
        references.append(
            _require_mapping(
                signal_mapping.get("array"),
                label=f"signals[{index}].array",
            )
        )
    quality = _require_mapping(manifest.get("quality"), label="quality")
    valid_mask = quality.get("valid_sample_mask")
    if valid_mask is not None:
        references.append(
            _require_mapping(valid_mask, label="quality.valid_sample_mask")
        )
    return references


def _validate_array_references(
    references: list[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    verified_files: Mapping[str, Path],
) -> str:
    archive_paths: set[str] = set()
    keys: set[str] = set()
    for reference in references:
        file_reference = _require_mapping(reference.get("file"), label="array.file")
        file_path = file_reference.get("path")
        key = reference.get("key")
        dtype = reference.get("dtype")
        shape = reference.get("shape")
        if not isinstance(file_path, str) or file_path not in verified_files:
            raise _error(f"unverified numeric archive: {file_path!r}")
        if not isinstance(key, str) or not key:
            raise _error("numeric array reference has no key")
        if key in keys:
            raise _error(f"duplicate numeric array key reference: {key!r}")
        keys.add(key)
        archive_paths.add(file_path)

        if key not in arrays:
            raise _error(f"referenced array is missing: {key!r}")
        array = arrays[key]
        if array.dtype.str != dtype:
            raise _error(
                f"dtype mismatch for {key}: manifest {dtype!r}, "
                f"archive {array.dtype.str!r}"
            )
        if not isinstance(shape, list) or tuple(shape) != array.shape:
            raise _error(
                f"shape mismatch for {key}: manifest {shape!r}, "
                f"archive {list(array.shape)!r}"
            )

    if len(archive_paths) != 1:
        raise _error(
            "all numeric array references must use one artifact-local NPZ archive"
        )
    return next(iter(archive_paths))


def _confirm_loaded_archive(
    archive_path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    loaded = load_numeric_npz(archive_path)
    if set(loaded) != set(arrays):
        raise _error(
            "provided arrays do not have the same keys as the referenced NPZ: "
            f"{sorted(arrays)!r} != {sorted(loaded)!r}"
        )
    for key, expected in loaded.items():
        actual = arrays[key]
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or actual.tobytes(order="C") != expected.tobytes(order="C")
        ):
            raise _error(f"provided array {key!r} does not match the referenced NPZ")


def _schema_manifest(
    manifest: Mapping[str, Any],
    artifact: str,
    schema_dir: str | Path | None,
) -> None:
    validate_schema(
        cast(JsonValue, dict(manifest)),
        artifact,
        schema_dir=schema_dir,
    )


def _validate_content_hash(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    declared = manifest.get("content_sha256")
    actual = content_sha256(manifest, arrays)
    if declared != actual:
        raise _error(
            f"scientific content SHA-256 mismatch: expected {declared}, got {actual}"
        )


def _integer_timestamp(value: Any, *, label: str) -> int:
    if not isinstance(value, str):
        raise _error(f"{label} must be an integer encoded as a string")
    try:
        return int(value)
    except ValueError as exc:
        raise _error(f"{label} is not an integer timestamp: {value!r}") from exc


def _validate_normalized_clocks(manifest: Mapping[str, Any]) -> None:
    canonical_clock = _require_mapping(
        manifest.get("canonical_clock"),
        label="canonical_clock",
    )
    canonical_clock_id = canonical_clock.get("clock_id")
    if not isinstance(canonical_clock_id, str):
        raise _error("canonical_clock.clock_id must be a string")

    alignment_by_source: dict[str, tuple[int, int]] = {}
    raw_alignments = manifest.get("clock_alignments")
    if not isinstance(raw_alignments, list):
        raise _error("clock_alignments must be an array")
    for index, raw_alignment in enumerate(raw_alignments):
        alignment = _require_mapping(
            raw_alignment,
            label=f"clock_alignments[{index}]",
        )
        source_clock_id = alignment.get("source_clock_id")
        target_clock_id = alignment.get("target_clock_id")
        if not isinstance(source_clock_id, str):
            raise _error(f"clock_alignments[{index}].source_clock_id is invalid")
        if target_clock_id != canonical_clock_id:
            raise _error(
                f"clock_alignments[{index}].target_clock_id must name the "
                "canonical clock"
            )
        if source_clock_id == canonical_clock_id:
            raise _error(
                f"clock_alignments[{index}] redundantly maps the canonical clock"
            )
        if source_clock_id in alignment_by_source:
            raise _error(f"duplicate clock alignment for {source_clock_id!r}")

        valid_start = _integer_timestamp(
            alignment.get("valid_source_start_ns"),
            label=f"clock_alignments[{index}].valid_source_start_ns",
        )
        valid_end = _integer_timestamp(
            alignment.get("valid_source_end_ns"),
            label=f"clock_alignments[{index}].valid_source_end_ns",
        )
        if valid_start > valid_end:
            raise _error(f"clock_alignments[{index}] has an inverted validity interval")
        rms = alignment.get("rms_residual_s")
        maximum = alignment.get("max_residual_s")
        if (
            isinstance(rms, (int, float))
            and isinstance(maximum, (int, float))
            and rms > maximum
        ):
            raise _error(
                f"clock_alignments[{index}].rms_residual_s exceeds max_residual_s"
            )
        alignment_by_source[source_clock_id] = (valid_start, valid_end)

    used_source_bounds: dict[str, tuple[int, int]] = {}
    signals = manifest.get("signals")
    if not isinstance(signals, list):
        raise _error("signals must be an array")
    for index, raw_signal in enumerate(signals):
        signal = _require_mapping(raw_signal, label=f"signals[{index}]")
        descriptor = _require_mapping(
            signal.get("descriptor"),
            label=f"signals[{index}].descriptor",
        )
        if descriptor.get("clock_id") != canonical_clock_id:
            raise _error(
                f"signals[{index}].descriptor.clock_id must name the canonical clock"
            )

        binding = _require_mapping(
            signal.get("source_binding"),
            label=f"signals[{index}].source_binding",
        )
        source_clock_id = binding.get("source_clock_id")
        if not isinstance(source_clock_id, str):
            raise _error(f"signals[{index}] has an invalid source clock")
        start_index = binding.get("start_message_index")
        end_index = binding.get("end_message_index_exclusive")
        if (
            not isinstance(start_index, int)
            or not isinstance(end_index, int)
            or start_index >= end_index
        ):
            raise _error(f"signals[{index}] has invalid source message bounds")
        start_timestamp = _integer_timestamp(
            binding.get("start_timestamp_ns"),
            label=f"signals[{index}].source_binding.start_timestamp_ns",
        )
        end_timestamp = _integer_timestamp(
            binding.get("end_timestamp_ns"),
            label=f"signals[{index}].source_binding.end_timestamp_ns",
        )
        if start_timestamp > end_timestamp:
            raise _error(f"signals[{index}] has inverted source timestamps")
        previous = used_source_bounds.get(source_clock_id)
        used_source_bounds[source_clock_id] = (
            start_timestamp if previous is None else min(previous[0], start_timestamp),
            end_timestamp if previous is None else max(previous[1], end_timestamp),
        )

    required_alignment_sources = set(used_source_bounds) - {canonical_clock_id}
    if set(alignment_by_source) != required_alignment_sources:
        missing = sorted(required_alignment_sources - set(alignment_by_source))
        unused = sorted(set(alignment_by_source) - required_alignment_sources)
        raise _error(
            "clock alignments must map every used noncanonical source exactly "
            f"once; missing={missing!r}, unused={unused!r}"
        )
    for source_clock_id in sorted(required_alignment_sources):
        used_start, used_end = used_source_bounds[source_clock_id]
        valid_start, valid_end = alignment_by_source[source_clock_id]
        if valid_start > used_start or valid_end < used_end:
            raise _error(
                f"clock alignment for {source_clock_id!r} does not cover all "
                "selected source samples"
            )


def validate_motion_protocol(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate a motion manifest and its authoritative compiled arrays."""
    _schema_manifest(manifest, "motion-protocol", schema_dir)
    _require_canonical_joints(manifest)
    _validate_primitive_arrays(arrays)

    root_path = Path(root).resolve()
    verified = _verify_file_references(manifest, root_path)
    array_manifest = _require_mapping(manifest["arrays"], label="arrays")
    for logical_name, expected_key in _MOTION_KEYS.items():
        if logical_name not in array_manifest:
            if logical_name == "desired_effort_feedforward":
                continue
            raise _error(f"motion array reference is missing: {logical_name}")
        reference = _require_mapping(
            array_manifest[logical_name],
            label=f"arrays.{logical_name}",
        )
        if reference.get("key") != expected_key:
            raise _error(
                f"arrays.{logical_name}.key must be {expected_key!r}, "
                f"got {reference.get('key')!r}"
            )
    if (
        manifest.get("command_interface") == "joint_effort"
        and "desired_effort_feedforward" not in array_manifest
    ):
        raise _error(
            "joint_effort motion protocols require arrays.desired_effort_feedforward"
        )

    references = _numeric_references(manifest, artifact="motion-protocol")
    archive_relative = _validate_array_references(references, arrays, verified)
    _confirm_loaded_archive(verified[archive_relative], arrays)
    _validate_content_hash(manifest, arrays)

    required_keys = {
        "time_from_start_ns",
        "q_rad",
        "dq_rad_s",
        "ddq_rad_s2",
    }
    if array_manifest.get("desired_effort_feedforward") is not None:
        required_keys.add("tau_feedforward_Nm")
    if set(arrays) != required_keys:
        raise _error(
            f"motion NPZ keys must be exactly {sorted(required_keys)!r}, "
            f"got {sorted(arrays)!r}"
        )

    times = arrays["time_from_start_ns"]
    if times.ndim != 1 or times.dtype.str != "<i8":
        raise _error("time_from_start_ns must have shape (N,) and dtype <i8")
    sample_count = times.shape[0]
    if sample_count < 2:
        raise _error("motion protocol must contain at least two trajectory knots")
    time_values = [int(value) for value in times]
    if time_values[0] != 0 or any(
        current <= previous for previous, current in pairwise(time_values)
    ):
        raise _error("time_from_start_ns must start at zero and increase strictly")
    period_ns = manifest.get("sample_period_ns")
    if not isinstance(period_ns, int) or any(
        current - previous != period_ns for previous, current in pairwise(time_values)
    ):
        raise _error("time_from_start_ns must use the declared exact sample period")

    for key in ("q_rad", "dq_rad_s", "ddq_rad_s2"):
        if arrays[key].shape != (sample_count, 7) or arrays[key].dtype.str != "<f8":
            raise _error(f"{key} must have shape (N, 7) and dtype <f8")
    if "tau_feedforward_Nm" in arrays and (
        arrays["tau_feedforward_Nm"].shape != (sample_count, 7)
        or arrays["tau_feedforward_Nm"].dtype.str != "<f8"
    ):
        raise _error("tau_feedforward_Nm must have shape (N, 7) and dtype <f8")

    for boundary_name, index in (("start_state", 0), ("end_state", -1)):
        boundary = _require_mapping(manifest.get(boundary_name), label=boundary_name)
        if not np.array_equal(
            arrays["q_rad"][index],
            np.asarray(boundary.get("position_rad"), dtype=np.float64),
        ):
            raise _error(f"{boundary_name}.position_rad does not match q_rad")
        if not np.array_equal(
            arrays["dq_rad_s"][index],
            np.asarray(boundary.get("velocity_rad_s"), dtype=np.float64),
        ):
            raise _error(f"{boundary_name}.velocity_rad_s does not match dq_rad_s")

    previous_end = 0
    segment_ids: set[str] = set()
    for segment in cast(list[Mapping[str, Any]], manifest["segments"]):
        segment_id = cast(str, segment["segment_id"])
        start = cast(int, segment["start_index"])
        end = cast(int, segment["end_index_exclusive"])
        if segment_id in segment_ids:
            raise _error(f"duplicate motion segment ID: {segment_id!r}")
        segment_ids.add(segment_id)
        if start < previous_end or start >= end or end > sample_count:
            raise _error(f"invalid or overlapping motion segment: {segment_id!r}")
        previous_end = end


def validate_normalized_trajectory(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    root: str | Path,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate one fixed-grid normalized M-transition FER trajectory."""
    _schema_manifest(manifest, "normalized-trajectory", schema_dir)
    _require_canonical_joints(manifest)
    _validate_primitive_arrays(arrays)
    _validate_normalized_clocks(manifest)

    root_path = Path(root).resolve()
    verified = _verify_file_references(manifest, root_path)
    references = _numeric_references(manifest, artifact="normalized-trajectory")
    archive_relative = _validate_array_references(references, arrays, verified)
    _confirm_loaded_archive(verified[archive_relative], arrays)
    _validate_content_hash(manifest, arrays)

    state_time_reference = _require_mapping(
        manifest["state_time"],
        label="state_time",
    )
    if state_time_reference.get("key") != "state_time_s":
        raise _error("normalized state_time.key must be 'state_time_s'")
    control_time_reference = _require_mapping(
        manifest["control_time"],
        label="control_time",
    )
    if control_time_reference.get("key") != "control_time_s":
        raise _error("normalized control_time.key must be 'control_time_s'")
    required_core = {"state_time_s", "control_time_s", "q_rad", "dq_rad_s"}
    if not required_core.issubset(arrays):
        raise _error(
            f"normalized NPZ is missing core arrays: "
            f"{sorted(required_core - set(arrays))!r}"
        )

    state_time = arrays["state_time_s"]
    control_time = arrays["control_time_s"]
    if state_time.ndim != 1 or state_time.dtype.str != "<f8":
        raise _error("state_time_s must have shape (M + 1,) and dtype <f8")
    if control_time.ndim != 1 or control_time.dtype.str != "<f8":
        raise _error("control_time_s must have shape (M,) and dtype <f8")
    transitions = control_time.shape[0]
    if transitions < 1 or state_time.shape != (transitions + 1,):
        raise _error("state_time_s/control_time_s must follow the M+1/M convention")
    if state_time[0] != 0.0 or control_time[0] != 0.0:
        raise _error("normalized state and control times must start exactly at zero")
    if not np.array_equal(control_time, state_time[:-1]):
        raise _error("control_time_s must equal state_time_s[:-1] exactly")

    sample_grid = _require_mapping(manifest.get("sample_grid"), label="sample_grid")
    if sample_grid.get("kind") != "uniform":
        raise _error("normalized trajectory must declare a uniform sample grid")
    period_ns = sample_grid.get("period_ns")
    if (
        not isinstance(period_ns, int)
        or period_ns <= 0
        or period_ns > np.iinfo(np.int64).max
    ):
        raise _error("uniform sample grid requires a positive int64 period_ns")
    expected_state_time = np.arange(transitions + 1, dtype=np.float64) * (
        period_ns * 1e-9
    )
    if not np.array_equal(state_time, expected_state_time):
        raise _error("state_time_s does not follow the declared exact fixed grid")

    q = arrays["q_rad"]
    dq = arrays["dq_rad_s"]
    expected_state_shape = (transitions + 1, 7)
    if q.shape != expected_state_shape or q.dtype.str != "<f8":
        raise _error("q_rad must have shape (M + 1, 7) and dtype <f8")
    if dq.shape != expected_state_shape or dq.dtype.str != "<f8":
        raise _error("dq_rad_s must have shape (M + 1, 7) and dtype <f8")

    signal_keys: set[str] = set()
    signal_names: set[str] = set()
    q_signals = 0
    dq_signals = 0
    control_effort_signals = 0
    for index, raw_signal in enumerate(cast(list[Any], manifest["signals"])):
        signal = _require_mapping(raw_signal, label=f"signals[{index}]")
        descriptor = _require_mapping(
            signal.get("descriptor"),
            label=f"signals[{index}].descriptor",
        )
        reference = _require_mapping(
            signal.get("array"),
            label=f"signals[{index}].array",
        )
        key = cast(str, reference["key"])
        signal_keys.add(key)
        signal_name = descriptor.get("name")
        if not isinstance(signal_name, str) or signal_name in signal_names:
            raise _error(f"signals[{index}] has a missing or duplicate signal name")
        signal_names.add(signal_name)
        if descriptor.get("joint_order") != list(FER_ARM_JOINT_ORDER):
            raise _error(f"signals[{index}] does not use canonical FER joint order")

        quantity = descriptor.get("quantity")
        sample_count = descriptor.get("sample_count")
        unit = descriptor.get("unit")
        if reference.get("unit") != unit:
            raise _error(
                f"signals[{index}] descriptor and array reference units disagree"
            )
        time_base = signal.get("time_base")
        if time_base == "state":
            expected_samples = transitions + 1
        elif time_base == "control":
            expected_samples = transitions
        else:
            raise _error(f"signals[{index}] has an invalid time_base")
        array = arrays[key]
        if array.shape != (expected_samples, 7) or array.dtype.str != "<f8":
            raise _error(
                f"signal array {key!r} must have shape "
                f"({expected_samples}, 7) for the {time_base} time base "
                "and dtype <f8"
            )
        if sample_count != expected_samples:
            raise _error(
                f"signals[{index}].descriptor.sample_count must be "
                f"{expected_samples} for the {time_base} time base"
            )

        if key == "q_rad":
            q_signals += 1
            if (
                quantity != "joint_position"
                or descriptor.get("semantic_role") != "measured_joint_position"
                or unit != "rad"
                or time_base != "state"
            ):
                raise _error(
                    "q_rad must be measured joint position in rad on the "
                    "state time base"
                )
        elif key == "dq_rad_s":
            dq_signals += 1
            if (
                quantity != "joint_velocity"
                or descriptor.get("semantic_role") != "measured_joint_velocity"
                or unit != "rad/s"
                or time_base != "state"
            ):
                raise _error(
                    "dq_rad_s must be measured joint velocity in rad/s on the "
                    "state time base"
                )
        elif quantity == "joint_effort":
            if unit != "N*m":
                raise _error(f"declared effort array {key!r} must use N*m")
            if time_base == "control":
                resampling = _require_mapping(
                    signal.get("resampling"),
                    label=f"signals[{index}].resampling",
                )
                if resampling.get("method") not in {"none", "zero_order_hold"}:
                    raise _error(
                        f"control-effort signal {key!r} must be native-aligned "
                        "or causally resampled with zero_order_hold"
                    )
                control_effort_signals += 1
        else:
            expected_unit = _STATE_QUANTITY_UNITS.get(cast(str, quantity))
            if expected_unit is None or unit != expected_unit:
                raise _error(f"signals[{index}] has an inconsistent quantity/unit pair")

    if q_signals != 1 or dq_signals != 1:
        raise _error("signals must declare q_rad and dq_rad_s exactly once")
    if control_effort_signals < 1:
        raise _error(
            "signals must declare at least one joint-effort candidate on the "
            "control time base"
        )

    initial_state = _require_mapping(
        manifest.get("initial_state"),
        label="initial_state",
    )
    if not np.array_equal(
        q[0],
        np.asarray(initial_state.get("position_rad"), dtype=np.float64),
    ):
        raise _error("initial_state.position_rad does not match q_rad[0]")
    if not np.array_equal(
        dq[0],
        np.asarray(initial_state.get("velocity_rad_s"), dtype=np.float64),
    ):
        raise _error("initial_state.velocity_rad_s does not match dq_rad_s[0]")

    quality = _require_mapping(manifest.get("quality"), label="quality")
    mask_reference = quality.get("valid_sample_mask")
    declared_keys = {"state_time_s", "control_time_s"} | signal_keys
    if mask_reference is not None:
        mask = _require_mapping(mask_reference, label="quality.valid_sample_mask")
        mask_key = cast(str, mask["key"])
        declared_keys.add(mask_key)
        if arrays[mask_key].shape != (transitions + 1,):
            raise _error("valid_sample_mask must have shape (M + 1,)")
        if arrays[mask_key].dtype.str != "|b1":
            raise _error("valid_sample_mask must have dtype |b1")
        if not np.all(arrays[mask_key]):
            raise _error("normalized trajectories may not contain invalid samples")

    if set(arrays) != declared_keys:
        raise _error(
            f"normalized NPZ contains undeclared or missing keys: "
            f"expected {sorted(declared_keys)!r}, got {sorted(arrays)!r}"
        )
