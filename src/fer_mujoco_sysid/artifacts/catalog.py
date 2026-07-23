"""Fail-closed resolution and cross-artifact dataset validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from fer_mujoco_sysid.artifacts.bundle import (
    CHECKSUM_MANIFEST_NAME,
    SEAL_NAME,
    verify_sealed_bundle,
)
from fer_mujoco_sysid.artifacts.content import content_sha256
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import (
    JsonValue,
    load_json,
    load_numeric_npz,
    resolve_contained_path,
    sha256_file,
)
from fer_mujoco_sysid.artifacts.schemas import validate_schema
from fer_mujoco_sysid.artifacts.semantic import (
    validate_acquisition_run,
    validate_fit_result,
    validate_identified_parameters,
    validate_splits,
)
from fer_mujoco_sysid.artifacts.validation import (
    FER_ARM_JOINT_ORDER,
    _validate_content_hash,
    _verify_file_references,
    validate_motion_protocol,
    validate_normalized_trajectory,
)

_SCHEMA_TO_ARTIFACT = {
    "fer-mujoco-sysid/acquisition-run@1": "acquisition-run",
    "fer-mujoco-sysid/dataset@1": "dataset",
    "fer-mujoco-sysid/fit-result@1": "fit-result",
    "fer-mujoco-sysid/identified-parameters@1": "identified-parameters",
    "fer-mujoco-sysid/motion-protocol@1": "motion-protocol",
    "fer-mujoco-sysid/normalized-trajectory@1": "normalized-trajectory",
    "fer-mujoco-sysid/splits@1": "splits",
}
_SIMULATION_BACKENDS = frozenset({"standalone_mujoco", "ros_mujoco"})
_SCIENTIFIC_PARTITIONS = frozenset({"fit", "development", "held_out_test"})
_REFERENCE_KEYS = frozenset(
    {
        "artifact_id",
        "schema",
        "manifest_sha256",
        "bundle_sha256",
        "locator",
    }
)
_REQUIRED_REFERENCE_KEYS = frozenset(
    {"artifact_id", "schema", "manifest_sha256", "locator"}
)
_RAW_DESCRIPTOR_FIELDS = (
    "quantity",
    "semantic_role",
    "unit",
    "joint_order",
    "positive_direction",
    "source",
)
_SELECTED_EFFORT_IDENTITY = {
    "controller_effort_command": ("controller", "generalized_coordinate"),
    "hardware_desired_link_effort": ("hardware", "link_side"),
    "measured_link_effort": ("sensor", "link_side"),
    "simulated_actuator_effort": ("simulator", "generalized_coordinate"),
}


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """One validated, recursively read-only manifest and its byte identities."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    bundle_root: Path
    manifest_sha256: str
    bundle_sha256: str | None


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    """Validated closure whose manifests and lookup indexes are read-only."""

    dataset: ResolvedArtifact
    splits: ResolvedArtifact
    protocols: Mapping[str, ResolvedArtifact]
    runs: Mapping[str, ResolvedArtifact]
    trajectories: Mapping[str, ResolvedArtifact]
    partition_by_trajectory: Mapping[str, str]
    catalog_root: Path


def _error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be a JSON object")
    return value


def _strict_manifest(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise _error(f"artifact manifest must contain a JSON object: {path}")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_thaw_json(child) for child in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reference(
    value: Any,
    *,
    label: str,
    expected_schema: str | None = None,
) -> Mapping[str, Any]:
    reference = _mapping(value, label=label)
    keys = set(reference)
    if not _REQUIRED_REFERENCE_KEYS.issubset(keys) or not keys.issubset(
        _REFERENCE_KEYS
    ):
        missing = sorted(_REQUIRED_REFERENCE_KEYS - keys)
        unknown = sorted(keys - _REFERENCE_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise _error(f"{label} has invalid fields ({'; '.join(details)})")

    artifact_id = reference["artifact_id"]
    schema = reference["schema"]
    if not isinstance(artifact_id, str) or not artifact_id:
        raise _error(f"{label}.artifact_id must be a non-empty string")
    if not isinstance(schema, str) or schema not in _SCHEMA_TO_ARTIFACT:
        raise _error(
            f"{label}.schema {schema!r} is unsupported; migrate the artifact "
            "to a supported schema version"
        )
    if expected_schema is not None and schema != expected_schema:
        raise _error(f"{label}.schema must be {expected_schema!r}, got {schema!r}")
    _digest(reference["manifest_sha256"], label=f"{label}.manifest_sha256")
    if "bundle_sha256" in reference:
        _digest(reference["bundle_sha256"], label=f"{label}.bundle_sha256")

    locator = _mapping(reference["locator"], label=f"{label}.locator")
    kind = locator.get("kind")
    if kind == "relative":
        if set(locator) != {"kind", "path"} or not isinstance(locator.get("path"), str):
            raise _error(f"{label}.locator is not a strict relative locator")
    elif kind == "external":
        if set(locator) != {"kind", "uri"} or not isinstance(locator.get("uri"), str):
            raise _error(f"{label}.locator is not a strict external locator")
    else:
        raise _error(f"{label}.locator has unsupported kind {kind!r}")
    return reference


def _bundle_digest(bundle_root: Path, *, required: bool) -> str | None:
    checksum_path = bundle_root / CHECKSUM_MANIFEST_NAME
    seal_path = bundle_root / SEAL_NAME
    if required:
        verify_sealed_bundle(bundle_root)
        return sha256_file(checksum_path)
    if checksum_path.exists() or seal_path.exists():
        verify_sealed_bundle(bundle_root)
        return sha256_file(checksum_path)
    return None


def _numeric_archive_path(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    schema: str,
) -> Path:
    raw_references: list[Any]
    if schema == "fer-mujoco-sysid/motion-protocol@1":
        arrays = _mapping(manifest.get("arrays"), label="arrays")
        raw_references = list(arrays.values())
    else:
        raw_references = [manifest.get("state_time"), manifest.get("control_time")]
        signals = manifest.get("signals")
        if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
            raise _error("signals must be an array")
        raw_references.extend(
            _mapping(signal, label=f"signals[{index}]").get("array")
            for index, signal in enumerate(signals)
        )
        quality = _mapping(manifest.get("quality"), label="quality")
        if "valid_sample_mask" in quality:
            raw_references.append(quality["valid_sample_mask"])

    archive_paths: set[str] = set()
    for index, raw_reference in enumerate(raw_references):
        reference = _mapping(raw_reference, label=f"numeric reference {index}")
        file_reference = _mapping(
            reference.get("file"),
            label=f"numeric reference {index}.file",
        )
        path = file_reference.get("path")
        if not isinstance(path, str):
            raise _error(f"numeric reference {index}.file.path must be a string")
        archive_paths.add(path)
    if len(archive_paths) != 1:
        raise _error(
            "all numeric array references must use one artifact-local NPZ archive"
        )
    return resolve_contained_path(
        root,
        archive_paths.pop(),
        label="numeric archive",
    )


def _load_protocol_arrays(protocol: ResolvedArtifact) -> dict[str, np.ndarray]:
    return load_numeric_npz(
        _numeric_archive_path(
            protocol.manifest,
            root=protocol.bundle_root,
            schema="fer-mujoco-sysid/motion-protocol@1",
        )
    )


def _protocol_interval_command_fingerprint(
    protocol: ResolvedArtifact,
    arrays: Mapping[str, np.ndarray],
    *,
    start_index: int,
    end_index_exclusive: int,
) -> str:
    """Fingerprint one executable interval independently of storage identity."""

    manifest = protocol.manifest
    raw_array_contracts = cast(
        Mapping[str, Mapping[str, Any]],
        manifest["arrays"],
    )
    logical_names = [
        "desired_position",
        "desired_velocity",
        "desired_acceleration",
    ]
    if "desired_effort_feedforward" in raw_array_contracts:
        logical_names.append("desired_effort_feedforward")

    array_contracts: dict[str, Any] = {}
    selected_arrays: dict[str, np.ndarray] = {}
    for logical_name in logical_names:
        reference = raw_array_contracts[logical_name]
        key = cast(str, reference["key"])
        stop = (
            end_index_exclusive - 1
            if logical_name == "desired_effort_feedforward"
            else end_index_exclusive
        )
        selected = np.array(
            arrays[key][start_index:stop],
            dtype=arrays[key].dtype,
            order="C",
            copy=True,
        )
        if selected.dtype.kind == "f":
            selected[selected == 0.0] = 0.0
        selected_arrays[key] = selected
        array_contracts[logical_name] = {
            "key": reference["key"],
            "dtype": reference["dtype"],
            "shape": list(selected.shape),
            "unit": reference["unit"],
        }

    execution_contract = {
        "joint_order": list(manifest["joint_order"]),
        "command_interface": manifest["command_interface"],
        "sample_period_ns": manifest["sample_period_ns"],
        "arrays": array_contracts,
    }
    return content_sha256(execution_contract, selected_arrays)


def _validate_intrinsic(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    schema: str,
) -> None:
    if schema == "fer-mujoco-sysid/motion-protocol@1":
        arrays = load_numeric_npz(
            _numeric_archive_path(manifest, root=root, schema=schema)
        )
        validate_motion_protocol(manifest, arrays, root=root)
    elif schema == "fer-mujoco-sysid/normalized-trajectory@1":
        arrays = load_numeric_npz(
            _numeric_archive_path(manifest, root=root, schema=schema)
        )
        validate_normalized_trajectory(manifest, arrays, root=root)
    elif schema == "fer-mujoco-sysid/acquisition-run@1":
        validate_acquisition_run(manifest, root=root)
    elif schema == "fer-mujoco-sysid/identified-parameters@1":
        validate_identified_parameters(manifest, root=root)
    elif schema == "fer-mujoco-sysid/splits@1":
        validate_splits(manifest, root=root)
    elif schema == "fer-mujoco-sysid/fit-result@1":
        validate_fit_result(manifest, root=root)
    elif schema == "fer-mujoco-sysid/dataset@1":
        validate_schema(cast(JsonValue, dict(manifest)), "dataset")
        _verify_file_references(manifest, root)
        _validate_content_hash(manifest, {})
    else:
        raise _error(
            f"unsupported artifact schema {schema!r}; migrate the artifact "
            "to a supported schema version"
        )


class LocalArtifactResolver:
    """Resolve only local catalog-relative manifests, without network fallback."""

    def __init__(
        self,
        catalog_root: str | Path,
        *,
        require_sealed: bool = True,
    ) -> None:
        root = Path(catalog_root).resolve()
        if not root.is_dir():
            raise _error(f"catalog root is not a directory: {root}")
        self._catalog_root = root
        self._require_sealed = require_sealed

    @property
    def catalog_root(self) -> Path:
        return self._catalog_root

    def resolve(
        self,
        reference: Mapping[str, Any],
        *,
        expected_schema: str | None = None,
    ) -> ResolvedArtifact:
        """Resolve and intrinsically validate one exact artifact reference."""

        checked = _reference(
            reference,
            label="artifact reference",
            expected_schema=expected_schema,
        )
        locator = cast(Mapping[str, Any], checked["locator"])
        if locator["kind"] == "external":
            raise _error(
                "external artifact locators are unsupported by "
                "LocalArtifactResolver; materialize and seal the artifact "
                "under the explicit catalog root, then use a relative locator"
            )

        manifest_path = resolve_contained_path(
            self._catalog_root,
            cast(str, locator["path"]),
            label="artifact locator",
        )
        if not manifest_path.is_file():
            raise _error(f"artifact manifest does not exist: {manifest_path}")
        actual_manifest_sha256 = sha256_file(manifest_path)
        expected_manifest_sha256 = cast(str, checked["manifest_sha256"])
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise _error(
                f"manifest SHA-256 mismatch for {checked['artifact_id']!r}: "
                f"expected {expected_manifest_sha256}, "
                f"got {actual_manifest_sha256}"
            )

        manifest = _strict_manifest(manifest_path)
        expected_artifact_id = checked["artifact_id"]
        if manifest.get("artifact_id") != expected_artifact_id:
            raise _error(
                f"resolved artifact ID mismatch: expected "
                f"{expected_artifact_id!r}, got {manifest.get('artifact_id')!r}"
            )
        expected_manifest_schema = cast(str, checked["schema"])
        if manifest.get("schema") != expected_manifest_schema:
            raise _error(
                f"resolved schema discriminator mismatch for "
                f"{expected_artifact_id!r}: expected "
                f"{expected_manifest_schema!r}, got {manifest.get('schema')!r}"
            )

        bundle_root = manifest_path.parent
        expected_bundle_sha256 = checked.get("bundle_sha256")
        bundle_sha256 = _bundle_digest(
            bundle_root,
            required=self._require_sealed or expected_bundle_sha256 is not None,
        )
        if (
            expected_bundle_sha256 is not None
            and bundle_sha256 != expected_bundle_sha256
        ):
            raise _error(
                f"bundle SHA-256 mismatch for {expected_artifact_id!r}: "
                f"expected {expected_bundle_sha256}, got {bundle_sha256}"
            )

        _validate_intrinsic(
            manifest,
            root=bundle_root,
            schema=expected_manifest_schema,
        )
        return ResolvedArtifact(
            manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
            manifest_path=manifest_path,
            bundle_root=bundle_root,
            manifest_sha256=actual_manifest_sha256,
            bundle_sha256=bundle_sha256,
        )


def _direct_artifact(
    manifest_path: Path,
    *,
    expected_schema: str,
    require_sealed: bool,
) -> ResolvedArtifact:
    if not manifest_path.is_file():
        raise _error(f"required artifact manifest does not exist: {manifest_path}")
    manifest = _strict_manifest(manifest_path)
    if manifest.get("schema") != expected_schema:
        raise _error(
            f"{manifest_path.name} must use schema {expected_schema!r}, "
            f"got {manifest.get('schema')!r}"
        )
    bundle_root = manifest_path.parent
    bundle_sha256 = _bundle_digest(bundle_root, required=require_sealed)
    _validate_intrinsic(manifest, root=bundle_root, schema=expected_schema)
    return ResolvedArtifact(
        manifest=cast(Mapping[str, Any], _freeze_json(manifest)),
        manifest_path=manifest_path,
        bundle_root=bundle_root,
        manifest_sha256=sha256_file(manifest_path),
        bundle_sha256=bundle_sha256,
    )


def _locator_path(reference: Mapping[str, Any], *, label: str) -> str:
    checked = _reference(reference, label=label)
    locator = cast(Mapping[str, Any], checked["locator"])
    if locator["kind"] == "external":
        raise _error(
            f"{label} uses an external artifact locator, which is unsupported; "
            "materialize and seal it under the local catalog root"
        )
    return cast(str, locator["path"])


def _resolve_collection(
    resolver: LocalArtifactResolver,
    raw_references: Any,
    *,
    label: str,
    expected_schema: str,
) -> tuple[dict[str, ResolvedArtifact], dict[str, Mapping[str, Any]]]:
    if not isinstance(raw_references, Sequence) or isinstance(
        raw_references, (str, bytes)
    ):
        raise _error(f"{label} must be an array")
    resolved: dict[str, ResolvedArtifact] = {}
    references: dict[str, Mapping[str, Any]] = {}
    for index, raw_reference in enumerate(raw_references):
        reference = _reference(
            raw_reference,
            label=f"{label}[{index}]",
            expected_schema=expected_schema,
        )
        artifact_id = cast(str, reference["artifact_id"])
        if artifact_id in references:
            raise _error(f"duplicate artifact ID in {label}: {artifact_id!r}")
        references[artifact_id] = reference
        resolved[artifact_id] = resolver.resolve(
            reference,
            expected_schema=expected_schema,
        )
    if set(resolved) != set(references):
        raise _error(f"{label} references and resolved artifacts differ")
    return resolved, references


def _same_reference(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    checked = _reference(actual, label=label)
    if dict(checked) != dict(expected):
        raise _error(f"{label} does not exactly match the dataset catalog reference")


def _split_member(entry: Mapping[str, Any], partition: str) -> Mapping[str, Any]:
    if partition == "excluded":
        return _mapping(entry.get("member"), label="excluded split member")
    return entry


def _expected_source_kind(runs: Mapping[str, ResolvedArtifact]) -> str:
    backends = {run.manifest["backend"] for run in runs.values()}
    has_real = "agimus_fer" in backends
    has_simulation = bool(backends & _SIMULATION_BACKENDS)
    if has_real and has_simulation:
        return "mixed"
    if has_real:
        return "real_robot"
    if has_simulation:
        return "simulation"
    raise _error(f"dataset runs contain unsupported backends: {sorted(backends)!r}")


def _validate_raw_bindings(
    trajectory_id: str,
    trajectory: Mapping[str, Any],
    run: Mapping[str, Any],
) -> None:
    raw_by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw_signal in enumerate(cast(Sequence[Any], run["signals"])):
        descriptor = _mapping(
            raw_signal,
            label=f"run signal {trajectory_id}[{index}]",
        )
        name = cast(str, descriptor["name"])
        if name in raw_by_name:
            raise _error(
                f"source run for trajectory {trajectory_id!r} has duplicate "
                f"raw signal name {name!r}"
            )
        raw_by_name[name] = descriptor

    for index, raw_signal in enumerate(cast(Sequence[Any], trajectory["signals"])):
        signal = _mapping(
            raw_signal,
            label=f"trajectory {trajectory_id!r} signal[{index}]",
        )
        descriptor = _mapping(
            signal["descriptor"],
            label=f"trajectory {trajectory_id!r} signal[{index}].descriptor",
        )
        binding = _mapping(
            signal["source_binding"],
            label=f"trajectory {trajectory_id!r} signal[{index}].source_binding",
        )
        raw_name = cast(str, binding["raw_signal_name"])
        raw_descriptor = raw_by_name.get(raw_name)
        if raw_descriptor is None:
            raise _error(
                f"trajectory {trajectory_id!r} signal "
                f"{descriptor['name']!r} binds unknown raw signal {raw_name!r}"
            )
        start = cast(int, binding["start_message_index"])
        end = cast(int, binding["end_message_index_exclusive"])
        sample_count = cast(int, raw_descriptor["sample_count"])
        if not 0 <= start < end <= sample_count:
            raise _error(
                f"trajectory {trajectory_id!r} signal "
                f"{descriptor['name']!r} has raw bounds [{start}, {end}) "
                f"outside sample_count {sample_count}"
            )
        if binding["source_clock_id"] != raw_descriptor["clock_id"]:
            raise _error(
                f"trajectory {trajectory_id!r} signal "
                f"{descriptor['name']!r} source clock does not match raw signal"
            )
        for field in _RAW_DESCRIPTOR_FIELDS:
            if descriptor[field] != raw_descriptor[field]:
                raise _error(
                    f"trajectory {trajectory_id!r} signal "
                    f"{descriptor['name']!r} field {field!r} does not match "
                    f"raw signal {raw_name!r}"
                )
        if (
            descriptor["quantity"] == "joint_effort"
            and descriptor["effort_semantics"] != raw_descriptor["effort_semantics"]
        ):
            raise _error(
                f"trajectory {trajectory_id!r} effort signal "
                f"{descriptor['name']!r} effort_semantics do not exactly "
                f"match raw signal {raw_name!r}"
            )
        resampling = _mapping(
            signal["resampling"],
            label=f"trajectory {trajectory_id!r} signal resampling",
        )
        if resampling["method"] == "none":
            normalized_count = cast(Sequence[Any], signal["array"]["shape"])[0]
            if end - start != normalized_count:
                raise _error(
                    f"trajectory {trajectory_id!r} signal "
                    f"{descriptor['name']!r} uses resampling 'none' but raw "
                    f"cardinality {end - start} differs from normalized "
                    f"cardinality {normalized_count}"
                )
            if resampling["applied_time_shift_s"] != 0.0:
                raise _error(
                    f"trajectory {trajectory_id!r} signal "
                    f"{descriptor['name']!r} uses resampling 'none' with a "
                    "nonzero applied time shift"
                )


def _timestamp_in_canonical_seconds(
    trajectory_id: str,
    trajectory: Mapping[str, Any],
    *,
    source_clock_id: str,
    timestamp_ns: int,
) -> tuple[float, float]:
    canonical_clock_id = cast(str, trajectory["canonical_clock"]["clock_id"])
    timestamp_s = timestamp_ns * 1e-9
    if source_clock_id == canonical_clock_id:
        return timestamp_s, 0.0

    alignments = [
        alignment
        for alignment in cast(
            Sequence[Mapping[str, Any]],
            trajectory["clock_alignments"],
        )
        if alignment["source_clock_id"] == source_clock_id
    ]
    if len(alignments) != 1:
        raise _error(
            f"trajectory {trajectory_id!r} source clock "
            f"{source_clock_id!r} must have exactly one canonical alignment"
        )
    alignment = alignments[0]
    valid_start_ns = int(alignment["valid_source_start_ns"])
    valid_end_ns = int(alignment["valid_source_end_ns"])
    if not valid_start_ns <= timestamp_ns <= valid_end_ns:
        raise _error(
            f"trajectory {trajectory_id!r} timestamp {timestamp_ns} on source "
            f"clock {source_clock_id!r} lies outside its alignment validity "
            f"interval [{valid_start_ns}, {valid_end_ns}]"
        )
    method = alignment["method"]
    if method == "identity":
        return timestamp_s, 0.0
    if method == "affine":
        scale = cast(float, alignment["scale"])
        offset_s = cast(float, alignment["offset_s"])
        return (
            scale * timestamp_s + offset_s,
            cast(float, alignment["max_residual_s"]),
        )
    raise _error(
        f"trajectory {trajectory_id!r} uses piecewise clock alignment for "
        f"{source_clock_id!r}; execution-window validation requires identity "
        "or affine alignment"
    )


def _binding_in_canonical_seconds(
    trajectory_id: str,
    trajectory: Mapping[str, Any],
    signal: Mapping[str, Any],
) -> tuple[float, float, float]:
    binding = _mapping(
        signal["source_binding"],
        label=f"trajectory {trajectory_id!r} signal source binding",
    )
    source_clock_id = cast(str, binding["source_clock_id"])
    start_s, start_error = _timestamp_in_canonical_seconds(
        trajectory_id,
        trajectory,
        source_clock_id=source_clock_id,
        timestamp_ns=int(binding["start_timestamp_ns"]),
    )
    end_s, end_error = _timestamp_in_canonical_seconds(
        trajectory_id,
        trajectory,
        source_clock_id=source_clock_id,
        timestamp_ns=int(binding["end_timestamp_ns"]),
    )
    return start_s, end_s, max(start_error, end_error)


def _validate_cross_signal_window(
    trajectory_id: str,
    trajectory: Mapping[str, Any],
    *,
    protocol_position_signal_name: str,
    protocol_timing: Mapping[str, Any],
    interval_start: int,
    interval_end: int,
    period_ns: int,
) -> None:
    signals = cast(Sequence[Mapping[str, Any]], trajectory["signals"])
    by_name = {cast(str, signal["descriptor"]["name"]): signal for signal in signals}
    anchor = by_name[protocol_position_signal_name]
    anchor_start, anchor_end, anchor_error = _binding_in_canonical_seconds(
        trajectory_id,
        trajectory,
        anchor,
    )
    period_s = period_ns * 1e-9
    protocol_start_s, protocol_timing_error = _timestamp_in_canonical_seconds(
        trajectory_id,
        trajectory,
        source_clock_id=cast(str, protocol_timing["clock_id"]),
        timestamp_ns=int(protocol_timing["protocol_start_timestamp_ns"]),
    )
    expected_anchor_start = protocol_start_s + interval_start * period_s
    expected_anchor_end = protocol_start_s + (interval_end - 1) * period_s
    anchor_tolerance_s = 0.5 * period_s + protocol_timing_error + anchor_error
    if not math.isclose(
        anchor_start,
        expected_anchor_start,
        rel_tol=0.0,
        abs_tol=anchor_tolerance_s,
    ) or not math.isclose(
        anchor_end,
        expected_anchor_end,
        rel_tol=0.0,
        abs_tol=anchor_tolerance_s,
    ):
        raise _error(
            f"trajectory {trajectory_id!r} scheduled-reference anchor "
            "timestamps do not match protocol_timing plus the selected "
            f"protocol knot interval within {anchor_tolerance_s:.9g} s"
        )

    for signal_name, signal in by_name.items():
        start_s, end_s, alignment_error = _binding_in_canonical_seconds(
            trajectory_id,
            trajectory,
            signal,
        )
        time_base = signal["time_base"]
        expected_end = anchor_end if time_base == "state" else anchor_end - period_s
        tolerance_s = 0.5 * period_s + anchor_error + alignment_error
        if not math.isclose(
            start_s,
            anchor_start,
            rel_tol=0.0,
            abs_tol=tolerance_s,
        ) or not math.isclose(
            end_s,
            expected_end,
            rel_tol=0.0,
            abs_tol=tolerance_s,
        ):
            raise _error(
                f"trajectory {trajectory_id!r} signal {signal_name!r} source "
                "timestamps do not align with the selected protocol execution "
                f"window within {tolerance_s:.9g} s"
            )


def validate_dataset_bundle(
    dataset_root: str | Path,
    *,
    catalog_root: str | Path | None = None,
    require_sealed: bool = True,
) -> ValidatedDataset:
    """Validate a dataset manifest and its complete local artifact closure."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise _error(f"dataset root is not a directory: {root}")
    dataset = _direct_artifact(
        root / "dataset.json",
        expected_schema="fer-mujoco-sysid/dataset@1",
        require_sealed=require_sealed,
    )
    catalog = root if catalog_root is None else Path(catalog_root).resolve()
    if not catalog.is_dir():
        raise _error(f"catalog root is not a directory: {catalog}")
    try:
        root.relative_to(catalog)
    except ValueError as exc:
        raise _error(
            f"dataset root {root} must be contained by catalog root {catalog}"
        ) from exc
    resolver = LocalArtifactResolver(catalog, require_sealed=require_sealed)
    manifest = dataset.manifest
    if "supersedes" in manifest:
        raise _error(
            "dataset.supersedes is not supported by the version-1 local closure "
            "validator; remove it or implement the superseded-dataset closure"
        )

    all_references: list[Mapping[str, Any]] = []
    for field, expected_schema in (
        ("protocols", "fer-mujoco-sysid/motion-protocol@1"),
        ("runs", "fer-mujoco-sysid/acquisition-run@1"),
        ("trajectories", "fer-mujoco-sysid/normalized-trajectory@1"),
    ):
        raw_references = manifest[field]
        if not isinstance(raw_references, Sequence) or isinstance(
            raw_references, (str, bytes)
        ):
            raise _error(f"dataset.{field} must be an array")
        all_references.extend(
            _reference(
                raw_reference,
                label=f"dataset.{field}[{index}]",
                expected_schema=expected_schema,
            )
            for index, raw_reference in enumerate(raw_references)
        )
    splits_reference = _reference(
        manifest["splits"],
        label="dataset.splits",
        expected_schema="fer-mujoco-sysid/splits@1",
    )
    all_references.append(splits_reference)

    artifact_ids = {cast(str, manifest["artifact_id"])}
    locators: set[str] = set()
    for index, reference in enumerate(all_references):
        artifact_id = cast(str, reference["artifact_id"])
        if artifact_id in artifact_ids:
            raise _error(f"duplicate dataset artifact ID: {artifact_id!r}")
        artifact_ids.add(artifact_id)
        locator = _locator_path(reference, label=f"dataset artifact reference {index}")
        if locator in locators:
            raise _error(f"duplicate dataset artifact locator: {locator!r}")
        locators.add(locator)

    protocols, protocol_refs = _resolve_collection(
        resolver,
        manifest["protocols"],
        label="dataset.protocols",
        expected_schema="fer-mujoco-sysid/motion-protocol@1",
    )
    runs, run_refs = _resolve_collection(
        resolver,
        manifest["runs"],
        label="dataset.runs",
        expected_schema="fer-mujoco-sysid/acquisition-run@1",
    )
    trajectories, trajectory_refs = _resolve_collection(
        resolver,
        manifest["trajectories"],
        label="dataset.trajectories",
        expected_schema="fer-mujoco-sysid/normalized-trajectory@1",
    )
    protocol_numeric_arrays = {
        protocol_id: _load_protocol_arrays(protocol)
        for protocol_id, protocol in protocols.items()
    }

    splits = resolver.resolve(
        splits_reference,
        expected_schema="fer-mujoco-sysid/splits@1",
    )
    split_manifest = splits.manifest
    if split_manifest["dataset_id"] != manifest["dataset_id"]:
        raise _error("splits.dataset_id does not match dataset.dataset_id")
    if split_manifest["dataset_version"] != manifest["version"]:
        raise _error("splits.dataset_version does not match dataset.version")

    expected_source_kind = _expected_source_kind(runs)
    if manifest["source_kind"] != expected_source_kind:
        raise _error(
            f"dataset.source_kind must be {expected_source_kind!r} for its "
            f"resolved run backends, got {manifest['source_kind']!r}"
        )

    for run_id, run_record in runs.items():
        run_manifest = run_record.manifest
        if "protocol_validation" in run_manifest:
            raise _error(
                f"run {run_id!r}.protocol_validation is not supported by the "
                "version-1 local closure validator"
            )
        protocol_id = cast(str, run_manifest["protocol"]["artifact_id"])
        expected_protocol = protocol_refs.get(protocol_id)
        if expected_protocol is None:
            raise _error(
                f"run {run_id!r} references protocol {protocol_id!r}, "
                "which is absent from dataset.protocols"
            )
        _same_reference(
            run_manifest["protocol"],
            expected_protocol,
            label=f"run {run_id!r}.protocol",
        )
        protocol_context = protocols[protocol_id].manifest["context"]
        robot_context = run_manifest["robot"]
        if robot_context["end_effector_id"] != protocol_context["end_effector_id"]:
            raise _error(
                f"run {run_id!r} end effector does not match protocol {protocol_id!r}"
            )
        if _thaw_json(robot_context["payload"]) != _thaw_json(
            protocol_context["payload"]
        ):
            raise _error(
                f"run {run_id!r} payload does not match protocol {protocol_id!r}"
            )

    segment_by_trajectory: dict[str, Mapping[str, Any]] = {}
    command_fingerprint_by_trajectory: dict[str, str] = {}
    for trajectory_id, trajectory_record in trajectories.items():
        trajectory_manifest = trajectory_record.manifest
        run_id = cast(str, trajectory_manifest["source_run"]["artifact_id"])
        protocol_id = cast(str, trajectory_manifest["protocol"]["artifact_id"])
        expected_run = run_refs.get(run_id)
        expected_protocol = protocol_refs.get(protocol_id)
        if expected_run is None:
            raise _error(
                f"trajectory {trajectory_id!r} references source run "
                f"{run_id!r}, which is absent from dataset.runs"
            )
        if expected_protocol is None:
            raise _error(
                f"trajectory {trajectory_id!r} references protocol "
                f"{protocol_id!r}, which is absent from dataset.protocols"
            )
        _same_reference(
            trajectory_manifest["source_run"],
            expected_run,
            label=f"trajectory {trajectory_id!r}.source_run",
        )
        _same_reference(
            trajectory_manifest["protocol"],
            expected_protocol,
            label=f"trajectory {trajectory_id!r}.protocol",
        )
        run_protocol_id = runs[run_id].manifest["protocol"]["artifact_id"]
        if run_protocol_id != protocol_id:
            raise _error(
                f"trajectory {trajectory_id!r} protocol {protocol_id!r} "
                f"does not match source run protocol {run_protocol_id!r}"
            )

        segments = {
            segment["segment_id"]: segment
            for segment in cast(
                Sequence[Mapping[str, Any]],
                protocols[protocol_id].manifest["segments"],
            )
        }
        segment_id = cast(str, trajectory_manifest["protocol_segment_id"])
        segment = segments.get(segment_id)
        if segment is None:
            raise _error(
                f"trajectory {trajectory_id!r} names unknown protocol segment "
                f"{segment_id!r}"
            )
        protocol_interval = _mapping(
            trajectory_manifest["protocol_sample_interval"],
            label=f"trajectory {trajectory_id!r}.protocol_sample_interval",
        )
        interval_start = cast(int, protocol_interval["start_index"])
        interval_end = cast(int, protocol_interval["end_index_exclusive"])
        if (
            interval_start < segment["start_index"]
            or interval_end > segment["end_index_exclusive"]
        ):
            raise _error(
                f"trajectory {trajectory_id!r} protocol sample interval "
                f"[{interval_start}, {interval_end}) lies outside segment "
                f"{segment_id!r} bounds [{segment['start_index']}, "
                f"{segment['end_index_exclusive']})"
            )
        protocol_period_ns = protocols[protocol_id].manifest["sample_period_ns"]
        protocol_duration_ns = (interval_end - interval_start - 1) * protocol_period_ns
        state_sample_count = trajectory_manifest["state_time"]["shape"][0]
        normalized_duration_ns = (state_sample_count - 1) * trajectory_manifest[
            "sample_grid"
        ]["period_ns"]
        if protocol_duration_ns != normalized_duration_ns:
            raise _error(
                f"trajectory {trajectory_id!r} normalized duration "
                f"{normalized_duration_ns} ns does not match declared protocol "
                f"sample interval duration {protocol_duration_ns} ns"
            )
        normalized_period_ns = trajectory_manifest["sample_grid"]["period_ns"]
        if protocol_period_ns != normalized_period_ns:
            raise _error(
                f"trajectory {trajectory_id!r} sample period does not match "
                f"protocol {protocol_id!r}"
            )
        if interval_end - interval_start != state_sample_count:
            raise _error(
                f"trajectory {trajectory_id!r} state sample count does not "
                "match its declared protocol knot interval"
            )

        _validate_raw_bindings(
            trajectory_id,
            trajectory_manifest,
            runs[run_id].manifest,
        )

        reference_names = _mapping(
            trajectory_manifest["protocol_reference_signals"],
            label=f"trajectory {trajectory_id!r}.protocol_reference_signals",
        )
        protocol_array_contracts = _mapping(
            protocols[protocol_id].manifest["arrays"],
            label=f"protocol {protocol_id!r}.arrays",
        )
        expected_reference_logical_names = {
            "desired_position",
            "desired_velocity",
            "desired_acceleration",
        }
        if "desired_effort_feedforward" in protocol_array_contracts:
            expected_reference_logical_names.add("desired_effort_feedforward")
        if set(reference_names) != expected_reference_logical_names:
            raise _error(
                f"trajectory {trajectory_id!r} protocol reference channels "
                f"must be exactly {sorted(expected_reference_logical_names)!r}"
            )
        trajectory_signals_by_name = {
            signal["descriptor"]["name"]: signal
            for signal in cast(
                Sequence[Mapping[str, Any]],
                trajectory_manifest["signals"],
            )
        }
        run_signals_by_name = {
            signal["name"]: signal
            for signal in cast(
                Sequence[Mapping[str, Any]],
                runs[run_id].manifest["signals"],
            )
        }
        trajectory_arrays = load_numeric_npz(
            _numeric_archive_path(
                trajectory_manifest,
                root=trajectory_record.bundle_root,
                schema="fer-mujoco-sysid/normalized-trajectory@1",
            )
        )
        protocol_knot_count = protocol_numeric_arrays[protocol_id][
            "time_from_start_ns"
        ].shape[0]
        run_protocol_timing = _mapping(
            runs[run_id].manifest["protocol_timing"],
            label=f"run {run_id!r}.protocol_timing",
        )
        run_reference_names = _mapping(
            runs[run_id].manifest["protocol_reference_signals"],
            label=f"run {run_id!r}.protocol_reference_signals",
        )
        if set(run_reference_names) != set(reference_names):
            raise _error(
                f"run {run_id!r} and trajectory {trajectory_id!r} protocol "
                "reference channel sets differ"
            )
        for logical_name, signal_name in reference_names.items():
            reference_signal = trajectory_signals_by_name[signal_name]
            binding = reference_signal["source_binding"]
            if binding["raw_signal_name"] != run_reference_names[logical_name]:
                raise _error(
                    f"trajectory {trajectory_id!r} protocol reference "
                    f"{logical_name!r} must bind the source run signal named "
                    "by protocol_reference_signals"
                )
            is_effort = logical_name == "desired_effort_feedforward"
            expected_source_start = interval_start
            expected_source_end = interval_end - 1 if is_effort else interval_end
            actual_source_bounds = (
                binding["start_message_index"],
                binding["end_message_index_exclusive"],
            )
            if actual_source_bounds != (
                expected_source_start,
                expected_source_end,
            ):
                raise _error(
                    f"trajectory {trajectory_id!r} protocol reference "
                    f"{logical_name!r} raw bounds {actual_source_bounds!r} "
                    f"must equal protocol bounds "
                    f"{(expected_source_start, expected_source_end)!r}"
                )
            raw_signal = run_signals_by_name[binding["raw_signal_name"]]
            expected_raw_count = (
                protocol_knot_count - 1 if is_effort else protocol_knot_count
            )
            if raw_signal["sample_count"] != expected_raw_count:
                raise _error(
                    f"run {run_id!r} protocol reference {logical_name!r} "
                    f"sample_count must be {expected_raw_count}, got "
                    f"{raw_signal['sample_count']}"
                )

            protocol_reference = protocol_array_contracts[logical_name]
            protocol_key = protocol_reference["key"]
            stop = interval_end - 1 if is_effort else interval_end
            expected_reference = protocol_numeric_arrays[protocol_id][protocol_key][
                interval_start:stop
            ]
            reference_key = reference_signal["array"]["key"]
            reference_array = trajectory_arrays[reference_key]
            if not np.array_equal(reference_array, expected_reference):
                raise _error(
                    f"trajectory {trajectory_id!r} protocol reference "
                    f"{logical_name!r} does not match protocol "
                    f"{protocol_id!r} over [{interval_start}, {stop})"
                )

        _validate_cross_signal_window(
            trajectory_id,
            trajectory_manifest,
            protocol_position_signal_name=cast(
                str,
                reference_names["desired_position"],
            ),
            protocol_timing=run_protocol_timing,
            interval_start=interval_start,
            interval_end=interval_end,
            period_ns=normalized_period_ns,
        )

        command_fingerprint_by_trajectory[trajectory_id] = (
            _protocol_interval_command_fingerprint(
                protocols[protocol_id],
                protocol_numeric_arrays[protocol_id],
                start_index=interval_start,
                end_index_exclusive=interval_end,
            )
        )
        segment_by_trajectory[trajectory_id] = segment

    partition_by_trajectory: dict[str, str] = {}
    scientific_partition_by_protocol: dict[str, tuple[str, str]] = {}
    partitions = _mapping(split_manifest["partitions"], label="splits.partitions")
    for partition, raw_entries in partitions.items():
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes)
        ):
            raise _error(f"splits.partitions.{partition} must be an array")
        for index, raw_entry in enumerate(raw_entries):
            entry = _mapping(
                raw_entry,
                label=f"splits.partitions.{partition}[{index}]",
            )
            member = _split_member(entry, partition)
            trajectory_id = cast(str, member["trajectory"]["artifact_id"])
            trajectory_record = trajectories.get(trajectory_id)
            if trajectory_record is None:
                raise _error(
                    f"split partition {partition!r} references trajectory "
                    f"{trajectory_id!r}, which is absent from dataset.trajectories"
                )
            if trajectory_id in partition_by_trajectory:
                raise _error(
                    f"trajectory {trajectory_id!r} occurs more than once "
                    "across split partitions"
                )
            partition_by_trajectory[trajectory_id] = partition
            _same_reference(
                member["trajectory"],
                trajectory_refs[trajectory_id],
                label=(f"splits.partitions.{partition}[{index}].trajectory"),
            )
            trajectory_manifest = trajectory_record.manifest
            expected_metadata = {
                "lineage_group_id": trajectory_manifest["lineage_group_id"],
                "protocol_artifact_id": trajectory_manifest["protocol"]["artifact_id"],
                "source_run_artifact_id": trajectory_manifest["source_run"][
                    "artifact_id"
                ],
            }
            for field, expected in expected_metadata.items():
                if member[field] != expected:
                    raise _error(
                        f"split metadata {field!r} for trajectory "
                        f"{trajectory_id!r} must be {expected!r}, "
                        f"got {member[field]!r}"
                    )

            if partition in _SCIENTIFIC_PARTITIONS:
                protocol_id = cast(
                    str,
                    trajectory_manifest["protocol"]["artifact_id"],
                )
                command_fingerprint = command_fingerprint_by_trajectory[trajectory_id]
                prior = scientific_partition_by_protocol.setdefault(
                    command_fingerprint,
                    (partition, protocol_id),
                )
                prior_partition, prior_protocol_id = prior
                if prior_partition != partition:
                    raise _error(
                        f"protocol intervals from {prior_protocol_id!r} and "
                        f"{protocol_id!r} have the same executable command "
                        "fingerprint but "
                        f"occur in scientific partitions {prior_partition!r} "
                        f"and {partition!r}"
                    )
                if not segment_by_trajectory[trajectory_id]["analysis_eligible"]:
                    raise _error(
                        f"trajectory {trajectory_id!r} is assigned to "
                        f"{partition!r} but its protocol segment is not "
                        "analysis eligible"
                    )
                run_id = cast(
                    str,
                    trajectory_manifest["source_run"]["artifact_id"],
                )
                if runs[run_id].manifest["outcome"]["status"] != "completed":
                    raise _error(
                        f"trajectory {trajectory_id!r} is assigned to "
                        f"{partition!r} but source run {run_id!r} did not complete"
                    )
                if trajectory_manifest["quality"]["status"] != "pass":
                    raise _error(
                        f"trajectory {trajectory_id!r} is assigned to "
                        f"{partition!r} but quality status is not 'pass'"
                    )

    dataset_trajectory_ids = set(trajectories)
    partitioned_trajectory_ids = set(partition_by_trajectory)
    if partitioned_trajectory_ids != dataset_trajectory_ids:
        missing = sorted(dataset_trajectory_ids - partitioned_trajectory_ids)
        unknown = sorted(partitioned_trajectory_ids - dataset_trajectory_ids)
        raise _error(
            "split coverage must contain every dataset trajectory exactly once; "
            f"missing={missing!r}, unknown={unknown!r}"
        )

    return ValidatedDataset(
        dataset=dataset,
        splits=splits,
        protocols=MappingProxyType(dict(protocols)),
        runs=MappingProxyType(dict(runs)),
        trajectories=MappingProxyType(dict(trajectories)),
        partition_by_trajectory=MappingProxyType(partition_by_trajectory),
        catalog_root=catalog,
    )


def _record_reference_matches(
    reference: Any,
    record: ResolvedArtifact,
    *,
    label: str,
    catalog_root: Path,
) -> None:
    checked = _reference(reference, label=label)
    expected = {
        "artifact_id": record.manifest["artifact_id"],
        "schema": record.manifest["schema"],
        "manifest_sha256": record.manifest_sha256,
    }
    for field, value in expected.items():
        if checked[field] != value:
            raise _error(f"{label}.{field} must be {value!r}, got {checked[field]!r}")
    if "bundle_sha256" in checked and checked["bundle_sha256"] != record.bundle_sha256:
        raise _error(
            f"{label}.bundle_sha256 does not match the resolved artifact bundle"
        )
    locator = cast(Mapping[str, Any], checked["locator"])
    if locator["kind"] == "external":
        raise _error(
            f"{label} uses an external locator, which cannot be checked against "
            "the validated local dataset"
        )
    resolved_path = resolve_contained_path(
        catalog_root,
        cast(str, locator["path"]),
        label=f"{label}.locator",
    )
    if resolved_path != record.manifest_path:
        raise _error(f"{label}.locator does not name the resolved manifest")


def validate_fit_against_dataset(
    fit_manifest: Mapping[str, Any] | ResolvedArtifact,
    *,
    root: str | Path,
    dataset: ValidatedDataset,
) -> None:
    """Validate a fit's selected trajectory closure and effort semantics."""

    fit_root = Path(root).resolve()
    if (
        isinstance(fit_manifest, ResolvedArtifact)
        and fit_root != fit_manifest.bundle_root.resolve()
    ):
        raise _error(
            "resolved fit root must equal its sealed artifact bundle root: "
            f"expected {fit_manifest.bundle_root.resolve()}, got {fit_root}"
        )
    raw_manifest = (
        fit_manifest.manifest
        if isinstance(fit_manifest, ResolvedArtifact)
        else fit_manifest
    )
    fit_manifest = cast(Mapping[str, Any], _thaw_json(raw_manifest))
    validate_fit_result(fit_manifest, root=fit_root)
    inputs = _mapping(fit_manifest["inputs"], label="fit.inputs")
    _record_reference_matches(
        inputs["dataset"],
        dataset.dataset,
        label="fit.inputs.dataset",
        catalog_root=dataset.catalog_root,
    )
    _record_reference_matches(
        inputs["splits"],
        dataset.splits,
        label="fit.inputs.splits",
        catalog_root=dataset.catalog_root,
    )

    trajectory_inputs = inputs["trajectories"]
    if not isinstance(trajectory_inputs, list):
        raise _error("fit.inputs.trajectories must be an array")
    fit_references: dict[str, Mapping[str, Any]] = {}
    for index, raw_reference in enumerate(trajectory_inputs):
        reference = _reference(
            raw_reference,
            label=f"fit.inputs.trajectories[{index}]",
            expected_schema="fer-mujoco-sysid/normalized-trajectory@1",
        )
        trajectory_id = cast(str, reference["artifact_id"])
        if trajectory_id in fit_references:
            raise _error(
                f"fit.inputs.trajectories contains duplicate trajectory "
                f"{trajectory_id!r}"
            )
        fit_references[trajectory_id] = reference

    expected_ids = {
        trajectory_id
        for trajectory_id, partition in dataset.partition_by_trajectory.items()
        if partition in _SCIENTIFIC_PARTITIONS
    }
    scientific_protocol_ids = {
        cast(
            str,
            dataset.trajectories[trajectory_id].manifest["protocol"]["artifact_id"],
        )
        for trajectory_id in expected_ids
    }
    protocol_sources = {
        protocol_id: dataset.protocols[protocol_id].manifest["context"]["source_model"]
        for protocol_id in scientific_protocol_ids
    }
    baseline_protocol_id = min(protocol_sources)
    baseline_source = protocol_sources[baseline_protocol_id]
    for protocol_id, source in protocol_sources.items():
        if _thaw_json(source) != _thaw_json(baseline_source):
            raise _error(
                f"scientific protocols {baseline_protocol_id!r} and "
                f"{protocol_id!r} do not share one canonical source model"
            )
    if _thaw_json(inputs["source_model"]) != _thaw_json(baseline_source):
        raise _error(
            "fit.inputs.source_model does not match the canonical source model "
            "of the scientific protocols"
        )
    if set(fit_references) != expected_ids:
        missing = sorted(expected_ids - set(fit_references))
        unexpected = sorted(set(fit_references) - expected_ids)
        raise _error(
            "fit trajectory references must exactly equal the fit, development, "
            f"and held-out dataset trajectories; missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
    for trajectory_id, reference in fit_references.items():
        _record_reference_matches(
            reference,
            dataset.trajectories[trajectory_id],
            label=f"fit trajectory {trajectory_id!r}",
            catalog_root=dataset.catalog_root,
        )

    forward_input = _mapping(
        fit_manifest["forward_model_input"],
        label="fit.forward_model_input",
    )
    signal_name = cast(str, forward_input["signal_name"])
    semantic_role = forward_input["semantic_role"]
    expected_joints = FER_ARM_JOINT_ORDER
    baseline_semantics: Mapping[str, Any] | None = None
    baseline_trajectory_id: str | None = None
    for trajectory_id in sorted(expected_ids):
        trajectory = dataset.trajectories[trajectory_id].manifest
        matches = [
            _mapping(
                raw_signal,
                label=f"trajectory {trajectory_id!r} signal",
            )
            for raw_signal in cast(Sequence[Any], trajectory["signals"])
            if _mapping(
                raw_signal,
                label=f"trajectory {trajectory_id!r} signal",
            )["descriptor"]["name"]
            == signal_name
        ]
        if len(matches) != 1:
            raise _error(
                f"selected effort signal {signal_name!r} must occur exactly once "
                f"in trajectory {trajectory_id!r}, got {len(matches)}"
            )
        selected = matches[0]
        descriptor = _mapping(
            selected["descriptor"],
            label=f"trajectory {trajectory_id!r} selected descriptor",
        )
        if descriptor["semantic_role"] != semantic_role:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} has "
                f"role {descriptor['semantic_role']!r}, expected {semantic_role!r}"
            )
        required_descriptor = {
            "quantity": "joint_effort",
            "unit": "N*m",
            "positive_direction": "same_as_joint_coordinate",
        }
        for field, expected in required_descriptor.items():
            if descriptor[field] != expected:
                raise _error(
                    f"selected effort signal in trajectory {trajectory_id!r} "
                    f"requires {field}={expected!r}, got {descriptor[field]!r}"
                )
        if tuple(descriptor["joint_order"]) != expected_joints:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} "
                f"requires joint_order={expected_joints!r}, "
                f"got {descriptor['joint_order']!r}"
            )
        if selected["time_base"] != "control":
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} "
                "must use the control time base"
            )
        if trajectory["quality"]["status"] != "pass":
            raise _error(
                f"fit trajectory {trajectory_id!r} must have quality status 'pass'"
            )
        numeric_transform = _mapping(
            selected["numeric_transform"],
            label=f"trajectory {trajectory_id!r} selected numeric transform",
        )
        if dict(numeric_transform) != {"scale": 1.0, "offset": 0.0}:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} "
                "requires an identity numeric transform until structured fit "
                "transform validation is implemented"
            )
        resampling = _mapping(
            selected["resampling"],
            label=f"trajectory {trajectory_id!r} selected resampling",
        )
        if resampling["method"] not in {"none", "zero_order_hold"}:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} "
                "requires causal none/zero_order_hold resampling"
            )
        if resampling["applied_time_shift_s"] != 0.0:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} "
                "requires zero applied time shift until structured delay "
                "validation is implemented"
            )

        semantics = _mapping(
            descriptor.get("effort_semantics"),
            label=f"trajectory {trajectory_id!r} selected effort semantics",
        )
        unknown_fields = [
            field
            for field in (
                "gravity",
                "coriolis",
                "friction_compensation",
                "rate_limit_position",
            )
            if semantics[field] == "unknown"
        ]
        if unknown_fields:
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} has "
                f"unknown semantics: {unknown_fields!r}"
            )
        if (
            semantics["stage"] == "controller"
            and semantics["rate_limit_position"] == "before"
        ):
            raise _error(
                f"selected effort signal in trajectory {trajectory_id!r} is a "
                "controller-stage command before rate limiting"
            )
        expected_identity = _SELECTED_EFFORT_IDENTITY[cast(str, semantic_role)]
        actual_identity = (semantics["stage"], semantics["location"])
        if actual_identity != expected_identity:
            raise _error(
                f"selected effort role {semantic_role!r} in trajectory "
                f"{trajectory_id!r} requires stage/location "
                f"{expected_identity!r}, got {actual_identity!r}"
            )
        if baseline_semantics is None:
            baseline_semantics = semantics
            baseline_trajectory_id = trajectory_id
        elif dict(semantics) != dict(baseline_semantics):
            raise _error(
                f"selected effort semantics are incoherent between trajectories "
                f"{baseline_trajectory_id!r} and {trajectory_id!r}"
            )
