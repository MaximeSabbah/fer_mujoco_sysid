"""Deterministic scientific-content fingerprints for portable artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError

_CONTENT_DOMAIN = b"fer-mujoco-sysid/content-sha256@1"
_NUMERIC_REFERENCE_FIELDS = {"file", "key", "dtype", "shape", "unit"}
_CANONICAL_DTYPES = {"<f8", "<i8", "<i4", "<u8", "<u4", "|b1"}


def _canonicalize_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if _NUMERIC_REFERENCE_FIELDS.issubset(value):
            file_reference = value["file"]
            if not isinstance(file_reference, Mapping):
                raise ArtifactValidationError(
                    "numeric array reference file must be an object"
                )
            path = file_reference.get("path")
            if not isinstance(path, str):
                raise ArtifactValidationError(
                    "numeric array reference file.path must be a string"
                )
            return {
                "file": {"path": path},
                "key": _canonicalize_manifest_value(value["key"]),
                "dtype": _canonicalize_manifest_value(value["dtype"]),
                "shape": _canonicalize_manifest_value(value["shape"]),
                "unit": _canonicalize_manifest_value(value["unit"]),
            }
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(
                    f"content manifest key is not a string: {key!r}"
                )
            result[key] = _canonicalize_manifest_value(child)
        return result
    if isinstance(value, list):
        return [_canonicalize_manifest_value(child) for child in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ArtifactValidationError(
        f"unsupported content manifest value: {type(value).__name__}"
    )


def _compact_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ArtifactValidationError(
            f"cannot canonicalize scientific content as JSON: {exc}"
        ) from exc


def _update_record_header(
    digest: Any,
    tag: bytes,
    payload_size: int,
) -> None:
    digest.update(tag)
    digest.update(b"\0")
    digest.update(payload_size.to_bytes(8, byteorder="big", signed=False))


def _update_record(digest: Any, tag: bytes, payload: bytes) -> None:
    _update_record_header(digest, tag, len(payload))
    digest.update(payload)


def _validate_content_array(name: str, array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise ArtifactValidationError(f"content array {name!r} is not an ndarray")
    if array.dtype.str not in _CANONICAL_DTYPES:
        raise ArtifactValidationError(
            f"content array {name!r} has noncanonical dtype {array.dtype.str}"
        )
    if array.dtype.itemsize > 1:
        byteorder = array.dtype.byteorder
        if byteorder == ">" or (byteorder == "=" and sys.byteorder != "little"):
            raise ArtifactValidationError(
                f"content array {name!r} is not little-endian"
            )
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ArtifactValidationError(
            f"content array {name!r} contains NaN or infinity"
        )
    if not array.flags.c_contiguous:
        raise ArtifactValidationError(f"content array {name!r} must be C-contiguous")


def content_sha256(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Return the version-1 scientific-content SHA-256 fingerprint.

    The byte stream is framed to prevent concatenation ambiguity:

    1. the ASCII domain ``fer-mujoco-sysid/content-sha256@1``;
    2. one ``manifest`` record containing compact, UTF-8, sorted-key JSON after
       removing top-level ``content_sha256`` and ``created_at``;
    3. for every NPZ array in sorted key order, an ``array-metadata`` record
       containing compact JSON with ``key``, little-endian ``dtype`` and
       ``shape``, followed by an ``array-data`` record containing its C-order
       bytes.

    Every record is encoded as ``tag + NUL + uint64_be(length) + payload``.
    Within each numeric-array reference, the manifest contribution retains
    ``key``, ``dtype``, ``shape``, ``unit`` and ``file.path`` but removes that
    file reference's SHA-256, size and media type. Consequently the fingerprint
    changes with scientific values or semantics, but not with JSON formatting,
    creation time, or the ZIP encoding of an equivalent NPZ archive. Artifact
    IDs, provenance, logical relative paths and locators remain part of the
    version-1 identity.
    """
    top_level = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "created_at"}
    }
    canonical_manifest = _canonicalize_manifest_value(top_level)
    manifest_bytes = _compact_json_bytes(canonical_manifest)

    digest = hashlib.sha256()
    digest.update(_CONTENT_DOMAIN)
    digest.update(b"\0")
    _update_record(digest, b"manifest", manifest_bytes)

    invalid_keys = [key for key in arrays if not isinstance(key, str) or not key]
    if invalid_keys:
        raise ArtifactValidationError(f"invalid content array key: {invalid_keys[0]!r}")

    for key in sorted(arrays):
        array = arrays[key]
        _validate_content_array(key, array)
        metadata = _compact_json_bytes(
            {
                "dtype": array.dtype.str,
                "key": key,
                "shape": list(array.shape),
            }
        )
        _update_record(digest, b"array-metadata", metadata)
        _update_record_header(digest, b"array-data", array.nbytes)
        digest.update(memoryview(array).cast("B"))

    return digest.hexdigest()
