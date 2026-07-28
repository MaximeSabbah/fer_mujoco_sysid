"""Reading and writing project artifacts: JSON manifests, NPZ arrays, hashes.

Deliberately small. An artifact is a directory holding a human-readable
``*.json`` manifest, one ``*.npz`` of numeric arrays, and a
``checksums.sha256`` file in ``sha256sum`` format. Nothing here enforces a
schema registry or a lineage graph — the manifest is documentation plus a
content fingerprint, and the fingerprint is what makes a result citable.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

CHECKSUM_FILE = "checksums.sha256"
_CONTENT_DOMAIN = b"fer-mujoco-sysid/content-sha256@2"


class ArtifactError(ValueError):
    """An artifact is malformed, inconsistent, or fails verification."""


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON with sorted keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(digest: hashlib._Hash, tag: bytes, payload: bytes) -> None:
    """Length-framed record, so concatenations cannot collide."""
    digest.update(tag + b"\0" + struct.pack(">Q", len(payload)) + payload)


def content_sha256(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """Fingerprint the scientific content of an artifact.

    Covers the manifest (minus its own fingerprint and creation time) and
    every array's dtype, shape and bytes. Two artifacts share a fingerprint
    exactly when they mean the same thing, regardless of JSON formatting or
    NPZ compression.
    """
    reduced = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "created_at"}
    }
    digest = hashlib.sha256()
    digest.update(_CONTENT_DOMAIN + b"\0")
    _record(
        digest,
        b"manifest",
        json.dumps(reduced, sort_keys=True, separators=(",", ":")).encode(),
    )
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        if not np.isfinite(array).all():
            raise ArtifactError(f"array {key!r} contains non-finite values")
        _record(
            digest,
            b"array-meta",
            json.dumps(
                {
                    "key": key,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        _record(digest, b"array-data", array.tobytes(order="C"))
    return digest.hexdigest()


def write_checksums(root: str | Path) -> Path:
    """Write ``checksums.sha256`` covering every other file in *root*."""
    root = Path(root)
    entries = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    )
    if not entries:
        raise ArtifactError(f"nothing to checksum in {root}")
    target = root / CHECKSUM_FILE
    target.write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in entries),
        encoding="utf-8",
    )
    return target


def verify_checksums(root: str | Path) -> None:
    """Raise :class:`ArtifactError` on any missing, extra or altered file."""
    root = Path(root)
    manifest = root / CHECKSUM_FILE
    if not manifest.is_file():
        raise ArtifactError(f"missing {CHECKSUM_FILE} in {root}")

    expected: dict[str, str] = {}
    for number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or not name:
            raise ArtifactError(f"{manifest}:{number}: malformed entry")
        expected[name] = digest

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    if present != set(expected):
        missing = sorted(set(expected) - present)
        extra = sorted(present - set(expected))
        raise ArtifactError(
            f"{root}: file set does not match {CHECKSUM_FILE} "
            f"(missing={missing}, unlisted={extra})"
        )
    for name, digest in expected.items():
        actual = sha256_file(root / name)
        if actual != digest:
            raise ArtifactError(f"{root / name}: SHA-256 mismatch")


def save_arrays(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}
