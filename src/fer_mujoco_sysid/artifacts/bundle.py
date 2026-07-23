"""Deterministic finalization and strict verification of sealed bundles."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import (
    load_checksum_manifest,
    load_json,
    sha256_file,
)
from fer_mujoco_sysid.artifacts.schemas import validate_schema

CHECKSUM_MANIFEST_NAME = "checksums.sha256"
SEAL_NAME = "seal.json"
SEAL_SCHEMA = "fer-mujoco-sysid/seal@1"

_CONTROL_FILES = frozenset({CHECKSUM_MANIFEST_NAME, SEAL_NAME})
_SEAL_KEYS = frozenset(
    {
        "schema",
        "checksum_manifest",
        "checksum_manifest_sha256",
    }
)


def _root_path(root: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(root)))
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArtifactValidationError(
            f"cannot inspect bundle root {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(mode):
        raise ArtifactValidationError(f"bundle root must not be a symlink: {path}")
    if not stat.S_ISDIR(mode):
        raise ArtifactValidationError(f"bundle root is not a directory: {path}")
    return path


def _relative_path(parts: tuple[str, ...]) -> str:
    relative = PurePosixPath(*parts).as_posix()
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
        or "\n" in relative
        or "\r" in relative
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ArtifactValidationError(
            f"bundle contains a non-portable relative path: {relative!r}"
        )
    return relative


def _regular_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise ArtifactValidationError(
                f"cannot enumerate bundle directory {directory}: {exc}"
            ) from exc

        for entry in ordered:
            parts = (*prefix, entry.name)
            relative = _relative_path(parts)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ArtifactValidationError(
                    f"cannot inspect bundle entry {relative!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ArtifactValidationError(
                    f"bundle entry must not be a symlink: {relative!r}"
                )
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), parts)
                continue
            if not stat.S_ISREG(mode):
                raise ArtifactValidationError(
                    f"bundle entry is not a regular file: {relative!r}"
                )
            files[relative] = Path(entry.path)

    visit(root, ())
    return files


def _canonical_manifest(entries: dict[str, str]) -> bytes:
    return "".join(
        f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_name: str | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as exc:
        raise ArtifactValidationError(f"cannot atomically write {path}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _payload_files(files: dict[str, Path]) -> dict[str, Path]:
    return {
        relative: path
        for relative, path in files.items()
        if relative not in _CONTROL_FILES
    }


def _verify_payload(
    root: Path,
    entries: dict[str, str],
    *,
    files: dict[str, Path] | None = None,
) -> None:
    if _CONTROL_FILES.intersection(entries):
        reserved = sorted(_CONTROL_FILES.intersection(entries))
        raise ArtifactValidationError(
            "checksum manifest lists reserved control files: "
            + ", ".join(repr(path) for path in reserved)
        )

    if files is None:
        files = _regular_files(root)
    payload = _payload_files(files)
    listed = set(entries)
    present = set(payload)
    missing = sorted(listed - present)
    unlisted = sorted(present - listed)
    if missing or unlisted:
        details: list[str] = []
        if missing:
            details.append(
                "missing listed files: " + ", ".join(repr(path) for path in missing)
            )
        if unlisted:
            details.append(
                "unlisted regular files: " + ", ".join(repr(path) for path in unlisted)
            )
        raise ArtifactValidationError(
            "sealed bundle file-set mismatch (" + "; ".join(details) + ")"
        )

    for relative in sorted(entries):
        actual = sha256_file(payload[relative])
        expected = entries[relative]
        if actual != expected:
            raise ArtifactValidationError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )


def _load_canonical_manifest(path: Path) -> tuple[dict[str, str], bytes]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(
            f"cannot read checksum manifest {path}: {exc}"
        ) from exc
    entries = load_checksum_manifest(path)
    canonical = _canonical_manifest(entries)
    if content != canonical:
        raise ArtifactValidationError(
            f"checksum manifest is not in canonical sorted form: {path}"
        )
    return entries, content


def _load_seal(path: Path) -> dict[str, str]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactValidationError("seal.json must contain a JSON object")
    validate_schema(value, "seal")
    keys = set(value)
    if keys != _SEAL_KEYS:
        missing = sorted(_SEAL_KEYS - keys)
        unknown = sorted(keys - _SEAL_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ArtifactValidationError(
            "invalid seal.json schema (" + "; ".join(details) + ")"
        )
    if value["schema"] != SEAL_SCHEMA:
        raise ArtifactValidationError(
            f"unsupported seal.json schema: {value['schema']!r}"
        )
    if value["checksum_manifest"] != CHECKSUM_MANIFEST_NAME:
        raise ArtifactValidationError(
            "seal.json checksum_manifest must be 'checksums.sha256'"
        )
    digest = value["checksum_manifest_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactValidationError(
            "seal.json checksum_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    return {
        "schema": SEAL_SCHEMA,
        "checksum_manifest": CHECKSUM_MANIFEST_NAME,
        "checksum_manifest_sha256": digest,
    }


def finalize_bundle(root: str | Path) -> dict[str, str]:
    """Seal an unsealed directory and return its payload checksum entries.

    Every regular payload file is listed. ``checksums.sha256`` and ``seal.json``
    are control files and are never part of that payload closure. An existing
    seal is never overwritten.
    """

    bundle_root = _root_path(root)
    files = _regular_files(bundle_root)
    seal_path = bundle_root / SEAL_NAME
    if SEAL_NAME in files or seal_path.exists():
        raise ArtifactValidationError(f"bundle is already sealed: {bundle_root}")

    checksum_path = bundle_root / CHECKSUM_MANIFEST_NAME
    if checksum_path.exists() and CHECKSUM_MANIFEST_NAME not in files:
        raise ArtifactValidationError(
            f"reserved checksum path is not a regular file: {checksum_path}"
        )

    payload = _payload_files(files)
    if not payload:
        raise ArtifactValidationError("cannot seal a bundle with no payload files")
    entries = {relative: sha256_file(payload[relative]) for relative in sorted(payload)}
    manifest_content = _canonical_manifest(entries)
    _atomic_write(checksum_path, manifest_content)

    # Recheck closure and bytes before publishing the final seal marker.
    _verify_payload(bundle_root, entries)
    manifest_digest = hashlib.sha256(manifest_content).hexdigest()
    seal = {
        "schema": SEAL_SCHEMA,
        "checksum_manifest": CHECKSUM_MANIFEST_NAME,
        "checksum_manifest_sha256": manifest_digest,
    }
    seal_content = (
        json.dumps(
            seal,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(seal_path, seal_content)
    try:
        return verify_sealed_bundle(bundle_root)
    except Exception:
        try:
            seal_path.unlink()
        except OSError as exc:
            raise ArtifactValidationError(
                "bundle finalization failed and its seal could not be rolled back: "
                f"{exc}"
            ) from exc
        raise


def verify_sealed_bundle(root: str | Path) -> dict[str, str]:
    """Verify seal metadata, manifest bytes, checksums, and exact file closure."""

    bundle_root = _root_path(root)
    files = _regular_files(bundle_root)
    for control_file in sorted(_CONTROL_FILES):
        if control_file not in files:
            raise ArtifactValidationError(
                f"sealed bundle is missing regular control file {control_file!r}"
            )

    seal_digest = sha256_file(files[SEAL_NAME])
    seal = _load_seal(files[SEAL_NAME])
    entries, manifest_content = _load_canonical_manifest(files[CHECKSUM_MANIFEST_NAME])
    manifest_digest = hashlib.sha256(manifest_content).hexdigest()
    if manifest_digest != seal["checksum_manifest_sha256"]:
        raise ArtifactValidationError(
            "SHA-256 mismatch for checksums.sha256: "
            f"expected {seal['checksum_manifest_sha256']}, got {manifest_digest}"
        )
    _verify_payload(bundle_root, entries, files=files)
    files_after = _regular_files(bundle_root)
    if set(files_after) != set(files):
        raise ArtifactValidationError(
            "sealed bundle file set changed during verification"
        )
    final_manifest_digest = sha256_file(files_after[CHECKSUM_MANIFEST_NAME])
    if final_manifest_digest != seal["checksum_manifest_sha256"]:
        raise ArtifactValidationError(
            "checksums.sha256 changed during sealed-bundle verification"
        )
    if sha256_file(files_after[SEAL_NAME]) != seal_digest:
        raise ArtifactValidationError("seal.json changed during bundle verification")
    return entries
