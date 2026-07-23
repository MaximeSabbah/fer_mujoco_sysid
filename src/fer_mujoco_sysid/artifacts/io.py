"""Strict, deterministic I/O for identification artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)

_CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64}) ([ *])(.+)$")


def _reject_json_constant(token: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON number is forbidden: {token}")


def _parse_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ArtifactValidationError(f"non-finite JSON number is forbidden: {token}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_json_value(
    value: Any,
    *,
    location: str = "$",
    active_containers: set[int] | None = None,
) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(
                f"{location}: non-finite JSON number is forbidden"
            )
        return

    if active_containers is None:
        active_containers = set()
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise ArtifactValidationError(f"{location}: cyclic JSON value")
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    location=f"{location}[{index}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active_containers:
            raise ArtifactValidationError(f"{location}: cyclic JSON value")
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ArtifactValidationError(
                        f"{location}: JSON object key is not a string: {key!r}"
                    )
                _validate_json_value(
                    item,
                    location=f"{location}.{key}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    raise ArtifactValidationError(
        f"{location}: unsupported JSON value type {type(value).__name__}"
    )


def _loads_json(text: str, *, source: str) -> JsonValue:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ArtifactValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON in {source}: {exc}") from exc
    _validate_json_value(value)
    return value


def load_json(path: str | Path) -> JsonValue:
    """Load strict UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(f"cannot read JSON {path}: {exc}") from exc
    return _loads_json(text, source=str(path))


def write_json(path: str | Path, value: JsonValue) -> None:
    """Atomically write deterministic UTF-8 JSON with sorted object keys."""
    _validate_json_value(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_numeric_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a numeric NPZ without pickle or object/string arrays."""
    path = Path(path)
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"cannot safely load NPZ {path}: {exc}") from exc

    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ArtifactValidationError(f"expected an NPZ archive: {path}")

    arrays: dict[str, np.ndarray] = {}
    try:
        names = loaded.files
        if len(names) != len(set(names)):
            raise ArtifactValidationError(f"duplicate array names in NPZ: {path}")
        for name in names:
            try:
                array = np.array(loaded[name], copy=True)
            except ValueError as exc:
                raise ArtifactValidationError(
                    f"unsafe array {name!r} in {path}: {exc}"
                ) from exc
            if array.dtype.kind not in "biufc":
                raise ArtifactValidationError(
                    f"array {name!r} in {path} is not numeric/boolean: {array.dtype}"
                )
            arrays[name] = array
    finally:
        loaded.close()
    return arrays


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactValidationError(f"checksum target is not a file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\x00" in value or "\\" in value:
        raise ArtifactValidationError(f"{label} is not a portable relative path")
    if value.split("/") != [part for part in value.split("/") if part not in ("", ".")]:
        raise ArtifactValidationError(f"{label} is not canonical: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ArtifactValidationError(f"{label} escapes its artifact root: {value!r}")
    return path


def resolve_contained_path(
    root: str | Path,
    relative_path: str,
    *,
    label: str = "artifact path",
) -> Path:
    """Resolve a portable relative path and reject root or symlink escape."""
    root = Path(root).resolve()
    relative = _validate_relative_path(relative_path, label=label)
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{label} escapes its artifact root: {relative_path!r}"
        ) from exc
    return candidate


def load_checksum_manifest(path: str | Path) -> dict[str, str]:
    """Parse strict sha256sum-compatible entries keyed by relative path."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(
            f"cannot read checksum manifest {path}: {exc}"
        ) from exc

    entries: dict[str, str] = {}
    manifest_order: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ArtifactValidationError(
                f"{path}:{line_number}: invalid SHA-256 manifest entry"
            )
        digest, _, relative_text = match.groups()
        relative = _validate_relative_path(
            relative_text,
            label=f"{path}:{line_number} path",
        ).as_posix()
        if relative in entries:
            raise ArtifactValidationError(
                f"{path}:{line_number}: duplicate checksum path {relative!r}"
            )
        entries[relative] = digest.lower()
        manifest_order.append(relative)
    if not entries:
        raise ArtifactValidationError(f"checksum manifest is empty: {path}")
    if manifest_order != sorted(manifest_order):
        raise ArtifactValidationError(f"checksum manifest paths are not sorted: {path}")
    return entries


def verify_checksum_manifest(
    root: str | Path,
    manifest: str | Path = "checksums.sha256",
) -> dict[str, str]:
    """Verify every relative file entry in a SHA-256 checksum manifest."""
    root = Path(root).resolve()
    manifest_input = Path(manifest)
    if manifest_input.is_absolute():
        manifest_path = manifest_input.resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"manifest escapes its artifact root: {manifest_path}"
            ) from exc
    else:
        manifest_path = resolve_contained_path(
            root,
            manifest_input.as_posix(),
            label="manifest",
        )
    entries = load_checksum_manifest(manifest_path)
    for relative, expected in entries.items():
        target = resolve_contained_path(root, relative, label="checksum path")
        actual = sha256_file(target)
        if actual != expected:
            raise ArtifactValidationError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
    return entries
