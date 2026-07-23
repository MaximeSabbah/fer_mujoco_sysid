"""Draft 2020-12 JSON Schema loading and validation."""

from __future__ import annotations

import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import JsonValue, _loads_json

_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _is_directory(candidate: Traversable) -> bool:
    try:
        return candidate.is_dir()
    except OSError:
        return False


def schema_directory(
    explicit: str | Path | None = None,
) -> Traversable:
    """Locate schemas in an explicit directory or installed package data."""
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_dir():
            raise ArtifactValidationError(f"schema directory not found: {candidate}")
        return candidate

    package_candidate = resources.files("fer_mujoco_sysid").joinpath("schemas")
    if _is_directory(package_candidate):
        return package_candidate
    raise ArtifactValidationError(
        "schema directory not found in the installed fer_mujoco_sysid package"
    )


def _schema_filename(artifact: str, version: int) -> str:
    if _ARTIFACT_NAME.fullmatch(artifact) is None:
        raise ArtifactValidationError(f"invalid artifact schema name: {artifact!r}")
    if version < 1:
        raise ArtifactValidationError(f"schema version must be positive: {version}")
    return f"{artifact}-v{version}.schema.json"


def _read_traversable_json(path: Traversable) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(f"cannot read schema {path}: {exc}") from exc
    value = _loads_json(text, source=str(path))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"schema root must be an object: {path}")
    return value


def _all_schemas(root: Traversable) -> list[tuple[str, str, dict[str, Any]]]:
    schemas: list[tuple[str, str, dict[str, Any]]] = []
    schema_ids: set[str] = set()
    for candidate in root.iterdir():
        if candidate.is_file() and candidate.name.endswith(".schema.json"):
            schema = _read_traversable_json(candidate)
            schema_id = schema.get("$id")
            if not isinstance(schema_id, str) or not schema_id:
                raise ArtifactValidationError(
                    f"schema has no non-empty $id: {candidate}"
                )
            if schema_id in schema_ids:
                raise ArtifactValidationError(f"duplicate schema $id: {schema_id}")
            schema_ids.add(schema_id)
            schemas.append((candidate.name, schema_id, schema))
    return schemas


def _jsonschema_api() -> tuple[Any, Any, Any, Any]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource
    except ImportError as exc:
        raise RuntimeError(
            "JSON Schema validation requires the project 'jsonschema' dependency"
        ) from exc
    return Draft202012Validator, FormatChecker, Registry, Resource


def _checked_schema(schema: dict[str, Any], *, source: str) -> None:
    dialect = schema.get("$schema")
    if dialect not in (_DRAFT_2020_12, f"{_DRAFT_2020_12}#"):
        raise ArtifactValidationError(
            f"{source} is not a Draft 2020-12 schema: {dialect!r}"
        )
    Draft202012Validator, _, _, _ = _jsonschema_api()
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ArtifactValidationError(f"invalid JSON Schema {source}: {exc}") from exc


def load_schema(
    artifact: str,
    version: int = 1,
    *,
    schema_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and metaschema-check one checked-in artifact schema."""
    root = schema_directory(schema_dir)
    filename = _schema_filename(artifact, version)
    candidate = root.joinpath(filename)
    if not candidate.is_file():
        raise ArtifactValidationError(f"artifact schema not found: {candidate}")
    schema = _read_traversable_json(candidate)
    _checked_schema(schema, source=str(candidate))
    return schema


def validate_schema(
    instance: JsonValue,
    artifact: str,
    version: int = 1,
    *,
    schema_dir: str | Path | None = None,
) -> None:
    """Validate an instance against a checked-in Draft 2020-12 schema."""
    root = schema_directory(schema_dir)
    schema = load_schema(artifact, version, schema_dir=schema_dir)
    all_schemas = _all_schemas(root)
    Draft202012Validator, FormatChecker, Registry, Resource = _jsonschema_api()

    registry = Registry()
    for filename, schema_id, registered_schema in all_schemas:
        _checked_schema(registered_schema, source=schema_id)
        resource = Resource.from_contents(registered_schema)
        registry = registry.with_resource(schema_id, resource)
        # Register the filename too so future schemas may use package-local
        # relative references while the canonical identifiers remain URNs.
        registry = registry.with_resource(filename, resource)

    validator = Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    try:
        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        raise ArtifactValidationError(
            f"could not resolve or evaluate the {artifact}@{version} schema: {exc}"
        ) from exc
    if errors:
        error = errors[0]
        instance_path = "$"
        for part in error.absolute_path:
            instance_path += f"[{part}]" if isinstance(part, int) else f".{part}"
        raise ArtifactValidationError(
            f"{artifact}@{version} schema violation at {instance_path}: {error.message}"
        )
