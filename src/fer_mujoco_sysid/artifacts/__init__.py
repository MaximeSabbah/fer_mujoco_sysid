"""Portable, pickle-free artifact I/O and semantic validation."""

from fer_mujoco_sysid.artifacts.bundle import (
    CHECKSUM_MANIFEST_NAME,
    SEAL_NAME,
    SEAL_SCHEMA,
    finalize_bundle,
    verify_sealed_bundle,
)
from fer_mujoco_sysid.artifacts.catalog import (
    LocalArtifactResolver,
    ResolvedArtifact,
    ValidatedDataset,
    validate_dataset_bundle,
    validate_fit_against_dataset,
)
from fer_mujoco_sysid.artifacts.content import content_sha256
from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import (
    JsonValue,
    load_checksum_manifest,
    load_json,
    load_numeric_npz,
    resolve_contained_path,
    sha256_file,
    verify_checksum_manifest,
    write_json,
)
from fer_mujoco_sysid.artifacts.schemas import (
    load_schema,
    schema_directory,
    validate_schema,
)
from fer_mujoco_sysid.artifacts.validation import (
    FER_ARM_JOINT_ORDER,
    validate_motion_protocol,
    validate_normalized_trajectory,
)

__all__ = [
    "CHECKSUM_MANIFEST_NAME",
    "FER_ARM_JOINT_ORDER",
    "SEAL_NAME",
    "SEAL_SCHEMA",
    "ArtifactValidationError",
    "JsonValue",
    "LocalArtifactResolver",
    "ResolvedArtifact",
    "ValidatedDataset",
    "content_sha256",
    "finalize_bundle",
    "load_checksum_manifest",
    "load_json",
    "load_numeric_npz",
    "load_schema",
    "resolve_contained_path",
    "schema_directory",
    "sha256_file",
    "validate_dataset_bundle",
    "validate_fit_against_dataset",
    "validate_motion_protocol",
    "validate_normalized_trajectory",
    "validate_schema",
    "verify_checksum_manifest",
    "verify_sealed_bundle",
    "write_json",
]
