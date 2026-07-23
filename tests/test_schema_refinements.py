from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

from fer_mujoco_sysid.artifacts import (
    FER_ARM_JOINT_ORDER,
    load_schema,
    validate_schema,
)

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_SHA256 = "a" * 64
_REVISION = "b" * 40


def _file_reference(path: str = "config.json") -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _SHA256,
        "size_bytes": 1,
    }


def _artifact_reference(
    *,
    artifact_id: str,
    schema: str,
    path: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "schema": schema,
        "manifest_sha256": _SHA256,
        "locator": {
            "kind": "relative",
            "path": path,
        },
    }


def _source_document() -> dict[str, Any]:
    return {
        "repository": "https://example.invalid/fer.git",
        "revision": _REVISION,
        "path": "models/fer.xml",
        "sha256": _SHA256,
    }


def _fragment_validator(
    artifact: str,
    fragment: str,
) -> Draft202012Validator:
    registry = Registry()
    target = load_schema(artifact)
    for schema_name in sorted({"common", artifact}):
        schema = load_schema(schema_name)
        registry = registry.with_resource(
            schema["$id"],
            Resource.from_contents(schema),
        )
    return Draft202012Validator(
        {
            "$schema": _DRAFT_2020_12,
            "$ref": f"{target['$id']}{fragment}",
        },
        registry=registry,
        format_checker=FormatChecker(),
    )


def _acquisition_run() -> dict[str, Any]:
    return {
        "schema": "fer-mujoco-sysid/acquisition-run@1",
        "artifact_id": "run-test",
        "started_at_utc": "2026-07-23T00:00:00Z",
        "ended_at_utc": "2026-07-23T00:00:01Z",
        "backend": "standalone_mujoco",
        "protocol": _artifact_reference(
            artifact_id="protocol-test",
            schema="fer-mujoco-sysid/motion-protocol@1",
            path="protocol/protocol.json",
        ),
        "protocol_reference_signals": {
            "desired_position": "scheduled_position",
            "desired_velocity": "scheduled_velocity",
            "desired_acceleration": "scheduled_acceleration",
        },
        "protocol_timing": {
            "clock_id": "simulation",
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
            "robot_id": "simulation-fer",
            "end_effector_id": "fer-hand",
            "payload": {
                "payload_id": "no-payload",
                "mass_kg": 0.0,
                "center_of_mass_m": [0.0, 0.0, 0.0],
            },
        },
        "controller": {
            "name": "trajectory-player",
            "type": "joint-trajectory",
            "update_rate_hz": 1000.0,
            "configuration": _file_reference(),
        },
        "source_model": _source_document(),
        "clock_domains": [
            {
                "clock_id": "simulation",
                "domain": "simulation_clock",
                "epoch": "run start",
                "tick_unit": "ns",
                "resolution_ns": 1,
                "timestamp_source": "mujoco data time",
            }
        ],
        "signals": [
            {
                "name": "scheduled_position",
                "quantity": "joint_position",
                "semantic_role": "desired_joint_position",
                "unit": "rad",
                "joint_order": list(FER_ARM_JOINT_ORDER),
                "positive_direction": "same_as_joint_coordinate",
                "clock_id": "simulation",
                "source": {
                    "kind": "protocol_player",
                    "protocol_artifact_id": "protocol-test",
                    "field": "q_rad",
                },
                "sample_count": 2,
            }
        ],
        "recording_files": [_file_reference("raw.npz")],
        "recorded_topics": [],
        "software": [
            {
                "name": "fer-mujoco-sysid",
                "version": "0.1.0",
            }
        ],
        "outcome": {
            "status": "completed",
            "reason_code": "completed",
        },
    }


def _parameter_component() -> dict[str, Any]:
    return {
        "name": "frictionloss",
        "value": 0.1,
        "unit": "N*m",
        "nominal_value": 0.05,
        "initial_value": 0.05,
        "bounds": {
            "lower": 0.0,
            "upper": 1.0,
        },
        "bound_status": "none",
        "identifiability_status": "retained",
        "export_policy": "mjcf",
    }


def _split_member() -> dict[str, Any]:
    return {
        "trajectory": _artifact_reference(
            artifact_id="trajectory-test",
            schema="fer-mujoco-sysid/normalized-trajectory@1",
            path="trajectories/trajectory.json",
        ),
        "lineage_group_id": "lineage-test",
        "protocol_artifact_id": "protocol-test",
        "source_run_artifact_id": "run-test",
    }


def test_acquisition_protocol_validation_is_optional_but_defined() -> None:
    schema = load_schema("acquisition-run")

    assert "protocol_validation" not in schema["required"]
    assert "protocol_validation" in schema["properties"]
    validate_schema(_acquisition_run(), "acquisition-run")


@pytest.mark.parametrize("bound_status", ["none", "lower", "upper", "both"])
def test_parameter_component_accepts_explicit_bound_status(
    bound_status: str,
) -> None:
    component = _parameter_component()
    component["bound_status"] = bound_status

    _fragment_validator("common", "#/$defs/parameterComponent").validate(component)


def test_parameter_component_requires_valid_bound_status() -> None:
    validator = _fragment_validator("common", "#/$defs/parameterComponent")
    missing = _parameter_component()
    del missing["bound_status"]
    invalid = _parameter_component()
    invalid["bound_status"] = "near_lower"

    with pytest.raises(ValidationError):
        validator.validate(missing)
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_uncertainty_none_has_no_confidence_interval() -> None:
    validator = _fragment_validator("common", "#/$defs/parameterComponent")
    component = _parameter_component()
    component["uncertainty"] = {"method": "none"}
    validator.validate(component)

    component["uncertainty"]["confidence_level"] = 0.95
    with pytest.raises(ValidationError):
        validator.validate(component)


@pytest.mark.parametrize("method", ["local_jacobian", "trajectory_bootstrap"])
def test_quantified_uncertainty_requires_complete_interval(method: str) -> None:
    validator = _fragment_validator("common", "#/$defs/parameterComponent")
    component = _parameter_component()
    component["uncertainty"] = {
        "method": method,
        "confidence_level": 0.95,
        "lower": 0.08,
        "upper": 0.12,
    }
    validator.validate(component)

    incomplete = deepcopy(component)
    del incomplete["uncertainty"]["lower"]
    with pytest.raises(ValidationError):
        validator.validate(incomplete)


def test_fit_data_use_is_required_and_exact() -> None:
    schema = load_schema("fit-result")
    validator = _fragment_validator("fit-result", "#/properties/data_use")
    valid = {
        "optimizer": "fit",
        "model_selection": ["fit", "development"],
        "final_evaluation": "held_out_test",
    }

    assert "data_use" in schema["required"]
    validator.validate(valid)

    for invalid in (
        {**valid, "optimizer": "development"},
        {**valid, "model_selection": ["development", "fit"]},
        {**valid, "final_evaluation": "development"},
        {**valid, "unexpected": True},
    ):
        with pytest.raises(ValidationError):
            validator.validate(invalid)


def test_other_exclusion_requires_nonempty_details() -> None:
    validator = _fragment_validator("splits", "#/$defs/excludedMember")
    other = {
        "member": _split_member(),
        "reason_code": "other",
    }

    with pytest.raises(ValidationError):
        validator.validate(other)
    with pytest.raises(ValidationError):
        validator.validate({**other, "details": ""})
    validator.validate({**other, "details": "Manually reviewed exclusion."})


def test_named_exclusion_does_not_require_details() -> None:
    _fragment_validator("splits", "#/$defs/excludedMember").validate(
        {
            "member": _split_member(),
            "reason_code": "quality_failed",
        }
    )
