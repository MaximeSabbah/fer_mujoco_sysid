"""Open-loop standalone MuJoCo execution of compiled FER effort protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.artifacts.errors import ArtifactValidationError
from fer_mujoco_sysid.artifacts.io import sha256_file
from fer_mujoco_sysid.artifacts.validation import validate_motion_protocol
from fer_mujoco_sysid.model_contract import (
    HYDRAX_ARM_JOINT_NAMES,
    build_hydrax_arm_model,
)

_ARM_DIMENSION = 7
_MINIMUM_TIME_TOLERANCE_S = 1e-12
_LIMIT_TOLERANCE_RAD = 1e-9
_FLOAT64_EPSILON = np.finfo(np.float64).eps

_MUJOCO_CALLBACK_GETTERS = (
    ("mjcb_passive", mujoco.get_mjcb_passive),
    ("mjcb_control", mujoco.get_mjcb_control),
    ("mjcb_contactfilter", mujoco.get_mjcb_contactfilter),
    ("mjcb_sensor", mujoco.get_mjcb_sensor),
    ("mjcb_time", mujoco.get_mjcb_time),
    ("mjcb_act_dyn", mujoco.get_mjcb_act_dyn),
    ("mjcb_act_gain", mujoco.get_mjcb_act_gain),
    ("mjcb_act_bias", mujoco.get_mjcb_act_bias),
)


@dataclass(frozen=True, slots=True)
class StandaloneRollout:
    """Copy-owned open-loop rollout following the M+1-state/M-control convention.

    ``state_time_s``, ``q_rad``, and ``dq_rad_s`` contain the initial boundary
    and every post-step boundary. The remaining arrays describe the interval
    beginning at ``control_time_s``. All arrays are C-contiguous, copy-owned,
    and marked read-only. This primitive applies only the protocol's desired
    effort feedforward; it does not emulate a ROS trajectory controller.
    """

    state_time_s: NDArray[np.float64]
    control_time_s: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    dq_rad_s: NDArray[np.float64]
    desired_effort_feedforward_Nm: NDArray[np.float64]
    simulated_actuator_effort_Nm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _ArmMapping:
    joint_ids: NDArray[np.int64]
    qpos_addresses: NDArray[np.int64]
    dof_addresses: NDArray[np.int64]
    actuator_ids: NDArray[np.int64]


def _immutable_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _model_error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(f"standalone MuJoCo model contract: {message}")


def _require_no_mujoco_callbacks(*, boundary: str) -> None:
    active = [name for name, getter in _MUJOCO_CALLBACK_GETTERS if getter() is not None]
    if active:
        raise ArtifactValidationError(
            f"{boundary}: standalone rollout requires all process-global MuJoCo "
            f"callbacks to be unset; active callbacks: {active!r}"
        )


def _time_tolerance_s(boundary_index: int, expected_time_s: float) -> float:
    # MuJoCo advances time through repeated binary64 additions. The standard
    # forward-error bound therefore grows with both the number of additions and
    # elapsed time. The factor of two leaves room for correctly rounded parsing
    # while remaining tight enough to detect a changed integration step.
    accumulated_roundoff = (
        2.0 * _FLOAT64_EPSILON * max(boundary_index, 1) * max(abs(expected_time_s), 1.0)
    )
    return max(_MINIMUM_TIME_TOLERANCE_S, accumulated_roundoff)


def _arm_mapping(model: mujoco.MjModel) -> _ArmMapping:
    dimensions = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "njnt": model.njnt,
    }
    unexpected = {
        name: value for name, value in dimensions.items() if value != _ARM_DIMENSION
    }
    if unexpected:
        raise _model_error(
            "expected exactly seven arm positions, velocities, joints, and "
            f"actuators; got {unexpected!r}"
        )

    unsupported_state = {
        "na": model.na,
        "nmocap": model.nmocap,
        "npluginstate": model.npluginstate,
        "nuserdata": model.nuserdata,
    }
    nonzero_state = {
        name: value for name, value in unsupported_state.items() if value != 0
    }
    if nonzero_state:
        raise _model_error(
            f"unsupported non-arm state dimensions are nonzero: {nonzero_state!r}"
        )
    if model.neq != 0:
        raise _model_error(
            "equality constraints are unsupported in the arm projection "
            f"(neq={model.neq})"
        )
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise _model_error("integrator must be mjINT_IMPLICITFAST")
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT:
        raise _model_error("contacts must be disabled")

    try:
        joint_ids = np.asarray(
            [model.joint(name).id for name in HYDRAX_ARM_JOINT_NAMES],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise _model_error(
            "canonical FER joints must map to model joints joint1 through joint7"
        ) from exc
    if len(set(int(value) for value in joint_ids)) != _ARM_DIMENSION:
        raise _model_error("joint1 through joint7 do not map one-to-one")
    if set(int(value) for value in joint_ids) != set(range(model.njnt)):
        raise _model_error("model contains joints outside joint1 through joint7")
    if any(
        model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
        for joint_id in joint_ids
    ):
        raise _model_error("joint1 through joint7 must all be scalar hinge joints")

    qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    dof_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    if set(int(value) for value in qpos_addresses) != set(range(model.nq)):
        raise _model_error("joint1 through joint7 do not cover qpos one-to-one")
    if set(int(value) for value in dof_addresses) != set(range(model.nv)):
        raise _model_error("joint1 through joint7 do not cover qvel one-to-one")

    actuator_ids = np.empty(_ARM_DIMENSION, dtype=np.int64)
    assigned_actuators: set[int] = set()
    for index, joint_id in enumerate(joint_ids):
        matching = [
            actuator_id
            for actuator_id in range(model.nu)
            if (
                model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[actuator_id, 0]) == int(joint_id)
            )
        ]
        if len(matching) != 1:
            joint_name = HYDRAX_ARM_JOINT_NAMES[index]
            raise _model_error(
                f"{joint_name} must have exactly one direct joint actuator; "
                f"found {len(matching)}"
            )
        actuator_id = matching[0]
        if actuator_id in assigned_actuators:
            raise _model_error("arm actuators do not map one-to-one to joints")
        assigned_actuators.add(actuator_id)
        actuator_ids[index] = actuator_id
        _validate_direct_effort_actuator(model, actuator_id)

    if assigned_actuators != set(range(model.nu)):
        raise _model_error("model contains actuators outside the seven arm joints")
    return _ArmMapping(
        joint_ids=joint_ids,
        qpos_addresses=qpos_addresses,
        dof_addresses=dof_addresses,
        actuator_ids=actuator_ids,
    )


def _validate_direct_effort_actuator(
    model: mujoco.MjModel,
    actuator_id: int,
) -> None:
    actuator_name = model.actuator(actuator_id).name
    if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE:
        raise _model_error(
            f"actuator {actuator_name!r} must have no activation dynamics"
        )
    if model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
        raise _model_error(f"actuator {actuator_name!r} must use fixed gain")
    if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
        raise _model_error(f"actuator {actuator_name!r} must have no bias")

    expected_gain = np.zeros(model.actuator_gainprm.shape[1])
    expected_gain[0] = 1.0
    if not np.array_equal(model.actuator_gainprm[actuator_id], expected_gain):
        raise _model_error(f"actuator {actuator_name!r} must have unit gain")
    if np.any(model.actuator_biasprm[actuator_id] != 0.0):
        raise _model_error(f"actuator {actuator_name!r} must have zero bias parameters")
    expected_gear = np.zeros(model.actuator_gear.shape[1])
    expected_gear[0] = 1.0
    if not np.array_equal(model.actuator_gear[actuator_id], expected_gear):
        raise _model_error(f"actuator {actuator_name!r} must have unit scalar gear")


def _check_joint_limits(
    model: mujoco.MjModel,
    mapping: _ArmMapping,
    q_rad: np.ndarray,
    *,
    label: str,
) -> None:
    for index, joint_id in enumerate(mapping.joint_ids):
        if not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        value = q_rad[index]
        if value < lower - _LIMIT_TOLERANCE_RAD or value > upper + _LIMIT_TOLERANCE_RAD:
            raise ArtifactValidationError(
                f"{label}: {HYDRAX_ARM_JOINT_NAMES[index]} position {value!r} "
                f"is outside model limits [{lower!r}, {upper!r}]"
            )


def _check_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: _ArmMapping,
    *,
    label: str,
) -> None:
    finite_fields = {
        "time": np.asarray([data.time]),
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "qfrc_actuator": data.qfrc_actuator,
    }
    invalid = [
        name
        for name, values in finite_fields.items()
        if not np.all(np.isfinite(values))
    ]
    if invalid:
        raise ArtifactValidationError(
            f"{label}: non-finite MuJoCo values in {', '.join(invalid)}"
        )
    warning_ids = [
        index for index, warning in enumerate(data.warning) if warning.number > 0
    ]
    if warning_ids:
        raise ArtifactValidationError(
            f"{label}: MuJoCo reported divergence warning IDs {warning_ids!r}"
        )
    _check_joint_limits(
        model,
        mapping,
        data.qpos[mapping.qpos_addresses],
        label=label,
    )


def _compile_source_model_at_timestep(
    source_path: Path,
    *,
    timestep_s: float,
) -> mujoco.MjModel:
    try:
        return build_hydrax_arm_model(source_path, timestep=timestep_s)
    except (mujoco.FatalError, OSError, RuntimeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"cannot compile standalone MuJoCo source model {source_path}: {exc}"
        ) from exc


def _validate_unsaturated_effort(
    model: mujoco.MjModel,
    mapping: _ArmMapping,
    desired_effort: np.ndarray,
) -> None:
    for joint_index, actuator_id in enumerate(mapping.actuator_ids):
        lower_bounds: list[tuple[str, float]] = []
        upper_bounds: list[tuple[str, float]] = []
        if model.actuator_ctrllimited[actuator_id]:
            lower, upper = model.actuator_ctrlrange[actuator_id]
            lower_bounds.append(("control", float(lower)))
            upper_bounds.append(("control", float(upper)))
        if model.actuator_forcelimited[actuator_id]:
            lower, upper = model.actuator_forcerange[actuator_id]
            lower_bounds.append(("force", float(lower)))
            upper_bounds.append(("force", float(upper)))

        for interval_index, value in enumerate(desired_effort[:, joint_index]):
            for limit_kind, lower in lower_bounds:
                if value < lower:
                    raise ArtifactValidationError(
                        f"control interval {interval_index}: desired effort "
                        f"{value!r} N*m for "
                        f"{HYDRAX_ARM_JOINT_NAMES[joint_index]} is below the "
                        f"actuator {limit_kind} limit {lower!r} N*m"
                    )
            for limit_kind, upper in upper_bounds:
                if value > upper:
                    raise ArtifactValidationError(
                        f"control interval {interval_index}: desired effort "
                        f"{value!r} N*m for "
                        f"{HYDRAX_ARM_JOINT_NAMES[joint_index]} is above the "
                        f"actuator {limit_kind} limit {upper!r} N*m"
                    )


def run_open_loop_effort_protocol(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    protocol_root: str | Path,
    model_source_path: str | Path,
    command_semantics: Literal["zero_order_hold"],
    schema_dir: str | Path | None = None,
) -> StandaloneRollout:
    """Execute feedforward effort without ROS, feedback, or wall-clock timing.

    Each protocol effort sample at knot ``i`` is held over exactly one model
    step, ``[t_i, t_(i+1))``. The terminal effort knot has no physical interval
    and is therefore validated as part of the protocol but is not emitted as a
    control sample. This low-level plant primitive deliberately ignores desired
    position, velocity, and acceleration after initialization. A separate
    controller layer is required before claiming parity with ROS effort-mode
    trajectory playback.
    """

    _require_no_mujoco_callbacks(boundary="before validation")
    validate_motion_protocol(
        manifest,
        arrays,
        root=protocol_root,
        schema_dir=schema_dir,
    )
    if manifest["command_interface"] != "joint_effort":
        raise ArtifactValidationError(
            "standalone simulation initially supports only joint_effort protocols"
        )
    if command_semantics != "zero_order_hold":
        raise ArtifactValidationError(
            "standalone effort commands require zero_order_hold semantics"
        )

    times_ns = arrays["time_from_start_ns"]
    commanded = arrays.get("tau_feedforward_Nm")
    if commanded is None:
        raise ArtifactValidationError(
            "joint_effort protocol is missing tau_feedforward_Nm"
        )
    expected_shape = (times_ns.shape[0], _ARM_DIMENSION)
    if commanded.shape != expected_shape:
        raise ArtifactValidationError(
            f"tau_feedforward_Nm must have shape {expected_shape!r}, "
            f"got {commanded.shape!r}"
        )

    source_path = Path(model_source_path).expanduser().resolve()
    source_digest_before = sha256_file(source_path)
    declared_source_digest = manifest["context"]["source_model"]["sha256"]
    if source_digest_before != declared_source_digest:
        raise ArtifactValidationError(
            "protocol source-model SHA-256 mismatch: "
            f"expected {declared_source_digest}, got {source_digest_before}"
        )

    expected_timestep = manifest["sample_period_ns"] * 1e-9
    simulation_model = _compile_source_model_at_timestep(
        source_path,
        timestep_s=expected_timestep,
    )
    mapping = _arm_mapping(simulation_model)
    if simulation_model.opt.timestep != expected_timestep:
        raise _model_error(
            f"timestep {simulation_model.opt.timestep!r} s does not match "
            f"protocol period {expected_timestep!r} s"
        )

    transition_count = times_ns.shape[0] - 1
    q = np.empty((transition_count + 1, _ARM_DIMENSION), dtype=np.float64)
    dq = np.empty((transition_count + 1, _ARM_DIMENSION), dtype=np.float64)
    realized = np.empty((transition_count, _ARM_DIMENSION), dtype=np.float64)
    desired_intervals = commanded[:-1]
    _validate_unsaturated_effort(
        simulation_model,
        mapping,
        desired_intervals,
    )

    data = mujoco.MjData(simulation_model)
    data.time = 0.0
    data.qpos[mapping.qpos_addresses] = np.asarray(
        manifest["start_state"]["position_rad"],
        dtype=np.float64,
    )
    data.qvel[mapping.dof_addresses] = np.asarray(
        manifest["start_state"]["velocity_rad_s"],
        dtype=np.float64,
    )
    data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation_model, data)
    _check_data(simulation_model, data, mapping, label="initial state")

    scheduled_time_s = (
        np.arange(times_ns.shape[0], dtype=np.float64) * expected_timestep
    )
    for boundary_index in range(transition_count + 1):
        expected_time = scheduled_time_s[boundary_index]
        if abs(data.time - expected_time) > _time_tolerance_s(
            boundary_index,
            expected_time,
        ):
            raise ArtifactValidationError(
                f"boundary {boundary_index}: MuJoCo time {data.time!r} s "
                f"does not match protocol time {expected_time!r} s"
            )
        q[boundary_index] = data.qpos[mapping.qpos_addresses]
        dq[boundary_index] = data.qvel[mapping.dof_addresses]
        if boundary_index == transition_count:
            break

        data.ctrl[:] = 0.0
        data.ctrl[mapping.actuator_ids] = commanded[boundary_index]
        mujoco.mj_forward(simulation_model, data)
        _check_data(
            simulation_model,
            data,
            mapping,
            label=f"control interval {boundary_index}",
        )
        realized[boundary_index] = data.qfrc_actuator[mapping.dof_addresses]
        if not np.array_equal(
            realized[boundary_index],
            desired_intervals[boundary_index],
        ):
            raise ArtifactValidationError(
                f"control interval {boundary_index}: simulated actuator effort "
                "does not exactly equal the unsaturated desired effort "
                "feedforward"
            )
        mujoco.mj_step(simulation_model, data)
        _check_data(
            simulation_model,
            data,
            mapping,
            label=f"post-step boundary {boundary_index + 1}",
        )

    _require_no_mujoco_callbacks(boundary="after rollout")
    source_digest_after = sha256_file(source_path)
    if source_digest_after != source_digest_before:
        raise ArtifactValidationError(
            f"MuJoCo source model changed during rollout: {source_path}"
        )

    return StandaloneRollout(
        state_time_s=_immutable_copy(scheduled_time_s),
        control_time_s=_immutable_copy(scheduled_time_s[:-1]),
        q_rad=_immutable_copy(q),
        dq_rad_s=_immutable_copy(dq),
        desired_effort_feedforward_Nm=_immutable_copy(desired_intervals),
        simulated_actuator_effort_Nm=_immutable_copy(realized),
    )
