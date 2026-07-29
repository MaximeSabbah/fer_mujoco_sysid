"""P3 slice 1 gates: friction-family excitation protocol generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.excitation import (
    FER_ACCELERATION_LIMIT_RAD_S2,
    FER_VELOCITY_LIMIT_RAD_S,
    FrictionProtocolSpec,
    InertialProtocolSpec,
    ProtocolLimitError,
    generate_friction_protocol,
    generate_inertial_protocol,
    validate_workspace_clearance,
    write_protocol_bundle,
)
from fer_mujoco_sysid.fitting import (
    CONDITIONING_RATIO_MINIMUM,
    CORRELATION_FREEZE_LIMIT,
    MeasuredRun,
    conditioning_report,
    friction_parameters,
    measurement_sequences,
)
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec

_SPEC = FrictionProtocolSpec(
    protocol_id="friction-test",
    seed=11,
    amplitudes_rad=(0.25,) * 7,
    cruise_speeds_rad_s=(0.05, 0.3),
    hold_s=0.5,
    settle_s=0.3,
)


@pytest.fixture(scope="module")
def nominal_model(model_paths: ModelPaths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True).compile()


def test_generation_is_deterministic(nominal_model: mujoco.MjModel) -> None:
    first = generate_friction_protocol(_SPEC, nominal_model)
    second = generate_friction_protocol(_SPEC, nominal_model)
    assert np.array_equal(first.q_rad, second.q_rad)
    assert np.array_equal(first.dq_rad_s, second.dq_rad_s)
    assert np.array_equal(first.ddq_rad_s2, second.ddq_rad_s2)
    assert first.segments == second.segments
    assert first.analysis_windows == second.analysis_windows

    reseeded = generate_friction_protocol(replace(_SPEC, seed=12), nominal_model)
    assert not np.array_equal(first.q_rad, reseeded.q_rad)


def test_sweeps_contain_constant_velocity_cruises(
    nominal_model: mujoco.MjModel,
) -> None:
    """The S-curve property: every excitation sweep holds its cruise speed."""
    compiled = generate_friction_protocol(_SPEC, nominal_model)
    amplitudes = np.max(np.abs(compiled.q_rad - np.asarray(_SPEC.home_qpos)), axis=0)
    leader = int(np.argmax(amplitudes))
    speed_by_tag = {
        f"{int(round(speed * 1000))}mrad_s": speed
        for speed in _SPEC.cruise_speeds_rad_s
    }
    checked = 0
    for segment in compiled.segments:
        if segment.kind != "excitation":
            continue
        tag = segment.segment_id.split("_to_")[0].removeprefix("sweep_")
        speed = speed_by_tag[tag]
        window = compiled.dq_rad_s[
            segment.start_index : segment.end_index_exclusive, leader
        ]
        cruise = np.abs(np.abs(window) - speed) < 1e-9
        assert cruise.sum() >= 100, segment.segment_id
        checked += 1
    assert checked == 2 * len(_SPEC.cruise_speeds_rad_s)
    assert len(compiled.analysis_windows) == checked
    for window in compiled.analysis_windows:
        np.testing.assert_allclose(
            compiled.ddq_rad_s2[window.start_index : window.end_index_exclusive],
            0.0,
        )
        np.testing.assert_allclose(
            compiled.dq_rad_s[window.start_index : window.end_index_exclusive],
            np.repeat(
                np.asarray(window.nominal_velocity_rad_s)[None, :],
                window.end_index_exclusive - window.start_index,
                axis=0,
            ),
        )


def test_rejects_excessive_cruise_speed(nominal_model: mujoco.MjModel) -> None:
    # Steep ramps so the cruise geometrically fits and the FER velocity
    # limit check is what trips.
    fast = replace(
        _SPEC,
        cruise_speeds_rad_s=(3.0,),
        amplitudes_rad=(0.5,) * 7,
        cruise_acceleration_rad_s2=40.0,
    )
    with pytest.raises(ProtocolLimitError, match="velocity"):
        generate_friction_protocol(fast, nominal_model)


def test_rejects_position_margin_violation(
    nominal_model: mujoco.MjModel,
) -> None:
    wide = replace(_SPEC, amplitudes_rad=(0.25,) * 3 + (1.5,) + (0.25,) * 3)
    with pytest.raises(ProtocolLimitError, match="position range"):
        generate_friction_protocol(wide, nominal_model)


def test_bundle_round_trip(nominal_model: mujoco.MjModel, tmp_path: Path) -> None:
    compiled = generate_friction_protocol(_SPEC, nominal_model)
    created_at = "2026-07-27T00:00:00Z"
    root = write_protocol_bundle(
        compiled,
        tmp_path / "protocols",
        model=nominal_model,
        created_at=created_at,
    )
    for name in ("protocol.json", "desired.npz", "checksums.sha256"):
        assert (root / name).is_file()

    # Immutability: the canonical bundle cannot be overwritten.
    with pytest.raises(FileExistsError):
        write_protocol_bundle(
            compiled,
            tmp_path / "protocols",
            model=nominal_model,
            created_at=created_at,
        )

    # Determinism: an identical build in a fresh root yields the same content.
    import json

    other = write_protocol_bundle(
        generate_friction_protocol(_SPEC, nominal_model),
        tmp_path / "other",
        model=nominal_model,
        created_at=created_at,
    )
    first_manifest = json.loads((root / "protocol.json").read_text())
    second_manifest = json.loads((other / "protocol.json").read_text())
    assert first_manifest["content_sha256"] == second_manifest["content_sha256"]


@pytest.mark.slow
def test_protocol_conditions_the_friction_block(
    model_paths: ModelPaths, nominal_model: mujoco.MjModel
) -> None:
    """Simulated playback of the protocol identifies all 14 friction params."""
    compiled = generate_friction_protocol(_SPEC, nominal_model)
    dt = nominal_model.opt.timestep
    stride = int(round(dt / _SPEC.sample_period_s))
    q_des = compiled.q_rad[::stride]
    dq_des = compiled.dq_rad_s[::stride]
    kp = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 25.0, 15.0])
    kd = np.array([10.0, 10.0, 10.0, 10.0, 4.0, 3.0, 2.0])

    data = mujoco.MjData(nominal_model)
    data.qpos[:] = compiled.q_rad[0]
    mujoco.mj_forward(nominal_model, data)
    steps = len(q_des)
    control_times = np.arange(steps) * dt
    control = np.empty((steps, 7))
    measured_times = np.empty(steps)
    measured = np.empty((steps, nominal_model.nsensordata))
    low = nominal_model.actuator_ctrlrange[:, 0]
    high = nominal_model.actuator_ctrlrange[:, 1]
    for k in range(steps):
        tau = np.clip(
            kp * (q_des[k] - data.qpos) + kd * (dq_des[k] - data.qvel),
            low,
            high,
        )
        control[k] = tau
        data.ctrl[:] = tau
        mujoco.mj_step(nominal_model, data)
        measured_times[k] = data.time
        measured[k] = data.sensordata
    run = MeasuredRun(
        label="protocol_playback",
        qpos0=compiled.q_rad[0],
        qvel0=np.zeros(7),
        control_times=control_times,
        control=control,
        measured_times=measured_times,
        measured=measured,
    )
    sequences = measurement_sequences(
        build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True),
        [run],
        window_s=0.5,
    )

    parameters = friction_parameters(nominal_model).move_off_bounds()
    report = conditioning_report(parameters, sequences)

    assert report.conditioning_ratio >= CONDITIONING_RATIO_MINIMUM
    correlations = report.parameter_correlations
    assert correlations.shape == (14, 14)
    for joint in range(7):
        pair = abs(correlations[2 * joint, 2 * joint + 1])
        assert pair < CORRELATION_FREEZE_LIMIT, f"joint{joint + 1}: {pair:.3f}"


_INERTIAL = InertialProtocolSpec(protocol_id="inertial-test", seed=77, candidates=8)


def test_inertial_protocol_starts_and_ends_at_rest(
    nominal_model: mujoco.MjModel,
) -> None:
    compiled = generate_inertial_protocol(_INERTIAL, nominal_model)
    assert np.abs(compiled.dq_rad_s[0]).max() == 0.0
    assert np.abs(compiled.dq_rad_s[-1]).max() == 0.0
    assert np.abs(compiled.ddq_rad_s2[0]).max() == 0.0
    assert np.abs(compiled.ddq_rad_s2[-1]).max() == 0.0
    assert len(compiled.analysis_windows) == 1
    assert compiled.analysis_windows[0].parent_segment_id == "fourier"


def test_inertial_protocol_decorrelates_the_joints(
    nominal_model: mujoco.MjModel,
) -> None:
    """The property the friction family lacks: joints must move independently
    or link inertias cannot be told apart."""
    inertial = generate_inertial_protocol(_INERTIAL, nominal_model)
    friction = generate_friction_protocol(_SPEC, nominal_model)

    def worst_cross_correlation(dq: np.ndarray) -> float:
        matrix = np.corrcoef(dq.T)
        return float(np.abs(matrix[~np.eye(7, dtype=bool)]).max())

    assert worst_cross_correlation(friction.dq_rad_s) > 0.99
    assert worst_cross_correlation(inertial.dq_rad_s) < 0.7


def test_inertial_protocol_excites_acceleration(
    nominal_model: mujoco.MjModel,
) -> None:
    """Armature and inertia are acceleration effects; the friction family
    barely accelerates, the inertial family must."""
    inertial = generate_inertial_protocol(_INERTIAL, nominal_model)
    friction = generate_friction_protocol(_SPEC, nominal_model)
    assert np.abs(inertial.ddq_rad_s2).max() > 5.0 * np.abs(friction.ddq_rad_s2).max()


def test_inertial_protocol_respects_limits(nominal_model: mujoco.MjModel) -> None:
    compiled = generate_inertial_protocol(_INERTIAL, nominal_model)
    margin = 1.0 - _INERTIAL.limit_margin_fraction
    assert np.all(
        np.abs(compiled.dq_rad_s).max(0) <= margin * FER_VELOCITY_LIMIT_RAD_S + 1e-9
    )
    assert np.all(
        np.abs(compiled.ddq_rad_s2).max(0)
        <= margin * FER_ACCELERATION_LIMIT_RAD_S2 + 1e-9
    )


def test_inertial_protocol_is_deterministic(nominal_model: mujoco.MjModel) -> None:
    first = generate_inertial_protocol(_INERTIAL, nominal_model)
    second = generate_inertial_protocol(_INERTIAL, nominal_model)
    assert np.array_equal(first.q_rad, second.q_rad)


def test_table_clearance_is_checked(
    model_paths: ModelPaths, nominal_model: mujoco.MjModel
) -> None:
    """A pose that reaches below the table must be rejected."""
    compiled = generate_friction_protocol(_SPEC, nominal_model)
    validate_workspace_clearance(model_paths.hydrax, compiled.q_rad)

    # Fold joint 2 fully forward with the elbow bent: the wrist swings 0.43 m
    # below the table plane the robot is bolted to.
    below = np.tile(np.array([0.0, 1.76, 0.0, -1.57, 0.0, 3.0, 0.0]), (5, 1))
    with pytest.raises(ProtocolLimitError, match="table"):
        validate_workspace_clearance(model_paths.hydrax, below, stride=1)


def test_committed_protocols_clear_the_table(
    model_paths: ModelPaths, nominal_model: mujoco.MjModel
) -> None:
    from fer_mujoco_sysid.campaign import CAMPAIGN, compile_protocol

    for spec in CAMPAIGN:
        compile_protocol(spec, nominal_model, model_paths.hydrax)
