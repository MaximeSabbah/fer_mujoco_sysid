"""P2 synthetic-recovery gates for the friction parameter group.

The hidden truth lives only in this test module: a perturbed model generates
position-tracked slow-reversal data (mirroring the D021 playback decision:
a position controller moves the robot, the applied torque is recorded), and
the fit sees nothing but the recorded signals and the nominal spec.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.model_contract import (
    ModelPaths,
    build_hydrax_arm_spec,
)
from fer_mujoco_sysid.sysid import (
    MeasuredRun,
    fit_parameters,
    friction_parameters,
    measurement_sequences,
)
from fer_mujoco_sysid.sysid.adapter import set_hinge_damping

_HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_KP = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 25.0, 15.0])
_KD = np.array([10.0, 10.0, 10.0, 10.0, 4.0, 3.0, 2.0])
_AMPLITUDE_RAD = 0.3
_FREQUENCY_HZ = 0.2  # slow reversals: peak |dq| ~ 0.38 rad/s
_RAMP_S = 1.0  # amplitude ramp-in avoids a step transient at t=0
_DURATION_S = 3.0
_RUN_PHASES = (0.0, 1.1)
_WINDOW_S = 0.5  # short rollout windows keep the residual well-conditioned


def _identification_spec(model_paths: ModelPaths) -> mujoco.MjSpec:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True)


def _hidden_truth_model(
    model_paths: ModelPaths,
    *,
    frictionloss: dict[str, float] | None = None,
    damping: dict[str, float] | None = None,
) -> mujoco.MjModel:
    """Test-owned ground truth; fitting code never sees these values."""
    spec = _identification_spec(model_paths)
    for joint_name, value in (frictionloss or {}).items():
        spec.joint(joint_name).frictionloss = value
    for joint_name, value in (damping or {}).items():
        set_hinge_damping(spec.joint(joint_name), value)
    return spec.compile()


def _pd_tracking_run(
    truth_model: mujoco.MjModel, phase: float, label: str
) -> MeasuredRun:
    """Track slow per-joint reversals with a PD law; record torque and state."""
    dt = truth_model.opt.timestep
    steps = int(round(_DURATION_S / dt))
    phases = phase + np.arange(7) * (np.pi / 3.0)
    omega = 2.0 * np.pi * _FREQUENCY_HZ
    ctrl_low = truth_model.actuator_ctrlrange[:, 0]
    ctrl_high = truth_model.actuator_ctrlrange[:, 1]

    data = mujoco.MjData(truth_model)
    data.qpos[:] = _HOME_QPOS
    mujoco.mj_forward(truth_model, data)

    control_times = np.empty(steps)
    control = np.empty((steps, truth_model.nu))
    measured_times = np.empty(steps)
    measured = np.empty((steps, truth_model.nsensordata))
    for k in range(steps):
        t = k * dt
        ramp = min(t / _RAMP_S, 1.0)
        ramp_rate = 1.0 / _RAMP_S if t < _RAMP_S else 0.0
        wave = np.sin(omega * t + phases)
        wave_rate = omega * np.cos(omega * t + phases)
        q_des = _HOME_QPOS + ramp * _AMPLITUDE_RAD * wave
        v_des = _AMPLITUDE_RAD * (ramp * wave_rate + ramp_rate * wave)
        tau = np.clip(
            _KP * (q_des - data.qpos) + _KD * (v_des - data.qvel),
            ctrl_low,
            ctrl_high,
        )
        control_times[k] = t
        control[k] = tau
        data.ctrl[:] = tau
        mujoco.mj_step(truth_model, data)
        measured_times[k] = data.time
        measured[k] = data.sensordata
    return MeasuredRun(
        label=label,
        qpos0=_HOME_QPOS.copy(),
        qvel0=np.zeros(7),
        control_times=control_times,
        control=control,
        measured_times=measured_times,
        measured=measured,
    )


def _sequences(model_paths: ModelPaths, truth_model: mujoco.MjModel):
    runs = [
        _pd_tracking_run(truth_model, phase, f"reversals_{index}")
        for index, phase in enumerate(_RUN_PHASES)
    ]
    return measurement_sequences(
        _identification_spec(model_paths), runs, window_s=_WINDOW_S
    )


def test_nominal_parameters_are_a_noop(model_paths: ModelPaths) -> None:
    """Applying nominal parameters changes no compiled physical quantity."""
    from mujoco import sysid

    reference = _identification_spec(model_paths).compile()
    parameters = friction_parameters(reference)
    compiled = sysid.apply_param_modifiers(
        parameters, _identification_spec(model_paths)
    )
    assert np.array_equal(compiled.dof_frictionloss, reference.dof_frictionloss)
    assert np.array_equal(compiled.dof_damping, reference.dof_damping)
    assert np.array_equal(compiled.dof_armature, reference.dof_armature)
    assert np.array_equal(compiled.body_mass, reference.body_mass)
    assert np.array_equal(compiled.body_ipos, reference.body_ipos)
    assert np.array_equal(compiled.body_inertia, reference.body_inertia)

    run_a = _pd_tracking_run(reference, 0.0, "noop_a")
    run_b = _pd_tracking_run(compiled, 0.0, "noop_b")
    assert np.array_equal(run_a.measured, run_b.measured)
    assert np.array_equal(run_a.control, run_b.control)


def test_single_frictionloss_recovery(model_paths: ModelPaths) -> None:
    """Noise-free single-parameter recovery within 1% (P2 exit gate)."""
    truth_value = 0.9
    truth_model = _hidden_truth_model(model_paths, frictionloss={"joint4": truth_value})
    sequences = _sequences(model_paths, truth_model)

    nominal_model = _identification_spec(model_paths).compile()
    parameters = friction_parameters(
        nominal_model, joints=("joint4",), include_damping=False
    )
    # The nominal frictionloss (0.0) sits on the lower box bound.
    parameters.move_off_bounds()

    result = fit_parameters(parameters, sequences, max_iters=40)

    fitted = result.values["joint4_frictionloss"][0]
    assert fitted == pytest.approx(truth_value, rel=0.01)
    assert result.objective_reduction >= 0.99


def test_single_damping_recovery(model_paths: ModelPaths) -> None:
    """Noise-free single-parameter recovery within 1% (P2 exit gate)."""
    truth_value = 2.5
    truth_model = _hidden_truth_model(model_paths, damping={"joint2": truth_value})
    sequences = _sequences(model_paths, truth_model)

    nominal_model = _identification_spec(model_paths).compile()
    parameters = friction_parameters(
        nominal_model, joints=("joint2",), include_frictionloss=False
    )
    result = fit_parameters(parameters, sequences, max_iters=40)

    fitted = result.values["joint2_damping"][0]
    assert fitted == pytest.approx(truth_value, rel=0.01)
    assert result.objective_reduction >= 0.99


def test_frictionloss_damping_separation(model_paths: ModelPaths) -> None:
    """Slow reversals separate dry friction from viscous damping."""
    truth_frictionloss = 0.8
    truth_damping = 2.0
    truth_model = _hidden_truth_model(
        model_paths,
        frictionloss={"joint4": truth_frictionloss},
        damping={"joint4": truth_damping},
    )
    sequences = _sequences(model_paths, truth_model)

    nominal_model = _identification_spec(model_paths).compile()
    parameters = friction_parameters(nominal_model, joints=("joint4",))
    parameters.move_off_bounds()

    result = fit_parameters(parameters, sequences, max_iters=60)

    fitted_frictionloss = result.values["joint4_frictionloss"][0]
    fitted_damping = result.values["joint4_damping"][0]
    assert fitted_frictionloss == pytest.approx(truth_frictionloss, rel=0.02)
    assert fitted_damping == pytest.approx(truth_damping, rel=0.02)

    # P2 freeze threshold: pairs correlated above |0.98| are not separable.
    assert result.parameter_correlations.shape == (2, 2)
    assert abs(result.parameter_correlations[0, 1]) < 0.98
