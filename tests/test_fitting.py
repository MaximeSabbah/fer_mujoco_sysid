"""P2 synthetic-recovery gates for the friction parameter group.

The hidden truth lives only in this test module: a perturbed model generates
position-tracked slow-reversal data (mirroring the D021 playback decision:
a position controller moves the robot, the applied torque is recorded), and
the fit sees nothing but the recorded signals and the nominal spec.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from mujoco import sysid

import fer_mujoco_sysid.fitting as fitting_module
from fer_mujoco_sysid.fitting import (
    CONDITIONING_RATIO_MINIMUM,
    CORRELATION_FREEZE_LIMIT,
    BodyInertialCorrection,
    ConditioningReport,
    FitStage,
    IdentificationAcceptanceError,
    MeasuredRun,
    MultistartResult,
    _representative_bounds,
    _resampling_safe_control_times,
    armature_parameters,
    body_full_inertia,
    cad_prior_inertial_parameters,
    combine_parameters,
    conditioning_acceptance,
    conditioning_report,
    fit_multistart,
    fit_parameters,
    fit_staged,
    friction_parameters,
    inertial_parameters,
    measurement_sequences,
    select_observable_subset,
    set_hinge_damping,
)
from fer_mujoco_sysid.model import (
    ModelPaths,
    build_hydrax_arm_spec,
)

_HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_KP = np.array([100.0, 100.0, 100.0, 100.0, 40.0, 25.0, 15.0])
_KD = np.array([10.0, 10.0, 10.0, 10.0, 4.0, 3.0, 2.0])
_AMPLITUDE_RAD = 0.3
_FREQUENCY_HZ = 0.2  # slow reversals: peak |dq| ~ 0.38 rad/s
# The dynamic protocol accelerates hard enough (~8.5 rad/s^2 peak) that
# armature torque dominates; the slow protocol barely excites it.
_DYNAMIC_AMPLITUDE_RAD = 0.15
_DYNAMIC_FREQUENCY_HZ = 1.2
_RAMP_S = 1.0  # amplitude ramp-in avoids a step transient at t=0
_DURATION_S = 3.0
_RUN_PHASES = (0.0, 1.1)
_WINDOW_S = 0.5  # short rollout windows keep the residual well-conditioned


def test_representative_windows_are_evenly_distributed() -> None:
    bounds = [(index * 10, (index + 1) * 10) for index in range(8)]
    assert _representative_bounds(bounds, maximum=1) == [bounds[4]]
    assert _representative_bounds(bounds, maximum=2) == [bounds[2], bounds[6]]
    assert _representative_bounds(bounds, maximum=None) == bounds


def test_window_grid_cannot_gain_an_interpolated_control_row(
    model_paths: ModelPaths,
) -> None:
    """Large recording timestamps must not distort a local SysID window."""
    model = _identification_spec(model_paths).compile()
    run = _pd_tracking_run(model, 0.0, "timestamp-regression")
    large_origin = 1_000_000.0
    shifted = replace(
        run,
        control_times=run.control_times + large_origin,
        measured_times=run.measured_times + large_origin,
    )
    sequences = measurement_sequences(
        _identification_spec(model_paths),
        [shifted],
        window_s=0.32,
        max_windows_per_run=1,
        measurement_stride=5,
    )

    original = sequences.control[0]
    resampled = original.resample(target_dt=model.opt.timestep)

    assert len(original) == 160
    assert len(resampled) == len(original)
    assert np.array_equal(resampled.data, original.data)
    np.testing.assert_array_equal(
        original.times,
        _resampling_safe_control_times(160, model.opt.timestep),
    )


def _identification_spec(model_paths: ModelPaths) -> mujoco.MjSpec:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True)


def _hidden_truth_model(
    model_paths: ModelPaths,
    *,
    frictionloss: dict[str, float] | None = None,
    damping: dict[str, float] | None = None,
    armature: dict[str, float] | None = None,
    body_mass_scale: dict[str, float] | None = None,
) -> mujoco.MjModel:
    """Test-owned ground truth; fitting code never sees these values."""
    spec = _identification_spec(model_paths)
    for joint_name, value in (frictionloss or {}).items():
        spec.joint(joint_name).frictionloss = value
    for joint_name, value in (damping or {}).items():
        set_hinge_damping(spec.joint(joint_name), value)
    for joint_name, value in (armature or {}).items():
        spec.joint(joint_name).armature = value
    for body_name, scale in (body_mass_scale or {}).items():
        spec.body(body_name).mass = float(spec.body(body_name).mass) * scale
    return spec.compile()


def _pd_tracking_run(
    truth_model: mujoco.MjModel,
    phase: float,
    label: str,
    *,
    frequency_hz: float = _FREQUENCY_HZ,
    amplitude_rad: float = _AMPLITUDE_RAD,
    extra_torque=None,
) -> MeasuredRun:
    """Track per-joint sinusoid reversals with a PD law; record torque/state.

    ``extra_torque(qvel) -> (7,)`` injects a hidden plant torque (via
    ``qfrc_applied``) that is deliberately absent from the recorded control —
    unmodeled physics from the fit's point of view.
    """
    dt = truth_model.opt.timestep
    steps = int(round(_DURATION_S / dt))
    phases = phase + np.arange(7) * (np.pi / 3.0)
    omega = 2.0 * np.pi * frequency_hz
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
        q_des = _HOME_QPOS + ramp * amplitude_rad * wave
        v_des = amplitude_rad * (ramp * wave_rate + ramp_rate * wave)
        tau = np.clip(
            _KP * (q_des - data.qpos) + _KD * (v_des - data.qvel),
            ctrl_low,
            ctrl_high,
        )
        control_times[k] = t
        control[k] = tau
        data.ctrl[:] = tau
        if extra_torque is not None:
            data.qfrc_applied[:] = extra_torque(np.asarray(data.qvel))
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


def _sequences(
    model_paths: ModelPaths,
    truth_model: mujoco.MjModel,
    *,
    frequency_hz: float = _FREQUENCY_HZ,
    amplitude_rad: float = _AMPLITUDE_RAD,
    phases: tuple[float, ...] = _RUN_PHASES,
    extra_torque=None,
):
    runs = [
        _pd_tracking_run(
            truth_model,
            phase,
            f"reversals_{index}",
            frequency_hz=frequency_hz,
            amplitude_rad=amplitude_rad,
            extra_torque=extra_torque,
        )
        for index, phase in enumerate(phases)
    ]
    return measurement_sequences(
        _identification_spec(model_paths), runs, window_s=_WINDOW_S
    )


def test_nominal_parameters_are_a_noop(model_paths: ModelPaths) -> None:
    """Applying nominal parameters changes no compiled physical quantity."""
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


def test_blind_seven_link_inertial_block_is_refused(
    model_paths: ModelPaths,
) -> None:
    spec = _identification_spec(model_paths)
    with pytest.raises(ValueError, match="blind seven-link"):
        inertial_parameters(spec, spec.compile())


def test_cad_prior_body_corrections_compose_with_armature(
    model_paths: ModelPaths,
) -> None:
    """Multiple COM axes and a joint armature survive one modifier block."""
    nominal_spec = _identification_spec(model_paths)
    nominal = nominal_spec.compile()
    body_parameters = cad_prior_inertial_parameters(
        nominal_spec,
        nominal,
        (
            BodyInertialCorrection(
                "link3",
                com_axes=(0, 1, 2),
                estimate_inertia_scale=True,
            ),
        ),
    )
    armature = armature_parameters(nominal, joints=("joint3",))
    parameters = combine_parameters(armature, body_parameters)

    parameters["joint3_armature"].update_from_vector(np.array([0.24]))
    parameters["link3_mass_scale"].update_from_vector(np.array([1.12]))
    offsets = (0.004, -0.006, 0.008)
    for axis, value in zip(("x", "y", "z"), offsets, strict=True):
        parameters[f"link3_com_{axis}_offset_m"].update_from_vector(np.array([value]))
    parameters["link3_inertia_scale"].update_from_vector(np.array([1.08]))

    compiled = sysid.apply_param_modifiers(
        parameters, _identification_spec(model_paths)
    )
    joint3_dof = int(compiled.joint("joint3").dofadr[0])
    assert compiled.dof_armature[joint3_dof] == pytest.approx(0.24)
    assert float(compiled.body("link3").mass[0]) == pytest.approx(
        float(nominal.body("link3").mass[0]) * 1.12
    )
    np.testing.assert_allclose(
        compiled.body("link3").ipos,
        np.asarray(nominal.body("link3").ipos) + offsets,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        body_full_inertia(compiled, "link3"),
        body_full_inertia(nominal, "link3") * 1.08,
        rtol=1e-9,
        atol=5e-12,
    )


def test_observable_subset_deterministically_freezes_a_collinear_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw correction is released only once for a duplicated direction."""
    parameters = sysid.ParameterDict()
    for name in ("mass_a", "mass_b", "inertia_scale"):
        parameters.add(sysid.Parameter(name, nominal=1.0, min_value=0.7, max_value=1.3))
    jacobian = np.array(
        (
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.0),
        )
    )
    synthetic = ConditioningReport(
        scaled_singular_values=np.linalg.svd(jacobian, compute_uv=False),
        parameter_correlations=np.eye(3),
        scaled_jacobian=jacobian,
        component_names=("mass_a", "mass_b", "inertia_scale"),
        objective=1.0,
    )
    monkeypatch.setattr(
        fitting_module, "conditioning_report", lambda *args, **kwargs: synthetic
    )

    selected = select_observable_subset(parameters, object())  # type: ignore[arg-type]

    assert selected.accepted_names == ("mass_a", "inertia_scale")
    assert selected.rejected_names == ("mass_b",)
    assert selected.parameters["mass_b"].frozen
    assert not parameters["mass_b"].frozen
    selected.require_nonempty()


def test_conditioning_acceptance_fails_closed() -> None:
    report = ConditioningReport(
        scaled_singular_values=np.array([1.0, 1e-12]),
        parameter_correlations=np.array(((1.0, 0.999), (0.999, 1.0))),
        scaled_jacobian=np.eye(2),
        component_names=("a", "b"),
        objective=1.0,
    )
    acceptance = conditioning_acceptance(report)
    assert not acceptance.accepted
    with pytest.raises(IdentificationAcceptanceError, match="rejected"):
        acceptance.require()


def test_multistart_spread_uses_box_span_for_zero_centered_correction() -> None:
    def result(value: float, objective: float) -> SimpleNamespace:
        parameters = sysid.ParameterDict()
        parameter = sysid.Parameter(
            "link3_com_x_offset_m",
            nominal=0.0,
            min_value=-0.02,
            max_value=0.02,
        )
        parameter.update_from_vector(np.array([value]))
        parameters.add(parameter)
        return SimpleNamespace(
            final_objective=objective,
            values={"link3_com_x_offset_m": np.array([value])},
            parameters=parameters,
        )

    multistart = MultistartResult(  # type: ignore[arg-type]
        results=[result(0.0, 1.0), result(2e-5, 1.1)]
    )

    assert multistart.value_spread["link3_com_x_offset_m"] == pytest.approx(2e-5 / 0.04)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_armature_single_recovery(model_paths: ModelPaths) -> None:
    """Noise-free armature recovery within 1% on dynamic excitation."""
    truth_value = 0.25
    truth_model = _hidden_truth_model(model_paths, armature={"joint3": truth_value})
    sequences = _sequences(
        model_paths,
        truth_model,
        frequency_hz=_DYNAMIC_FREQUENCY_HZ,
        amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
    )

    nominal_model = _identification_spec(model_paths).compile()
    parameters = armature_parameters(nominal_model, joints=("joint3",))
    result = fit_parameters(parameters, sequences, max_iters=30)

    assert result.values["joint3_armature"][0] == pytest.approx(truth_value, rel=0.01)
    assert result.objective_reduction >= 0.99


@pytest.mark.slow
def test_staged_friction_then_armature(model_paths: ModelPaths) -> None:
    """Plan ordering rule: friction on slow data, then armature on dynamic
    data on top of the accepted friction result, then a friction refit."""
    truth_frictionloss = 0.8
    truth_damping = 2.0
    truth_armature = 0.25
    truth_model = _hidden_truth_model(
        model_paths,
        frictionloss={"joint4": truth_frictionloss},
        damping={"joint4": truth_damping},
        armature={"joint4": truth_armature},
    )
    slow = _sequences(model_paths, truth_model)
    dynamic = _sequences(
        model_paths,
        truth_model,
        frequency_hz=_DYNAMIC_FREQUENCY_HZ,
        amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
    )

    nominal_model = _identification_spec(model_paths).compile()
    stages = [
        FitStage(
            friction_parameters(nominal_model, joints=("joint4",)).move_off_bounds(),
            slow,
        ),
        FitStage(armature_parameters(nominal_model, joints=("joint4",)), dynamic),
        FitStage(
            friction_parameters(nominal_model, joints=("joint4",)).move_off_bounds(),
            slow,
        ),
    ]
    first, armature_stage, refit = fit_staged(stages, max_iters=40)

    # Stage 1 fits friction against a wrong armature: close but biased
    # (measured ~13%/19% bias on frictionloss/damping).
    assert first.values["joint4_frictionloss"][0] == pytest.approx(
        truth_frictionloss, rel=0.25
    )
    assert first.values["joint4_damping"][0] == pytest.approx(truth_damping, rel=0.25)
    # Stage 2 inherits the accepted friction values; still carries part of
    # the stage-1 bias (measured ~5%).
    assert armature_stage.values["joint4_armature"][0] == pytest.approx(
        truth_armature, rel=0.08
    )
    # Stage 3 refits friction with armature corrected and must land tight.
    assert refit.values["joint4_frictionloss"][0] == pytest.approx(
        truth_frictionloss, rel=0.02
    )
    assert refit.values["joint4_damping"][0] == pytest.approx(truth_damping, rel=0.02)
    # The refinement loop must strictly improve on the first pass.
    assert abs(refit.values["joint4_frictionloss"][0] - truth_frictionloss) < abs(
        first.values["joint4_frictionloss"][0] - truth_frictionloss
    )
    assert abs(refit.values["joint4_damping"][0] - truth_damping) < abs(
        first.values["joint4_damping"][0] - truth_damping
    )


@pytest.mark.slow
def test_multistart_agreement_and_conditioning(model_paths: ModelPaths) -> None:
    """Independent starts agree, and the block passes the P2 conditioning
    thresholds (ratio above minimum, no bound hits, correlation below the
    freeze limit)."""
    truth_frictionloss = 0.8
    truth_damping = 2.0
    truth_model = _hidden_truth_model(
        model_paths,
        frictionloss={"joint4": truth_frictionloss},
        damping={"joint4": truth_damping},
    )
    sequences = _sequences(model_paths, truth_model)

    nominal_model = _identification_spec(model_paths).compile()
    parameters = friction_parameters(
        nominal_model, joints=("joint4",)
    ).move_off_bounds()

    multistart = fit_multistart(parameters, sequences, n_starts=3, seed=7, max_iters=40)
    best = multistart.best

    assert best.objective_reduction >= 0.99
    assert best.values["joint4_frictionloss"][0] == pytest.approx(
        truth_frictionloss, rel=0.02
    )
    assert best.values["joint4_damping"][0] == pytest.approx(truth_damping, rel=0.02)
    for spread in multistart.value_spread.values():
        assert spread <= 0.01
    assert best.conditioning_ratio >= CONDITIONING_RATIO_MINIMUM
    assert not best.bound_hits
    assert abs(best.parameter_correlations[0, 1]) < CORRELATION_FREEZE_LIMIT


def test_link_mass_recovery(model_paths: ModelPaths) -> None:
    """Noise-free single-body mass recovery within 1% on dynamic data."""
    truth_scale = 1.15
    truth_model = _hidden_truth_model(
        model_paths, body_mass_scale={"link3": truth_scale}
    )
    sequences = _sequences(
        model_paths,
        truth_model,
        frequency_hz=_DYNAMIC_FREQUENCY_HZ,
        amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
        phases=(0.0,),
    )

    spec = _identification_spec(model_paths)
    nominal_model = spec.compile()
    parameters = inertial_parameters(
        spec, nominal_model, bodies=("link3",), inertia_type=sysid.InertiaType.Mass
    )
    result = fit_parameters(parameters, sequences, max_iters=25)

    true_mass = float(nominal_model.body("link3").mass[0]) * truth_scale
    assert result.values["link3_inertia"][0] == pytest.approx(true_mass, rel=0.01)
    assert result.objective_reduction >= 0.99


def test_tool_composite_unidentifiability_detected(model_paths: ModelPaths) -> None:
    """Estimating hand inertia alongside link7 must be reported unidentifiable.

    Everything downstream of joint 7 is one rigid composite, so splitting its
    inertia between link7 and hand adds a pure nullspace. The pre-fit
    conditioning report must reject that block while accepting the
    link7-composite convention (measured: ratio ~5e-25 vs ~3e-4).
    """
    truth_model = _hidden_truth_model(model_paths)
    sequences = _sequences(
        model_paths,
        truth_model,
        frequency_hz=_DYNAMIC_FREQUENCY_HZ,
        amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
        phases=(0.0,),
    )

    spec = _identification_spec(model_paths)
    nominal_model = spec.compile()
    pair = inertial_parameters(spec, nominal_model, bodies=("link7", "hand"))
    solo = inertial_parameters(spec, nominal_model, bodies=("link7",))

    pair_report = conditioning_report(pair, sequences)
    solo_report = conditioning_report(solo, sequences)

    assert pair_report.conditioning_ratio < CONDITIONING_RATIO_MINIMUM
    assert solo_report.conditioning_ratio > CONDITIONING_RATIO_MINIMUM
    assert pair_report.conditioning_ratio < 1e-6 * solo_report.conditioning_ratio


@pytest.mark.slow
def test_inadequate_friction_model_detected(model_paths: ModelPaths) -> None:
    """An out-of-class friction truth must be detected, not absorbed.

    The hidden plant applies quadratic velocity drag on joint 4 — outside the
    damping+frictionloss class. Fitted across two amplitude regimes, the
    in-class model must stall visibly above the in-class floor (measured
    ~178x), and a subsequent armature stage must stay near nominal instead of
    absorbing the residual (measured 6.7% drift explaining 0.6%).
    """

    def hidden_drag(qvel: np.ndarray) -> np.ndarray:
        torque = np.zeros(7)
        torque[3] = -1.5 * qvel[3] * abs(qvel[3])
        return torque

    truth_model = _hidden_truth_model(model_paths)
    slow = _sequences(model_paths, truth_model, phases=(0.0,), extra_torque=hidden_drag)
    dynamic = _sequences(
        model_paths,
        truth_model,
        frequency_hz=_DYNAMIC_FREQUENCY_HZ,
        amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
        phases=(0.4,),
        extra_torque=hidden_drag,
    )

    nominal_model = _identification_spec(model_paths).compile()
    friction_stage, armature_stage = fit_staged(
        [
            FitStage(
                friction_parameters(
                    nominal_model, joints=("joint4",)
                ).move_off_bounds(),
                [slow, dynamic],
            ),
            FitStage(armature_parameters(nominal_model, joints=("joint4",)), dynamic),
        ],
        max_iters=40,
    )

    # In-class control: same pipeline on a truth inside the model class.
    control_truth = _hidden_truth_model(
        model_paths, frictionloss={"joint4": 0.8}, damping={"joint4": 2.0}
    )
    control = fit_parameters(
        friction_parameters(nominal_model, joints=("joint4",)).move_off_bounds(),
        [
            _sequences(model_paths, control_truth, phases=(0.0,)),
            _sequences(
                model_paths,
                control_truth,
                frequency_hz=_DYNAMIC_FREQUENCY_HZ,
                amplitude_rad=_DYNAMIC_AMPLITUDE_RAD,
                phases=(0.4,),
            ),
        ],
        max_iters=40,
    )
    assert control.objective_reduction >= 0.99

    # Detectable structured residual: far above the in-class floor.
    assert friction_stage.objective_reduction < 0.95
    assert friction_stage.final_objective > 50.0 * control.final_objective
    # Armature is not corrupted by the unexplained residual.
    nominal_armature = float(nominal_model.dof_armature[3])
    assert (
        abs(armature_stage.values["joint4_armature"][0] - nominal_armature)
        <= 0.1 * nominal_armature
    )
    assert armature_stage.objective_reduction < 0.1
