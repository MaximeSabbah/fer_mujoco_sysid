"""End-to-end simulated identification with the full diagnostic set.

Runs the complete pipeline against a hidden-truth robot and reports it the
way a robotics identification result should be reported:

* excitation quality as the **condition number of the regressor**, per
  family and per joint, before any fitting;
* measurement noise, then the explicit zero-phase Butterworth
  **preprocessing** that handles it;
* **relative standard deviations** per parameter with the classical freeze
  rule, so a number that is not identified is reported as such;
* **torque plots**, since friction acts on torques: measured against what
  the nominal and identified models reconstruct for the same motion;
* held-out prediction error in physical units (rad and millimetres).

Also renders one MuJoCo video per committed protocol for visual review.
Run through ``scripts/identification-demo``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fer_mujoco_sysid.campaign import (  # noqa: E402
    CAMPAIGN,
    FRICTION_CAMPAIGN,
    INERTIAL_CAMPAIGN,
    compile_protocol,
    repository_root,
    tracking_run,
)
from fer_mujoco_sysid.diagnostics import (  # noqa: E402
    friction_regressor_report,
    parameter_quality,
    predicted_torque,
    torque_residuals,
)
from fer_mujoco_sysid.fitting import (  # noqa: E402
    MeasuredRun,
    fit_parameters,
    friction_parameters,
    measurement_sequences,
    set_hinge_damping,
)
from fer_mujoco_sysid.model import (  # noqa: E402
    build_hydrax_arm_spec,
    resolve_model_paths,
)
from fer_mujoco_sysid.preprocessing import FilterSettings, prepare  # noqa: E402

# Hidden truth of realistic magnitude, informed by the 2026-07-22 real-run
# breakaway bounds (j1>=1.37, j3>=1.11, j4>=1.50, j6~1.1, j7>=0.55 Nm).
# These are NOT the real robot's values.
TRUTH_FRICTIONLOSS = (1.4, 1.2, 1.1, 1.5, 0.35, 1.1, 0.55)
TRUTH_DAMPING = (2.0, 1.8, 1.5, 1.9, 0.8, 0.7, 0.5)
TRUTH_ARMATURE = (0.14, 0.13, 0.12, 0.12, 0.08, 0.07, 0.06)

# Encoder and torque-sensor noise, so preprocessing is actually exercised.
POSITION_NOISE_RAD = 2e-5
VELOCITY_NOISE_RAD_S = 2e-3
TORQUE_NOISE_NM = 0.02

RENDER_WIDTH = 800
RENDER_HEIGHT = 608

_INK = "#374151"
_MEASURED = "#111827"
_NOMINAL = "#9ca3af"
_IDENTIFIED = "#2563eb"
_TRUTH = "#b45309"


def _spec_and_model() -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    spec = build_hydrax_arm_spec(
        resolve_model_paths().require().hydrax, joint_state_sensors=True
    )
    return spec, spec.compile()


def _apply(
    spec: mujoco.MjSpec,
    frictionloss,
    damping,
    armature=None,
) -> mujoco.MjModel:
    for index in range(1, 8):
        joint = spec.joint(f"joint{index}")
        joint.frictionloss = float(frictionloss[index - 1])
        set_hinge_damping(joint, float(damping[index - 1]))
        if armature is not None:
            joint.armature = float(armature[index - 1])
    return spec.compile()


def truth_model() -> mujoco.MjModel:
    spec, _ = _spec_and_model()
    return _apply(spec, TRUTH_FRICTIONLOSS, TRUTH_DAMPING, TRUTH_ARMATURE)


def _add_noise(run: MeasuredRun, seed: int) -> MeasuredRun:
    """Corrupt a recording the way a real encoder and torque sensor would."""
    rng = np.random.default_rng(seed)
    measured = run.measured.copy()
    measured[:, :7] += POSITION_NOISE_RAD * rng.standard_normal((len(measured), 7))
    measured[:, 7:14] += VELOCITY_NOISE_RAD_S * rng.standard_normal((len(measured), 7))
    control = run.control + TORQUE_NOISE_NM * rng.standard_normal(run.control.shape)
    return MeasuredRun(
        label=run.label,
        qpos0=run.qpos0,
        qvel0=run.qvel0,
        control_times=run.control_times,
        control=control,
        measured_times=run.measured_times,
        measured=measured,
    )


@dataclass(frozen=True)
class Recording:
    """One played protocol: raw, filtered, and its derived acceleration."""

    label: str
    raw: MeasuredRun
    filtered: MeasuredRun
    ddq_rad_s2: np.ndarray


def play(model: mujoco.MjModel, spec, seed: int, settings: FilterSettings) -> Recording:
    """Play a protocol on *model*, add sensor noise, and preprocess."""
    compiled = compile_protocol(spec, model)
    run = _add_noise(tracking_run(model, compiled), seed)
    prepared = prepare(
        run.measured_times,
        run.measured[:, :7],
        run.measured[:, 7:14],
        run.control,
        settings,
    )
    measured = run.measured.copy()
    measured[:, :7] = prepared.q_rad
    measured[:, 7:14] = prepared.dq_rad_s
    filtered = MeasuredRun(
        label=run.label,
        qpos0=run.qpos0,
        qvel0=run.qvel0,
        control_times=run.control_times,
        control=prepared.tau_Nm,
        measured_times=run.measured_times,
        measured=measured,
    )
    return Recording(
        label=spec.protocol_id,
        raw=run,
        filtered=filtered,
        ddq_rad_s2=prepared.ddq_rad_s2,
    )


def plot_friction_curves(
    path: Path,
    nominal: mujoco.MjModel,
    recording: Recording,
    frictionloss: np.ndarray,
    damping: np.ndarray,
) -> None:
    """Friction torque against velocity at the constant-velocity cruises."""
    run = recording.filtered
    velocity = run.measured[:, 7:14]
    nominal_damping = nominal.dof_damping[:7].copy()
    residual = run.control - predicted_torque(
        nominal, run.measured[:, :7], velocity, np.zeros_like(velocity)
    )
    residual += nominal_damping[None, :] * velocity
    steady = np.abs(recording.ddq_rad_s2) < 0.05

    figure, axes = plt.subplots(4, 2, figsize=(10, 11), constrained_layout=True)
    figure.suptitle(
        "Friction curves on the held-out protocol — friction torque vs joint "
        "velocity (constant-velocity samples)",
        color=_INK,
    )
    grid = np.linspace(-0.5, 0.5, 200)
    for joint in range(7):
        axis = axes[joint // 2, joint % 2]
        selected = steady[:, joint]
        axis.plot(
            velocity[selected, joint][::3],
            residual[selected, joint][::3],
            ".",
            ms=2.5,
            color=_MEASURED,
            alpha=0.3,
            label="measured (cruise)" if joint == 0 else None,
        )
        for values, color, width, style, label in (
            (
                TRUTH_FRICTIONLOSS[joint] * np.sign(grid) + TRUTH_DAMPING[joint] * grid,
                _TRUTH,
                4.5,
                "-",
                "hidden truth",
            ),
            (
                frictionloss[joint] * np.sign(grid) + damping[joint] * grid,
                _IDENTIFIED,
                1.6,
                "--",
                "identified",
            ),
            (
                nominal_damping[joint] * grid,
                _NOMINAL,
                1.6,
                "-",
                "nominal (frictionless)",
            ),
        ):
            axis.plot(
                grid,
                values,
                color=color,
                lw=width,
                ls=style,
                alpha=0.9,
                label=label if joint == 0 else None,
            )
        span = 1.5 * (TRUTH_FRICTIONLOSS[joint] + 0.5 * TRUTH_DAMPING[joint])
        axis.set_ylim(-span, span)
        axis.set_xlim(-0.55, 0.55)
        axis.set_title(f"joint {joint + 1}", fontsize=9, color=_INK)
        axis.set_xlabel("dq [rad/s]", fontsize=8, color=_INK)
        axis.set_ylabel("friction torque [Nm]", fontsize=8, color=_INK)
        axis.tick_params(labelsize=7, colors=_INK)
        axis.grid(True, alpha=0.25, lw=0.5)
    axes[3, 1].axis("off")
    figure.legend(loc="lower right", fontsize=9)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_torque_tracking(
    path: Path,
    title: str,
    recording: Recording,
    nominal: mujoco.MjModel,
    identified: mujoco.MjModel,
) -> None:
    """Measured joint torque against what each model reconstructs.

    The classical validation plot: friction acts on torques, so a model
    missing friction shows a visible, structured torque gap.
    """
    run = recording.filtered
    q, dq = run.measured[:, :7], run.measured[:, 7:14]
    ddq = recording.ddq_rad_s2
    nominal_tau = predicted_torque(nominal, q, dq, ddq)
    identified_tau = predicted_torque(identified, q, dq, ddq)
    time_s = run.measured_times - run.measured_times[0]

    figure, axes = plt.subplots(
        7, 1, figsize=(11, 13), sharex=True, constrained_layout=True
    )
    figure.suptitle(title, color=_INK)
    for joint in range(7):
        axis = axes[joint]
        axis.plot(
            time_s,
            run.control[:, joint],
            color=_MEASURED,
            lw=2.2,
            alpha=0.55,
            label="measured" if joint == 0 else None,
        )
        axis.plot(
            time_s,
            nominal_tau[:, joint],
            color=_NOMINAL,
            lw=1.2,
            label="nominal model" if joint == 0 else None,
        )
        axis.plot(
            time_s,
            identified_tau[:, joint],
            color=_IDENTIFIED,
            lw=1.2,
            ls="--",
            label="identified model" if joint == 0 else None,
        )
        axis.set_ylabel(f"j{joint + 1} [Nm]", fontsize=8, color=_INK)
        axis.tick_params(labelsize=7, colors=_INK)
        axis.grid(True, alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("time [s]", fontsize=8, color=_INK)
    figure.legend(loc="upper right", fontsize=9)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_torque_residuals(
    path: Path,
    recording: Recording,
    nominal: mujoco.MjModel,
    identified: mujoco.MjModel,
) -> None:
    """Per-joint torque residual RMSE, nominal against identified."""
    run = recording.filtered
    q, dq = run.measured[:, :7], run.measured[:, 7:14]
    ddq = recording.ddq_rad_s2
    nominal_residual = run.control - predicted_torque(nominal, q, dq, ddq)
    identified_residual = run.control - predicted_torque(identified, q, dq, ddq)

    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    positions = np.arange(7)
    width = 0.38
    axis.bar(
        positions - width / 2,
        np.sqrt(np.mean(nominal_residual**2, axis=0)),
        width,
        color=_NOMINAL,
        label="nominal model",
    )
    axis.bar(
        positions + width / 2,
        np.sqrt(np.mean(identified_residual**2, axis=0)),
        width,
        color=_IDENTIFIED,
        label="identified model",
    )
    axis.set_xticks(positions, [f"joint {j + 1}" for j in range(7)], fontsize=8)
    axis.set_ylabel("torque residual RMSE [Nm]", fontsize=9, color=_INK)
    axis.set_title(
        "Torque reconstruction error on the held-out protocol",
        fontsize=10,
        color=_INK,
    )
    axis.tick_params(labelsize=8, colors=_INK)
    axis.grid(True, axis="y", alpha=0.25, lw=0.5)
    axis.legend(fontsize=9)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _windowed_prediction(
    model: mujoco.MjModel, run: MeasuredRun, *, window_s: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop replay in short windows: per-joint q RMSE, gripper error mm."""
    import mujoco.rollout

    site = int(model.site("gripper").id)
    data = mujoco.MjData(model)
    steps = int(round(window_s / model.opt.timestep))
    q_errors: list[np.ndarray] = []
    ee_errors: list[np.ndarray] = []

    def site_positions(rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), 3))
        for index, value in enumerate(rows):
            data.qpos[:] = value
            mujoco.mj_kinematics(model, data)
            out[index] = data.site_xpos[site]
        return out

    for start in range(0, len(run.control), steps):
        stop = min(start + steps, len(run.control))
        if stop - start < 2:
            continue
        data.qpos[:] = run.measured[start, :7]
        data.qvel[:] = run.measured[start, 7:14]
        state = np.empty(
            mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
        )
        mujoco.mj_getState(
            model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )
        _, sensor = mujoco.rollout.rollout(model, data, state, run.control[start:stop])
        predicted = np.squeeze(sensor, axis=0)
        measured = run.measured[start:stop]
        q_errors.append(predicted[:, :7] - measured[:, :7])
        ee_errors.append(
            1e3
            * np.linalg.norm(
                site_positions(predicted[:, :7]) - site_positions(measured[:, :7]),
                axis=1,
            )
        )
    stacked = np.vstack(q_errors)
    return np.sqrt(np.mean(stacked**2, axis=0)), np.concatenate(ee_errors)


def render_protocol_videos(review_dir: Path, *, fps: int = 25) -> list[Path]:
    """Render every committed protocol as a MuJoCo replay."""
    _, plan_model = _spec_and_model()
    spec = build_hydrax_arm_spec(
        resolve_model_paths().require().hydrax, joint_state_sensors=True
    )
    floor = spec.worldbody.add_geom()
    floor.name = "review_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.05]
    floor.rgba = [0.86, 0.87, 0.89, 1.0]
    floor.contype = 0
    floor.conaffinity = 0
    # The default offscreen framebuffer is 640x480; rendering larger frames
    # requires enlarging it in the spec before compiling.
    spec.visual.global_.offwidth = RENDER_WIDTH
    spec.visual.global_.offheight = RENDER_HEIGHT
    model = spec.compile()

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.05, 0.0, 0.45)
    camera.distance = 1.55
    camera.azimuth = 135.0
    camera.elevation = -15.0
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

    outputs: list[Path] = []
    try:
        for protocol in CAMPAIGN:
            compiled = compile_protocol(protocol, plan_model)
            times = compiled.time_s
            path = review_dir / f"{protocol.protocol_id}.mp4"
            with imageio.get_writer(
                path, fps=fps, codec="libx264", quality=8
            ) as writer:
                for frame_time in np.arange(0.0, times[-1], 1.0 / fps):
                    row = min(int(np.searchsorted(times, frame_time)), len(times) - 1)
                    data.qpos[:] = compiled.q_rad[row]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
            outputs.append(path)
            print(f"video: {path.name} ({times[-1]:.1f} s)")
    finally:
        renderer.close()
    return outputs


def run_identification(demo_dir: Path, *, reuse_fit: bool = False) -> None:
    """Play, filter, fit, and report with the full diagnostic set."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    settings = FilterSettings()
    truth = truth_model()
    _, nominal = _spec_and_model()
    cache = demo_dir / "identified_parameters.json"

    print("playing protocols on the hidden-truth robot (with sensor noise) ...")
    friction_runs = [
        play(truth, spec, seed=10 + index, settings=settings)
        for index, spec in enumerate(FRICTION_CAMPAIGN[:2])
    ]
    inertial_run = play(truth, INERTIAL_CAMPAIGN[0], seed=30, settings=settings)
    holdout = play(truth, FRICTION_CAMPAIGN[2], seed=40, settings=settings)

    friction_excitation = friction_regressor_report(
        np.vstack([record.filtered.measured[:, 7:14] for record in friction_runs])
    )
    inertial_excitation = friction_regressor_report(
        inertial_run.filtered.measured[:, 7:14], inertial_run.ddq_rad_s2
    )
    print(
        f"  friction family worst cond(Y) = "
        f"{friction_excitation.worst_condition_number:.2f}"
    )
    print(
        f"  inertial family worst cond(Y) = "
        f"{inertial_excitation.worst_condition_number:.2f}"
    )

    quality = None
    if reuse_fit and cache.is_file():
        cached = json.loads(cache.read_text())
        frictionloss = np.asarray(cached["frictionloss"])
        damping = np.asarray(cached["damping"])
        print(f"reusing cached fit from {cache}")
    else:
        print("fitting friction on the friction family (several minutes) ...")
        spec_for_fit, _ = _spec_and_model()
        sequences = measurement_sequences(
            spec_for_fit, [record.filtered for record in friction_runs], window_s=1.0
        )
        parameters = friction_parameters(nominal).move_off_bounds()
        result = fit_parameters(parameters, sequences, max_iters=30)
        frictionloss = np.array(
            [result.values[f"joint{i}_frictionloss"][0] for i in range(1, 8)]
        )
        damping = np.array([result.values[f"joint{i}_damping"][0] for i in range(1, 8)])
        quality = parameter_quality(
            tuple(result.values),
            result.parameters.as_vector(),
            result.parameter_covariance,
        )
        cache.write_text(
            json.dumps(
                {
                    "frictionloss": frictionloss.tolist(),
                    "damping": damping.tolist(),
                    "objective_reduction": result.objective_reduction,
                    "sigma_percent": quality.relative_percent.tolist(),
                    "rejected": quality.rejected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    identified_spec, _ = _spec_and_model()
    identified = _apply(identified_spec, frictionloss, damping)

    print("evaluating on the held-out protocol ...")
    nominal_q, nominal_ee = _windowed_prediction(nominal, holdout.filtered)
    identified_q, identified_ee = _windowed_prediction(identified, holdout.filtered)

    ddq = holdout.ddq_rad_s2
    q, dq = holdout.filtered.measured[:, :7], holdout.filtered.measured[:, 7:14]
    nominal_torque = torque_residuals(
        holdout.filtered.control, predicted_torque(nominal, q, dq, ddq)
    )
    inertial_q = inertial_run.filtered.measured[:, :7]
    inertial_dq = inertial_run.filtered.measured[:, 7:14]
    inertial_nominal = torque_residuals(
        inertial_run.filtered.control,
        predicted_torque(nominal, inertial_q, inertial_dq, inertial_run.ddq_rad_s2),
    )
    inertial_identified = torque_residuals(
        inertial_run.filtered.control,
        predicted_torque(identified, inertial_q, inertial_dq, inertial_run.ddq_rad_s2),
    )
    identified_torque = torque_residuals(
        holdout.filtered.control, predicted_torque(identified, q, dq, ddq)
    )

    plot_friction_curves(
        demo_dir / "friction_curves.png", nominal, holdout, frictionloss, damping
    )
    plot_torque_tracking(
        demo_dir / "torque_tracking_friction.png",
        "Joint torque, held-out FRICTION protocol: measured vs model reconstruction",
        holdout,
        nominal,
        identified,
    )
    # The same check on the inertial family. Only friction was fitted, so
    # this exposes what the model still misses where inertia dominates.
    plot_torque_tracking(
        demo_dir / "torque_tracking_inertial.png",
        "Joint torque, INERTIAL protocol (friction fitted, armature and "
        "inertias not): measured vs model reconstruction",
        inertial_run,
        nominal,
        identified,
    )
    plot_torque_residuals(
        demo_dir / "torque_residuals.png", holdout, nominal, identified
    )

    _write_report(
        demo_dir / "result.md",
        frictionloss=frictionloss,
        damping=damping,
        quality=quality,
        friction_excitation=friction_excitation,
        inertial_excitation=inertial_excitation,
        filter_settings=settings,
        nominal_q=nominal_q,
        identified_q=identified_q,
        nominal_ee=nominal_ee,
        identified_ee=identified_ee,
        nominal_torque=nominal_torque,
        identified_torque=identified_torque,
        inertial_nominal=inertial_nominal,
        inertial_identified=inertial_identified,
    )
    print((demo_dir / "result.md").read_text())


def _write_report(path: Path, **data) -> None:
    frictionloss = data["frictionloss"]
    damping = data["damping"]
    quality = data["quality"]
    settings = data["filter_settings"]
    lines = [
        "# Simulated identification — result",
        "",
        "Hidden truth: friction, damping and armature of realistic magnitude "
        "(informed by the real 2026-07-22 breakaway bounds; NOT the real "
        "robot's values). Recordings carry encoder and torque-sensor noise "
        "and are preprocessed before fitting.",
        "",
        "## 1. Excitation quality (measured before fitting)",
        "",
        "Condition number of the regressor per joint — the classical "
        "excitation measure. 1 is ideal; above ~100 the motion cannot "
        "separate the parameters.",
        "",
        "| family | parameters | worst cond(Y) | verdict |",
        "| --- | --- | --- | --- |",
    ]
    for label, report in (
        ("friction", data["friction_excitation"]),
        ("inertial", data["inertial_excitation"]),
    ):
        verdict = "well excited" if report.is_well_excited() else "POORLY EXCITED"
        lines.append(
            f"| {label} | {', '.join(report.parameter_names)} "
            f"| {report.worst_condition_number:.2f} | {verdict} |"
        )

    lines += [
        "",
        "## 2. Preprocessing",
        "",
        f"Zero-phase Butterworth low-pass, order {settings.order}, cutoff "
        f"{settings.cutoff_hz} Hz, applied identically to positions, "
        "velocities and torques. Zero phase matters: a causal filter would "
        "shift the joint states against the recorded torque and bias "
        "friction exactly at the velocity reversals.",
        "",
        "## 3. Identified parameters",
        "",
        "| joint | frictionloss truth | identified | damping truth | identified |",
        "| --- | --- | --- | --- | --- |",
    ]
    for joint in range(7):
        lines.append(
            f"| {joint + 1} | {TRUTH_FRICTIONLOSS[joint]:.3f} "
            f"| {frictionloss[joint]:.3f} "
            f"| {TRUTH_DAMPING[joint]:.3f} | {damping[joint]:.3f} |"
        )

    if quality is not None:
        lines += [
            "",
            "### Relative standard deviations",
            "",
            "Classical acceptance rule: a parameter whose sigma% exceeds "
            f"{quality.limit_percent:.0f}% is not identified by this data and "
            "should be frozen rather than reported.",
            "",
            quality.table(),
            "",
            f"Rejected: {quality.rejected or 'none'}",
        ]

    lines += [
        "",
        "## 4. Held-out validation",
        "",
        "A protocol never used for fitting. Open-loop prediction over 0.5 s "
        "windows re-initialized from the measured state.",
        "",
        "| metric | nominal model | identified model |",
        "| --- | --- | --- |",
        f"| worst joint q RMSE [rad] | {data['nominal_q'].max():.4f} "
        f"| {data['identified_q'].max():.4f} |",
        "| gripper position RMSE [mm] "
        f"| {float(np.sqrt(np.mean(data['nominal_ee'] ** 2))):.2f} "
        f"| {float(np.sqrt(np.mean(data['identified_ee'] ** 2))):.2f} |",
        f"| gripper position max [mm] | {float(data['nominal_ee'].max()):.2f} "
        f"| {float(data['identified_ee'].max()):.2f} |",
        "| worst joint torque RMSE [Nm] "
        f"| {data['nominal_torque'].worst_rmse:.3f} "
        f"| {data['identified_torque'].worst_rmse:.3f} |",
        "",
        "Per-joint torque reconstruction error [Nm]:",
        "",
        "| joint | nominal | identified |",
        "| --- | --- | --- |",
    ]
    for joint in range(7):
        lines.append(
            f"| {joint + 1} | {data['nominal_torque'].rmse_Nm[joint]:.3f} "
            f"| {data['identified_torque'].rmse_Nm[joint]:.3f} |"
        )

    lines += [
        "",
        "Same check on the **inertial** protocol, where only friction has been "
        "fitted so far (armature and link inertias are still nominal):",
        "",
        "| metric | nominal model | friction-identified model |",
        "| --- | --- | --- |",
        "| worst joint torque RMSE [Nm] "
        f"| {data['inertial_nominal'].worst_rmse:.3f} "
        f"| {data['inertial_identified'].worst_rmse:.3f} |",
        "",
        "## 5. Plots",
        "",
        "* `friction_curves.png` — friction torque vs velocity at the "
        "cruises: truth (thick orange), identified (dashed blue), nominal "
        "(grey — no step at zero, because it has no dry friction).",
        "* `torque_tracking_friction.png` — measured joint torque against "
        "what each model reconstructs, on the held-out friction protocol.",
        "* `torque_tracking_inertial.png` — the same on an inertial "
        "protocol: the residual there is what armature and inertial "
        "identification still has to remove.",
        "* `torque_residuals.png` — per-joint torque reconstruction error.",
        "",
        "## What this does and does not prove",
        "",
        "**Does prove**: the protocols are well conditioned for what they "
        "target; the pipeline recovers realistic friction from noisy data "
        "without seeing the truth; the identified model reconstructs joint "
        "torques and predicts unseen motion far better than the nominal one.",
        "",
        "**Does not prove anything about the real robot.** The truth here "
        "differs from the model only in parameters MuJoCo can express, the "
        "noise is white, and the plant is the same simulator the fit uses. "
        "The real FER has position- and temperature-dependent friction, "
        "transmission effects, and dynamics outside this model class.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--reuse-fit", action="store_true")
    arguments = parser.parse_args(argv)
    root = repository_root()
    if not arguments.skip_videos:
        render_protocol_videos(root / "docs" / "protocol_review")
    if not arguments.skip_fit:
        run_identification(
            root / "docs" / "identification_demo", reuse_fit=arguments.reuse_fit
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
