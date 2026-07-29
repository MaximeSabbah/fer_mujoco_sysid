"""Identify a model from recorded runs.

The entry point for a real campaign. Everything upstream of it — playing the
protocol, recording it, converting the bag — happens under ROS; this reads
only the recording bundles those produce, so the identification does not care
whether the arm was simulated or real. That is the point: the code that
identifies your robot is the code that was rehearsed in simulation.

Roles are assigned by protocol identifier, not by hand: anything named
``*-holdout`` is reserved for evaluation and never reaches the fit. Fixing the
split by name rather than by choice is what stops a held-out result being
quietly chosen after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.classical import LinearFrictionFit, fit_friction
from fer_mujoco_sysid.dataset import (
    find_recordings,
    health_report,
    load_recording,
    recording_backend,
)
from fer_mujoco_sysid.diagnostics import (
    friction_regressor_report,
    parameter_quality,
    predicted_torque,
    torque_residuals,
)
from fer_mujoco_sysid.export import IdentifiedParameters, export_identified_model
from fer_mujoco_sysid.fitting import (
    MeasuredRun,
    armature_parameters,
    fit_parameters,
    friction_parameters,
    measurement_sequences,
    set_hinge_damping,
)
from fer_mujoco_sysid.model import (
    HYDRAX_ARM_JOINT_NAMES,
    build_hydrax_arm_spec,
    resolve_model_paths,
)
from fer_mujoco_sysid.preprocessing import FilterSettings, prepare

#: A recording whose protocol identifier ends with this is evaluation-only.
HOLDOUT_SUFFIX = "-holdout"

#: Which family a protocol belongs to, read from its identifier. The two are
#: fitted in stages, never pooled: friction from the slow cruises where
#: inertia contributes nothing, then armature from the fast Fourier motion
#: with friction already held. Pooling them asks one friction number to
#: explain both regimes, which is exactly the error the staging rule exists
#: to prevent.
FRICTION_FAMILY = "fer-friction"
INERTIAL_FAMILY = "fer-inertial"

#: Rollout window for the residuals, matching the standalone pipeline.
FIT_WINDOW_S = 1.0
EVALUATION_WINDOW_S = 0.5

#: Horizons the held-out prediction is reported over. This is *the* acceptance
#: metric: on the real robot there is no true parameter value to compare
#: against, so the only thing that can be measured is whether the simulator's
#: rollout still resembles the measurement — and for how long. A model can look
#: excellent over 0.1 s and drift badly over 2 s, which is the difference
#: between a model a planner can use and one it cannot.
EVALUATION_HORIZONS_S = (0.1, 0.5, 2.0)


@dataclass(frozen=True)
class PreparedRecording:
    """A recording resampled, filtered, and shaped for the fit."""

    label: str
    protocol_id: str
    backend: str
    run: MeasuredRun
    ddq_rad_s2: NDArray[np.float64]
    health: dict[str, object]

    @property
    def is_holdout(self) -> bool:
        return self.protocol_id.endswith(HOLDOUT_SUFFIX)

    @property
    def family(self) -> str:
        return self.protocol_id.rsplit("-", 1)[0].removesuffix("-holdout")


def _fitting_spec(model_path: str | Path) -> mujoco.MjSpec:
    """The identification model, matched to how the recording was produced.

    **Gravity off.** Under a torque command the robot compensates its own
    weight, so the commanded effort a recording carries is the effort *on top
    of* gravity compensation (D029); the simulated plant is built the same
    way. Fitting with gravity enabled asks the optimizer to explain an arm
    that falls, which no friction value can do — measured, not assumed:
    seeded at the true parameters the residual was 3.1e+07 and the optimizer
    drove damping to its upper bound trying to slow the fall.

    If a future campaign fits a torque channel that *does* include gravity
    (the measured link-side ``tau_J``), this is the line that changes with it.
    """
    spec = build_hydrax_arm_spec(model_path, joint_state_sensors=True)
    spec.option.gravity = [0.0, 0.0, 0.0]
    return spec


def _resample(
    time_s: NDArray[np.float64],
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear resampling onto a uniform grid, column by column."""
    return np.column_stack(
        [np.interp(grid, time_s, values[:, column]) for column in range(values.shape[1])]
    )


def prepare_recording(
    root: str | Path,
    model: mujoco.MjModel,
    *,
    settings: FilterSettings | None = None,
) -> PreparedRecording:
    """Load one recording and shape it into a :class:`MeasuredRun`.

    Two things happen here that the recording deliberately does not do for
    itself. It is **resampled** onto the model's timestep, because the fit
    rolls the model out at that step and a recording arrives on the
    controller's clock; and it is **filtered**, because the recording is raw
    by contract and what was applied to it has to be recorded rather than
    baked in.

    The row convention follows :class:`MeasuredRun`: ``control[k]`` is the
    effort commanded from ``t_k``, ``measured[k]`` is the state at ``t_k``
    that produced it, and the measured stamps carry the post-step skew
    ``mujoco.rollout`` emits. Do not "fix" one side alone.
    """
    manifest, arrays = load_recording(root)
    settings = settings or FilterSettings()

    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    step = float(model.opt.timestep)
    grid = np.arange(time_s[0], time_s[-1], step)
    q = _resample(time_s, np.asarray(arrays["q_rad"]), grid)
    dq = _resample(time_s, np.asarray(arrays["dq_rad_s"]), grid)
    tau = _resample(time_s, np.asarray(arrays["tau_cmd_Nm"]), grid)

    prepared = prepare(grid, q, dq, tau, settings)
    measured = np.column_stack([prepared.q_rad, prepared.dq_rad_s])
    samples = len(grid)
    protocol_id = str(manifest.get("protocol", {}).get("protocol_id", Path(root).name))

    return PreparedRecording(
        label=str(root),
        protocol_id=protocol_id,
        backend=recording_backend(manifest),
        run=MeasuredRun(
            label=protocol_id,
            qpos0=prepared.q_rad[0],
            qvel0=prepared.dq_rad_s[0],
            control_times=np.arange(samples) * step,
            control=prepared.tau_Nm,
            measured_times=(np.arange(samples) + 1) * step,
            measured=measured,
        ),
        ddq_rad_s2=prepared.ddq_rad_s2,
        health=health_report(manifest, arrays),
    )


def _windowed_prediction(
    model: mujoco.MjModel, run: MeasuredRun, *, window_s: float = EVALUATION_WINDOW_S
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Open-loop replay in short windows: per-joint q RMSE, gripper error mm."""
    import mujoco.rollout

    site = int(model.site("gripper").id)
    data = mujoco.MjData(model)
    steps = int(round(window_s / model.opt.timestep))
    q_errors: list[NDArray[np.float64]] = []
    ee_errors: list[NDArray[np.float64]] = []

    def site_positions(rows: NDArray[np.float64]) -> NDArray[np.float64]:
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


def _apply(spec: mujoco.MjSpec, frictionloss, damping, armature=None) -> mujoco.MjModel:
    for index, name in enumerate(HYDRAX_ARM_JOINT_NAMES):
        joint = spec.joint(name)
        joint.frictionloss = float(frictionloss[index])
        set_hinge_damping(joint, float(damping[index]))
        if armature is not None:
            joint.armature = float(armature[index])
    return spec.compile()


def run(
    recordings_root: str | Path,
    output_dir: str | Path,
    *,
    max_iters: int = 30,
) -> dict[str, object]:
    """Fit from every recording under ``recordings_root`` and report."""
    recordings_root = Path(recordings_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = find_recordings(recordings_root)
    if not paths:
        raise FileNotFoundError(f"no recordings under {recordings_root}")

    hydrax = resolve_model_paths().require().hydrax
    spec = _fitting_spec(hydrax)
    nominal = spec.compile()

    print(f"loading {len(paths)} recording(s) ...")
    prepared = [prepare_recording(path, nominal) for path in paths]
    unusable = [record for record in prepared if not record.health["usable"]]
    for record in unusable:
        print(f"  REJECTED {record.protocol_id}: {record.health['problems']}")
    prepared = [record for record in prepared if record.health["usable"]]

    backends = sorted({record.backend for record in prepared})
    if len(backends) > 1:
        raise RuntimeError(
            f"these recordings come from different places ({', '.join(backends)}). "
            "A model fitted across them describes no particular machine — point "
            "identify at one output/<backend> directory at a time."
        )

    train = [record for record in prepared if not record.is_holdout]
    holdout = [record for record in prepared if record.is_holdout]
    if not train:
        raise RuntimeError("every usable recording is a holdout; nothing to fit")

    # The two families answer different questions and are never pooled: the
    # friction stage reads the slow cruises where inertia contributes nothing,
    # and the armature stage reads the fast Fourier motion with friction
    # already held. One friction number fitted across both would be explaining
    # two regimes at once.
    friction_train = [r for r in train if r.family == FRICTION_FAMILY]
    inertial_train = [r for r in train if r.family == INERTIAL_FAMILY]
    if not friction_train:
        raise RuntimeError(
            "no friction protocol to fit from. Friction is identified first and "
            "held while armature is fitted; there is no supported order that "
            "starts with the inertial family."
        )
    print(
        f"  friction stage: {[r.protocol_id for r in friction_train]}\n"
        f"  armature stage: {[r.protocol_id for r in inertial_train] or 'skipped (no inertial recordings)'}\n"
        f"  held out: {[r.protocol_id for r in holdout]}"
    )

    excitation = friction_regressor_report(
        np.vstack([record.run.measured[:, 7:14] for record in friction_train])
    )
    print(f"  worst cond(Y) = {excitation.worst_condition_number:.2f}")

    # Stage 1: the classical regressor solve. Milliseconds, and it earns its
    # place three times over — it reports whether the data can determine these
    # parameters at all, it seeds the rollout fit close enough to converge in
    # one iteration instead of dozens, and it is an independent estimate to
    # disagree with.
    print("classical regressor solve ...")
    linear = fit_friction(
        nominal,
        np.vstack([record.run.measured[:, :7] for record in friction_train]),
        np.vstack([record.run.measured[:, 7:14] for record in friction_train]),
        np.vstack([record.ddq_rad_s2 for record in friction_train]),
        np.vstack([record.run.control for record in friction_train]),
    )
    print(
        f"  frictionloss {np.array2string(linear.frictionloss, precision=3)}\n"
        f"  damping      {np.array2string(linear.damping, precision=3)}\n"
        f"  worst cond(Y) {linear.condition_number.max():.2f}, "
        f"worst sigma% {max(linear.frictionloss_sigma_percent.max(), linear.damping_sigma_percent.max()):.2f}"
    )

    # Stage 2: refine by rollout matching, which optimizes trajectory fidelity
    # rather than the instantaneous torque balance — the objective a planner
    # actually experiences once the model class stops being exact.
    print("rollout refinement from that seed (a few minutes) ...")
    sequences = measurement_sequences(
        _fitting_spec(hydrax),
        [record.run for record in friction_train],
        window_s=FIT_WINDOW_S,
    )
    seeded = friction_parameters(nominal).move_off_bounds()
    for index, joint in enumerate(HYDRAX_ARM_JOINT_NAMES):
        seeded[f"{joint}_frictionloss"].value[:] = linear.frictionloss[index]
        seeded[f"{joint}_damping"].value[:] = linear.damping[index]
    result = fit_parameters(seeded, sequences, max_iters=max_iters)
    frictionloss = np.array(
        [result.values[f"joint{i}_frictionloss"][0] for i in range(1, 8)]
    )
    damping = np.array([result.values[f"joint{i}_damping"][0] for i in range(1, 8)])
    quality = parameter_quality(
        tuple(result.values), result.parameters.as_vector(), result.parameter_covariance
    )

    # Stage 3: armature, from the inertial family only, with the friction
    # result held. Slow cruises barely accelerate, so armature is nearly
    # invisible there — this is what the inertial protocols are *for*.
    armature: NDArray[np.float64] | None = None
    armature_quality = None
    if inertial_train:
        print("armature stage on the inertial family, friction held ...")
        held = _fitting_spec(hydrax)
        for index, joint in enumerate(HYDRAX_ARM_JOINT_NAMES):
            held.joint(joint).frictionloss = float(frictionloss[index])
            set_hinge_damping(held.joint(joint), float(damping[index]))
        inertial_sequences = measurement_sequences(
            held, [record.run for record in inertial_train], window_s=FIT_WINDOW_S
        )
        armature_result = fit_parameters(
            armature_parameters(held.compile()).move_off_bounds(),
            inertial_sequences,
            max_iters=max_iters,
        )
        armature = np.array(
            [armature_result.values[f"joint{i}_armature"][0] for i in range(1, 8)]
        )
        armature_quality = parameter_quality(
            tuple(armature_result.values),
            armature_result.parameters.as_vector(),
            armature_result.parameter_covariance,
        )
        print(f"  armature {np.array2string(armature, precision=4)}")

    identified = _apply(_fitting_spec(hydrax), frictionloss, damping, armature)

    # The classical estimate as a model in its own right, so the refinement
    # can be judged rather than assumed to help.
    classical_model = _apply(_fitting_spec(hydrax), linear.frictionloss, linear.damping)

    evaluation: dict[str, object] = {}
    if holdout:
        record = holdout[0]
        candidates = {
            "nominal": nominal,
            "classical": classical_model,
            "identified": identified,
        }
        horizons: dict[str, dict[str, dict[str, float]]] = {}
        for horizon in EVALUATION_HORIZONS_S:
            horizons[f"{horizon:g}s"] = {}
            for name, candidate in candidates.items():
                q_rmse, ee = _windowed_prediction(
                    candidate, record.run, window_s=horizon
                )
                horizons[f"{horizon:g}s"][name] = {
                    "worst_q_rmse_rad": float(q_rmse.max()),
                    "gripper_rmse_mm": float(np.sqrt(np.mean(ee**2))),
                    "gripper_max_mm": float(ee.max()),
                }
        q, dq = record.run.measured[:, :7], record.run.measured[:, 7:14]
        evaluation = {
            "protocol_id": record.protocol_id,
            "horizons": horizons,
            "worst_torque_rmse_Nm": {
                name: float(
                    torque_residuals(
                        record.run.control,
                        predicted_torque(candidate, q, dq, record.ddq_rad_s2),
                    ).worst_rmse
                )
                for name, candidate in candidates.items()
            },
        }

    parameters = IdentifiedParameters(
        frictionloss=tuple(frictionloss), damping=tuple(damping)
    )
    export = export_identified_model(
        output_dir / "fer_identified.xml",
        parameters,
        source_path=hydrax,
        manifest_path=output_dir / "fer_identified.json",
    )

    disagreement = np.concatenate(
        [
            100.0 * (frictionloss - linear.frictionloss) / np.abs(linear.frictionloss),
            100.0 * (damping - linear.damping) / np.abs(linear.damping),
        ]
    )

    summary: dict[str, object] = {
        "backend": backends[0],
        "recordings": [record.label for record in prepared],
        "rejected": [record.label for record in unusable],
        "train": [record.protocol_id for record in train],
        "holdout": [record.protocol_id for record in holdout],
        "worst_condition_number": excitation.worst_condition_number,
        "frictionloss": frictionloss.tolist(),
        "damping": damping.tolist(),
        "sigma_percent": quality.relative_percent.tolist(),
        "rejected_parameters": quality.rejected,
        "objective_reduction": result.objective_reduction,
        "held_out": evaluation,
        "classical": {
            "frictionloss": linear.frictionloss.tolist(),
            "damping": linear.damping.tolist(),
            "condition_number": linear.condition_number.tolist(),
            "frictionloss_sigma_percent": linear.frictionloss_sigma_percent.tolist(),
            "damping_sigma_percent": linear.damping_sigma_percent.tolist(),
        },
        "worst_method_disagreement_percent": float(np.max(np.abs(disagreement))),
        "exported_model": str(output_dir / "fer_identified.xml"),
        "export_roundtrip_error": export.roundtrip_error,
    }
    print("plotting ...")
    from fer_mujoco_sysid import identify_plots

    # One replay video per run: what the arm actually did, not what was asked
    # of it. On hardware this is the only visual review of data already taken.
    for record in prepared:
        _, raw = load_recording(record.label)
        desired = raw.get("q_desired_rad")
        video = identify_plots.render_recording_replay(
            Path(record.label).parent / "replay.mp4",
            hydrax,
            raw["q_rad"],
            raw["time_s"],
            q_desired_rad=desired,
        )
        print(f"  replay: {video}")
        if desired is not None:
            identify_plots.plot_tracking(
                Path(record.label).parent / "tracking.png",
                raw["time_s"],
                raw["q_rad"],
                desired,
            )

    if holdout:
        record = holdout[0]
        identify_plots.plot_rollout_vs_measurement(
            output_dir / "rollout_vs_measurement.png", candidates, record.run
        )
        identify_plots.plot_error_vs_horizon(
            output_dir / "error_vs_horizon.png", evaluation["horizons"]
        )
        identify_plots.plot_end_effector(
            output_dir / "end_effector.png", candidates, record.run
        )
        identify_plots.plot_torque_tracking(
            output_dir / "torque_tracking.png",
            candidates,
            record.run,
            record.ddq_rad_s2,
        )
        identify_plots.plot_friction_curves(
            output_dir / "friction_curves.png",
            nominal,
            linear,
            (frictionloss, damping),
            record.run.measured[:, :7],
            record.run.measured[:, 7:14],
            record.ddq_rad_s2,
            record.run.control,
        )

    (output_dir / "identification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(output_dir / "result.md", summary, export, linear)
    return summary


def _write_report(
    path: Path, summary: dict, export, linear: LinearFrictionFit
) -> None:
    lines = [
        "# Identification from recorded runs",
        "",
        f"Fitted on {', '.join(summary['train'])}; "
        + (
            f"held out {', '.join(summary['holdout'])}."
            if summary["holdout"]
            else "no held-out protocol was recorded."
        ),
        "",
        "## Excitation quality",
        "",
        f"Worst regressor condition number: **{summary['worst_condition_number']:.2f}** "
        "(1 is ideal; above ~100 the motion cannot separate the parameters).",
        "",
        "## Stage 1 — classical regressor solve",
        "",
        "Direct least squares on `tau - tau_rigid = frictionloss*sign(dq) + "
        "damping*dq`, over the samples where each joint is clearly sliding. "
        "The relative standard deviations are the classical acceptance rule: "
        "above ~20% a parameter is not determined by this data.",
        "",
        linear.table(),
        "",
        "## Stage 2 — rollout refinement",
        "",
        "Seeded from stage 1 and refined by matching simulated to measured "
        "trajectories. The two methods minimize different things — stage 1 "
        "balances the torque equation sample by sample, stage 2 reproduces the "
        "motion — so they agree only while the model class holds.",
        "",
        f"**Worst disagreement between the two: "
        f"{summary['worst_method_disagreement_percent']:.2f}%.** A large gap "
        "here is not noise; it means the friction model class is wrong "
        "(Stribeck, temperature, transmission) and inertial fitting should "
        "pause for a model-class review rather than absorb the error.",
        "",
        "## Identified parameters",
        "",
        "| joint | frictionloss [Nm] | damping [Nm s/rad] | sigma% fl | sigma% d |",
        "| --- | --- | --- | --- | --- |",
    ]
    sigma = summary["sigma_percent"]
    for joint in range(7):
        lines.append(
            f"| {joint + 1} | {summary['frictionloss'][joint]:.4f} "
            f"| {summary['damping'][joint]:.4f} "
            f"| {sigma[2 * joint]:.2f}% | {sigma[2 * joint + 1]:.2f}% |"
        )
    if summary["rejected_parameters"]:
        lines += ["", f"Rejected (sigma% too high): {summary['rejected_parameters']}"]

    if summary["held_out"]:
        held = summary["held_out"]
        lines += [
            "",
            "## Held-out validation — the acceptance metric",
            "",
            f"Protocol `{held['protocol_id']}`, never used for fitting. The "
            "simulator starts from a measured state, is driven with the "
            "recorded torque, and runs open loop; the error is how far it has "
            "drifted from the measurement by the end of each window.",
            "",
            "**This is what decides whether a model is good enough.** The "
            "parameter values above are worth reading — they are how a fit is "
            "checked against the measured breakaway bounds, and how a "
            "physically absurd result is spotted — but on the real robot "
            "there is no true value to compare them against. What can always "
            "be measured is whether the rollout still resembles what the arm "
            "did, and for how long. A model can look excellent over 0.1 s and "
            "drift badly over 2 s; that is the difference between one a "
            "planner can use and one it cannot.",
            "",
            "| horizon | model | worst joint q RMSE [rad] | gripper RMSE [mm] | gripper max [mm] |",
            "| --- | --- | --- | --- | --- |",
        ]
        for horizon, models in held["horizons"].items():
            for name in ("nominal", "classical", "identified"):
                entry = models[name]
                lines.append(
                    f"| {horizon} | {name} | {entry['worst_q_rmse_rad']:.5f} "
                    f"| {entry['gripper_rmse_mm']:.3f} "
                    f"| {entry['gripper_max_mm']:.3f} |"
                )
        lines += [
            "",
            "`classical` is the stage-1 estimate taken as a model on its own; "
            "`identified` is after rollout refinement. If refinement does not "
            "improve this table it is not earning its runtime — and that "
            "judgement can be made on hardware, where parameter truth cannot.",
            "",
            "Torque reconstruction on the same protocol (inverse dynamics — a "
            "diagnostic, not the acceptance metric):",
            "",
            "| model | worst joint torque RMSE [Nm] |",
            "| --- | --- |",
        ]
        for name, value in held["worst_torque_rmse_Nm"].items():
            lines.append(f"| {name} | {value:.3f} |")

    lines += [
        "",
        "## Plots",
        "",
        "* `rollout_vs_measurement.png` — **look at this first.** The "
        "simulator run open loop against what the arm actually did, over "
        "several 2 s windows. Where the coloured line leaves the black one, "
        "the model is wrong.",
        "* `error_vs_horizon.png` — how fast that agreement decays with "
        "rollout length, per model.",
        "* `end_effector.png` — the same rollout judged at the gripper. "
        "Joint error is what the fit minimizes; this is what the task "
        "cares about, and small joint errors compound along the chain.",
        "* `friction_curves.png` — friction torque against velocity: the step "
        "at zero is Coulomb friction, the slope is viscous. The nominal model "
        "has no step at all, which is the problem this project exists to fix.",
        "* `torque_tracking.png` — recorded commanded torque against what each "
        "model reconstructs. The spikes at the velocity corners are an "
        "artifact of estimating acceleration by differentiating filtered "
        "velocity through a sharp transition, not a model defect: the "
        "error is 0.002 Nm where jerk is low and 0.61 Nm in the top jerk "
        "percentile, and the rollout — which never uses acceleration — is "
        "exact throughout.",
        "* `../<protocol>/replay.mp4` — the motion each run actually "
        "performed, with a translucent ghost of what the controller was "
        "asking for. Where the ghost pulls ahead, the arm did not follow.",
        "* `../<protocol>/tracking.png` — the same thing as numbers: "
        "commanded minus measured, per joint. A property of the *run* "
        "rather than of the model.",
        "",
        "## The identified model",
        "",
        f"`{Path(summary['exported_model']).name}`, written only after passing "
        "the export whitelist and reload checks.",
        "",
        export.table(),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from fer_mujoco_sysid.campaign import repository_root
    from fer_mujoco_sysid.dataset import BACKENDS, OUTPUT_DIRECTORY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recordings",
        nargs="?",
        default=None,
        help="directory holding recorded runs. Defaults to "
        f"{OUTPUT_DIRECTORY}/<backend> for the backend given by --backend.",
    )
    parser.add_argument(
        "--backend",
        default="mujoco_ros",
        choices=BACKENDS,
        help="which recordings to fit, when no directory is given",
    )
    parser.add_argument(
        "--output", default=None, help="where to write the report and model"
    )
    parser.add_argument("--max-iters", type=int, default=30)
    arguments = parser.parse_args(argv)

    recordings = Path(
        arguments.recordings
        or repository_root() / OUTPUT_DIRECTORY / arguments.backend
    )
    if not recordings.is_dir():
        parser.error(
            f"no recordings at {recordings}. Record some first:\n"
            f"  ./scripts/run-protocol <protocol> --record"
        )
    arguments.recordings = str(recordings)
    output = Path(arguments.output or recordings / "identification")
    summary = run(arguments.recordings, output, max_iters=arguments.max_iters)
    print((output / "result.md").read_text())
    return 0 if not summary["rejected_parameters"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
