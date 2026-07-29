"""Fit, validate, report, and export a MuJoCo model from recorded campaigns.

The release criterion is deliberately behavioral: starting from a withheld
measured state and driven by the recorded applied-torque channel, does the
candidate simulator reproduce the robot over MPPI-relevant horizons?  The
parameter values remain essential diagnostics, but no hardware experiment
provides a hidden parameter truth against which they could be accepted.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import mujoco
import numpy as np

from fer_mujoco_sysid.dataset import (
    BACKENDS,
    OUTPUT_DIRECTORY,
    find_recordings,
    load_recording,
)
from fer_mujoco_sysid.diagnostics import predicted_torque, torque_residuals
from fer_mujoco_sysid.export import (
    ConsumerModelExportCheck,
    ExportCheck,
    export_consumer_model,
)
from fer_mujoco_sysid.io import read_json, write_json
from fer_mujoco_sysid.model import resolve_model_paths
from fer_mujoco_sysid.preparation import (
    PreparedRecording,
    prepare_recording,
    validate_campaign_lineage,
)
from fer_mujoco_sysid.protocol import FRICTION_FAMILY, INERTIAL_FAMILY
from fer_mujoco_sysid.stages import (
    DEFAULT_BODY_CORRECTIONS,
    BodyInertialCorrection,
    StageResult,
    fit_stages,
    fitting_spec,
)
from fer_mujoco_sysid.validation import (
    AcceptanceThresholds,
    ReproductionAcceptance,
    RolloutMetrics,
    accept_reproduction,
    evaluate_models,
)

# Compatibility for callers that used the earlier private model projection.
_fitting_spec = fitting_spec

# Kept as a compatibility constant only. Roles now come from signed protocol
# metadata and are never inferred from this suffix.
HOLDOUT_SUFFIX = "-holdout"

# 0.32 s is the deployed Hydrax Feedback-MPPI plan horizon (8 x 40 ms).
# Shorter/longer views expose one-step quality and open-loop drift without
# replacing the actual consumer-facing gate.
EVALUATION_HORIZONS_S = (0.1, 0.32, 0.5, 2.0)
EXPORT_Q_PARITY_TOLERANCE_RAD = 1e-5
EXPORT_DQ_PARITY_TOLERANCE_RAD_S = 1e-4
EXPORT_EE_PARITY_TOLERANCE_MM = 0.05
RESULT_FORMAT = "fer-mujoco-sysid/identification-result@3"
STATUS_FORMAT = "fer-mujoco-sysid/identification-status@1"
STATUS_FILENAME = "identification_status.json"
CONSUMER_ARTIFACTS = ("fer_identified.xml", "fer_identified.json")


def _serializable_horizons(
    horizons: dict[str, dict[str, RolloutMetrics]],
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        horizon: {
            model_name: metrics.as_dict() for model_name, metrics in models.items()
        }
        for horizon, models in horizons.items()
    }


def _holdout_evaluation(
    recording: PreparedRecording,
    candidates: dict[str, mujoco.MjModel],
) -> tuple[
    dict[str, dict[str, RolloutMetrics]],
    dict[str, float],
    dict[str, object],
]:
    run = recording.protocol_run()
    ddq = recording.protocol_acceleration()
    torque = recording.protocol_classical_torque()
    q = run.measured[:, :7]
    dq = run.measured[:, 7:14]
    horizons = evaluate_models(
        candidates,
        run,
        horizons_s=EVALUATION_HORIZONS_S,
    )
    torque_rmse = {
        name: float(
            torque_residuals(
                torque,
                predicted_torque(model, q, dq, ddq),
            ).worst_rmse
        )
        for name, model in candidates.items()
    }
    serialized: dict[str, object] = {
        "protocol_id": recording.protocol_id,
        "family": recording.family,
        "control_channel": recording.control_channel,
        "horizons": _serializable_horizons(horizons),
        "worst_torque_rmse_Nm": torque_rmse,
    }
    return horizons, torque_rmse, serialized


def _export_behavior_parity(
    fitted: mujoco.MjModel,
    exported: mujoco.MjModel,
    holdouts: list[PreparedRecording],
) -> dict[str, float]:
    worst_q = 0.0
    worst_dq = 0.0
    worst_ee = 0.0
    for recording in holdouts:
        compared = evaluate_models(
            {"fitted": fitted, "exported": exported},
            recording.protocol_run(),
            horizons_s=EVALUATION_HORIZONS_S,
        )
        for models in compared.values():
            before = models["fitted"]
            after = models["exported"]
            worst_q = max(
                worst_q,
                float(np.max(np.abs(before.q_rmse_rad - after.q_rmse_rad))),
            )
            worst_dq = max(
                worst_dq,
                float(np.max(np.abs(before.dq_rmse_rad_s - after.dq_rmse_rad_s))),
            )
            worst_ee = max(
                worst_ee,
                abs(before.gripper_rmse_mm - after.gripper_rmse_mm),
            )
    if (
        worst_q > EXPORT_Q_PARITY_TOLERANCE_RAD
        or worst_dq > EXPORT_DQ_PARITY_TOLERANCE_RAD_S
        or worst_ee > EXPORT_EE_PARITY_TOLERANCE_MM
    ):
        raise RuntimeError(
            "exported model does not reproduce the accepted in-memory model "
            f"(q={worst_q:.3g} rad, dq={worst_dq:.3g} rad/s, "
            f"gripper={worst_ee:.3g} mm)"
        )
    return {
        "worst_q_rmse_difference_rad": worst_q,
        "worst_dq_rmse_difference_rad_s": worst_dq,
        "worst_gripper_rmse_difference_mm": worst_ee,
    }


def _recording_summary(recording: PreparedRecording) -> dict[str, object]:
    return {
        "path": recording.label,
        "recording_sha256": recording.content_sha256,
        "protocol_id": recording.protocol_id,
        "protocol_sha256": recording.protocol_content_sha256,
        "family": recording.family,
        "role": recording.role,
        "backend": recording.backend,
        "control_channel": recording.control_channel,
        "source_model_sha256": recording.source_model_sha256,
        "preprocessing": recording.preprocessing,
        "health": recording.health,
        "classical_samples": int(recording.stage.classical_mask.sum()),
        "analysis_samples": int(recording.stage.analysis_mask.sum()),
        "protocol_samples": int(recording.stage.protocol_mask.sum()),
    }


def _write_attempt_status(
    output_dir: Path,
    *,
    run_id: str,
    state: str,
    error: BaseException | None = None,
) -> None:
    """Publish the authoritative state last, so stale XML is never ambiguous."""
    value: dict[str, object] = {
        "format": STATUS_FORMAT,
        "run_id": run_id,
        "state": state,
        "result_manifest": (
            "identification.json" if state in {"accepted", "rejected"} else None
        ),
        "consumer_model_current": state == "accepted",
    }
    if error is not None:
        value["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_json(output_dir / STATUS_FILENAME, value)


def _archive_previous_consumer(output_dir: Path, *, run_id: str) -> list[str]:
    """Move only known generated model files aside before publishing a result."""
    present = [
        output_dir / name
        for name in CONSUMER_ARTIFACTS
        if (output_dir / name).is_file()
    ]
    if not present:
        return []
    archive = output_dir / "previous_consumer_models" / f"superseded-by-{run_id}"
    archive.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for source in present:
        destination = archive / source.name
        os.replace(source, destination)
        archived.append(str(destination.relative_to(output_dir)))
    return archived


def _publish_staged_attempt(
    staging_dir: Path,
    output_dir: Path,
    *,
    run_id: str,
) -> list[str]:
    """Atomically replace each generated file after the attempt is complete.

    The final status file is deliberately not part of this operation. The
    caller writes it only after every staged file has reached its final path.
    """
    archived = _archive_previous_consumer(output_dir, run_id=run_id)
    for source in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(staging_dir)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    return archived


def _rewrite_staged_consumer_manifest(
    manifest_path: Path,
    *,
    final_model_path: Path,
) -> None:
    """Replace the temporary export path with the stable delivered path."""
    manifest = read_json(manifest_path)
    consumer = manifest.get("consumer_model")
    if not isinstance(consumer, dict):
        raise RuntimeError("consumer export manifest has no consumer_model mapping")
    consumer["path"] = str(final_model_path)
    write_json(manifest_path, manifest)


def _relative_media_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _render_outputs(
    output_dir: Path,
    model_path: Path,
    prepared: list[PreparedRecording],
    holdouts: list[PreparedRecording],
    stages: StageResult,
    evaluations: dict[str, dict[str, object]],
) -> list[str]:
    """Write the expected videos and metric plots, returning relative paths."""
    from fer_mujoco_sysid import identify_plots

    written: list[str] = []
    for recording in prepared:
        _, raw = load_recording(recording.root)
        desired = raw.get("q_desired_rad")
        recording_dir = output_dir / "media" / recording.protocol_id
        video = identify_plots.render_recording_replay(
            recording_dir / "replay.mp4",
            model_path,
            raw["q_rad"],
            raw["time_s"],
            q_desired_rad=desired,
        )
        written.append(_relative_media_path(video, output_dir))
        if desired is not None:
            tracking = identify_plots.plot_tracking(
                recording_dir / "tracking.png",
                raw["time_s"],
                raw["q_rad"],
                desired,
            )
            written.append(_relative_media_path(tracking, output_dir))

    candidates = {
        "nominal": stages.nominal,
        "classical": stages.classical,
        "identified": stages.identified,
    }
    for recording in holdouts:
        tag = recording.protocol_id.replace("-", "_")
        run = recording.protocol_run()
        ddq = recording.protocol_acceleration()
        evaluation = evaluations[recording.protocol_id]
        paths = (
            identify_plots.plot_rollout_vs_measurement(
                output_dir / f"rollout_vs_measurement_{tag}.png",
                candidates,
                run,
            ),
            identify_plots.plot_error_vs_horizon(
                output_dir / f"error_vs_horizon_{tag}.png",
                evaluation["horizons"],
            ),
            identify_plots.plot_end_effector(
                output_dir / f"end_effector_{tag}.png",
                candidates,
                run,
            ),
            identify_plots.plot_torque_tracking(
                output_dir / f"torque_tracking_{tag}.png",
                candidates,
                run,
                ddq,
                recording.protocol_classical_torque(),
                torque_label=recording.control_channel,
            ),
        )
        written.extend(_relative_media_path(path, output_dir) for path in paths)

        if recording.family == FRICTION_FAMILY:
            mask = recording.stage.classical_mask
            friction = identify_plots.plot_friction_curves(
                output_dir / "friction_curves.png",
                stages.nominal,
                stages.linear,
                (
                    np.asarray(stages.parameters.frictionloss),
                    np.asarray(stages.parameters.damping),
                ),
                recording.run.measured[mask, :7],
                recording.run.measured[mask, 7:14],
                recording.ddq_rad_s2[mask],
                recording.stage.classical_torque_Nm[mask],
            )
            written.append(_relative_media_path(friction, output_dir))
    return written


def _run_attempt(
    recordings_root: str | Path,
    output_dir: Path,
    *,
    run_id: str,
    max_iters: int = 30,
    dynamic_starts: int = 3,
    body_corrections: tuple[BodyInertialCorrection, ...] = DEFAULT_BODY_CORRECTIONS,
    acceptance_thresholds: AcceptanceThresholds | None = None,
    render_media: bool = True,
) -> tuple[dict[str, object], ReproductionAcceptance]:
    """Build one complete attempt in staging, then publish its files."""
    recordings_root = Path(recordings_root)
    paths = find_recordings(recordings_root)
    if not paths:
        raise FileNotFoundError(f"no recordings under {recordings_root}")

    model_path = resolve_model_paths().require().hydrax
    nominal = fitting_spec(model_path).compile()
    print(f"loading and binding {len(paths)} recording(s) ...")
    prepared = [
        prepare_recording(
            path,
            nominal,
            expected_source_model=model_path,
        )
        for path in paths
    ]
    unusable = [record for record in prepared if not record.health["usable"]]
    if unusable:
        details = "; ".join(
            f"{record.protocol_id}: {record.health['problems']}" for record in unusable
        )
        raise RuntimeError(f"campaign contains unusable recordings: {details}")
    validate_campaign_lineage(prepared, require_holdouts=True)

    training = [record for record in prepared if record.role == "train"]
    holdouts = [record for record in prepared if record.role == "holdout"]
    print(
        "  friction train: "
        f"{[r.protocol_id for r in training if r.family == FRICTION_FAMILY]}\n"
        "  inertial train: "
        f"{[r.protocol_id for r in training if r.family == INERTIAL_FAMILY]}\n"
        f"  holdouts: {[r.protocol_id for r in holdouts]}"
    )

    print("fitting friction, observable dynamic corrections, then friction ...")
    stages = fit_stages(
        model_path,
        [record.stage for record in prepared],
        max_iters=max_iters,
        dynamic_starts=dynamic_starts,
        body_corrections=body_corrections,
    )
    candidates = {
        "nominal": stages.nominal,
        "classical": stages.classical,
        "identified": stages.identified,
    }

    evaluation_objects: dict[str, dict[str, dict[str, RolloutMetrics]]] = {}
    torque_objects: dict[str, dict[str, float]] = {}
    evaluations: dict[str, dict[str, object]] = {}
    print("evaluating every family on withheld measured outputs ...")
    for recording in holdouts:
        horizons, torque, serialized = _holdout_evaluation(recording, candidates)
        evaluation_objects[recording.protocol_id] = horizons
        torque_objects[recording.protocol_id] = torque
        evaluations[recording.protocol_id] = serialized

    acceptance = accept_reproduction(
        evaluation_objects,
        required_families=tuple(record.protocol_id for record in holdouts),
        torque_rmse_Nm=torque_objects,
        thresholds=acceptance_thresholds,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output_dir.name}-staging-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_dir.parent) as temporary:
        staging_dir = Path(temporary)
        exported_model = staging_dir / CONSUMER_ARTIFACTS[0]
        exported_manifest = staging_dir / CONSUMER_ARTIFACTS[1]
        export: ConsumerModelExportCheck | None = None
        export_parity: dict[str, float] | None = None

        if acceptance.accepted:
            # `stages.identified` is the exact gravity-free projection that
            # passed the withheld gate. The consumer exporter installs those
            # parameters into the gravity-enabled full Panda and verifies the
            # ordinary and Hydrax/MPPI load paths before anything is released.
            print("staging the accepted gravity-enabled consumer model ...")
            export = export_consumer_model(
                exported_model,
                stages.parameters,
                accepted_fitted_model=stages.identified,
                source_path=model_path,
                manifest_path=exported_manifest,
            )
            _rewrite_staged_consumer_manifest(
                exported_manifest,
                final_model_path=output_dir / CONSUMER_ARTIFACTS[0],
            )
            reloaded_fit_projection = fitting_spec(exported_model).compile()
            export_parity = _export_behavior_parity(
                stages.identified,
                reloaded_fit_projection,
                holdouts,
            )

        summary: dict[str, object] = {
            "format": RESULT_FORMAT,
            "run_id": run_id,
            "status": "accepted" if acceptance.accepted else "rejected",
            "backend": prepared[0].backend,
            "source_model": {
                "path": str(model_path),
                "sha256": prepared[0].source_model_sha256,
            },
            "recordings": [_recording_summary(record) for record in prepared],
            "train": [record.protocol_id for record in training],
            "holdout": [record.protocol_id for record in holdouts],
            "stages": stages.summary(),
            "held_out": evaluations,
            "acceptance": acceptance.as_dict(),
            "torque_and_gravity_conventions": {
                "identification": (
                    "gravity-free effective joint effort on top of Franka "
                    "internal gravity compensation"
                ),
                "consumer_model": (
                    "gravity-enabled full Panda; Hydrax MPPI inverse dynamics "
                    "and rollouts include gravity"
                ),
                "mujoco_simulation_command": "send the full modeled joint torque",
                "real_robot_command": (
                    "subtract the model gravity contribution only in the final "
                    "LFC adapter because Franka compensates it internally"
                ),
            },
            "exported_model": (
                str(output_dir / CONSUMER_ARTIFACTS[0]) if acceptance.accepted else None
            ),
            "consumer_model_manifest": (
                str(output_dir / CONSUMER_ARTIFACTS[1]) if acceptance.accepted else None
            ),
            "export_roundtrip_error": (
                export.roundtrip_error if export is not None else None
            ),
            "export_behavior_parity": export_parity,
            "consumer_export_verification": (
                {
                    "gravity_enabled": True,
                    "ordinary_full_mjmodel_reload": True,
                    "hydrax_planning_derivation_reload": True,
                    "planning_compiled_max_abs_error": (export.planning_compiled_error),
                    "planning_behavior_max_abs_error": (export.planning_behavior_error),
                    "gravity_normalized_fit_behavior_max_abs_error": (
                        export.fit_convention_behavior_error
                    ),
                }
                if export is not None
                else None
            ),
            "media": [],
        }

        if render_media:
            print("rendering recording videos and per-family metric plots ...")
            try:
                summary["media"] = _render_outputs(
                    staging_dir,
                    model_path,
                    prepared,
                    holdouts,
                    stages,
                    evaluations,
                )
            except Exception as error:
                if acceptance.accepted:
                    # A nominally accepted fit is not deliverable without the
                    # requested evidence. Its staged XML is discarded.
                    raise
                summary["media_generation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

        write_json(staging_dir / "identification.json", summary)
        _write_report(
            staging_dir / "result.md",
            summary,
            stages,
            export,
            acceptance,
        )
        _publish_staged_attempt(staging_dir, output_dir, run_id=run_id)
    return summary, acceptance


def run(
    recordings_root: str | Path,
    output_dir: str | Path,
    *,
    max_iters: int = 30,
    dynamic_starts: int = 3,
    body_corrections: tuple[BodyInertialCorrection, ...] = DEFAULT_BODY_CORRECTIONS,
    acceptance_thresholds: AcceptanceThresholds | None = None,
    render_media: bool = True,
) -> dict[str, object]:
    """Fit a campaign and publish one unambiguous accepted or rejected result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    _write_attempt_status(output_dir, run_id=run_id, state="running")
    try:
        summary, acceptance = _run_attempt(
            recordings_root,
            output_dir,
            run_id=run_id,
            max_iters=max_iters,
            dynamic_starts=dynamic_starts,
            body_corrections=body_corrections,
            acceptance_thresholds=acceptance_thresholds,
            render_media=render_media,
        )
        _write_attempt_status(
            output_dir,
            run_id=run_id,
            state="accepted" if acceptance.accepted else "rejected",
        )
    except BaseException as error:
        _write_attempt_status(
            output_dir,
            run_id=run_id,
            state="failed",
            error=error,
        )
        raise
    acceptance.require()
    return summary


def _write_report(
    path: Path,
    summary: dict[str, object],
    stages: StageResult,
    export: ExportCheck | None,
    acceptance: ReproductionAcceptance,
) -> None:
    recordings = summary["recordings"]
    assert isinstance(recordings, list)
    lines = [
        "# Identification from recorded runs",
        "",
        "The release target is simulator reproduction of withheld robot "
        "measurements over MPPI-relevant horizons.",
        "",
        "The protocol fit and its held-out rollouts use the **gravity-free "
        "effective-effort convention**: commanded effort on top of Franka's "
        "internal gravity compensation. This is a projection of the physical "
        "model, not the delivered model itself.",
        "",
        f"Fitted on {', '.join(summary['train'])}; "
        f"held out {', '.join(summary['holdout'])}.",
        "",
        "## Recording and lineage",
        "",
        "| protocol | family | role | control used | classical / rollout rows "
        "| health |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for raw in recordings:
        record = raw
        lines.append(
            f"| {record['protocol_id']} | {record['family']} | "
            f"{record['role']} | `{record['control_channel']}` | "
            f"{record['classical_samples']} / {record['analysis_samples']} | "
            f"{'usable' if record['health']['usable'] else 'REJECTED'} |"
        )

    lines += [
        "",
        "## Stage 1 — friction on true cruise plateaus",
        "",
        "Only explicitly marked constant-velocity windows enter this solve; "
        "approach, acceleration, reversal, hold, return, and filter-edge "
        "samples are excluded.",
        "",
        f"Uncertainty: `{stages.linear.covariance_method}`, "
        f"{stages.linear.covariance_lags} HAC lags.",
        "",
        stages.linear.table(),
        "",
        "## Stage 2 — rollout-refined friction",
        "",
        f"Objective reduction: **{stages.first_friction.objective_reduction:.2%}**.",
        "",
        "## Stage 3 — observable armature and link inertials",
        "",
    ]
    if stages.dynamic is None:
        lines += [
            "No inertial training recording was supplied; armature and body "
            "inertials remain nominal.",
            "",
        ]
    else:
        observable = stages.observable_dynamic
        assert observable is not None
        lines += [
            "Candidate corrections are tightly bounded around CAD. The local "
            "recording Jacobian releases only scalar directions that are "
            "sufficiently sensitive, conditioned, and decorrelated.",
            "",
            f"Accepted: {', '.join(observable.accepted_names) or 'none'}.",
            "",
            f"Left nominal: {', '.join(observable.rejected_names) or 'none'}.",
            "",
            f"Objective reduction: **{stages.dynamic.objective_reduction:.2%}**; "
            f"conditioning ratio: **{stages.dynamic.conditioning_ratio:.3g}**.",
            "",
        ]
        if stages.parameters.armature is not None:
            lines += [
                "| joint | armature [kg m²] |",
                "| --- | ---: |",
                *[
                    f"| {index + 1} | {value:.6g} |"
                    for index, value in enumerate(stages.parameters.armature)
                ],
                "",
            ]
        if stages.parameters.body_inertials:
            lines += [
                "| body | mass [kg] | COM [m] | full inertia tensor |",
                "| --- | ---: | --- | --- |",
            ]
            for body in stages.parameters.body_inertials:
                lines.append(
                    f"| {body.body} | {body.mass:.6g} | "
                    f"`{np.array(body.ipos)}` | `{np.array(body.inertia)}` |"
                )
            lines.append("")

    lines += [
        "## Stage 4 — friction refit with accepted dynamics held",
        "",
        f"Objective reduction: "
        f"**{stages.friction_refit.objective_reduction:.2%}**; "
        "worst classical/rollout disagreement: "
        f"**{stages.method_disagreement_percent:.2f}%**.",
        "",
        "The friction and inertial protocols remain separate solves. They were "
        f"alternated for **{stages.coupling_refinement_rounds}** refinement "
        "round(s) until the largest shared-model parameter change was "
        f"**{stages.coupling_max_change_fraction:.3g}** of its allowed span.",
        "",
        "| joint | frictionloss [Nm] | damping [Nm s/rad] |",
        "| --- | ---: | ---: |",
    ]
    for index, (friction, damping) in enumerate(
        zip(
            stages.parameters.frictionloss or (),
            stages.parameters.damping or (),
            strict=True,
        )
    ):
        lines.append(f"| {index + 1} | {friction:.6g} | {damping:.6g} |")

    lines += [
        "",
        "## Held-out simulator reproduction — release gate",
        "",
        f"Verdict: **{'ACCEPTED' if acceptance.accepted else 'REJECTED'}**.",
        "",
    ]
    held_out = summary["held_out"]
    assert isinstance(held_out, dict)
    for protocol_id, raw in held_out.items():
        evaluation = raw
        lines += [
            f"### `{protocol_id}` ({evaluation['family']})",
            "",
            "| horizon | model | worst q RMSE [rad] | worst dq RMSE [rad/s] "
            "| gripper RMSE [mm] | gripper max [mm] |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for horizon, models in evaluation["horizons"].items():
            for name in ("nominal", "classical", "identified"):
                metrics = models[name]
                lines.append(
                    f"| {horizon} | {name} | "
                    f"{metrics['worst_q_rmse_rad']:.6g} | "
                    f"{metrics['worst_dq_rmse_rad_s']:.6g} | "
                    f"{metrics['gripper_rmse_mm']:.6g} | "
                    f"{metrics['gripper_max_mm']:.6g} |"
                )
        lines += [
            "",
            "Worst inverse-dynamics torque RMSE:",
            "",
            "| model | RMSE [Nm] |",
            "| --- | ---: |",
        ]
        for name, value in evaluation["worst_torque_rmse_Nm"].items():
            lines.append(f"| {name} | {value:.6g} |")
        lines.append("")

    lines += ["## Delivered MuJoCo model", ""]
    if export is None:
        lines += [
            "**No consumer XML was published.** The fit failed the held-out "
            "release gate; the tables and plots are retained as rejection "
            "diagnostics.",
            "",
        ]
    else:
        exported_model = summary["exported_model"]
        assert isinstance(exported_model, str)
        lines += [
            f"`{Path(exported_model).name}` is the **gravity-enabled full "
            "Panda consumer model**. It carries the physical parameters from "
            "the accepted gravity-free fit projection, but it is not a "
            "gravity-free model.",
            "",
            "The export passed parameter round-trip, ordinary full-model "
            "reload, the exact Hydrax seven-joint planning derivation, and "
            "gravity-normalized parity with the accepted fit. Hydrax/MPPI "
            "therefore performs inverse dynamics and rollouts with gravity. "
            "MuJoCo simulation receives the full torque; on the real Franka, "
            "the final LFC adapter subtracts the gravity contribution because "
            "the robot supplies it internally.",
            "",
            export.table(),
            "",
        ]

    lines += [
        "## Output media",
        "",
        "- `media/<protocol>/replay.mp4`: measured motion with desired ghost.",
        "- `media/<protocol>/tracking.png`: commanded-minus-measured tracking.",
        "- `rollout_vs_measurement_<holdout>.png`: open-loop simulator versus robot.",
        "- `error_vs_horizon_<holdout>.png`: joint, velocity, and gripper error.",
        "- `end_effector_<holdout>.png`: Cartesian reproduction.",
        "- `torque_tracking_<holdout>.png`: recorded torque versus inverse dynamics.",
        "- `friction_curves.png`: plateau torque versus joint velocity.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from fer_mujoco_sysid.campaign import repository_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recordings",
        nargs="?",
        default=None,
        help="recording campaign root; defaults to output/<backend>",
    )
    parser.add_argument(
        "--backend",
        default="mujoco_ros",
        choices=BACKENDS,
        help="default recording partition",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument(
        "--dynamic-starts",
        type=int,
        default=3,
        help="independent starts for the armature/body stage",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="skip videos/plots for a diagnostic run; normal deliverables include them",
    )
    arguments = parser.parse_args(argv)

    recordings = Path(
        arguments.recordings or repository_root() / OUTPUT_DIRECTORY / arguments.backend
    )
    if not recordings.is_dir():
        parser.error(
            f"no recordings at {recordings}. Record the six-protocol campaign first."
        )
    output = Path(arguments.output or recordings / "identification")
    try:
        summary = run(
            recordings,
            output,
            max_iters=arguments.max_iters,
            dynamic_starts=arguments.dynamic_starts,
            render_media=not arguments.no_media,
        )
    except RuntimeError:
        status_path = output / STATUS_FILENAME
        if (
            status_path.is_file()
            and read_json(status_path).get("state") == "rejected"
            and (output / "result.md").is_file()
        ):
            print((output / "result.md").read_text())
            return 2
        raise
    print((output / "result.md").read_text())
    return 0 if summary["acceptance"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
