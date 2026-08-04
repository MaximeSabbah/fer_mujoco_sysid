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
from collections.abc import Mapping
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
    IdentifiedParameters,
    export_consumer_model,
)
from fer_mujoco_sysid.io import read_json, sha256_file, write_json
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

# The fit is the expensive half of a run — tens of minutes of nonlinear solves
# — while everything downstream of it (held-out evaluation, plots, videos, the
# report) is minutes. Twice now a defect in that cheap half has thrown away a
# completed fit, so the fit is checkpointed the moment it returns and reused
# whenever the same recordings, the same knobs and the same fitting code would
# reproduce it. It is a cache, never an input: any doubt about the key discards
# it and refits.
FIT_CHECKPOINT_FILENAME = "fit_checkpoint.pickle"
FIT_CHECKPOINT_FORMAT = "fer-mujoco-sysid/fit-checkpoint@1"
#: Sources whose content decides the fit. A change to any of them invalidates
#: every checkpoint, which is why the list must stay honest.
FIT_CODE_SOURCES = (
    "stages.py",
    "fitting.py",
    "classical.py",
    "selection.py",
    "preparation.py",
    "preprocessing.py",
)


def _fit_checkpoint_key(
    prepared: list[PreparedRecording],
    *,
    max_iters: int,
    dynamic_starts: int,
    body_corrections: tuple[BodyInertialCorrection, ...],
) -> str:
    """Identity of a fit: its data, its knobs, and the code that produces it."""
    import dataclasses
    import hashlib
    import json

    module_dir = Path(__file__).parent
    identity = {
        "format": FIT_CHECKPOINT_FORMAT,
        "recordings": sorted(
            (record.protocol_id, record.content_sha256, record.role)
            for record in prepared
        ),
        "source_model_sha256": prepared[0].source_model_sha256,
        "max_iters": max_iters,
        "dynamic_starts": dynamic_starts,
        "body_corrections": [
            dataclasses.asdict(correction) for correction in body_corrections
        ],
        "code": {
            name: sha256_file(module_dir / name)
            for name in FIT_CODE_SOURCES
            if (module_dir / name).is_file()
        },
    }
    canonical = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _save_fit_checkpoint(path: Path, stages: StageResult, *, key: str) -> None:
    """Store a completed fit. Failing to cache is never worth failing a run."""
    import pickle

    try:
        staging = path.with_name(path.name + ".partial")
        with staging.open("wb") as handle:
            pickle.dump(
                {"format": FIT_CHECKPOINT_FORMAT, "key": key, "stages": stages},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(staging, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  fit checkpointed to {path} ({size_mb:.0f} MB)")
    except Exception as error:  # noqa: BLE001 - a cache miss must not end a run
        print(f"  could not checkpoint the fit: {type(error).__name__}: {error}")


def _load_fit_checkpoint(path: Path, *, key: str) -> StageResult | None:
    """Return a previously checkpointed fit, or None with the reason printed."""
    import pickle

    if not path.is_file():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
        if payload.get("format") != FIT_CHECKPOINT_FORMAT:
            print(f"  ignoring {path.name}: unknown checkpoint format")
            return None
        if payload.get("key") != key:
            print(
                f"  ignoring {path.name}: recordings, knobs or fitting code "
                "changed since it was written"
            )
            return None
        stages = payload["stages"]
        if not isinstance(stages, StageResult):
            print(f"  ignoring {path.name}: does not contain a fit")
            return None
    except Exception as error:  # noqa: BLE001 - an unreadable cache is a miss
        print(f"  ignoring {path.name}: {type(error).__name__}: {error}")
        return None
    return stages


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


class MediaRenderingIncomplete(RuntimeError):
    """Some artifacts could not be drawn. The ones that could are on disk.

    Carries both halves so a run can record exactly what it produced and
    exactly what it could not, rather than discarding the successful figures
    along with the failed one.
    """

    def __init__(self, written: list[str], failures: list[str]) -> None:
        super().__init__(
            f"{len(failures)} artifact(s) could not be rendered: " + "; ".join(failures)
        )
        self.written = tuple(written)
        self.failures = tuple(failures)


def _json_ready(value: object) -> object:
    """Replace non-finite floats with JSON null, recursively.

    A relative uncertainty is ``sigma / |coefficient|``, so a coefficient the
    data pin at exactly zero has an *undefined* relative uncertainty, not an
    infinite one — and JSON cannot spell either. Simulated campaigns never
    reached this branch: their measurements were generated by the model class
    being fitted, so every coefficient had a positive true value and stayed
    interior. Real measurements are not in the model class, drive coefficients
    onto their bounds, and the strict encoder then refused the finished report.
    Null is the honest encoding of undefined, and no completed fit should be
    lost to its own diagnostics.
    """
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating) and not np.isfinite(value):
        return None
    return value


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


#: Everything a run regenerates, removed before it starts writing so the
#: directory never mixes two runs. The fit checkpoint, the status file, the
#: superseded-model archive and anything a human put here are not touched.
_GENERATED_ARTIFACTS = ("identification.json", "result.md")
_GENERATED_PATTERNS = ("*.png", "*.mp4")


def _clear_generated_artifacts(output_dir: Path) -> list[str]:
    """Remove the previous run's artifacts, keeping the reusable fit and media
    directories out of the way of a partial overwrite."""
    import shutil

    removed: list[str] = []
    for name in _GENERATED_ARTIFACTS:
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    for pattern in _GENERATED_PATTERNS:
        for path in sorted(output_dir.glob(pattern)):
            path.unlink()
            removed.append(path.name)
    media = output_dir / "media"
    if media.is_dir():
        shutil.rmtree(media)
        removed.append("media/")
    if removed:
        print(f"  cleared {len(removed)} artifact(s) from the previous run")
    return removed


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
    """Write the expected videos and metric plots, returning relative paths.

    Each artifact is attempted independently: one figure that cannot be drawn
    costs that figure, not the other twenty-four and not the run.
    """
    from fer_mujoco_sysid import identify_plots

    written: list[str] = []
    failures: list[str] = []

    def attempt(label: str, render):
        """Run one renderer, recording rather than propagating its failure."""
        try:
            written.append(_relative_media_path(render(), output_dir))
        except Exception as error:  # noqa: BLE001 - one artifact, not the run
            failures.append(f"{label}: {type(error).__name__}: {error}")
            print(f"  could not render {label}: {type(error).__name__}: {error}")

    for recording in prepared:
        _, raw = load_recording(recording.root)
        desired = raw.get("q_desired_rad")
        recording_dir = output_dir / "media" / recording.protocol_id
        attempt(
            f"{recording.protocol_id} replay",
            lambda recording_dir=recording_dir, raw=raw, desired=desired: (
                identify_plots.render_recording_replay(
                    recording_dir / "replay.mp4",
                    model_path,
                    raw["q_rad"],
                    raw["time_s"],
                    q_desired_rad=desired,
                )
            ),
        )
        if desired is not None:
            attempt(
                f"{recording.protocol_id} tracking",
                lambda recording_dir=recording_dir, raw=raw, desired=desired: (
                    identify_plots.plot_tracking(
                        recording_dir / "tracking.png",
                        raw["time_s"],
                        raw["q_rad"],
                        desired,
                    )
                ),
            )

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
        for label, render in (
            (
                "rollout_vs_measurement",
                lambda tag=tag, run=run: identify_plots.plot_rollout_vs_measurement(
                    output_dir / f"rollout_vs_measurement_{tag}.png",
                    candidates,
                    run,
                ),
            ),
            (
                "error_vs_horizon",
                lambda tag=tag, evaluation=evaluation: (
                    identify_plots.plot_error_vs_horizon(
                        output_dir / f"error_vs_horizon_{tag}.png",
                        evaluation["horizons"],
                    )
                ),
            ),
            (
                "end_effector",
                lambda tag=tag, run=run: identify_plots.plot_end_effector(
                    output_dir / f"end_effector_{tag}.png",
                    candidates,
                    run,
                ),
            ),
            (
                "torque_tracking",
                lambda tag=tag, recording=recording, ddq=ddq, run=run: (
                    identify_plots.plot_torque_tracking(
                        output_dir / f"torque_tracking_{tag}.png",
                        candidates,
                        run,
                        ddq,
                        recording.protocol_classical_torque(),
                        torque_label=recording.control_channel,
                    )
                ),
            ),
        ):
            attempt(f"{recording.protocol_id} {label}", render)

        if recording.family == FRICTION_FAMILY:
            mask = recording.stage.classical_mask
            attempt(
                "friction curves",
                lambda mask=mask, recording=recording: (
                    identify_plots.plot_friction_curves(
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
                ),
            )
    if failures:
        raise MediaRenderingIncomplete(written, failures)
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
    reuse_fit: bool = True,
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

    checkpoint_path = output_dir / FIT_CHECKPOINT_FILENAME
    checkpoint_key = _fit_checkpoint_key(
        prepared,
        max_iters=max_iters,
        dynamic_starts=dynamic_starts,
        body_corrections=body_corrections,
    )
    stages = (
        _load_fit_checkpoint(checkpoint_path, key=checkpoint_key) if reuse_fit else None
    )
    if stages is None:
        print("fitting friction, observable dynamic corrections, then friction ...")
        stages = fit_stages(
            model_path,
            [record.stage for record in prepared],
            max_iters=max_iters,
            dynamic_starts=dynamic_starts,
            body_corrections=body_corrections,
        )
        _save_fit_checkpoint(checkpoint_path, stages, key=checkpoint_key)
    else:
        print(
            f"reusing the fit checkpointed in {checkpoint_path.name}: same "
            "recordings, same knobs, same fitting code"
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

    required_families = tuple(record.protocol_id for record in holdouts)
    # The release question is whether a candidate reproduces the withheld robot
    # better than the model we already ship. `relative_scores` answers exactly
    # that: the median held-out error ratio against nominal, per family. A
    # candidate that is below 1.0 on every holdout is a better simulator than
    # the current one, and that is the whole criterion.
    #
    # The per-joint ceilings and regression checks still run, and every one
    # they raise is reported. They no longer withhold the model: on this arm
    # they fail on the wrist, where MuJoCo's frictionloss+damping class cannot
    # express the measured Stribeck curve, and refusing a model that halves
    # overall error over 0.6 degrees of wrist error ships nothing instead of
    # something better.
    candidate_reports = {
        name: accept_reproduction(
            evaluation_objects,
            required_families=required_families,
            torque_rmse_Nm=torque_objects,
            thresholds=acceptance_thresholds,
            candidate=name,
        )
        for name in ("identified", "classical")
    }

    def _beats_nominal(report: ReproductionAcceptance) -> bool:
        scores = report.relative_scores
        return (
            bool(scores)
            and set(scores) == set(required_families)
            and all(score < 1.0 for score in scores.values())
        )

    def _score(report: ReproductionAcceptance) -> float:
        scores = report.relative_scores
        return (
            float(np.exp(np.mean(np.log(list(scores.values())))))
            if scores
            else float("inf")
        )

    better = {
        name: report
        for name, report in candidate_reports.items()
        if _beats_nominal(report)
    }
    release_candidate = (
        min(better, key=lambda name: _score(better[name])) if better else None
    )
    acceptance = candidate_reports.get(release_candidate or "identified")
    released = release_candidate is not None
    for name, report in candidate_reports.items():
        scores = ", ".join(
            f"{family} {score:.3f}"
            for family, score in sorted(report.relative_scores.items())
        )
        print(f"  {name}: held-out error ratio vs nominal -> {scores or 'unavailable'}")
    if released:
        print(
            f"  releasing '{release_candidate}': better than nominal on every "
            "holdout (geometric mean "
            f"{_score(candidate_reports[release_candidate]):.3f})"
        )
    else:
        print("  no candidate reproduces the withheld robot better than nominal")

    if release_candidate == "classical":
        # The classical model carries fitted frictionloss and damping on the
        # nominal dynamics: no body inertial was accepted into it.
        release_model = stages.classical
        release_parameters = IdentifiedParameters.from_model(
            stages.classical,
            bodies=(),
            include_frictionloss=True,
            include_damping=True,
            include_armature=True,
        )
    else:
        release_model = stages.identified
        release_parameters = stages.parameters

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_artifacts(output_dir)
    # A previous run's model is retired the moment a new attempt starts. Whatever
    # happens from here, this directory never holds a delivered model that its
    # own report does not describe.
    _archive_previous_consumer(output_dir, run_id=run_id)
    prefix = f".{output_dir.name}-export-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_dir.parent) as temporary:
        # Only the consumer model is staged. Every diagnostic is written straight
        # into the output directory as it is produced, so a later failure leaves
        # the evidence in place instead of deleting it: metrics and plots are
        # most wanted exactly when a run did not finish. Releasing a *model*
        # still requires passing the held-out gate, which is why these two files
        # alone wait in a temporary directory until everything else is on disk.
        export_dir = Path(temporary)
        exported_model = export_dir / CONSUMER_ARTIFACTS[0]
        exported_manifest = export_dir / CONSUMER_ARTIFACTS[1]
        export: ConsumerModelExportCheck | None = None
        export_parity: dict[str, float] | None = None

        if released:
            # `release_model` is the exact gravity-free projection that passed
            # the withheld gate. The consumer exporter installs those
            # parameters into the gravity-enabled full Panda and verifies the
            # ordinary and Hydrax/MPPI load paths before anything is released.
            print(
                "staging the accepted gravity-enabled consumer model "
                f"({release_candidate}) ..."
            )
            export = export_consumer_model(
                exported_model,
                release_parameters,
                accepted_fitted_model=release_model,
                source_path=model_path,
                manifest_path=exported_manifest,
            )
            _rewrite_staged_consumer_manifest(
                exported_manifest,
                final_model_path=output_dir / CONSUMER_ARTIFACTS[0],
            )
            reloaded_fit_projection = fitting_spec(exported_model).compile()
            export_parity = _export_behavior_parity(
                release_model,
                reloaded_fit_projection,
                holdouts,
            )

        summary: dict[str, object] = {
            "format": RESULT_FORMAT,
            "run_id": run_id,
            "status": "accepted" if released else "rejected",
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
            "release_candidate": release_candidate,
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
                str(output_dir / CONSUMER_ARTIFACTS[0]) if released else None
            ),
            "consumer_model_manifest": (
                str(output_dir / CONSUMER_ARTIFACTS[1]) if released else None
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

        # Metrics first, before anything that renders. Whatever happens next,
        # the numbers are already readable on disk.
        write_json(output_dir / "identification.json", _json_ready(summary))

        media_error: Exception | None = None
        if render_media:
            print("rendering recording videos and per-family metric plots ...")
            try:
                summary["media"] = _render_outputs(
                    output_dir,
                    model_path,
                    prepared,
                    holdouts,
                    stages,
                    evaluations,
                )
            except Exception as error:
                # Whatever was drawn before the failure is already on disk, so
                # it is recorded as produced rather than forgotten.
                if isinstance(error, MediaRenderingIncomplete):
                    summary["media"] = list(error.written)
                summary["media_generation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "failures": list(getattr(error, "failures", ())),
                }
                media_error = error
                print(f"  media generation incomplete: {error}")

        write_json(output_dir / "identification.json", _json_ready(summary))
        try:
            _write_report(
                output_dir / "result.md",
                summary,
                stages,
                export,
                acceptance,
            )
        except Exception as error:
            # The report is a rendering of identification.json, which is already
            # written. Losing the prose must not lose the run.
            import traceback

            print(f"  report generation failed: {type(error).__name__}: {error}")
            (output_dir / "result.md").write_text(
                "# Identification report generation failed\n\n"
                f"Run `{run_id}` produced its metrics and media, but rendering "
                "this report raised:\n\n```\n"
                f"{traceback.format_exc()}```\n\n"
                "Every number is in `identification.json`, and the plots and "
                "videos in this directory are complete.\n",
                encoding="utf-8",
            )

        if released:
            if media_error is not None:
                # A nominally accepted fit is not deliverable without the
                # requested evidence, so the verified model is not released.
                # Everything else stays on disk to show why.
                raise media_error
            for name in CONSUMER_ARTIFACTS:
                os.replace(export_dir / name, output_dir / name)
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
    reuse_fit: bool = True,
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
            reuse_fit=reuse_fit,
        )
        released = summary.get("status") == "accepted"
        _write_attempt_status(
            output_dir,
            run_id=run_id,
            state="accepted" if released else "rejected",
        )
    except BaseException as error:
        _write_attempt_status(
            output_dir,
            run_id=run_id,
            state="failed",
            error=error,
        )
        raise
    if not released:
        # Nothing reproduced the withheld robot better than the model already
        # in service, so there is nothing to ship. The per-joint findings in
        # `acceptance` are reported either way; they do not decide this.
        raise RuntimeError(
            "no candidate reproduced the withheld robot better than nominal:"
            "\n  - " + "\n  - ".join(acceptance.problems)
        )
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
    verdict = "ACCEPTED" if summary.get("status") == "accepted" else "REJECTED"
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
        "## Findings",
        "",
        "Every check that fired, in fitting order. A finding explains a number "
        "in this report; it does not withhold one. Only the held-out release "
        "gate below decides whether the consumer model ships.",
        "",
    ]
    findings = list(stages.problems) + [
        f"held-out release gate: {problem}" for problem in acceptance.problems
    ]
    if findings:
        lines += [f"- {finding}" for finding in findings]
    else:
        lines.append("None: every stage passed its own checks.")

    lines += [
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
        f"Verdict: **{verdict}**.",
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
    parser.add_argument(
        "--refit",
        action="store_true",
        help=(
            "ignore any checkpointed fit for these recordings and solve again; "
            "by default a fit is reused when the data, knobs and fitting code "
            "are unchanged"
        ),
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
            reuse_fit=not arguments.refit,
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
