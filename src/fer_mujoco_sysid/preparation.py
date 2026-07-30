"""Convert immutable recording bundles into stage-ready synchronized runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.dataset import (
    BOUND_RECORDING_FORMAT,
    health_report,
    load_recording,
    recording_analysis_mask,
    recording_backend,
    recording_protocol_family,
    recording_rollout_mask,
)
from fer_mujoco_sysid.fitting import MeasuredRun
from fer_mujoco_sysid.io import sha256_file
from fer_mujoco_sysid.preprocessing import FilterSettings, prepare
from fer_mujoco_sysid.protocol import FRICTION_FAMILY, INERTIAL_FAMILY
from fer_mujoco_sysid.selection import runs_from_mask
from fer_mujoco_sysid.stages import StageRecording

CLASSICAL_SAMPLE_RATE_HZ = 100.0


@dataclass(frozen=True)
class PreparedRecording:
    """A checksummed recording plus its synchronized fitting representation."""

    root: Path
    stage: StageRecording
    backend: str
    manifest: dict[str, object]
    health: dict[str, object]
    preprocessing: dict[str, object]
    control_channel: str
    content_sha256: str
    source_model_sha256: str
    protocol_content_sha256: str

    @property
    def label(self) -> str:
        return str(self.root)

    @property
    def protocol_id(self) -> str:
        return self.stage.protocol_id

    @property
    def family(self) -> str:
        return self.stage.family

    @property
    def role(self) -> str:
        return self.stage.role

    @property
    def is_holdout(self) -> bool:
        return self.role == "holdout"

    @property
    def run(self) -> MeasuredRun:
        return self.stage.run

    @property
    def ddq_rad_s2(self) -> NDArray[np.float64]:
        return self.stage.ddq_rad_s2

    def protocol_run(self) -> MeasuredRun:
        runs = runs_from_mask(self.run, self.stage.protocol_mask)
        if len(runs) != 1:
            raise RuntimeError(
                f"{self.protocol_id}: protocol mask produced {len(runs)} "
                "disjoint intervals"
            )
        return runs[0]

    def protocol_acceleration(self) -> NDArray[np.float64]:
        return self.ddq_rad_s2[self.stage.protocol_mask]

    def protocol_classical_torque(self) -> NDArray[np.float64]:
        return self.stage.classical_torque_Nm[self.stage.protocol_mask]


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _default_filter_settings(
    manifest: Mapping[str, object],
) -> FilterSettings:
    """Select the declared acquisition contract, never infer from backend."""
    simulation = manifest.get("simulation")
    if not isinstance(simulation, Mapping):
        return FilterSettings()
    mode = simulation.get("preprocessing_mode")
    if mode == "raw_engine_exact":
        return FilterSettings(enabled=False)
    if mode in (None, "hardware_like"):
        return FilterSettings()
    raise ValueError(f"unknown simulation preprocessing_mode {mode!r}")


def _linear_resample(
    source_time_s: NDArray[np.float64],
    values: NDArray[np.float64],
    target_time_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.column_stack(
        [
            np.interp(target_time_s, source_time_s, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _causal_resample(
    source_time_s: NDArray[np.float64],
    values: NDArray[np.float64],
    target_time_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Previous-sample hold for controls and discrete recording decisions."""
    indices = np.searchsorted(source_time_s, target_time_s, side="right") - 1
    indices = np.clip(indices, 0, len(source_time_s) - 1)
    return np.asarray(values)[indices]


def _causal_event_resample(
    source_time_s: NDArray[np.float64],
    values: NDArray[np.bool_],
    target_time_s: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Map sparse sample events without turning one 100 Hz row into a plateau."""
    indices = np.searchsorted(source_time_s, target_time_s, side="right") - 1
    indices = np.clip(indices, 0, len(source_time_s) - 1)
    selected = np.asarray(values, dtype=np.bool_)[indices].copy()
    selected[1:][indices[1:] == indices[:-1]] = False
    return selected


def _thin_event_mask(
    mask: NDArray[np.bool_],
    *,
    step_s: float,
    maximum_rate_hz: float = CLASSICAL_SAMPLE_RATE_HZ,
) -> NDArray[np.bool_]:
    """Keep at most one selected row per native-rate diagnostic interval."""
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be finite and positive")
    if not np.isfinite(maximum_rate_hz) or maximum_rate_hz <= 0.0:
        raise ValueError("maximum_rate_hz must be finite and positive")
    spacing = max(int(round(1.0 / (maximum_rate_hz * step_s))), 1)
    selected = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    kept = np.zeros_like(mask, dtype=np.bool_)
    last = -spacing
    for index in selected:
        if int(index) - last >= spacing:
            kept[index] = True
            last = int(index)
    return kept


def _protocol_role(manifest: Mapping[str, object]) -> str:
    protocol = _mapping(manifest.get("protocol"), "recording protocol")
    role = str(protocol.get("role", ""))
    if role not in ("train", "holdout"):
        raise ValueError(f"recording protocol role must be train/holdout, got {role!r}")
    return role


def _protocol_mask(
    arrays: Mapping[str, NDArray],
) -> NDArray[np.bool_]:
    if "protocol_segment_index" not in arrays:
        raise ValueError("recording has no protocol row binding; reconvert its raw bag")
    return np.asarray(arrays["protocol_segment_index"], dtype=np.int64) >= 0


def _control_channel(
    backend: str,
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray],
) -> tuple[str, NDArray[np.float64]]:
    if backend == "real":
        # Every backend fits the trajectory controller's commanded effort, and
        # the fitting model is gravity-free, because that pair *is* the FCI
        # convention: the effort a controller sends is the effort on top of the
        # robot's internal gravity compensation. Using a different channel on
        # hardware would identify a different quantity than the simulated runs
        # that validated this pipeline.
        #
        # tau_J_d and tau_J stay recorded as the cross-check: `health_report`
        # compares tau_cmd against tau_J_d, which is how a limiter clamp or a
        # gravity-composition surprise announces itself. Neither is a model
        # input — they are link-side, past the transmission, so joint friction
        # and rotor inertia do not appear in them the way MuJoCo's frictionloss
        # and armature do.
        for name in ("tau_J_d_Nm", "tau_J_Nm"):
            if name not in arrays:
                raise ValueError(
                    f"real recording has no causally aligned {name} telemetry; "
                    "the commanded effort cannot be cross-checked, so the run "
                    "is not fit-eligible"
                )
    return "tau_cmd_Nm", np.asarray(arrays["tau_cmd_Nm"], dtype=np.float64)


def prepare_recording(
    root: str | Path,
    model: mujoco.MjModel,
    *,
    settings: FilterSettings | None = None,
    expected_source_model: str | Path | None = None,
    require_usable: bool = True,
) -> PreparedRecording:
    """Load, lineage-check, resample and filter one protocol-bound recording."""
    root = Path(root)
    manifest, arrays = load_recording(root)
    if manifest.get("format") != BOUND_RECORDING_FORMAT:
        raise ValueError(
            f"{root}: recording predates protocol binding; reconvert the raw "
            "bag with the current converter"
        )
    backend = recording_backend(manifest)
    health = health_report(manifest, arrays)
    if require_usable and not bool(health["usable"]):
        problems = health.get("problems", [])
        detail = "; ".join(str(problem) for problem in problems)
        raise ValueError(f"{root}: recording health rejected: {detail}")
    family = recording_protocol_family(manifest)
    role = _protocol_role(manifest)
    protocol = _mapping(manifest.get("protocol"), "recording protocol")
    source_model = _mapping(manifest.get("source_model"), "source model")
    source_hash = str(source_model.get("sha256", ""))
    if expected_source_model is not None:
        expected_hash = sha256_file(expected_source_model)
        if source_hash != expected_hash:
            raise ValueError(
                f"{root}: recording source model {source_hash} does not match "
                f"the model being identified {expected_hash}"
            )

    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    step = float(model.opt.timestep)
    grid = np.arange(time_s[0], time_s[-1] + 0.5 * step, step)
    q = _linear_resample(time_s, np.asarray(arrays["q_rad"], dtype=np.float64), grid)
    dq = _linear_resample(
        time_s, np.asarray(arrays["dq_rad_s"], dtype=np.float64), grid
    )
    channel_name, source_control = _control_channel(backend, manifest, arrays)
    # An actuator command is piecewise constant until the next controller or
    # 100 Hz Franka sample. Linear interpolation would use a future torque and
    # manufacture an input the robot never received.
    control = _causal_resample(time_s, source_control, grid)

    settings = settings or _default_filter_settings(manifest)
    filtered = prepare(grid, q, dq, control, settings)
    measured = np.column_stack([filtered.q_rad, filtered.dq_rad_s])
    samples = len(grid)
    analysis = _causal_resample(
        time_s,
        recording_rollout_mask(
            manifest,
            arrays,
            require_telemetry=backend == "real",
        ),
        grid,
    ).astype(np.bool_)
    classical = _causal_event_resample(
        time_s,
        recording_analysis_mask(
            manifest,
            arrays,
            require_telemetry=backend == "real",
        ),
        grid,
    )
    # Keep uncertainty and HAC correlation duration comparable across
    # backends. Real telemetry is natively 100 Hz; simulation is deliberately
    # thinned to the same diagnostic rate rather than claiming extra evidence.
    classical = _thin_event_mask(classical, step_s=step)
    protocol_mask = _causal_resample(time_s, _protocol_mask(arrays), grid).astype(
        np.bool_
    )

    run = MeasuredRun(
        label=str(protocol["protocol_id"]),
        qpos0=filtered.q_rad[0],
        qvel0=filtered.dq_rad_s[0],
        control_times=np.arange(samples, dtype=np.float64) * step,
        control=control,
        measured_times=(np.arange(samples, dtype=np.float64) + 1.0) * step,
        measured=measured,
    )
    stage = StageRecording(
        label=str(root),
        protocol_id=str(protocol["protocol_id"]),
        family=family,
        role=role,
        run=run,
        ddq_rad_s2=filtered.ddq_rad_s2,
        classical_torque_Nm=filtered.tau_Nm,
        classical_mask=classical,
        analysis_mask=analysis,
        protocol_mask=protocol_mask,
    )
    stage.validate()
    preprocessing = dict(filtered.settings)
    preprocessing["classical_sample_rate_hz"] = CLASSICAL_SAMPLE_RATE_HZ
    return PreparedRecording(
        root=root,
        stage=stage,
        backend=backend,
        manifest=dict(manifest),
        health=health,
        preprocessing=preprocessing,
        control_channel=channel_name,
        content_sha256=str(manifest.get("content_sha256", "")),
        source_model_sha256=source_hash,
        protocol_content_sha256=str(protocol["content_sha256"]),
    )


def validate_campaign_lineage(
    recordings: list[PreparedRecording],
    *,
    require_holdouts: bool = True,
) -> None:
    """Reject mixed, duplicated, or scientifically incomplete campaigns."""
    if not recordings:
        raise ValueError("campaign contains no recordings")
    backends = {record.backend for record in recordings}
    if len(backends) != 1:
        raise ValueError(
            "recordings come from different backends: " + ", ".join(sorted(backends))
        )
    source_hashes = {record.source_model_sha256 for record in recordings}
    if len(source_hashes) != 1:
        raise ValueError("recordings were generated from different nominal models")
    labels = [record.protocol_id for record in recordings]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "campaign contains duplicate protocol recordings: " + ", ".join(duplicates)
        )
    fingerprints = [record.content_sha256 for record in recordings]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("campaign contains duplicate recording fingerprints")

    required = (FRICTION_FAMILY, INERTIAL_FAMILY)
    training_families = {
        record.family for record in recordings if record.role == "train"
    }
    missing_training = sorted(set(required) - training_families)
    if missing_training:
        raise ValueError(
            "complete identification requires training data for "
            + ", ".join(missing_training)
        )
    if require_holdouts:
        holdout_families = {
            record.family for record in recordings if record.role == "holdout"
        }
        missing_holdouts = sorted(set(required) - holdout_families)
        if missing_holdouts:
            raise ValueError(
                "complete identification requires held-out reproduction for "
                + ", ".join(missing_holdouts)
            )
