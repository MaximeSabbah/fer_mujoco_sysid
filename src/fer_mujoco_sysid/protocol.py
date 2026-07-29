"""The compiled-protocol bundle format: what it is, and how to read one.

Split out of :mod:`fer_mujoco_sysid.excitation` deliberately. *Writing* a
protocol needs MuJoCo — the manifest records the payload read off the model
and the motion is validated by simulating it. *Reading* one needs nothing but
NumPy, and the reader is what runs on the robot: the ROS player imports this
module and never pulls MuJoCo into a real-time-adjacent process. That also
means playback keeps working in a plain ROS environment, where the only
Python packages guaranteed present are the ones ROS itself ships.

A bundle is a directory holding ``protocol.json``, ``desired.npz`` and
``checksums.sha256``. It is immutable once written and identified by its
content SHA-256. Git history records changes without parallel ``rN``
directories.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.io import (
    ArtifactError,
    content_sha256,
    load_arrays,
    read_json,
    verify_checksums,
)

LEGACY_PROTOCOL_FORMAT = "fer-mujoco-sysid/motion-protocol@2"
PROTOCOL_FORMAT = "fer-mujoco-sysid/motion-protocol@3"
SUPPORTED_PROTOCOL_FORMATS = (LEGACY_PROTOCOL_FORMAT, PROTOCOL_FORMAT)

FRICTION_FAMILY = "fer-friction"
INERTIAL_FAMILY = "fer-inertial"
PROTOCOL_FAMILIES = (FRICTION_FAMILY, INERTIAL_FAMILY)

TRAIN_ROLE = "train"
HOLDOUT_ROLE = "holdout"
PROTOCOL_ROLES = (TRAIN_ROLE, HOLDOUT_ROLE)

FRICTION_CRUISE = "friction_cruise"
INERTIAL_EXCITATION = "inertial_excitation"
ANALYSIS_KINDS = (FRICTION_CRUISE, INERTIAL_EXCITATION)

# Zero-phase filtering leaks a transition into samples on both sides of it.
# Keeping this much recording-clock time away from an analysis-window boundary
# excludes the ramp/cruise corner at the default 8 Hz preprocessing cutoff.
# It is ten samples at the intentionally reliable 100 Hz robot telemetry rate.
DEFAULT_ANALYSIS_EDGE_GUARD_S = 0.1

#: Joint names as the ROS stack knows them, and the order every array column
#: and every trajectory point follows.
ROS_ARM_JOINT_NAMES = tuple(f"fer_joint{i}" for i in range(1, 8))

ARRAY_UNITS = {
    "time_s": "s",
    "q_rad": "rad",
    "dq_rad_s": "rad/s",
    "ddq_rad_s2": "rad/s^2",
}


@dataclass(frozen=True)
class AnalysisWindow:
    """A protocol interval that may enter one specific identification stage.

    Indices follow the protocol arrays and use the usual half-open convention.
    Friction windows carry the designed per-joint cruise velocity; inertial
    windows deliberately do not, because their velocity varies continuously.
    """

    window_id: str
    kind: str
    parent_segment_id: str
    start_index: int
    end_index_exclusive: int
    nominal_velocity_rad_s: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ProtocolAlignment:
    """Where a compiled protocol starts on a recording's clock."""

    protocol_id: str
    content_sha256: str
    start_time_s: float
    speed_scale: float
    normalized_rmse: float | None = None

    def validate_for(self, manifest: Mapping[str, object]) -> None:
        if self.protocol_id != str(manifest.get("protocol_id", "")):
            raise ArtifactError(
                f"alignment is for {self.protocol_id!r}, not "
                f"{manifest.get('protocol_id')!r}"
            )
        expected = str(manifest.get("content_sha256", ""))
        if self.content_sha256 != expected:
            raise ArtifactError(
                "alignment protocol fingerprint does not match the manifest"
            )
        if not np.isfinite(self.start_time_s):
            raise ArtifactError("alignment start_time_s must be finite")
        if not np.isfinite(self.speed_scale) or not 0.0 < self.speed_scale <= 1.0:
            raise ArtifactError(
                f"alignment speed_scale must lie in (0, 1], got {self.speed_scale}"
            )


@dataclass(frozen=True)
class MappedProtocolInterval:
    """One protocol segment or analysis window on the recording clock."""

    interval_id: str
    kind: str
    start_time_s: float
    end_time_s: float
    parent_segment_id: str | None = None
    analysis_eligible: bool = False
    nominal_velocity_rad_s: tuple[float, ...] | None = None

    def mask(self, recording_time_s: NDArray[np.float64]) -> NDArray[np.bool_]:
        times = np.asarray(recording_time_s, dtype=np.float64)
        return (times >= self.start_time_s) & (times <= self.end_time_s)


def protocol_family(manifest: Mapping[str, object]) -> str:
    """Return an explicitly declared family; never infer one from an id."""
    family = str(manifest.get("family", ""))
    if family not in PROTOCOL_FAMILIES:
        raise ArtifactError(
            f"protocol family must be one of {PROTOCOL_FAMILIES}, got {family!r}"
        )
    return family


def protocol_role(manifest: Mapping[str, object]) -> str:
    """Return an explicitly declared train/holdout role."""
    role = str(manifest.get("role", ""))
    if role not in PROTOCOL_ROLES:
        raise ArtifactError(
            f"protocol role must be one of {PROTOCOL_ROLES}, got {role!r}"
        )
    return role


def _validate_segments(
    manifest: Mapping[str, object], samples: int
) -> dict[str, Mapping[str, object]]:
    raw_segments = manifest.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ArtifactError("protocol declares no segments")

    segments: dict[str, Mapping[str, object]] = {}
    cursor = 0
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise ArtifactError("every protocol segment must be an object")
        segment_id = str(raw.get("segment_id", ""))
        if not segment_id or segment_id in segments:
            raise ArtifactError(f"invalid or duplicate segment id {segment_id!r}")
        start = int(raw.get("start_index", -1))
        stop = int(raw.get("end_index_exclusive", -1))
        if start != cursor:
            raise ArtifactError(f"segment {segment_id} is not contiguous")
        if stop <= start or stop > samples:
            raise ArtifactError(
                f"segment {segment_id} has invalid bounds [{start}, {stop})"
            )
        if not isinstance(raw.get("analysis_eligible"), bool):
            raise ArtifactError(
                f"segment {segment_id} must declare boolean analysis_eligible"
            )
        segments[segment_id] = raw
        cursor = stop
    if cursor != samples:
        raise ArtifactError("segments do not cover every sample")
    return segments


def _validate_analysis_windows(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    segments: Mapping[str, Mapping[str, object]],
) -> None:
    family = protocol_family(manifest)
    role = protocol_role(manifest)
    del role  # Its presence and vocabulary are the contract.

    raw_windows = manifest.get("analysis_windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ArtifactError("protocol declares no analysis_windows")

    expected_kind = {
        FRICTION_FAMILY: FRICTION_CRUISE,
        INERTIAL_FAMILY: INERTIAL_EXCITATION,
    }[family]
    samples = len(arrays["time_s"])
    previous_stop = -1
    ids: set[str] = set()
    for raw in raw_windows:
        if not isinstance(raw, Mapping):
            raise ArtifactError("every analysis window must be an object")
        window_id = str(raw.get("window_id", ""))
        if not window_id or window_id in ids:
            raise ArtifactError(f"invalid or duplicate analysis window {window_id!r}")
        ids.add(window_id)
        kind = str(raw.get("kind", ""))
        if kind != expected_kind:
            raise ArtifactError(
                f"{family} window {window_id} has kind {kind!r}, "
                f"expected {expected_kind!r}"
            )
        parent_id = str(raw.get("parent_segment_id", ""))
        parent = segments.get(parent_id)
        if parent is None:
            raise ArtifactError(
                f"analysis window {window_id} names unknown segment {parent_id!r}"
            )
        if not bool(parent["analysis_eligible"]):
            raise ArtifactError(
                f"analysis window {window_id} belongs to ineligible segment {parent_id}"
            )

        start = int(raw.get("start_index", -1))
        stop = int(raw.get("end_index_exclusive", -1))
        parent_start = int(parent["start_index"])
        parent_stop = int(parent["end_index_exclusive"])
        if not parent_start <= start < stop <= parent_stop or stop > samples:
            raise ArtifactError(
                f"analysis window {window_id} has invalid bounds [{start}, {stop})"
            )
        if stop - start < 2:
            raise ArtifactError(
                f"analysis window {window_id} must contain at least two samples"
            )
        if start < previous_stop:
            raise ArtifactError("analysis windows overlap or are not time ordered")
        previous_stop = stop

        velocity = raw.get("nominal_velocity_rad_s")
        if kind == FRICTION_CRUISE:
            vector = np.asarray(velocity, dtype=np.float64)
            if (
                vector.shape != (7,)
                or not np.all(np.isfinite(vector))
                or np.any(vector == 0.0)
            ):
                raise ArtifactError(
                    f"friction window {window_id} needs 7 finite, nonzero "
                    "nominal velocities"
                )
            actual = np.asarray(arrays["dq_rad_s"])[start:stop]
            if not np.allclose(
                actual,
                vector[None, :],
                rtol=0.0,
                atol=1e-10,
            ):
                raise ArtifactError(
                    f"friction window {window_id} is not a true "
                    "constant-velocity plateau"
                )
            acceleration = np.asarray(arrays["ddq_rad_s2"])[start:stop]
            if not np.allclose(acceleration, 0.0, rtol=0.0, atol=1e-10):
                raise ArtifactError(
                    f"friction window {window_id} contains acceleration"
                )
        elif velocity is not None:
            raise ArtifactError(
                f"inertial window {window_id} must not declare a fixed velocity"
            )


def validate_protocol_manifest(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.float64]]
) -> None:
    """Check a protocol manifest against its arrays (structure and meaning)."""
    artifact_format = manifest.get("format")
    if artifact_format not in SUPPORTED_PROTOCOL_FORMATS:
        raise ArtifactError(f"unsupported protocol format: {manifest.get('format')!r}")
    missing = set(ARRAY_UNITS) - set(arrays)
    if missing:
        raise ArtifactError(f"missing arrays: {sorted(missing)}")

    time_s = np.asarray(arrays["time_s"])
    samples = len(time_s)
    if (
        samples < 2
        or not np.all(np.isfinite(time_s))
        or not np.all(np.diff(time_s) > 0)
    ):
        raise ArtifactError("time_s must be strictly increasing with >= 2 samples")
    for key in ("q_rad", "dq_rad_s", "ddq_rad_s2"):
        array = np.asarray(arrays[key])
        if array.shape != (samples, 7):
            raise ArtifactError(
                f"{key} has shape {array.shape}, expected {(samples, 7)}"
            )
        if not np.all(np.isfinite(array)):
            raise ArtifactError(f"{key} holds non-finite samples")

    playback = manifest.get("playback", {})
    if not isinstance(playback, Mapping):
        raise ArtifactError("protocol playback section must be an object")
    if list(playback.get("joint_order", [])) != list(ROS_ARM_JOINT_NAMES):
        raise ArtifactError("joint_order does not match the canonical FER order")
    if int(playback.get("samples", -1)) != samples:
        raise ArtifactError("playback.samples disagrees with the arrays")

    segments = _validate_segments(manifest, samples)
    if artifact_format == PROTOCOL_FORMAT:
        _validate_analysis_windows(manifest, arrays, segments)


def protocol_analysis_windows(
    manifest: Mapping[str, object],
) -> tuple[AnalysisWindow, ...]:
    """Read validated v3 analysis windows from a protocol manifest."""
    if manifest.get("format") != PROTOCOL_FORMAT:
        raise ArtifactError(
            "legacy protocol has no explicit analysis windows; use a current "
            "canonical protocol bundle"
        )
    protocol_family(manifest)
    protocol_role(manifest)
    windows = []
    for raw in manifest.get("analysis_windows", []):
        velocity = raw.get("nominal_velocity_rad_s")
        windows.append(
            AnalysisWindow(
                window_id=str(raw["window_id"]),
                kind=str(raw["kind"]),
                parent_segment_id=str(raw["parent_segment_id"]),
                start_index=int(raw["start_index"]),
                end_index_exclusive=int(raw["end_index_exclusive"]),
                nominal_velocity_rad_s=(
                    tuple(float(value) for value in velocity)
                    if velocity is not None
                    else None
                ),
            )
        )
    return tuple(windows)


def alignment_from_start_time(
    manifest: Mapping[str, object],
    *,
    start_time_s: float,
    speed_scale: float = 1.0,
) -> ProtocolAlignment:
    """Build a checked alignment when acquisition recorded the action start."""
    protocol_family(manifest)
    protocol_role(manifest)
    alignment = ProtocolAlignment(
        protocol_id=str(manifest.get("protocol_id", "")),
        content_sha256=str(manifest.get("content_sha256", "")),
        start_time_s=float(start_time_s),
        speed_scale=float(speed_scale),
    )
    alignment.validate_for(manifest)
    return alignment


def align_protocol_reference(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    recording_time_s: NDArray[np.float64],
    q_reference_rad: NDArray[np.float64],
    *,
    speed_scale: float = 1.0,
    probe_samples: int = 512,
    maximum_normalized_rmse: float = 0.02,
) -> ProtocolAlignment:
    """Locate protocol start by matching its path to the recorded reference.

    The trajectory controller reference is used rather than measured position:
    tracking error is physics, while this function answers only *when the
    reviewed protocol began*. Candidate starts lie on the recording clock and
    the full path is probed, so the mandatory move-to-start cannot masquerade
    as part of the protocol.
    """
    validate_protocol_manifest(manifest, arrays)
    if manifest.get("format") != PROTOCOL_FORMAT:
        raise ArtifactError("reference alignment requires a current protocol")
    if not 0.0 < speed_scale <= 1.0:
        raise ArtifactError(f"speed_scale must lie in (0, 1], got {speed_scale}")
    if probe_samples < 16:
        raise ArtifactError("probe_samples must be at least 16")
    if not np.isfinite(maximum_normalized_rmse) or maximum_normalized_rmse <= 0.0:
        raise ArtifactError("maximum_normalized_rmse must be positive and finite")

    recording_times = np.asarray(recording_time_s, dtype=np.float64)
    reference = np.asarray(q_reference_rad, dtype=np.float64)
    if (
        recording_times.ndim != 1
        or len(recording_times) < 2
        or not np.all(np.isfinite(recording_times))
        or not np.all(np.diff(recording_times) > 0.0)
    ):
        raise ArtifactError("recording_time_s must be strictly increasing")
    if reference.shape != (len(recording_times), 7):
        raise ArtifactError(
            f"q_reference_rad has shape {reference.shape}, expected "
            f"{(len(recording_times), 7)}"
        )
    if not np.all(np.isfinite(reference)):
        raise ArtifactError("q_reference_rad holds non-finite samples")

    protocol_times = np.asarray(arrays["time_s"], dtype=np.float64)
    protocol_q = np.asarray(arrays["q_rad"], dtype=np.float64)
    played_duration = float(protocol_times[-1] / speed_scale)
    latest_start = float(recording_times[-1] - played_duration)
    candidates = recording_times[recording_times <= latest_start]
    if not len(candidates):
        raise ArtifactError(
            "recording is shorter than the time-scaled protocol; it cannot be aligned"
        )

    count = min(int(probe_samples), len(protocol_times))
    probe_indices = np.unique(
        np.linspace(0, len(protocol_times) - 1, count, dtype=np.intp)
    )
    probe_times = protocol_times[probe_indices] / speed_scale
    probe_q = protocol_q[probe_indices]
    scale = np.maximum(np.ptp(protocol_q, axis=0), 0.05)

    errors = np.empty(len(candidates), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        sampled = np.column_stack(
            [
                np.interp(candidate + probe_times, recording_times, reference[:, joint])
                for joint in range(7)
            ]
        )
        errors[index] = np.sqrt(np.mean(((sampled - probe_q) / scale) ** 2))

    best = int(np.argmin(errors))
    error = float(errors[best])
    if error > maximum_normalized_rmse:
        raise ArtifactError(
            "recorded controller reference does not match the reviewed protocol "
            f"(best normalized RMSE {error:.4f} > {maximum_normalized_rmse:.4f})"
        )
    alignment = ProtocolAlignment(
        protocol_id=str(manifest["protocol_id"]),
        content_sha256=str(manifest["content_sha256"]),
        start_time_s=float(candidates[best]),
        speed_scale=float(speed_scale),
        normalized_rmse=error,
    )
    alignment.validate_for(manifest)
    return alignment


def _recording_interval(
    protocol_times: NDArray[np.float64],
    start_index: int,
    end_index_exclusive: int,
    alignment: ProtocolAlignment,
    edge_guard_s: float,
) -> tuple[float, float]:
    if not np.isfinite(edge_guard_s) or edge_guard_s < 0.0:
        raise ArtifactError("edge_guard_s must be finite and non-negative")
    start = (
        alignment.start_time_s
        + float(protocol_times[start_index]) / alignment.speed_scale
        + edge_guard_s
    )
    stop = (
        alignment.start_time_s
        + float(protocol_times[end_index_exclusive - 1]) / alignment.speed_scale
        - edge_guard_s
    )
    if stop <= start:
        raise ArtifactError(
            f"edge guard {edge_guard_s:g} s removes an entire protocol interval"
        )
    return start, stop


def _recording_segment_interval(
    protocol_times: NDArray[np.float64],
    start_index: int,
    end_index_exclusive: int,
    alignment: ProtocolAlignment,
) -> tuple[float, float]:
    """Continuous segment bounds, including interpolation between knots."""
    start = (
        alignment.start_time_s
        + float(protocol_times[start_index]) / alignment.speed_scale
    )
    stop_index = min(end_index_exclusive, len(protocol_times) - 1)
    stop = (
        alignment.start_time_s
        + float(protocol_times[stop_index]) / alignment.speed_scale
    )
    if stop <= start:
        raise ArtifactError("protocol segment has an empty recording interval")
    return start, stop


def map_protocol_segments(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    alignment: ProtocolAlignment,
) -> tuple[MappedProtocolInterval, ...]:
    """Map every declared protocol segment onto the recording clock."""
    validate_protocol_manifest(manifest, arrays)
    alignment.validate_for(manifest)
    times = np.asarray(arrays["time_s"], dtype=np.float64)
    mapped = []
    for segment in manifest["segments"]:
        start, stop = _recording_segment_interval(
            times,
            int(segment["start_index"]),
            int(segment["end_index_exclusive"]),
            alignment,
        )
        mapped.append(
            MappedProtocolInterval(
                interval_id=str(segment["segment_id"]),
                kind=str(segment["kind"]),
                start_time_s=start,
                end_time_s=stop,
                analysis_eligible=bool(segment["analysis_eligible"]),
            )
        )
    return tuple(mapped)


def map_analysis_windows(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    alignment: ProtocolAlignment,
    *,
    kind: str | None = None,
    edge_guard_s: float = DEFAULT_ANALYSIS_EDGE_GUARD_S,
) -> tuple[MappedProtocolInterval, ...]:
    """Map selected analysis windows with a transient-safe default guard."""
    validate_protocol_manifest(manifest, arrays)
    alignment.validate_for(manifest)
    if kind is not None and kind not in ANALYSIS_KINDS:
        raise ArtifactError(f"unknown analysis kind {kind!r}")
    times = np.asarray(arrays["time_s"], dtype=np.float64)
    mapped = []
    for window in protocol_analysis_windows(manifest):
        if kind is not None and window.kind != kind:
            continue
        start, stop = _recording_interval(
            times,
            window.start_index,
            window.end_index_exclusive,
            alignment,
            edge_guard_s,
        )
        mapped.append(
            MappedProtocolInterval(
                interval_id=window.window_id,
                kind=window.kind,
                start_time_s=start,
                end_time_s=stop,
                parent_segment_id=window.parent_segment_id,
                analysis_eligible=True,
                nominal_velocity_rad_s=(
                    tuple(
                        alignment.speed_scale * value
                        for value in window.nominal_velocity_rad_s
                    )
                    if window.nominal_velocity_rad_s is not None
                    else None
                ),
            )
        )
    return tuple(mapped)


def analysis_masks_by_window(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    recording_time_s: NDArray[np.float64],
    alignment: ProtocolAlignment,
    *,
    kind: str,
    edge_guard_s: float = DEFAULT_ANALYSIS_EDGE_GUARD_S,
) -> dict[str, NDArray[np.bool_]]:
    """One recording-row mask per protocol analysis window."""
    recording_times = np.asarray(recording_time_s, dtype=np.float64)
    if (
        recording_times.ndim != 1
        or len(recording_times) < 2
        or not np.all(np.isfinite(recording_times))
        or not np.all(np.diff(recording_times) > 0.0)
    ):
        raise ArtifactError("recording_time_s must be strictly increasing")
    return {
        window.interval_id: window.mask(recording_times)
        for window in map_analysis_windows(
            manifest,
            arrays,
            alignment,
            kind=kind,
            edge_guard_s=edge_guard_s,
        )
    }


def _combined_analysis_mask(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    recording_time_s: NDArray[np.float64],
    alignment: ProtocolAlignment,
    *,
    family: str,
    kind: str,
    edge_guard_s: float,
) -> NDArray[np.bool_]:
    actual_family = protocol_family(manifest)
    if actual_family != family:
        raise ArtifactError(
            f"{kind} selection requires family {family!r}, got {actual_family!r}"
        )
    masks = analysis_masks_by_window(
        manifest,
        arrays,
        recording_time_s,
        alignment,
        kind=kind,
        edge_guard_s=edge_guard_s,
    )
    if not masks:
        raise ArtifactError(f"protocol has no {kind} windows")
    return np.logical_or.reduce(tuple(masks.values()))


def friction_cruise_mask(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    recording_time_s: NDArray[np.float64],
    alignment: ProtocolAlignment,
    *,
    edge_guard_s: float = DEFAULT_ANALYSIS_EDGE_GUARD_S,
) -> NDArray[np.bool_]:
    """Rows strictly inside true constant-velocity friction plateaus."""
    return _combined_analysis_mask(
        manifest,
        arrays,
        recording_time_s,
        alignment,
        family=FRICTION_FAMILY,
        kind=FRICTION_CRUISE,
        edge_guard_s=edge_guard_s,
    )


def inertial_excitation_mask(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    recording_time_s: NDArray[np.float64],
    alignment: ProtocolAlignment,
    *,
    edge_guard_s: float = DEFAULT_ANALYSIS_EDGE_GUARD_S,
) -> NDArray[np.bool_]:
    """Rows inside the Fourier excitation, excluding settle and edge leakage."""
    return _combined_analysis_mask(
        manifest,
        arrays,
        recording_time_s,
        alignment,
        family=INERTIAL_FAMILY,
        kind=INERTIAL_EXCITATION,
        edge_guard_s=edge_guard_s,
    )


def load_protocol_bundle(
    root: str | Path,
) -> tuple[dict[str, object], dict[str, NDArray[np.float64]]]:
    """Load a committed bundle, verifying checksums and content fingerprint.

    Three independent checks, all fail-closed: the files match
    ``checksums.sha256``, the manifest agrees with the arrays, and the content
    fingerprint recomputes. What reaches the robot is then provably the
    protocol that was reviewed.
    """
    root = Path(root)
    verify_checksums(root)
    manifest = read_json(root / "protocol.json")
    arrays = load_arrays(root / "desired.npz")
    validate_protocol_manifest(manifest, arrays)
    recomputed = content_sha256(manifest, arrays)
    if recomputed != manifest["content_sha256"]:
        raise ArtifactError(
            f"{root}: content fingerprint mismatch "
            f"({recomputed[:16]}... != {str(manifest['content_sha256'])[:16]}...)"
        )
    return manifest, arrays
