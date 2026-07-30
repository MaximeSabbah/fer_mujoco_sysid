"""The immutable, ROS-free representation of one measured protocol run.

A recording bundle contains ``recording.json``, ``measured.npz`` and
``checksums.sha256``.  The raw bag remains next to it.  Version 2 binds every
row to the reviewed protocol interval and keeps the Franka broadcaster's
native 100 Hz samples as well as a causal alignment onto the controller clock.

Torque channel names are deliberately literal.  ``tau_cmd_Nm`` is the
trajectory controller's output request; it is not asserted to be the
post-limiter or physically applied torque.  Real recordings retain
``tau_J_d`` and ``tau_J`` so identification can choose a channel whose
semantics are appropriate instead of silently treating them as equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.io import (
    ArtifactError,
    content_sha256,
    load_arrays,
    read_json,
    save_arrays,
    verify_checksums,
    write_checksums,
    write_json,
)
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

LEGACY_RECORDING_FORMAT = "fer-mujoco-sysid/recording@1"
# Keep this public name on @1 so existing standalone simulation fixtures and
# previously generated recordings remain readable.  ROS conversion opts in to
# the stricter, protocol-bound format explicitly.
RECORDING_FORMAT = LEGACY_RECORDING_FORMAT
BOUND_RECORDING_FORMAT = "fer-mujoco-sysid/recording@2"
SUPPORTED_RECORDING_FORMATS = (LEGACY_RECORDING_FORMAT, BOUND_RECORDING_FORMAT)

# Where a recording came from.  These paths must not be fitted together.
BACKENDS = ("mujoco", "mujoco_ros", "real")
PROTOCOL_FAMILIES = ("fer-friction", "fer-inertial")
PROTOCOL_ROLES = ("train", "holdout")
OUTPUT_DIRECTORY = "output"


class RecordingError(ArtifactError):
    """A recording is malformed, incomplete, or unusable for fitting."""


def output_root(repository: str | Path, backend: str) -> Path:
    """Return ``<repository>/output/<backend>``, creating it if needed."""
    if backend not in BACKENDS:
        raise RecordingError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    root = Path(repository) / OUTPUT_DIRECTORY / backend
    root.mkdir(parents=True, exist_ok=True)
    return root


def recording_backend(manifest: Mapping[str, object]) -> str:
    """Return the declared recording provenance."""
    backend = str(manifest.get("backend", ""))
    if backend not in BACKENDS:
        raise RecordingError(
            f"recording declares backend {backend!r}; expected one of {BACKENDS}"
        )
    return backend


def recording_protocol_family(manifest: Mapping[str, object]) -> str:
    """Return the explicit protocol family; never infer it from an id."""
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RecordingError("recording has no protocol object")
    family = str(protocol.get("family", ""))
    if family not in PROTOCOL_FAMILIES:
        raise RecordingError(
            f"recording protocol family must be one of {PROTOCOL_FAMILIES}, "
            f"got {family!r}"
        )
    return family


# Arrays on the controller-state clock.
REQUIRED_ARRAYS = {
    "time_s": "s",
    "q_rad": "rad",
    "dq_rad_s": "rad/s",
    "tau_cmd_Nm": "Nm",
}
ALIGNED_JOINT_ARRAYS = {
    "q_desired_rad": "rad",
    "tau_J_Nm": "Nm",
    "tau_J_d_Nm": "Nm",
    "theta_rad": "rad",
    "dtheta_rad_s": "rad/s",
}
# Retained as a compatibility name for callers that only know @1.
OPTIONAL_ARRAYS = ALIGNED_JOINT_ARRAYS

# One scalar per controller row.
ROW_ARRAYS = {
    "protocol_segment_index": "index",
    "analysis_eligible": "bool",
    "franka_sample_age_s": "s",
    "franka_new_sample": "bool",
    "telemetry_transient_free": "bool",
    "telemetry_fit_eligible": "bool",
}

# Native Franka broadcaster clock and channels.  M is intentionally unrelated
# to the N controller rows: the broadcaster is kept at its proven 100 Hz rate.
NATIVE_TIME_ARRAYS = {"franka_time_ns": "ns"}
NATIVE_JOINT_ARRAYS = {
    "tau_J_native_Nm": "Nm",
    "tau_J_d_native_Nm": "Nm",
    "theta_native_rad": "rad",
    "dtheta_native_rad_s": "rad/s",
}
ARRAY_UNITS = {
    **REQUIRED_ARRAYS,
    **ALIGNED_JOINT_ARRAYS,
    **ROW_ARRAYS,
    **NATIVE_TIME_ARRAYS,
    **NATIVE_JOINT_ARRAYS,
}

# Health limits for controller-state acquisition.
GAP_THRESHOLD_FRACTION = 1.5
MAX_GAP_S = 0.05
MISSING_FRACTION_LIMIT = 0.01

# A simulated plant executes every control cycle, so a missed one means the
# acquisition is wrong and MISSING_FRACTION_LIMIT applies. Real hardware on a
# machine without an RT kernel takes occasional late cycles — the first FER
# campaign held 938 Hz against a 1 kHz loop, 1.4% of cycles late — and that
# loses no information: every recorded row is still a true (state, commanded
# torque, timestamp) triple. What the fit actually depends on is reconstruction
# on the model's 2 ms grid, so for hardware the assertions are that no gap
# spans more than a few grid steps (measured worst case 7.6 ms) and that the
# achieved rate stays far above the grid rate. That is stricter than the 50 ms
# interpolation limit it replaces, not looser.
REAL_MAX_GAP_S = 0.010
REAL_MINIMUM_RATE_HZ = 250.0
COVERAGE_MINIMUM = 0.98
BOUND_COVERAGE_MAXIMUM = 1.02


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecordingError(f"{description} must be an object")
    return value


def _validate_intervals(value: object, description: str) -> None:
    if not isinstance(value, list) or not value:
        raise RecordingError(f"{description} must be a non-empty list")
    previous_start = -np.inf
    for interval in value:
        item = _mapping(interval, f"each {description} entry")
        if not str(item.get("interval_id", "")):
            raise RecordingError(f"each {description} entry needs interval_id")
        start = float(item.get("start_time_s", np.nan))
        end = float(item.get("end_time_s", np.nan))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise RecordingError(
                f"{description} interval {item.get('interval_id')!r} has invalid bounds"
            )
        if start < previous_start:
            raise RecordingError(f"{description} are not time ordered")
        previous_start = start


def _validate_bound_manifest(manifest: Mapping[str, object]) -> None:
    if list(manifest.get("joint_order", [])) != list(ROS_ARM_JOINT_NAMES):
        raise RecordingError("joint_order does not match the canonical FER order")

    protocol = _mapping(manifest.get("protocol"), "protocol")
    required = {
        "protocol_id",
        "family",
        "role",
        "content_sha256",
        "duration_s",
        "segments",
        "analysis_windows",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise RecordingError(f"bound recording protocol is missing {missing}")
    recording_protocol_family(manifest)
    if str(protocol.get("role")) not in PROTOCOL_ROLES:
        raise RecordingError(f"recording protocol role must be one of {PROTOCOL_ROLES}")
    if not _is_sha256(protocol.get("content_sha256")):
        raise RecordingError("recording protocol content_sha256 is malformed")
    if float(protocol.get("duration_s", 0.0)) <= 0.0:
        raise RecordingError("recording protocol duration_s must be positive")
    if not isinstance(protocol.get("segments"), list) or not protocol["segments"]:
        raise RecordingError("recording protocol must preserve its segments")
    if (
        not isinstance(protocol.get("analysis_windows"), list)
        or not protocol["analysis_windows"]
    ):
        raise RecordingError("recording protocol must preserve its analysis_windows")

    source = _mapping(manifest.get("source_model"), "source_model")
    for key in ("path", "revision", "sha256"):
        if not str(source.get(key, "")):
            raise RecordingError(f"source_model must preserve {key}")
    if not _is_sha256(source.get("sha256")):
        raise RecordingError("source_model sha256 is malformed")

    timing = _mapping(manifest.get("protocol_timing"), "protocol_timing")
    for key in (
        "start_stamp_ns",
        "end_stamp_ns",
        "selected_first_stamp_ns",
        "selected_last_stamp_ns",
    ):
        value = timing.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RecordingError(f"protocol_timing.{key} must be an integer")
    if int(timing["start_stamp_ns"]) >= int(timing["end_stamp_ns"]):
        raise RecordingError("protocol start marker is not before its end marker")
    if not (
        int(timing["start_stamp_ns"])
        <= int(timing["selected_first_stamp_ns"])
        <= int(timing["selected_last_stamp_ns"])
        <= int(timing["end_stamp_ns"])
    ):
        raise RecordingError("selected controller rows lie outside protocol markers")
    speed_scale = float(timing.get("speed_scale", np.nan))
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise RecordingError("protocol_timing.speed_scale must lie in (0, 1]")
    if not str(timing.get("marker_topic", "")):
        raise RecordingError("protocol_timing must preserve marker_topic")

    _validate_intervals(manifest.get("mapped_segments"), "mapped_segments")
    _validate_intervals(
        manifest.get("mapped_analysis_windows"), "mapped_analysis_windows"
    )

    torque_channels = _mapping(manifest.get("torque_channels"), "torque_channels")
    command = _mapping(torque_channels.get("tau_cmd_Nm"), "tau_cmd_Nm semantics")
    if command.get("application_status") != "not_asserted":
        raise RecordingError(
            "tau_cmd_Nm must explicitly say that applied/post-limiter status "
            "is not asserted"
        )

    if recording_backend(manifest) == "real":
        telemetry = _mapping(manifest.get("telemetry_alignment"), "telemetry_alignment")
        if telemetry.get("method") != "causal_previous_sample_hold":
            raise RecordingError(
                "real telemetry must use causal previous-sample alignment"
            )
        if int(telemetry.get("expected_rate_hz", 0)) != 100:
            raise RecordingError(
                "real telemetry expected_rate_hz must preserve the proven 100 Hz"
            )
        maximum_age = float(telemetry.get("maximum_allowed_age_s", np.nan))
        if not np.isfinite(maximum_age) or maximum_age <= 0.0:
            raise RecordingError(
                "real telemetry must declare a positive maximum_allowed_age_s"
            )


def validate_recording(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[Any]]
) -> None:
    """Check that recording arrays and their lineage are structurally sound."""
    artifact_format = manifest.get("format")
    if artifact_format not in SUPPORTED_RECORDING_FORMATS:
        raise RecordingError(f"unsupported recording format: {artifact_format!r}")

    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise RecordingError(f"recording is missing {missing}")

    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) < 2 or not np.all(np.isfinite(time_s)):
        raise RecordingError("time_s must be a finite 1-D vector with two samples")
    if not np.all(np.diff(time_s) > 0.0):
        raise RecordingError("time_s must be strictly increasing")

    joints = len(manifest.get("joint_order", []))
    if joints != 7:
        raise RecordingError(f"expected 7 joints, manifest declares {joints}")
    for name in (*REQUIRED_ARRAYS, *ALIGNED_JOINT_ARRAYS):
        if name == "time_s" or name not in arrays:
            continue
        array = np.asarray(arrays[name])
        if array.shape != (len(time_s), joints):
            raise RecordingError(
                f"{name} has shape {array.shape}, expected {(len(time_s), joints)}"
            )
        if not np.all(np.isfinite(array)):
            raise RecordingError(f"{name} holds non-finite samples")

    for name in ROW_ARRAYS:
        if name not in arrays:
            continue
        array = np.asarray(arrays[name])
        if array.shape != (len(time_s),):
            raise RecordingError(
                f"{name} has shape {array.shape}, expected {(len(time_s),)}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise RecordingError(f"{name} holds non-finite samples")

    if "protocol_segment_index" in arrays:
        segment_index = np.asarray(arrays["protocol_segment_index"])
        if segment_index.dtype.kind not in "iu":
            raise RecordingError("protocol_segment_index must be an integer array")
    for name in (
        "analysis_eligible",
        "franka_new_sample",
        "telemetry_transient_free",
        "telemetry_fit_eligible",
    ):
        if name in arrays and np.asarray(arrays[name]).dtype.kind != "b":
            raise RecordingError(f"{name} must be a boolean array")
    if "franka_sample_age_s" in arrays:
        age = np.asarray(arrays["franka_sample_age_s"], dtype=np.float64)
        if np.any(age < 0.0):
            raise RecordingError("franka_sample_age_s cannot be negative")

    native_count: int | None = None
    if "franka_time_ns" in arrays:
        native_time = np.asarray(arrays["franka_time_ns"])
        if native_time.ndim != 1 or len(native_time) < 2:
            raise RecordingError("franka_time_ns must hold at least two samples")
        if native_time.dtype.kind not in "iu":
            raise RecordingError("franka_time_ns must preserve integer nanoseconds")
        if not np.all(np.diff(native_time.astype(np.int64)) > 0):
            raise RecordingError("franka_time_ns must be strictly increasing")
        native_count = len(native_time)
    for name in NATIVE_JOINT_ARRAYS:
        if name not in arrays:
            continue
        if native_count is None:
            raise RecordingError(f"{name} exists without franka_time_ns")
        array = np.asarray(arrays[name])
        if array.shape != (native_count, joints):
            raise RecordingError(
                f"{name} has shape {array.shape}, expected {(native_count, joints)}"
            )
        if not np.all(np.isfinite(array)):
            raise RecordingError(f"{name} holds non-finite samples")

    if artifact_format == BOUND_RECORDING_FORMAT:
        _validate_bound_manifest(manifest)
        for name in ("protocol_segment_index", "analysis_eligible"):
            if name not in arrays:
                raise RecordingError(f"bound recording is missing {name}")
        segment_count = len(_mapping(manifest["protocol"], "protocol")["segments"])
        indices = np.asarray(arrays["protocol_segment_index"], dtype=np.int64)
        if np.any(indices < -1) or np.any(indices >= segment_count):
            raise RecordingError("protocol_segment_index contains an invalid index")

        if recording_backend(manifest) == "real":
            required_real = {
                "tau_J_Nm",
                "tau_J_d_Nm",
                "theta_rad",
                "dtheta_rad_s",
                "franka_sample_age_s",
                "franka_new_sample",
                "telemetry_transient_free",
                "telemetry_fit_eligible",
                "franka_time_ns",
                *NATIVE_JOINT_ARRAYS,
            }
            missing_real = sorted(required_real - set(arrays))
            if missing_real:
                raise RecordingError(
                    f"real bound recording is missing telemetry {missing_real}"
                )
            telemetry = _mapping(
                manifest.get("telemetry_alignment"), "telemetry_alignment"
            )
            age_limit = float(telemetry.get("maximum_allowed_age_s", np.nan))
            allowed = (
                np.asarray(arrays["analysis_eligible"], dtype=np.bool_)
                & np.asarray(arrays["telemetry_transient_free"], dtype=np.bool_)
                & np.asarray(arrays["franka_new_sample"], dtype=np.bool_)
                & (
                    np.asarray(arrays["franka_sample_age_s"], dtype=np.float64)
                    <= age_limit
                )
            )
            selected = np.asarray(arrays["telemetry_fit_eligible"], dtype=np.bool_)
            if np.any(selected & ~allowed):
                raise RecordingError(
                    "telemetry_fit_eligible includes a stale, transient, "
                    "non-analysis, or repeated telemetry row"
                )


def recording_analysis_mask(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
    *,
    require_telemetry: bool = False,
) -> NDArray[np.bool_]:
    """Return protocol analysis rows, sparse at native telemetry events on real."""
    validate_recording(manifest, arrays)
    if "analysis_eligible" not in arrays:
        # Legacy recordings predate explicit interval binding.
        raise RecordingError(
            "recording has no analysis_eligible mask; reconvert its raw bag"
        )
    mask = np.asarray(arrays["analysis_eligible"], dtype=np.bool_).copy()
    if require_telemetry:
        if "telemetry_fit_eligible" not in arrays:
            raise RecordingError(
                "recording has no telemetry_fit_eligible mask; real telemetry "
                "cannot be selected safely"
            )
        mask &= np.asarray(arrays["telemetry_fit_eligible"], dtype=np.bool_)
    return mask


def recording_rollout_mask(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
    *,
    require_telemetry: bool = False,
) -> NDArray[np.bool_]:
    """Return the continuous protocol mask suitable for dynamic rollout.

    Real torque is held causally from the native 100 Hz telemetry.  Rows
    remain valid between updates, but not across a reviewed transient or after
    the held sample becomes stale.  The sparse :func:`recording_analysis_mask`
    is the stricter selector for classical/regressor observations.
    """
    validate_recording(manifest, arrays)
    if "analysis_eligible" not in arrays:
        raise RecordingError(
            "recording has no analysis_eligible mask; reconvert its raw bag"
        )
    mask = np.asarray(arrays["analysis_eligible"], dtype=np.bool_).copy()
    if not require_telemetry:
        return mask
    for name in ("telemetry_transient_free", "franka_sample_age_s"):
        if name not in arrays:
            raise RecordingError(
                f"recording has no {name}; real telemetry rollout is unsafe"
            )
    telemetry = _mapping(manifest.get("telemetry_alignment"), "telemetry_alignment")
    age_limit = float(telemetry.get("maximum_allowed_age_s", np.nan))
    if not np.isfinite(age_limit) or age_limit <= 0.0:
        raise RecordingError("telemetry alignment has no valid maximum age")
    mask &= np.asarray(arrays["telemetry_transient_free"], dtype=np.bool_)
    mask &= np.asarray(arrays["franka_sample_age_s"], dtype=np.float64) <= age_limit
    return mask


def health_report(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[Any]]
) -> dict[str, object]:
    """Report acquisition completeness, gaps, torque slew, and telemetry health."""
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    steps = np.diff(time_s)
    period = float(np.median(steps))
    duration = float(time_s[-1] - time_s[0])
    protocol = manifest.get("protocol", {})
    expected = (
        float(protocol.get("duration_s", duration))
        if isinstance(protocol, Mapping)
        else duration
    )
    coverage = duration / expected if expected > 0.0 else 0.0

    jitter = float(np.max(np.abs(steps - period))) if len(steps) else 0.0
    long_steps = steps[steps > GAP_THRESHOLD_FRACTION * period]
    gaps = int(len(long_steps))
    longest_gap = float(long_steps.max()) if gaps else 0.0
    missing = float(np.sum(long_steps - period)) if gaps else 0.0
    missing_fraction = missing / duration if duration > 0.0 else 0.0

    command = np.asarray(arrays["tau_cmd_Nm"], dtype=np.float64)
    torque = np.abs(command)
    limits = np.asarray(
        manifest.get("torque_limit_Nm", [87.0] * 4 + [12.0] * 3),
        dtype=np.float64,
    )
    saturated = torque >= limits[None, :] - 1e-6
    saturated_fraction = saturated.mean(axis=0)

    # One control cycle produces one command increment, whatever time its
    # timestamp happens to carry. When the loop catches up after a late cycle,
    # two cycles land microseconds apart: on the first hardware campaign that
    # turned an ordinary 0.49 Nm increment arriving 44 us later into a reported
    # 11069 Nm/s, while the command itself stayed smooth. Rates are therefore
    # measured over at least the nominal period; genuine limiter-relevant slew
    # shows up as a large increment, which this still reports faithfully.
    slew = np.abs(np.diff(command, axis=0) / np.maximum(steps, period)[:, None])
    torque_slew = {
        "maximum_Nm_s": np.max(slew, axis=0).tolist(),
        "p99_9_Nm_s": np.percentile(slew, 99.9, axis=0).tolist(),
    }

    problems: list[str] = []
    warnings: list[str] = []
    if coverage < COVERAGE_MINIMUM:
        problems.append(
            f"covers {coverage:.1%} of the protocol's {expected:.1f} s — the run "
            "did not complete"
        )
    if (
        manifest.get("format") == BOUND_RECORDING_FORMAT
        and coverage > BOUND_COVERAGE_MAXIMUM
    ):
        problems.append(
            f"covers {coverage:.1%} of the protocol — rows outside the marked "
            "protocol interval were retained"
        )
    real = recording_backend(manifest) == "real"
    achieved_rate_hz = (len(time_s) - 1) / duration if duration > 0.0 else 0.0
    gap_limit = REAL_MAX_GAP_S if real else MAX_GAP_S
    if longest_gap > gap_limit:
        problems.append(
            f"a {longest_gap * 1e3:.0f} ms gap in the recording, beyond the "
            f"{gap_limit * 1e3:.0f} ms limit"
        )
    if not real:
        if missing_fraction > MISSING_FRACTION_LIMIT:
            problems.append(
                f"{missing_fraction:.2%} of the run is missing across {gaps} gaps"
            )
    elif achieved_rate_hz < REAL_MINIMUM_RATE_HZ:
        problems.append(
            f"the control loop averaged {achieved_rate_hz:.0f} Hz, below the "
            f"{REAL_MINIMUM_RATE_HZ:.0f} Hz this identification needs"
        )
    elif missing_fraction > MISSING_FRACTION_LIMIT:
        warnings.append(
            f"{missing_fraction:.2%} of cycles were late across {gaps} gaps; "
            f"the loop averaged {achieved_rate_hz:.0f} Hz"
        )
    if saturated_fraction.max() > 0.0:
        joint = int(np.argmax(saturated_fraction))
        problems.append(
            f"joint{joint + 1} commanded at its torque limit for "
            f"{saturated_fraction[joint]:.2%} of the run"
        )

    slew_warning = manifest.get("torque_slew_warning_Nm_s")
    if slew_warning is not None:
        threshold = np.asarray(slew_warning, dtype=np.float64)
        maximum = np.max(slew, axis=0)
        if threshold.shape == (7,) and np.any(maximum > threshold):
            joint = int(np.argmax(maximum / threshold))
            warnings.append(
                f"joint{joint + 1} controller-request slew "
                f"{maximum[joint]:.0f} Nm/s exceeds its diagnostic threshold "
                f"{threshold[joint]:.0f} Nm/s"
            )

    telemetry_mismatch: dict[str, object] | None = None
    timing = manifest.get("protocol_timing")
    if (
        "tau_J_d_native_Nm" in arrays
        and "franka_time_ns" in arrays
        and isinstance(timing, Mapping)
        and isinstance(timing.get("start_stamp_ns"), int)
    ):
        controller_ns = int(timing["start_stamp_ns"]) + np.rint(time_s * 1e9).astype(
            np.int64
        )
        native_ns = np.asarray(arrays["franka_time_ns"], dtype=np.int64)
        native_desired = np.asarray(arrays["tau_J_d_native_Nm"], dtype=np.float64)
        inside = (native_ns >= controller_ns[0]) & (native_ns <= controller_ns[-1])
        native_ns = native_ns[inside]
        native_desired = native_desired[inside]
        if len(native_ns):
            insertion = np.searchsorted(controller_ns, native_ns, side="left")
            lower = np.clip(insertion - 1, 0, len(controller_ns) - 1)
            upper = np.clip(insertion, 0, len(controller_ns) - 1)
            use_upper = np.abs(controller_ns[upper] - native_ns) < np.abs(
                native_ns - controller_ns[lower]
            )
            nearest = np.where(use_upper, upper, lower)
            difference = command[nearest] - native_desired
            telemetry_mismatch = {
                "samples": int(len(difference)),
                "bias_Nm": np.mean(difference, axis=0).tolist(),
                "rmse_Nm": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                "maximum_absolute_Nm": np.max(np.abs(difference), axis=0).tolist(),
                "alignment": (
                    "nearest controller row to each native telemetry sample; "
                    "diagnostic comparison only"
                ),
                "interpretation": (
                    "diagnostic difference only; channel equivalence and "
                    "post-limiter/applied semantics are not asserted"
                ),
            }
    elif "tau_J_d_Nm" in arrays:
        desired = np.asarray(arrays["tau_J_d_Nm"], dtype=np.float64)
        selected = np.ones(len(time_s), dtype=bool)
        if "franka_new_sample" in arrays:
            selected &= np.asarray(arrays["franka_new_sample"], dtype=bool)
        if np.any(selected):
            difference = command[selected] - desired[selected]
            telemetry_mismatch = {
                "samples": int(np.count_nonzero(selected)),
                "bias_Nm": np.mean(difference, axis=0).tolist(),
                "rmse_Nm": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                "maximum_absolute_Nm": np.max(np.abs(difference), axis=0).tolist(),
                "alignment": ("causal aligned rows (legacy fallback; diagnostic only)"),
                "interpretation": (
                    "diagnostic difference only; channel equivalence and "
                    "post-limiter/applied semantics are not asserted"
                ),
            }

    telemetry = manifest.get("telemetry_alignment")
    if isinstance(telemetry, Mapping):
        max_age = float(telemetry.get("maximum_age_s", np.inf))
        age_limit = float(telemetry.get("maximum_allowed_age_s", 0.0))
        max_gap = float(telemetry.get("maximum_native_gap_s", np.inf))
        gap_limit = float(telemetry.get("maximum_allowed_native_gap_s", 0.0))
        if age_limit > 0.0 and max_age > age_limit:
            problems.append(
                f"Franka telemetry is stale by {max_age * 1e3:.1f} ms "
                f"(limit {age_limit * 1e3:.1f} ms)"
            )
        if gap_limit > 0.0 and max_gap > gap_limit:
            problems.append(
                f"Franka telemetry has a {max_gap * 1e3:.1f} ms native gap "
                f"(limit {gap_limit * 1e3:.1f} ms)"
            )

    fit_rows = (
        int(np.count_nonzero(arrays["telemetry_fit_eligible"]))
        if "telemetry_fit_eligible" in arrays
        else None
    )
    return {
        "samples": int(len(time_s)),
        "duration_s": duration,
        "expected_duration_s": expected,
        "coverage": coverage,
        "rate_hz": 1.0 / period if period > 0.0 else 0.0,
        "worst_jitter_ms": jitter * 1e3,
        "gaps": gaps,
        "longest_gap_ms": longest_gap * 1e3,
        "missing_fraction": missing_fraction,
        "achieved_rate_hz": achieved_rate_hz,
        "saturated_fraction": saturated_fraction.tolist(),
        "torque_slew": torque_slew,
        "tau_cmd_minus_tau_J_d": telemetry_mismatch,
        "telemetry_fit_rows": fit_rows,
        "warnings": warnings,
        "problems": problems,
        "usable": not problems,
    }


def write_recording(
    root: str | Path,
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
) -> Path:
    """Write a recording bundle; fail closed unless it validates."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise RecordingError(f"recording directory already holds a run: {root}")

    payload = dict(manifest)
    payload.setdefault("format", RECORDING_FORMAT)
    payload["arrays"] = {
        name: {
            "unit": ARRAY_UNITS.get(name, ""),
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
        }
        for name, value in arrays.items()
    }
    payload["content_sha256"] = ""
    validate_recording(payload, arrays)
    payload["health"] = health_report(payload, arrays)
    payload["content_sha256"] = content_sha256(payload, arrays)

    root.mkdir(parents=True, exist_ok=True)
    save_arrays(root / "measured.npz", arrays)
    write_json(root / "recording.json", payload)
    write_checksums(root)
    verify_checksums(root)
    return root


def load_recording(
    root: str | Path,
) -> tuple[dict[str, object], dict[str, NDArray[Any]]]:
    """Load a recording bundle, verifying checksums and its fingerprint."""
    root = Path(root)
    verify_checksums(root)
    manifest = read_json(root / "recording.json")
    arrays = load_arrays(root / "measured.npz")
    validate_recording(manifest, arrays)

    stated = str(manifest.get("content_sha256", ""))
    probe = dict(manifest)
    probe["content_sha256"] = ""
    if content_sha256(probe, arrays) != stated:
        raise RecordingError(f"{root}: content fingerprint mismatch")
    return manifest, arrays


def find_recordings(root: str | Path) -> list[Path]:
    """Return every recording bundle under ``root`` in stable order."""
    root = Path(root)
    return sorted(
        path.parent for path in root.rglob("recording.json") if path.is_file()
    )
