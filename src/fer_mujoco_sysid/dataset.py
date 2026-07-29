"""What a recorded run looks like once it is out of ROS.

The boundary between the two worlds this project lives in. ROS deserializes
messages and writes a recording bundle; the fitting side reads it with NumPy
and never learns that ROS existed. Like :mod:`fer_mujoco_sysid.protocol`, this
module imports nothing heavier than NumPy so both sides can use it.

A recording is a directory holding ``recording.json``, ``measured.npz`` and
``checksums.sha256``, and it is immutable: the raw bag stays next to it, so a
dataset can always be rebuilt without re-running the robot.

Which torque is *the* torque, decided in D027/D028: the trajectory
controller's **commanded effort**. It is what actually drove the joint, it is
known exactly rather than inferred, and it is the same quantity in simulation
and on hardware. On hardware the measured link-side ``tau_J``, the desired
``tau_J_d`` and the motor-side ``theta`` are recorded alongside it — not
because the fit uses them, but because re-running the robot to recover a
channel we chose not to record is the expensive mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

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

RECORDING_FORMAT = "fer-mujoco-sysid/recording@1"

#: Where a recording came from. Three distinct paths, and they must never be
#: fitted together: "mujoco" is the standalone in-process pipeline (no ROS),
#: "mujoco_ros" is played through the real ROS stack against a simulated
#: plant, and "real" is the robot. A model fitted from a mix of them would be
#: describing no particular machine.
BACKENDS = ("mujoco", "mujoco_ros", "real")

#: Everything this project generates lands under one root, partitioned by
#: provenance: ``output/<backend>/``. Nothing in it is committed — it is all
#: reproducible from the protocols and the recordings — and the partition is
#: what makes "which robot is this model of?" answerable by looking at a path.
OUTPUT_DIRECTORY = "output"


def output_root(repository: str | Path, backend: str) -> Path:
    """``<repository>/output/<backend>``, created if needed."""
    if backend not in BACKENDS:
        raise RecordingError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    root = Path(repository) / OUTPUT_DIRECTORY / backend
    root.mkdir(parents=True, exist_ok=True)
    return root


def recording_backend(manifest: Mapping[str, object]) -> str:
    """The provenance a recording declares."""
    backend = str(manifest.get("backend", ""))
    if backend not in BACKENDS:
        raise RecordingError(
            f"recording declares backend {backend!r}; expected one of {BACKENDS}"
        )
    return backend

#: Arrays every recording must carry, with their units.
REQUIRED_ARRAYS = {
    "time_s": "s",
    "q_rad": "rad",
    "dq_rad_s": "rad/s",
    "tau_cmd_Nm": "Nm",
}

#: Arrays only hardware can provide. Their absence is normal in simulation.
OPTIONAL_ARRAYS = {
    # What the controller was aiming at, against what the arm did. Not fitted;
    # it is how tracking quality is judged, and it is the "expected robot" in
    # the replay video.
    "q_desired_rad": "rad",
    "tau_J_Nm": "Nm",
    "tau_J_d_Nm": "Nm",
    "theta_rad": "rad",
    "dtheta_rad_s": "rad/s",
}

#: A step longer than this counts as a gap rather than jitter.
GAP_THRESHOLD_FRACTION = 1.5

#: Longest single gap a recording may contain. Chosen physically, not
#: cosmetically: the fit resamples onto a uniform grid, so a gap matters only
#: when interpolating across it could misrepresent the motion. At the fastest
#: protocol velocity (2.04 rad/s) 50 ms is 0.1 rad — the point where a
#: straight line between samples stops being the truth. Recording a bag drops
#: the odd message; a handful of 8 ms holes in a 45 s run is not a defect.
MAX_GAP_S = 0.05

#: Total missing time a recording may carry, as a fraction of its duration.
MISSING_FRACTION_LIMIT = 0.01

#: Fraction of the protocol's duration a recording must cover to be usable.
COVERAGE_MINIMUM = 0.98


class RecordingError(ArtifactError):
    """A recording is malformed, incomplete, or unusable for fitting."""


def validate_recording(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.float64]]
) -> None:
    """Structural check: the arrays are present, aligned, and well formed."""
    if manifest.get("format") != RECORDING_FORMAT:
        raise RecordingError(f"unsupported recording format: {manifest.get('format')!r}")

    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise RecordingError(f"recording is missing {missing}")

    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) < 2:
        raise RecordingError("time_s must be a 1-D vector with at least two samples")
    if not np.all(np.diff(time_s) > 0.0):
        raise RecordingError("time_s must be strictly increasing")

    joints = len(manifest.get("joint_order", []))
    if joints != 7:
        raise RecordingError(f"expected 7 joints, manifest declares {joints}")
    for name in (*REQUIRED_ARRAYS, *OPTIONAL_ARRAYS):
        if name in ("time_s",) or name not in arrays:
            continue
        array = np.asarray(arrays[name])
        if array.shape != (len(time_s), joints):
            raise RecordingError(
                f"{name} has shape {array.shape}, expected {(len(time_s), joints)}"
            )
        if not np.all(np.isfinite(array)):
            raise RecordingError(f"{name} holds non-finite samples")


def health_report(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.float64]]
) -> dict[str, object]:
    """Judge whether a recording is fit to identify from.

    Deliberately separate from :func:`validate_recording`: a recording can be
    structurally perfect and still useless — half the protocol missing,
    samples dropped, a joint pinned at its torque limit. Those are the
    failures worth catching before a fit spends ten minutes explaining them.
    """
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    steps = np.diff(time_s)
    period = float(np.median(steps))
    duration = float(time_s[-1] - time_s[0])
    expected = float(manifest.get("protocol", {}).get("duration_s", duration))  # type: ignore[union-attr]
    coverage = duration / expected if expected > 0 else 0.0

    jitter = float(np.max(np.abs(steps - period))) if len(steps) else 0.0
    long_steps = steps[steps > GAP_THRESHOLD_FRACTION * period]
    gaps = int(len(long_steps))
    longest_gap = float(long_steps.max()) if gaps else 0.0
    missing = float(np.sum(long_steps - period)) if gaps else 0.0
    missing_fraction = missing / duration if duration > 0 else 0.0

    torque = np.abs(np.asarray(arrays["tau_cmd_Nm"], dtype=np.float64))
    limits = np.asarray(
        manifest.get("torque_limit_Nm", [87.0] * 4 + [12.0] * 3), dtype=np.float64
    )
    saturated = torque >= limits[None, :] - 1e-6
    saturated_fraction = saturated.mean(axis=0)

    problems: list[str] = []
    if coverage < COVERAGE_MINIMUM:
        problems.append(
            f"covers {coverage:.1%} of the protocol's {expected:.1f} s — the run "
            "did not complete"
        )
    if longest_gap > MAX_GAP_S:
        problems.append(
            f"a {longest_gap * 1e3:.0f} ms gap in the recording, beyond the "
            f"{MAX_GAP_S * 1e3:.0f} ms that can be interpolated across without "
            "misrepresenting the motion"
        )
    if missing_fraction > MISSING_FRACTION_LIMIT:
        problems.append(
            f"{missing_fraction:.2%} of the run is missing across {gaps} gaps"
        )
    if saturated_fraction.max() > 0.0:
        joint = int(np.argmax(saturated_fraction))
        problems.append(
            f"joint{joint + 1} commanded at its torque limit for "
            f"{saturated_fraction[joint]:.2%} of the run"
        )

    return {
        "samples": int(len(time_s)),
        "duration_s": duration,
        "expected_duration_s": expected,
        "coverage": coverage,
        "rate_hz": 1.0 / period if period > 0 else 0.0,
        "worst_jitter_ms": jitter * 1e3,
        "gaps": gaps,
        "longest_gap_ms": longest_gap * 1e3,
        "missing_fraction": missing_fraction,
        "saturated_fraction": saturated_fraction.tolist(),
        "problems": problems,
        "usable": not problems,
    }


def write_recording(
    root: str | Path,
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
) -> Path:
    """Write a recording bundle; fail closed unless it validates."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise RecordingError(f"recording directory already holds a run: {root}")

    payload = dict(manifest)
    payload["format"] = RECORDING_FORMAT
    payload["arrays"] = {
        name: {
            "unit": {**REQUIRED_ARRAYS, **OPTIONAL_ARRAYS}.get(name, ""),
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
) -> tuple[dict[str, object], dict[str, NDArray[np.float64]]]:
    """Load a recording bundle, verifying checksums and content fingerprint."""
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
    """Every recording bundle under ``root``, in a stable order."""
    root = Path(root)
    return sorted(
        path.parent for path in root.rglob("recording.json") if path.is_file()
    )
