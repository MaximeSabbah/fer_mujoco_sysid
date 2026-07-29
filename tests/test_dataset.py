"""Gates for the recording format and its health checks.

The recording is the boundary between ROS and the fit. These run without ROS,
which is the point: the format has to be checkable from the side that
consumes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fer_mujoco_sysid.dataset import (
    COVERAGE_MINIMUM,
    MAX_GAP_S,
    MISSING_FRACTION_LIMIT,
    RECORDING_FORMAT,
    RecordingError,
    find_recordings,
    health_report,
    load_recording,
    validate_recording,
    write_recording,
)
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

_RATE_HZ = 500.0
_DURATION_S = 4.0
_TORQUE_LIMIT = [87.0] * 4 + [12.0] * 3


def _arrays(
    duration_s: float = _DURATION_S, rate_hz: float = _RATE_HZ
) -> dict[str, np.ndarray]:
    time_s = np.arange(int(duration_s * rate_hz)) / rate_hz
    phase = 2 * np.pi * 0.25 * time_s
    q = np.column_stack([0.3 * np.sin(phase + joint) for joint in range(7)])
    dq = np.column_stack(
        [0.3 * 2 * np.pi * 0.25 * np.cos(phase + joint) for joint in range(7)]
    )
    tau = 1.2 * np.sign(dq) + 0.8 * dq
    return {"time_s": time_s, "q_rad": q, "dq_rad_s": dq, "tau_cmd_Nm": tau}


def _manifest(duration_s: float = _DURATION_S) -> dict[str, object]:
    return {
        "format": RECORDING_FORMAT,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "mujoco",
        "torque_limit_Nm": _TORQUE_LIMIT,
        "protocol": {"protocol_id": "fer-friction-a", "duration_s": duration_s},
    }


def test_recording_round_trips(tmp_path: Path) -> None:
    arrays = _arrays()
    root = write_recording(tmp_path / "run", _manifest(), arrays)
    manifest, loaded = load_recording(root)

    assert manifest["format"] == RECORDING_FORMAT
    assert manifest["health"]["usable"] is True
    for name, value in arrays.items():
        np.testing.assert_allclose(loaded[name], value)


def test_a_tampered_recording_is_refused(tmp_path: Path) -> None:
    """Checksums are only worth having if a changed file actually fails."""
    root = write_recording(tmp_path / "run", _manifest(), _arrays())
    text = (root / "recording.json").read_text()
    (root / "recording.json").write_text(text.replace('"mujoco"', '"real"'))

    with pytest.raises(Exception, match="SHA-256 mismatch"):
        load_recording(root)


def test_a_recording_is_never_silently_overwritten(tmp_path: Path) -> None:
    write_recording(tmp_path / "run", _manifest(), _arrays())
    with pytest.raises(RecordingError, match="already holds a run"):
        write_recording(tmp_path / "run", _manifest(), _arrays())


def test_structural_defects_are_caught() -> None:
    arrays = _arrays()
    with pytest.raises(RecordingError, match="missing"):
        validate_recording(_manifest(), {"time_s": arrays["time_s"]})

    backwards = dict(arrays)
    backwards["time_s"] = arrays["time_s"][::-1].copy()
    with pytest.raises(RecordingError, match="strictly increasing"):
        validate_recording(_manifest(), backwards)

    short = dict(arrays)
    short["q_rad"] = arrays["q_rad"][:-5]
    with pytest.raises(RecordingError, match="shape"):
        validate_recording(_manifest(), short)

    broken = dict(arrays)
    broken["tau_cmd_Nm"] = arrays["tau_cmd_Nm"].copy()
    broken["tau_cmd_Nm"][3, 2] = np.nan
    with pytest.raises(RecordingError, match="non-finite"):
        validate_recording(_manifest(), broken)


def test_a_truncated_run_is_reported_unusable() -> None:
    """A run that stopped early must not be fitted as if it had finished."""
    arrays = _arrays(duration_s=2.0)
    report = health_report(_manifest(duration_s=_DURATION_S), arrays)

    assert not report["usable"]
    assert report["coverage"] < COVERAGE_MINIMUM
    assert any("did not complete" in problem for problem in report["problems"])


def test_a_long_gap_is_rejected() -> None:
    """One hole too long to interpolate across makes the run unusable."""
    arrays = _arrays()
    keep = np.ones(len(arrays["time_s"]), dtype=bool)
    keep[400:520] = False  # 240 ms missing, far beyond MAX_GAP_S
    gapped = {name: value[keep] for name, value in arrays.items()}

    report = health_report(_manifest(), gapped)
    assert report["gaps"] == 1
    assert report["longest_gap_ms"] > MAX_GAP_S * 1e3
    assert not report["usable"]
    assert any("gap" in problem for problem in report["problems"])


def test_occasional_dropped_messages_do_not_reject_a_run() -> None:
    """Bag recording drops the odd message; that is not a defect.

    Measured on a real simulated campaign: 12 holes of at most 8 ms in a 45 s
    run, 0.05% of samples. At these speeds that is a few milliradians, and the
    fit resamples onto a uniform grid anyway. Rejecting it would throw away
    good hardware data for a cosmetic reason.
    """
    arrays = _arrays()
    keep = np.ones(len(arrays["time_s"]), dtype=bool)
    for start in range(200, 1800, 400):
        keep[start : start + 3] = False  # 6 ms each, ~0.6% of the run
    dropped = {name: value[keep] for name, value in arrays.items()}

    report = health_report(_manifest(), dropped)
    assert report["gaps"] > 0
    assert report["longest_gap_ms"] < MAX_GAP_S * 1e3
    assert report["missing_fraction"] < MISSING_FRACTION_LIMIT
    assert report["usable"], report["problems"]


def test_a_joint_pinned_at_its_torque_limit_is_reported() -> None:
    """Saturation means the commanded effort is not what the joint asked for."""
    arrays = _arrays()
    arrays["tau_cmd_Nm"] = arrays["tau_cmd_Nm"].copy()
    arrays["tau_cmd_Nm"][100:300, 5] = 12.0

    report = health_report(_manifest(), arrays)
    assert not report["usable"]
    assert any("joint6" in problem for problem in report["problems"])
    assert report["saturated_fraction"][5] > 0.0


def test_a_clean_run_reports_its_rate(tmp_path: Path) -> None:
    report = health_report(_manifest(), _arrays())
    assert report["usable"]
    assert report["rate_hz"] == pytest.approx(_RATE_HZ, rel=1e-6)
    assert report["gaps"] == 0
    assert report["worst_jitter_ms"] < 1e-6


def test_recordings_are_discovered_under_a_campaign_directory(tmp_path: Path) -> None:
    for protocol in ("fer-friction-a", "fer-friction-holdout"):
        write_recording(
            tmp_path / protocol / "recording", _manifest(), _arrays(duration_s=1.0)
        )
    found = find_recordings(tmp_path)
    assert [path.parent.name for path in found] == [
        "fer-friction-a",
        "fer-friction-holdout",
    ]
