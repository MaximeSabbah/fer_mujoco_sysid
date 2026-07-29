"""The compiled-protocol bundle format: what it is, and how to read one.

Split out of :mod:`fer_mujoco_sysid.excitation` deliberately. *Writing* a
protocol needs MuJoCo — the manifest records the payload read off the model
and the motion is validated by simulating it. *Reading* one needs nothing but
NumPy, and the reader is what runs on the robot: the ROS player imports this
module and never pulls MuJoCo into a real-time-adjacent process. That also
means playback keeps working in a plain ROS environment, where the only
Python packages guaranteed present are the ones ROS itself ships.

A bundle is a directory holding ``protocol.json``, ``desired.npz`` and
``checksums.sha256``. It is immutable once written; a changed protocol is a
new revision.
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
    verify_checksums,
)

PROTOCOL_FORMAT = "fer-mujoco-sysid/motion-protocol@2"

#: Joint names as the ROS stack knows them, and the order every array column
#: and every trajectory point follows.
ROS_ARM_JOINT_NAMES = tuple(f"fer_joint{i}" for i in range(1, 8))

ARRAY_UNITS = {
    "time_s": "s",
    "q_rad": "rad",
    "dq_rad_s": "rad/s",
    "ddq_rad_s2": "rad/s^2",
}


def validate_protocol_manifest(
    manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.float64]]
) -> None:
    """Check a protocol manifest against its arrays (structure and meaning)."""
    if manifest.get("format") != PROTOCOL_FORMAT:
        raise ArtifactError(f"unsupported protocol format: {manifest.get('format')!r}")
    missing = set(ARRAY_UNITS) - set(arrays)
    if missing:
        raise ArtifactError(f"missing arrays: {sorted(missing)}")

    time_s = np.asarray(arrays["time_s"])
    samples = len(time_s)
    if samples < 2 or not np.all(np.diff(time_s) > 0):
        raise ArtifactError("time_s must be strictly increasing with >= 2 samples")
    for key in ("q_rad", "dq_rad_s", "ddq_rad_s2"):
        array = np.asarray(arrays[key])
        if array.shape != (samples, 7):
            raise ArtifactError(
                f"{key} has shape {array.shape}, expected {(samples, 7)}"
            )

    playback = manifest.get("playback", {})
    if list(playback.get("joint_order", [])) != list(ROS_ARM_JOINT_NAMES):
        raise ArtifactError("joint_order does not match the canonical FER order")
    if int(playback.get("samples", -1)) != samples:
        raise ArtifactError("playback.samples disagrees with the arrays")

    segments = manifest.get("segments", [])
    if not segments:
        raise ArtifactError("protocol declares no segments")
    cursor = 0
    for segment in segments:
        if segment["start_index"] != cursor:
            raise ArtifactError(f"segment {segment['segment_id']} is not contiguous")
        cursor = segment["end_index_exclusive"]
    if cursor != samples:
        raise ArtifactError("segments do not cover every sample")


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


def latest_revision(protocol_root: str | Path) -> Path:
    """The newest committed revision directory of a protocol."""
    root = Path(protocol_root)
    revisions = sorted(path for path in root.iterdir() if path.is_dir())
    if not revisions:
        raise ArtifactError(f"{root} holds no protocol revision")
    return revisions[-1]
