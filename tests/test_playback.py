"""Gates for protocol playback: time scaling, approach, and preflight checks.

These cover the decisions the ROS player makes before and while the arm
moves. They run without ROS on purpose — the reason the logic lives in
:mod:`fer_mujoco_sysid.playback` is so that it can be gated here rather than
only on hardware.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fer_mujoco_sysid.campaign import CAMPAIGN, repository_root
from fer_mujoco_sysid.excitation import (
    FER_ACCELERATION_LIMIT_RAD_S2,
    FER_VELOCITY_LIMIT_RAD_S,
    load_protocol_bundle,
)
from fer_mujoco_sysid.playback import (
    APPROACH_MIN_DURATION_S,
    PlaybackError,
    PlaybackPlan,
    approach_playback,
    check_claimed_interfaces,
    check_start_state,
    joint_order,
    protocol_playback,
    reorder_to,
    start_qpos,
)

_JOINTS = tuple(f"fer_joint{index}" for index in range(1, 8))


def _bundle(protocol_id: str = "fer-friction-a"):
    root = repository_root() / "protocols" / protocol_id
    revision = sorted(path for path in root.iterdir() if path.is_dir())[-1]
    return load_protocol_bundle(revision)


# -- time scaling -----------------------------------------------------------


def test_playback_starts_after_zero_and_keeps_the_path() -> None:
    manifest, arrays = _bundle()
    plan = protocol_playback(manifest, arrays)

    assert plan.time_s[0] > 0.0, "time_from_start must be strictly positive"
    assert plan.joint_names == _JOINTS
    # The path itself is untouched: only the first (settle) sample is dropped.
    assert plan.samples == len(arrays["time_s"]) - 1
    np.testing.assert_allclose(plan.q_rad, arrays["q_rad"][1:])


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0])
def test_speed_scale_stretches_time_and_scales_derivatives(scale: float) -> None:
    manifest, arrays = _bundle()
    nominal = protocol_playback(manifest, arrays)
    scaled = protocol_playback(manifest, arrays, speed_scale=scale)

    assert scaled.duration_s == pytest.approx(nominal.duration_s / scale)
    # Same positions, so every position-based validation done at generation
    # time (joint ranges, table clearance) still holds.
    np.testing.assert_allclose(scaled.q_rad, nominal.q_rad)
    np.testing.assert_allclose(scaled.dq_rad_s, nominal.dq_rad_s * scale)
    np.testing.assert_allclose(scaled.ddq_rad_s2, nominal.ddq_rad_s2 * scale**2)


def test_speeding_a_protocol_up_is_refused() -> None:
    manifest, arrays = _bundle()
    with pytest.raises(PlaybackError, match="speed_scale must lie in"):
        protocol_playback(manifest, arrays, speed_scale=1.5)
    with pytest.raises(PlaybackError, match="speed_scale must lie in"):
        protocol_playback(manifest, arrays, speed_scale=0.0)


def test_every_committed_protocol_plays_within_the_fer_limits() -> None:
    """What is sent to the controller obeys the limits, not just what was compiled."""
    velocity_limit = np.asarray(FER_VELOCITY_LIMIT_RAD_S)
    acceleration_limit = np.asarray(FER_ACCELERATION_LIMIT_RAD_S2)
    for spec in CAMPAIGN:
        manifest, arrays = _bundle(spec.protocol_id)
        plan = protocol_playback(manifest, arrays)
        assert np.all(np.abs(plan.dq_rad_s) <= velocity_limit), spec.protocol_id
        assert np.all(np.abs(plan.ddq_rad_s2) <= acceleration_limit), spec.protocol_id
        assert plan.time_s[0] > 0.0, spec.protocol_id
        assert np.all(np.diff(plan.time_s) > 0.0), spec.protocol_id


# -- approach ---------------------------------------------------------------


def test_approach_is_rest_to_rest_and_lands_on_target() -> None:
    current = np.zeros(7)
    target = np.asarray(start_qpos(_bundle()[0]))
    plan = approach_playback(current, target, _JOINTS)

    np.testing.assert_allclose(plan.q_rad[-1], target, atol=1e-9)
    np.testing.assert_allclose(plan.dq_rad_s[-1], 0.0, atol=1e-9)
    np.testing.assert_allclose(plan.ddq_rad_s2[-1], 0.0, atol=1e-9)
    # Monotone in each joint: the quintic never overshoots and comes back.
    travel = target - current
    progress = (plan.q_rad - current) @ np.sign(travel + 1e-12)
    assert np.all(np.diff(progress) >= -1e-12)


def test_approach_respects_the_speed_cap() -> None:
    current = np.zeros(7)
    target = np.full(7, 1.0)
    speed = 0.2
    plan = approach_playback(current, target, _JOINTS, speed_rad_s=speed)

    peak = float(np.max(np.abs(plan.dq_rad_s)))
    assert peak <= speed * (1.0 + 1e-6), peak
    # The quintic's peak is 1.875*dq/T, so the cap should be nearly reached
    # rather than the motion being needlessly slow.
    assert peak > 0.9 * speed


def test_short_approach_still_takes_the_minimum_duration() -> None:
    plan = approach_playback(np.zeros(7), np.full(7, 1e-4), _JOINTS)
    assert plan.duration_s == pytest.approx(APPROACH_MIN_DURATION_S)


def test_approach_rejects_mismatched_shapes() -> None:
    with pytest.raises(PlaybackError, match="matching 1-D poses"):
        approach_playback(np.zeros(7), np.zeros(6), _JOINTS)
    with pytest.raises(PlaybackError, match="joint values for"):
        approach_playback(np.zeros(6), np.zeros(6), _JOINTS)


# -- preflight --------------------------------------------------------------


def test_start_state_check_measures_the_worst_joint() -> None:
    manifest, _ = _bundle()
    expected = start_qpos(manifest)

    at_start = check_start_state(expected, manifest)
    assert at_start.satisfied
    assert at_start.worst_error_rad == pytest.approx(0.0)

    off = expected.copy()
    off[3] += 0.5
    check = check_start_state(off, manifest)
    assert not check.satisfied
    assert check.worst_joint == 3
    assert check.worst_error_rad == pytest.approx(0.5)
    assert "fer_joint4" in check.describe()


def test_joint_state_is_read_by_name_not_by_position() -> None:
    """Publishers order joints freely and add the fingers; names are the contract."""
    reading = {
        "fer_finger_joint1": 0.04,
        "fer_joint7": 7.0,
        "fer_joint1": 1.0,
        "fer_joint2": 2.0,
        "fer_joint3": 3.0,
        "fer_joint4": 4.0,
        "fer_joint5": 5.0,
        "fer_joint6": 6.0,
    }
    np.testing.assert_allclose(
        reorder_to(reading, _JOINTS), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )

    del reading["fer_joint3"]
    with pytest.raises(PlaybackError, match="missing fer_joint3"):
        reorder_to(reading, _JOINTS)


def test_claimed_interfaces_must_cover_every_joint_in_effort_mode() -> None:
    effort = [f"{joint}/effort" for joint in _JOINTS]
    check_claimed_interfaces(effort, _JOINTS)

    with pytest.raises(PlaybackError, match="fer_joint7/effort"):
        check_claimed_interfaces(effort[:-1], _JOINTS)
    # A position-mode controller claims the joints but not in the mode the
    # identification requires.
    with pytest.raises(PlaybackError, match="effort-mode control"):
        check_claimed_interfaces([f"{joint}/position" for joint in _JOINTS], _JOINTS)


def test_plan_rejects_a_non_monotonic_schedule() -> None:
    with pytest.raises(PlaybackError, match="strictly increasing"):
        PlaybackPlan(
            label="broken",
            joint_names=_JOINTS,
            time_s=np.array([0.1, 0.3, 0.2]),
            q_rad=np.zeros((3, 7)),
            dq_rad_s=np.zeros((3, 7)),
        )


def test_joint_order_comes_from_the_manifest() -> None:
    manifest, _ = _bundle()
    assert joint_order(manifest) == _JOINTS
    with pytest.raises(PlaybackError, match="no 'playback' section"):
        joint_order({})


def test_no_protocol_contains_a_position_step() -> None:
    """Every position change must be explained by the velocity that produced it.

    The r1 inertial protocols failed this: their Fourier segment stopped
    mid-cycle and the trailing settle segment snapped back to the home pose
    in one 10 ms sample, up to 0.59 rad. Played through the trajectory
    controller in ROS simulation that saturated four joints and aborted the
    goal. A step is invisible in position, velocity and acceleration limit
    checks taken separately — only their consistency catches it.
    """
    for spec in CAMPAIGN:
        manifest, arrays = _bundle(spec.protocol_id)
        period = float(manifest["playback"]["sample_period_s"])
        step = np.max(np.abs(np.diff(arrays["q_rad"], axis=0)))
        explained = np.max(np.abs(arrays["dq_rad_s"])) * period
        assert step <= explained * 1.05, (
            f"{spec.protocol_id}: {step:.4f} rad step against {explained:.4f} "
            "rad of travel at peak velocity"
        )


def test_every_protocol_returns_to_its_start_pose() -> None:
    """The arm must end where the next protocol expects to begin."""
    for spec in CAMPAIGN:
        manifest, arrays = _bundle(spec.protocol_id)
        start = np.asarray(manifest["start_state"]["q_rad"])
        end = np.asarray(manifest["end_state"]["q_rad"])
        np.testing.assert_allclose(end, start, atol=1e-9, err_msg=spec.protocol_id)
        np.testing.assert_allclose(arrays["q_rad"][-1], start, atol=1e-9)
        np.testing.assert_allclose(arrays["dq_rad_s"][-1], 0.0, atol=1e-9)


def test_the_robot_side_import_surface_needs_no_mujoco() -> None:
    """What runs under ROS must import with only rclpy and NumPy available.

    A ROS environment carries whatever ``mujoco`` its own packages put on the
    path — in this workspace a ``build/`` data directory shadows the real
    package as a namespace package. The player must not care, so the modules
    it imports are checked to pull in neither MuJoCo nor SciPy.
    """
    source = (
        "import sys;"
        "import fer_mujoco_sysid.protocol, fer_mujoco_sysid.playback, fer_mujoco_sysid.io;"
        "heavy = sorted(m for m in ('mujoco', 'scipy', 'matplotlib') if m in sys.modules);"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
        cwd=repository_root(),
    )
    assert result.stdout.strip() == "", f"robot-side imports pulled in {result.stdout}"


def test_launch_and_config_are_present() -> None:
    """The player is useless without the pieces the launch wires together."""
    root: Path = repository_root()
    for relative in (
        "config/fer_sysid_controllers.yaml",
        "launch/play_protocol.launch.py",
        "urdf/fer_sysid_real.urdf.xacro",
        "urdf/fer_sysid_mujoco.urdf.xacro",
        "scripts/run-protocol",
    ):
        assert (root / relative).is_file(), relative
