"""Turning a compiled protocol into something a trajectory controller can run.

Everything here is pure: arrays in, arrays out, no ROS and no MuJoCo. The ROS
player (:mod:`fer_mujoco_sysid.ros.player_node`) is a thin shell around these
functions, so the decisions that matter for safety — how fast the arm is
allowed to move, whether it is close enough to the protocol start, whether the
controller really owns the joints — are testable without a robot or a
simulator.

Two plans are produced:

* the **approach**, a rest-to-rest quintic from wherever the arm currently is
  to the protocol's first sample. Its duration comes from a deliberately slow
  speed cap, not from the protocol;
* the **protocol playback** itself, optionally time-scaled.

Time scaling is the safety valve for the first hardware run. Scaling by
``s`` replays the identical *path* over ``1/s`` times as long: velocities
scale by ``s`` and accelerations by ``s**2``, so the workspace clearance
validated at generation time still holds exactly while the torque demand
drops quadratically. Speeding a protocol up is refused — the limits were
validated at ``s = 1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Joint-space speed cap for the move to the protocol start. Far below the
#: protocol speeds: this motion is unvalidated (it depends on where the arm
#: happens to be) and is the one a human watches with a hand on the stop.
APPROACH_SPEED_RAD_S = 0.2

#: No approach is ever shorter than this, however small the distance.
APPROACH_MIN_DURATION_S = 2.0

#: Sample period of the generated approach trajectory.
APPROACH_PERIOD_S = 0.01

#: How far the arm may be from the protocol's first sample before playback is
#: allowed to start. The approach is expected to close the gap to well inside
#: this; it is a verification threshold, not a motion tolerance.
START_TOLERANCE_RAD = 0.02


class PlaybackError(RuntimeError):
    """A protocol cannot be played as requested, or the robot is not ready."""


@dataclass(frozen=True)
class PlaybackPlan:
    """A time-stamped joint trajectory, ready to become controller messages.

    ``time_s`` is strictly increasing and starts strictly above zero, which is
    what ``trajectory_msgs`` requires of ``time_from_start``: a point at zero
    would be scheduled in the past by the time the controller receives it.
    """

    label: str
    joint_names: tuple[str, ...]
    time_s: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    dq_rad_s: NDArray[np.float64]
    ddq_rad_s2: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        samples = len(self.time_s)
        joints = len(self.joint_names)
        for name, array in (
            ("q_rad", self.q_rad),
            ("dq_rad_s", self.dq_rad_s),
            ("ddq_rad_s2", self.ddq_rad_s2),
        ):
            if array is None:
                continue
            if array.shape != (samples, joints):
                raise PlaybackError(
                    f"{self.label}: {name} has shape {array.shape}, "
                    f"expected {(samples, joints)}"
                )
        if samples < 2:
            raise PlaybackError(f"{self.label}: a plan needs at least two points")
        if not np.all(np.diff(self.time_s) > 0.0):
            raise PlaybackError(f"{self.label}: time_s must be strictly increasing")
        if self.time_s[0] <= 0.0:
            raise PlaybackError(
                f"{self.label}: the first point is at t={self.time_s[0]:.4f} s; "
                "time_from_start must be strictly positive"
            )

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def samples(self) -> int:
        return len(self.time_s)

    def peak_velocity_rad_s(self) -> NDArray[np.float64]:
        return np.max(np.abs(self.dq_rad_s), axis=0)

    def summary(self) -> str:
        peak = self.peak_velocity_rad_s()
        return (
            f"{self.label}: {self.samples} points, {self.duration_s:.1f} s, "
            f"peak |dq| {np.max(peak):.3f} rad/s (joint {int(np.argmax(peak)) + 1})"
        )


def joint_order(manifest: Mapping[str, object]) -> tuple[str, ...]:
    """The controller joint order a protocol was compiled against."""
    playback = manifest.get("playback")
    if not isinstance(playback, Mapping):
        raise PlaybackError("manifest has no 'playback' section")
    names = playback.get("joint_order")
    if not isinstance(names, Sequence) or isinstance(names, str) or not names:
        raise PlaybackError("manifest playback.joint_order is missing or empty")
    return tuple(str(name) for name in names)


def start_qpos(manifest: Mapping[str, object]) -> NDArray[np.float64]:
    """The joint positions a protocol expects the arm to be at."""
    start = manifest.get("start_state")
    if not isinstance(start, Mapping) or "q_rad" not in start:
        raise PlaybackError("manifest has no 'start_state.q_rad'")
    return np.asarray(start["q_rad"], dtype=np.float64)


def protocol_playback(
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[np.float64]],
    *,
    speed_scale: float = 1.0,
) -> PlaybackPlan:
    """Build the playback plan for a compiled protocol bundle.

    ``speed_scale`` must lie in ``(0, 1]``. The path is untouched; only its
    schedule changes, so every position-based validation done at generation
    time (joint ranges, table clearance) transfers unchanged.
    """
    if not 0.0 < speed_scale <= 1.0:
        raise PlaybackError(
            f"speed_scale must lie in (0, 1], got {speed_scale}. Playing a "
            "protocol faster than compiled would exceed the limits it was "
            "validated against."
        )
    names = joint_order(manifest)
    try:
        time_s = np.asarray(arrays["time_s"], dtype=np.float64)
        q_rad = np.asarray(arrays["q_rad"], dtype=np.float64)
        dq_rad_s = np.asarray(arrays["dq_rad_s"], dtype=np.float64)
        ddq_rad_s2 = np.asarray(arrays["ddq_rad_s2"], dtype=np.float64)
    except KeyError as error:  # pragma: no cover - guarded by the bundle loader
        raise PlaybackError(f"protocol bundle is missing array {error}") from error

    # The compiled grid starts at t=0; drop that sample rather than shifting
    # the whole schedule, so the remaining stamps stay on the original grid.
    # Sample 0 is inside the initial settle segment and carries no excitation.
    label = str(manifest.get("protocol_id", "protocol"))
    return PlaybackPlan(
        label=label,
        joint_names=names,
        time_s=time_s[1:] / speed_scale,
        q_rad=q_rad[1:],
        dq_rad_s=dq_rad_s[1:] * speed_scale,
        ddq_rad_s2=ddq_rad_s2[1:] * speed_scale**2,
    )


def approach_playback(
    current_q_rad: NDArray[np.float64],
    target_q_rad: NDArray[np.float64],
    joint_names: Sequence[str],
    *,
    speed_rad_s: float = APPROACH_SPEED_RAD_S,
    min_duration_s: float = APPROACH_MIN_DURATION_S,
    period_s: float = APPROACH_PERIOD_S,
) -> PlaybackPlan:
    """A rest-to-rest quintic from the current pose to the protocol start.

    The quintic ``10s**3 - 15s**4 + 6s**5`` starts and ends with zero velocity
    *and* zero acceleration, so the arm neither jerks on departure nor
    overshoots on arrival. Its peak velocity is ``1.875 * dq / T``, which is
    what sets the duration for the requested speed cap.
    """
    current = np.asarray(current_q_rad, dtype=np.float64)
    target = np.asarray(target_q_rad, dtype=np.float64)
    if current.shape != target.shape or current.ndim != 1:
        raise PlaybackError(
            f"approach needs matching 1-D poses, got {current.shape} and {target.shape}"
        )
    if len(current) != len(joint_names):
        raise PlaybackError(
            f"approach got {len(current)} joint values for {len(joint_names)} joints"
        )
    if speed_rad_s <= 0.0:
        raise PlaybackError(f"approach speed must be positive, got {speed_rad_s}")

    delta = target - current
    travel = float(np.max(np.abs(delta)))
    duration = max(min_duration_s, 1.875 * travel / speed_rad_s)
    steps = max(int(np.ceil(duration / period_s)), 2)
    # Excludes t=0 (the current pose, already reached) and lands exactly on T.
    time_s = np.linspace(0.0, duration, steps + 1)[1:]

    s = time_s / duration
    shape = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    rate = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    accel = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2

    return PlaybackPlan(
        label="approach",
        joint_names=tuple(str(name) for name in joint_names),
        time_s=time_s,
        q_rad=current[None, :] + shape[:, None] * delta[None, :],
        dq_rad_s=rate[:, None] * delta[None, :],
        ddq_rad_s2=accel[:, None] * delta[None, :],
    )


def reorder_to(
    values: Mapping[str, float], joint_names: Sequence[str]
) -> NDArray[np.float64]:
    """Pick out ``joint_names`` from a name-keyed reading, in protocol order.

    Joint-state publishers are free to order their arrays however they like
    and to carry extra joints (the gripper fingers). Reading by name is the
    only safe way to build a protocol-ordered vector.
    """
    missing = [name for name in joint_names if name not in values]
    if missing:
        raise PlaybackError(
            "joint state is missing " + ", ".join(missing) + "; "
            f"it reports {sorted(values)}"
        )
    return np.asarray([values[name] for name in joint_names], dtype=np.float64)


@dataclass(frozen=True)
class StartStateCheck:
    """How far the arm is from where a protocol expects it to be."""

    joint_names: tuple[str, ...]
    error_rad: NDArray[np.float64]
    tolerance_rad: float

    @property
    def worst_joint(self) -> int:
        return int(np.argmax(np.abs(self.error_rad)))

    @property
    def worst_error_rad(self) -> float:
        return float(np.abs(self.error_rad[self.worst_joint]))

    @property
    def satisfied(self) -> bool:
        return self.worst_error_rad <= self.tolerance_rad

    def describe(self) -> str:
        return (
            f"worst start-state error {self.worst_error_rad * 1e3:.1f} mrad on "
            f"{self.joint_names[self.worst_joint]} "
            f"(tolerance {self.tolerance_rad * 1e3:.1f} mrad)"
        )


def check_start_state(
    measured_q_rad: NDArray[np.float64],
    manifest: Mapping[str, object],
    *,
    tolerance_rad: float = START_TOLERANCE_RAD,
) -> StartStateCheck:
    """Compare a measured pose against the protocol's declared start state."""
    names = joint_order(manifest)
    expected = start_qpos(manifest)
    measured = np.asarray(measured_q_rad, dtype=np.float64)
    if measured.shape != expected.shape:
        raise PlaybackError(
            f"measured pose has {measured.shape} values, "
            f"protocol start state has {expected.shape}"
        )
    return StartStateCheck(
        joint_names=names,
        error_rad=measured - expected,
        tolerance_rad=tolerance_rad,
    )


def check_claimed_interfaces(
    claimed: Sequence[str],
    joint_names: Sequence[str],
    *,
    interface: str = "effort",
) -> None:
    """Assert a controller claims exactly the command interfaces we need.

    ``claimed`` is ``ControllerState.claimed_interfaces`` as reported by
    ``controller_manager/list_controllers``: entries like
    ``fer_joint1/effort``. ros2_control guarantees a command interface is
    claimed by at most one active controller, so this is simultaneously the
    "controller is configured for the right joints", "it is running in the
    mode we expect" and "nothing else is fighting it" check.
    """
    required = {f"{name}/{interface}" for name in joint_names}
    missing = sorted(required - set(claimed))
    if missing:
        raise PlaybackError(
            "controller does not command " + ", ".join(missing) + ". It claims "
            + (", ".join(sorted(claimed)) if claimed else "nothing")
            + f". Expected {interface}-mode control of every protocol joint."
        )
