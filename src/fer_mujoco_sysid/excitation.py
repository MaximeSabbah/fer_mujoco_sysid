"""Friction-family excitation protocols (P3 slice 1).

Excitation sweeps are jerk-limited trapezoids (S-curves): smoothstep
velocity ramps into a constant-velocity cruise, one bidirectional pass per
cruise speed, with holds at the stops and reversals at matched
configurations. Constant-velocity cruises make friction directly readable
(inertial torque vanishes, so measured torque is gravity plus friction) and
plottable per speed — the human-checkable assessment the acceptance rule
requires. Generation is deterministic from spec + seed, validated
fail-closed against FER limits with margins, and written as a compiled
bundle that must pass the frozen P1 motion-protocol validator and checksum
verification before the writer returns.
"""

from __future__ import annotations

import datetime
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.io import (
    content_sha256,
    save_arrays,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
)
from fer_mujoco_sysid.model import build_hydrax_arm_spec
from fer_mujoco_sysid.protocol import (
    ARRAY_UNITS,
    FRICTION_CRUISE,
    FRICTION_FAMILY,
    INERTIAL_EXCITATION,
    INERTIAL_FAMILY,
    PROTOCOL_FORMAT,
    PROTOCOL_ROLES,
    ROS_ARM_JOINT_NAMES,
    TRAIN_ROLE,
    AnalysisWindow,
    validate_protocol_manifest,
)
from fer_mujoco_sysid.protocol import (
    load_protocol_bundle as load_protocol_bundle,
)

GENERATOR_NAME = "fer-mujoco-sysid/excitation"
GENERATOR_VERSION = "0.2.0"

# FER joint limits (joint1..7) from the Franka Control Interface
# documentation. The generator's margins keep protocols well inside them;
# the hardware preflight (P6) revalidates against the robot's own limits.
FER_VELOCITY_LIMIT_RAD_S = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
FER_ACCELERATION_LIMIT_RAD_S2 = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
FER_JERK_LIMIT_RAD_S3 = np.array(
    [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]
)
FER_TORQUE_LIMIT_NM = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# The robot is bolted to a table, so the table surface is the plane through
# the robot base. Nothing may reach below it.
TABLE_HEIGHT_M = 0.0
TABLE_CLEARANCE_M = 0.02

_HOME_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# Smoothstep velocity ramp peaks: |ddq| <= 1.5 v / T_ramp,
# |dddq| <= 6 v / T_ramp^2.
_RAMP_PEAK_ACCELERATION = 1.5


class ProtocolLimitError(ValueError):
    """A compiled motion violates a declared FER limit or margin."""


@dataclass(frozen=True)
class FrictionProtocolSpec:
    """Deterministic description of one friction-family protocol.

    Each entry of ``cruise_speeds_rad_s`` produces one bidirectional
    sweep pass whose largest-amplitude joint cruises at that speed
    (smaller-amplitude joints cruise proportionally slower on the shared
    schedule). Low speeds are deliberately over-represented: that is the
    regime of the pregrasp standoff.
    """

    protocol_id: str
    seed: int
    family_id: str = FRICTION_FAMILY
    role: str = TRAIN_ROLE
    amplitudes_rad: tuple[float, ...] = (0.3,) * 7
    amplitude_jitter: float = 0.1
    cruise_speeds_rad_s: tuple[float, ...] = (0.05, 0.15, 0.4)
    cruise_acceleration_rad_s2: float = 1.0
    transit_speed_rad_s: float = 0.3
    hold_s: float = 1.0
    settle_s: float = 0.5
    sample_period_s: float = 0.001
    home_qpos: tuple[float, ...] = _HOME_QPOS
    position_margin_rad: float = 0.05
    limit_margin_fraction: float = 0.2


@dataclass(frozen=True)
class Segment:
    segment_id: str
    kind: str
    start_index: int
    end_index_exclusive: int
    analysis_eligible: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class CompiledProtocol:
    """Compiled protocol: dense arrays, segments, and fit-eligible windows."""

    spec: FrictionProtocolSpec | InertialProtocolSpec
    time_s: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    dq_rad_s: NDArray[np.float64]
    ddq_rad_s2: NDArray[np.float64]
    segments: tuple[Segment, ...]
    analysis_windows: tuple[AnalysisWindow, ...]

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        return {
            "time_s": self.time_s,
            "q_rad": self.q_rad,
            "dq_rad_s": self.dq_rad_s,
            "ddq_rad_s2": self.ddq_rad_s2,
        }


def _validate_protocol_identity(
    spec: FrictionProtocolSpec | InertialProtocolSpec, *, expected_family: str
) -> None:
    """Require identity metadata from the typed spec, never from its name."""
    if not spec.protocol_id.strip():
        raise ValueError("protocol_id cannot be empty")
    if spec.family_id != expected_family:
        raise ValueError(
            f"{type(spec).__name__} family must be {expected_family!r}, "
            f"got {spec.family_id!r}"
        )
    if spec.role not in PROTOCOL_ROLES:
        raise ValueError(
            f"protocol role must be one of {PROTOCOL_ROLES}, got {spec.role!r}"
        )


def _scurve_profile(
    distance: float, cruise_speed: float, acceleration: float, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalized S-curve move of *distance* > 0: (p, dp, ddp) sampled at dt.

    Smoothstep velocity ramp (zero boundary acceleration, bounded jerk) into
    a constant-velocity cruise and a mirrored ramp out. The ramp covers
    ``cruise_speed * T_ramp / 2`` of distance on each side.
    """
    ramp_time = _RAMP_PEAK_ACCELERATION * cruise_speed / acceleration
    cruise_distance = distance - cruise_speed * ramp_time
    if cruise_distance <= 0:
        raise ValueError(
            f"move of {distance:.3f} rad is too short for a "
            f"{cruise_speed:.3f} rad/s cruise with "
            f"{acceleration:.2f} rad/s^2 ramps; lower the speed or raise "
            "the amplitude"
        )
    cruise_time = cruise_distance / cruise_speed
    total = 2 * ramp_time + cruise_time
    steps = max(int(round(total / dt)), 4)
    t = np.arange(steps) * dt

    position = np.empty(steps)
    velocity = np.empty(steps)
    acceleration_out = np.empty(steps)

    ramp_in = t < ramp_time
    cruise = (t >= ramp_time) & (t < ramp_time + cruise_time)
    ramp_out = ~ramp_in & ~cruise

    u = t[ramp_in] / ramp_time
    velocity[ramp_in] = cruise_speed * (3 * u**2 - 2 * u**3)
    acceleration_out[ramp_in] = cruise_speed * 6 * (u - u**2) / ramp_time
    position[ramp_in] = cruise_speed * ramp_time * (u**3 - 0.5 * u**4)

    velocity[cruise] = cruise_speed
    acceleration_out[cruise] = 0.0
    position[cruise] = cruise_speed * (0.5 * ramp_time + (t[cruise] - ramp_time))

    w = (t[ramp_out] - ramp_time - cruise_time) / ramp_time
    w = np.clip(w, 0.0, 1.0)
    velocity[ramp_out] = cruise_speed * (1.0 - (3 * w**2 - 2 * w**3))
    acceleration_out[ramp_out] = -cruise_speed * 6 * (w - w**2) / ramp_time
    position[ramp_out] = cruise_speed * (
        0.5 * ramp_time + cruise_time + ramp_time * (w - w**3 + 0.5 * w**4)
    )

    return position, velocity, acceleration_out


def generate_friction_protocol(
    spec: FrictionProtocolSpec,
    model: mujoco.MjModel,
    model_path: str | Path | None = None,
) -> CompiledProtocol:
    """Compile and fail-closed-validate one friction protocol against *model*."""
    _validate_protocol_identity(spec, expected_family=FRICTION_FAMILY)
    if not spec.cruise_speeds_rad_s:
        raise ValueError("at least one cruise speed is required")
    dt = spec.sample_period_s
    home = np.asarray(spec.home_qpos, dtype=np.float64)
    rng = np.random.default_rng(spec.seed)
    jitter = 1.0 + spec.amplitude_jitter * rng.uniform(-1.0, 1.0, size=7)
    amplitudes = np.asarray(spec.amplitudes_rad, dtype=np.float64) * jitter
    largest = float(np.max(np.abs(amplitudes)))

    pieces: list[tuple[str, str, NDArray, NDArray, NDArray, bool, str | None]] = []
    plateau_by_segment: dict[str, tuple[int, int, tuple[float, ...]]] = {}

    def hold_piece(
        kind: str, position: NDArray[np.float64], duration: float, label: str
    ) -> None:
        steps = max(int(round(duration / dt)), 1)
        eligible = False
        reason = (
            "stationary hold is outside sliding-friction analysis"
            if kind == "hold"
            else "startup or shutdown transient"
        )
        pieces.append(
            (
                label,
                kind,
                np.repeat(position[None, :], steps, axis=0),
                np.zeros((steps, 7)),
                np.zeros((steps, 7)),
                eligible,
                reason,
            )
        )

    def sweep_piece(
        label: str,
        kind: str,
        start_offset: float,
        end_offset: float,
        cruise_speed: float,
    ) -> None:
        span = abs(end_offset - start_offset)
        profile, dprofile, ddprofile = _scurve_profile(
            span * largest, cruise_speed, spec.cruise_acceleration_rad_s2, dt
        )
        # Shared schedule: each joint moves by its own amplitude, cruising
        # proportionally slower than the largest-amplitude joint.
        direction = np.sign(end_offset - start_offset)
        per_joint = direction * amplitudes / largest
        start = home + start_offset * amplitudes
        eligible = kind == "excitation"
        reason = None if eligible else "transit back to home"
        if eligible:
            cruise = (dprofile == cruise_speed) & (ddprofile == 0.0)
            cruise_indices = np.flatnonzero(cruise)
            if not len(cruise_indices) or not np.all(np.diff(cruise_indices) == 1):
                raise RuntimeError(
                    f"{label} did not compile to one constant-velocity plateau"
                )
            plateau_by_segment[label] = (
                int(cruise_indices[0]),
                int(cruise_indices[-1]) + 1,
                tuple(float(value) for value in cruise_speed * per_joint),
            )
        pieces.append(
            (
                label,
                kind,
                start[None, :] + profile[:, None] * per_joint[None, :],
                dprofile[:, None] * per_joint[None, :],
                ddprofile[:, None] * per_joint[None, :],
                eligible,
                reason,
            )
        )

    hold_piece("settle", home, spec.settle_s, "settle_start")
    offset = 0.0
    for speed in spec.cruise_speeds_rad_s:
        tag = f"{int(round(speed * 1000))}mrad_s"
        for target in (1.0, -1.0):
            sweep_piece(
                f"sweep_{tag}_to_{'pos' if target > 0 else 'neg'}",
                "excitation",
                offset,
                target,
                speed,
            )
            offset = target
            hold_piece(
                "hold",
                home + offset * amplitudes,
                spec.hold_s,
                f"hold_{tag}_{'pos' if offset > 0 else 'neg'}",
            )
    sweep_piece("return_home", "return", offset, 0.0, spec.transit_speed_rad_s)
    hold_piece("settle", home, spec.settle_s, "settle_end")

    segments: list[Segment] = []
    analysis_windows: list[AnalysisWindow] = []
    q_parts, dq_parts, ddq_parts = [], [], []
    cursor = 0
    for label, kind, q, dq, ddq, eligible, reason in pieces:
        q_parts.append(q)
        dq_parts.append(dq)
        ddq_parts.append(ddq)
        segments.append(
            Segment(
                segment_id=label,
                kind=kind,
                start_index=cursor,
                end_index_exclusive=cursor + len(q),
                analysis_eligible=eligible,
                exclusion_reason=reason,
            )
        )
        plateau = plateau_by_segment.get(label)
        if plateau is not None:
            relative_start, relative_stop, velocity = plateau
            analysis_windows.append(
                AnalysisWindow(
                    window_id=f"{label}_cruise",
                    kind=FRICTION_CRUISE,
                    parent_segment_id=label,
                    start_index=cursor + relative_start,
                    end_index_exclusive=cursor + relative_stop,
                    nominal_velocity_rad_s=velocity,
                )
            )
        cursor += len(q)

    q_all = np.vstack(q_parts)
    dq_all = np.vstack(dq_parts)
    ddq_all = np.vstack(ddq_parts)
    time_s = (np.arange(len(q_all), dtype=np.float64) * dt).astype("<f8")

    validate_protocol_limits(
        model,
        q_all,
        dq_all,
        ddq_all,
        dt,
        position_margin_rad=spec.position_margin_rad,
        limit_margin_fraction=spec.limit_margin_fraction,
    )
    if model_path is not None:
        validate_workspace_clearance(model_path, q_all)
    return CompiledProtocol(
        spec=spec,
        time_s=time_s,
        q_rad=q_all.astype("<f8"),
        dq_rad_s=dq_all.astype("<f8"),
        ddq_rad_s2=ddq_all.astype("<f8"),
        segments=tuple(segments),
        analysis_windows=tuple(analysis_windows),
    )


def validate_protocol_limits(
    model: mujoco.MjModel,
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    ddq: NDArray[np.float64],
    dt: float,
    *,
    position_margin_rad: float,
    limit_margin_fraction: float,
    check_consistency: bool = True,
) -> None:
    """Raise :class:`ProtocolLimitError` on any limit or margin violation."""
    scale = 1.0 - limit_margin_fraction
    lower = model.jnt_range[:7, 0] + position_margin_rad
    upper = model.jnt_range[:7, 1] - position_margin_rad

    def report(quantity: str, values: NDArray[np.float64], bound) -> None:
        margin = np.abs(values).max(axis=0) - bound
        if (margin > 0).any():
            joint = int(np.argmax(margin))
            raise ProtocolLimitError(
                f"{quantity} limit violated on joint{joint + 1}: "
                f"max {np.abs(values[:, joint]).max():.4f} exceeds "
                f"{np.asarray(bound).reshape(-1)[joint]:.4f}"
            )

    outside = np.maximum(lower[None, :] - q, q - upper[None, :])
    if (outside > 0).any():
        joint = int(np.argmax(outside.max(axis=0)))
        raise ProtocolLimitError(
            f"position range (with {position_margin_rad} rad margin) violated "
            f"on joint{joint + 1}"
        )
    report("velocity", dq, FER_VELOCITY_LIMIT_RAD_S * scale)
    report("acceleration", ddq, FER_ACCELERATION_LIMIT_RAD_S2 * scale)
    jerk = np.diff(ddq, axis=0) / dt
    report("jerk", jerk, FER_JERK_LIMIT_RAD_S3 * scale)

    consistency = (
        np.abs(np.diff(q, axis=0) - 0.5 * dt * (dq[:-1] + dq[1:])).max()
        if check_consistency
        else 0.0
    )
    # The trapezoid rule is exact only up to O(dt^3 * |dddq|); scale the
    # tolerance with the observed jerk so coarse (e.g. 100 Hz) grids of a
    # correct trajectory still pass while wrong arrays still fail.
    jerk_bound = float(np.abs(jerk).max()) if len(jerk) else 0.0
    consistency_tolerance = max(1e-6, dt**3 * jerk_bound / 3.0)
    if consistency > consistency_tolerance:
        raise ProtocolLimitError(
            f"position/velocity arrays are inconsistent (max {consistency:.2e} "
            f"rad against the trapezoid integral, tolerance "
            f"{consistency_tolerance:.2e})"
        )

    data = mujoco.MjData(model)
    torque_bound = FER_TORQUE_LIMIT_NM * scale
    for index in range(len(q)):
        data.qpos[:] = q[index]
        data.qvel[:] = dq[index]
        data.qacc[:] = ddq[index]
        mujoco.mj_inverse(model, data)
        excess = np.abs(data.qfrc_inverse[:7]) - torque_bound
        if (excess > 0).any():
            joint = int(np.argmax(excess))
            raise ProtocolLimitError(
                f"predicted torque exceeds margin on joint{joint + 1} at "
                f"sample {index}: {abs(data.qfrc_inverse[joint]):.2f} Nm > "
                f"{torque_bound[joint]:.2f} Nm"
            )


def _clearance_model(model_path: str | Path) -> mujoco.MjModel:
    """Identification geometry plus the table, with contacts enabled.

    The identification model runs contact-free (it never touches anything),
    so workspace clearance is checked on a separate copy that has the table
    plane and collision detection switched back on.
    """
    spec = build_hydrax_arm_spec(model_path)
    spec.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    table = spec.worldbody.add_geom()
    table.name = "table_surface"
    table.type = mujoco.mjtGeom.mjGEOM_PLANE
    table.size = [3.0, 3.0, 0.05]
    table.pos = [0.0, 0.0, TABLE_HEIGHT_M - TABLE_CLEARANCE_M]
    return spec.compile()


def validate_workspace_clearance(
    model_path: str | Path,
    q: NDArray[np.float64],
    *,
    stride: int = 5,
) -> None:
    """Raise :class:`ProtocolLimitError` if the arm reaches the table.

    Checked kinematically on every ``stride``-th knot: the trajectories are
    smooth and slow relative to the knot rate, so intermediate poses cannot
    dip through the plane between checked samples.
    """
    model = _clearance_model(model_path)
    data = mujoco.MjData(model)
    for index in range(0, len(q), stride):
        data.qpos[:] = q[index]
        mujoco.mj_forward(model, data)
        if data.ncon:
            geom = model.geom(int(data.contact[0].geom2)).name or "arm geom"
            raise ProtocolLimitError(
                f"protocol reaches the table at sample {index}: {geom} is "
                f"within {TABLE_CLEARANCE_M} m of the base plane"
            )


def _nominal_hydrax_source(workspace_root: str | Path | None) -> dict[str, str]:
    """Return the provenance-pinned source model, verifying the file hash."""
    contract_path = Path(__file__).resolve().parents[2] / (
        "contracts/nominal_sources.toml"
    )
    contract = tomllib.loads(contract_path.read_text())["models"]["hydrax"]

    from fer_mujoco_sysid.model import resolve_model_paths

    hydrax_path = resolve_model_paths(workspace_root).require().hydrax
    actual = sha256_file(hydrax_path)
    if actual != contract["sha256"]:
        raise ValueError(
            "hydrax nominal model hash drifted from "
            f"contracts/nominal_sources.toml: {actual} != {contract['sha256']}"
        )
    return {
        "repository": contract["repository"],
        "revision": contract["revision"],
        "path": contract["path"],
        "sha256": contract["sha256"],
    }


def _manifest(
    compiled: CompiledProtocol,
    model: mujoco.MjModel,
    *,
    created_at: str,
    workspace_root: str | Path | None,
) -> dict[str, object]:
    spec = compiled.spec
    hand = model.body("hand")
    arrays = compiled.arrays()
    return {
        "format": PROTOCOL_FORMAT,
        "protocol_id": spec.protocol_id,
        "family": spec.family_id,
        "role": spec.role,
        "created_at": created_at,
        "content_sha256": "",
        "description": (
            (
                "Jerk-limited trapezoidal sweeps with constant-velocity "
                "cruises at several low speeds, holds at the stops, and "
                "reversals at matched configurations, for "
                "frictionloss/damping identification."
            )
            if isinstance(spec, FrictionProtocolSpec)
            else (
                "Per-joint finite Fourier series at distinct base "
                "frequencies, scaled to the kinematic limits and selected "
                "for regressor conditioning, for armature and inertial "
                "identification."
            )
        ),
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "seed": spec.seed,
            "spec": asdict(spec),
        },
        "playback": {
            # D021: protocols are played by a position-mode trajectory
            # controller; effort is recorded, never commanded.
            "command_interface": "joint_trajectory",
            "joint_order": list(ROS_ARM_JOINT_NAMES),
            "sample_period_s": spec.sample_period_s,
            "duration_s": float(compiled.time_s[-1]),
            "samples": int(len(compiled.time_s)),
        },
        "arrays": {
            key: {
                "unit": ARRAY_UNITS[key],
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
            for key, value in arrays.items()
        },
        "segments": [
            {
                "segment_id": segment.segment_id,
                "kind": segment.kind,
                "start_index": segment.start_index,
                "end_index_exclusive": segment.end_index_exclusive,
                "analysis_eligible": segment.analysis_eligible,
                **(
                    {"exclusion_reason": segment.exclusion_reason}
                    if segment.exclusion_reason
                    else {}
                ),
            }
            for segment in compiled.segments
        ],
        "analysis_windows": [asdict(window) for window in compiled.analysis_windows],
        "start_state": {
            "q_rad": compiled.q_rad[0].tolist(),
            "dq_rad_s": compiled.dq_rad_s[0].tolist(),
        },
        "end_state": {
            "q_rad": compiled.q_rad[-1].tolist(),
            "dq_rad_s": compiled.dq_rad_s[-1].tolist(),
        },
        "source_model": _nominal_hydrax_source(workspace_root),
        "payload": {
            "end_effector": "fer-hand",
            "mass_kg": float(model.body_subtreemass[hand.id]),
            "com_m": np.asarray(hand.ipos).tolist(),
        },
        "limits": {
            "position_margin_rad": spec.position_margin_rad,
            "limit_margin_fraction": spec.limit_margin_fraction,
            "velocity_rad_s": FER_VELOCITY_LIMIT_RAD_S.tolist(),
            "acceleration_rad_s2": FER_ACCELERATION_LIMIT_RAD_S2.tolist(),
            "jerk_rad_s3": FER_JERK_LIMIT_RAD_S3.tolist(),
            "torque_Nm": FER_TORQUE_LIMIT_NM.tolist(),
        },
    }


def write_protocol_bundle(
    compiled: CompiledProtocol,
    protocols_root: str | Path,
    *,
    model: mujoco.MjModel,
    workspace_root: str | Path | None = None,
    created_at: str | None = None,
) -> Path:
    """Write one content-addressed bundle; fail closed unless it validates."""
    root = Path(protocols_root) / compiled.spec.protocol_id
    if root.exists():
        raise FileExistsError(f"protocol bundle already exists (immutable): {root}")
    if created_at is None:
        created_at = (
            datetime.datetime.now(datetime.UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    arrays = compiled.arrays()
    manifest = _manifest(
        compiled,
        model,
        created_at=created_at,
        workspace_root=workspace_root,
    )
    manifest["content_sha256"] = content_sha256(manifest, arrays)
    validate_protocol_manifest(manifest, arrays)

    root.mkdir(parents=True)
    save_arrays(root / "desired.npz", arrays)
    write_json(root / "protocol.json", manifest)
    write_checksums(root)
    verify_checksums(root)
    return root


# --------------------------------------------------------------------------
# Inertial family: per-joint finite Fourier series
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InertialProtocolSpec:
    """A periodic, per-joint finite Fourier excitation (Swevers-style).

    Each joint follows its own truncated Fourier series about ``home_qpos``,
    with **a different base frequency per joint**, so the joints do not move
    proportionally. That decorrelation is the whole point: identifying link
    inertias requires the links to move independently, which the friction
    family (one shared schedule) deliberately does not do.

    The series starts and ends at rest at ``home_qpos`` by construction, but
    only because the base frequencies are **commensurate**: every joint's
    frequency is an integer multiple of ``1 / duration``, so every joint
    completes a whole number of cycles and lands back where it started.
    Incommensurate frequencies leave each joint wherever its own cycle
    happened to reach, and the trajectory then ends with a step back to the
    home pose — which a trajectory controller answers with a torque spike.
    :func:`_inertial_candidate` refuses to build such a spec.

    Distinct integer multiples still decorrelate the joints, which is the
    reason for per-joint frequencies in the first place.
    """

    protocol_id: str
    seed: int
    family_id: str = INERTIAL_FAMILY
    role: str = TRAIN_ROLE
    harmonics: int = 5
    #: Multiples 2..8 of the 0.05 Hz fundamental implied by
    #: ``periods / min(base_frequency_hz)`` = 20 s.
    base_frequency_hz: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    )
    periods: int = 2
    amplitude_rad: tuple[float, ...] = (0.45, 0.35, 0.45, 0.35, 0.6, 0.6, 0.7)
    sample_period_s: float = 0.01
    settle_s: float = 0.5
    home_qpos: tuple[float, ...] = _HOME_QPOS
    position_margin_rad: float = 0.05
    limit_margin_fraction: float = 0.2
    candidates: int = 24


def _fourier_series(
    coefficients: NDArray[np.float64],
    base_frequency: float,
    times: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Position, velocity and acceleration of one joint's Fourier series.

    ``coefficients`` holds ``[a_1..a_H, b_1..b_H]`` for
    ``sum_h a_h sin(h w t) + b_h (cos(h w t) - 1)``. The ``-1`` makes the
    position start at zero; every term is periodic, so the motion returns to
    rest at the end of each period.
    """
    harmonics = len(coefficients) // 2
    omega = 2.0 * np.pi * base_frequency
    position = np.zeros_like(times)
    velocity = np.zeros_like(times)
    acceleration = np.zeros_like(times)
    for index in range(harmonics):
        order = index + 1
        rate = order * omega
        sine, cosine = np.sin(rate * times), np.cos(rate * times)
        a, b = coefficients[index], coefficients[harmonics + index]
        position += a * sine + b * (cosine - 1.0)
        velocity += rate * (a * cosine - b * sine)
        acceleration += rate**2 * (-a * sine - b * cosine)
    return position, velocity, acceleration


def _inertial_candidate(
    spec: InertialProtocolSpec, rng: np.random.Generator
) -> tuple[NDArray[np.float64], ...]:
    """One random candidate, scaled so every kinematic limit is respected.

    A Fourier series with several harmonics reaches high velocities and
    accelerations for a modest position amplitude, so the scale of each
    joint is set by whichever of position, velocity or acceleration binds
    first. Candidates are therefore feasible by construction and the search
    can spend its draws on conditioning rather than on rejection.
    """
    dt = spec.sample_period_s
    duration = spec.periods / min(spec.base_frequency_hz)
    cycles = np.asarray(spec.base_frequency_hz, dtype=np.float64) * duration
    incommensurate = np.abs(cycles - np.round(cycles)) > 1e-9
    if incommensurate.any():
        joint = int(np.argmax(incommensurate))
        raise ValueError(
            f"base_frequency_hz[{joint}] = {spec.base_frequency_hz[joint]} Hz "
            f"completes {cycles[joint]:.4f} cycles over the {duration:g} s "
            "protocol, not a whole number. The joint would stop mid-cycle and "
            "the trajectory would end with a step back to the home pose. Use "
            f"multiples of the {1.0 / duration:g} Hz fundamental."
        )

    # Include the endpoint t = duration. There the series is exactly back at
    # its start — position, velocity and acceleration all zero — so the
    # trailing settle segment joins on continuously.
    steps = int(round(duration / dt))
    times = np.arange(steps + 1) * dt

    q = np.empty((len(times), 7))
    dq = np.empty((len(times), 7))
    ddq = np.empty((len(times), 7))
    home = np.asarray(spec.home_qpos, dtype=np.float64)
    orders = np.arange(1, spec.harmonics + 1, dtype=np.float64)
    for joint in range(7):
        sine = rng.normal(size=spec.harmonics)
        cosine = rng.normal(size=spec.harmonics)
        # Start and end at rest: velocity(0) ~ sum(h*a_h) and
        # acceleration(0) ~ sum(h^2*b_h) must vanish. Project them out; the
        # series is periodic, so the same holds at the end.
        sine = sine - orders * np.dot(orders, sine) / np.dot(orders, orders)
        weights = orders**2
        cosine = cosine - weights * np.dot(weights, cosine) / np.dot(weights, weights)
        coefficients = np.concatenate([sine, cosine])
        position, velocity, acceleration = _fourier_series(
            coefficients, spec.base_frequency_hz[joint], times
        )
        margin = 1.0 - spec.limit_margin_fraction
        limits = (
            spec.amplitude_rad[joint] / max(np.abs(position).max(), 1e-12),
            margin
            * FER_VELOCITY_LIMIT_RAD_S[joint]
            / max(np.abs(velocity).max(), 1e-12),
            margin
            * FER_ACCELERATION_LIMIT_RAD_S2[joint]
            / max(np.abs(acceleration).max(), 1e-12),
        )
        scale = float(min(limits))
        q[:, joint] = home[joint] + scale * position
        dq[:, joint] = scale * velocity
        ddq[:, joint] = scale * acceleration
    return times, q, dq, ddq


def generate_inertial_protocol(
    spec: InertialProtocolSpec,
    model: mujoco.MjModel,
    model_path: str | Path | None = None,
) -> CompiledProtocol:
    """Search random Fourier candidates for the best-conditioned feasible one.

    Candidates that violate a limit are discarded; among the survivors the
    one whose per-joint friction+armature regressor is best conditioned is
    kept. Selection therefore optimizes excitation quality directly, which is
    what a condition-number-based design is supposed to do.
    """
    from fer_mujoco_sysid.diagnostics import friction_regressor_report

    _validate_protocol_identity(spec, expected_family=INERTIAL_FAMILY)
    if spec.candidates < 1:
        raise ValueError("candidates must be at least 1")
    rng = np.random.default_rng(spec.seed)
    settle = max(int(round(spec.settle_s / spec.sample_period_s)), 1)
    home = np.asarray(spec.home_qpos, dtype=np.float64)

    best: tuple[float, NDArray[np.float64], ...] | None = None
    failures: list[str] = []
    for _ in range(spec.candidates):
        times, q, dq, ddq = _inertial_candidate(spec, rng)
        pad = np.zeros((settle, 7))
        rest = np.repeat(home[None, :], settle, axis=0)
        q_all = np.vstack([rest, q, rest])
        dq_all = np.vstack([pad, dq, pad])
        ddq_all = np.vstack([pad, ddq, pad])
        try:
            validate_protocol_limits(
                model,
                q_all,
                dq_all,
                ddq_all,
                spec.sample_period_s,
                position_margin_rad=spec.position_margin_rad,
                limit_margin_fraction=spec.limit_margin_fraction,
            )
        except ProtocolLimitError as error:
            failures.append(str(error))
            continue
        score = friction_regressor_report(dq_all, ddq_all).worst_condition_number
        if best is None or score < best[0]:
            best = (score, q_all, dq_all, ddq_all)

    if best is None:
        raise ProtocolLimitError(
            f"no feasible inertial candidate in {spec.candidates} draws; "
            f"last failure: {failures[-1] if failures else 'unknown'}"
        )

    _, q_all, dq_all, ddq_all = best
    if model_path is not None:
        validate_workspace_clearance(model_path, q_all)
    samples = len(q_all)
    time_s = (np.arange(samples, dtype=np.float64) * spec.sample_period_s).astype("<f8")
    segments = (
        Segment("settle_start", "settle", 0, settle, False, "startup transient"),
        Segment("fourier", "excitation", settle, samples - settle, True, None),
        Segment(
            "settle_end",
            "settle",
            samples - settle,
            samples,
            False,
            "shutdown transient",
        ),
    )
    return CompiledProtocol(
        spec=spec,
        time_s=time_s,
        q_rad=q_all.astype("<f8"),
        dq_rad_s=dq_all.astype("<f8"),
        ddq_rad_s2=ddq_all.astype("<f8"),
        segments=segments,
        analysis_windows=(
            AnalysisWindow(
                window_id="fourier_excitation",
                kind=INERTIAL_EXCITATION,
                parent_segment_id="fourier",
                start_index=settle,
                end_index_exclusive=samples - settle,
            ),
        ),
    )
