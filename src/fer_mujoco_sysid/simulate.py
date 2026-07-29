"""Play the campaign in-process and write recordings, with no ROS involved.

The point is a controlled comparison. ``output/mujoco_ros/`` comes from the
real ROS stack — trajectory controller, 500 Hz state publication, message
timestamps, bag recording — driving a simulated plant. This module drives the
*same* plant with the *same* control law at the model timestep, writes the
*same* recording format, and is fitted by the *same* ``identify``. What is
left between the two directories is exactly the cost of going through ROS.

It is also the fastest way to exercise the whole fitting chain: no ROS
environment, no launch, no 45 seconds of wall clock per protocol.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.campaign import (
    CAMPAIGN,
    compile_protocol,
    repository_root,
)
from fer_mujoco_sysid.dataset import BOUND_RECORDING_FORMAT, write_recording
from fer_mujoco_sysid.excitation import FER_TORQUE_LIMIT_NM
from fer_mujoco_sysid.identify import _fitting_spec
from fer_mujoco_sysid.model import resolve_model_paths
from fer_mujoco_sysid.protocol import (
    DEFAULT_ANALYSIS_EDGE_GUARD_S,
    FRICTION_FAMILY,
    INERTIAL_FAMILY,
    ROS_ARM_JOINT_NAMES,
    alignment_from_start_time,
    friction_cruise_mask,
    inertial_excitation_mask,
    load_protocol_bundle,
    map_analysis_windows,
    map_protocol_segments,
    protocol_family,
    protocol_role,
)

#: The trajectory controller's gains, so the simulated backend differs from
#: the ROS one only in the plumbing and not in the control law.
JTC_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
JTC_KD = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0])

#: Optional hardware-like perturbations. The release-critical simulation
#: oracle uses zero noise; these values are retained for a separate robustness
#: run and must never replace the numerical-closure gate.
HARDWARE_LIKE_POSITION_NOISE_RAD = 2e-5
HARDWARE_LIKE_VELOCITY_NOISE_RAD_S = 2e-3

SYNTHETIC_MARKER_FORMAT = "fer-mujoco-sysid/synthetic-protocol-event@1"
SYNTHETIC_MARKER_TOPIC = "in_process_protocol_boundary"
SYNTHETIC_START_STAMP_NS = 1_000_000_000


def play_protocol(
    model: mujoco.MjModel,
    compiled,
    *,
    seed: int,
    position_noise_rad: float = 0.0,
    velocity_noise_rad_s: float = 0.0,
) -> dict[str, NDArray[np.float64]]:
    """Track a compiled protocol under the trajectory controller's law."""
    if position_noise_rad < 0.0 or velocity_noise_rad_s < 0.0:
        raise ValueError("simulation noise standard deviations must be nonnegative")
    step = float(model.opt.timestep)
    protocol_time = compiled.time_s
    steps = int(round(protocol_time[-1] / step))
    times = np.arange(steps) * step

    q_desired = np.column_stack(
        [np.interp(times, protocol_time, compiled.q_rad[:, j]) for j in range(7)]
    )
    dq_desired = np.column_stack(
        [np.interp(times, protocol_time, compiled.dq_rad_s[:, j]) for j in range(7)]
    )

    data = mujoco.MjData(model)
    data.qpos[:] = compiled.q_rad[0]
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(seed)
    q = np.empty((steps, 7))
    dq = np.empty((steps, 7))
    tau = np.empty((steps, 7))
    for k in range(steps):
        q_measured = data.qpos[:7] + position_noise_rad * rng.standard_normal(7)
        dq_measured = data.qvel[:7] + velocity_noise_rad_s * rng.standard_normal(7)
        command = np.clip(
            JTC_KP * (q_desired[k] - q_measured)
            + JTC_KD * (dq_desired[k] - dq_measured),
            -FER_TORQUE_LIMIT_NM,
            FER_TORQUE_LIMIT_NM,
        )
        q[k] = q_measured
        dq[k] = dq_measured
        tau[k] = command
        data.ctrl[:7] = command
        mujoco.mj_step(model, data)

    return {
        "time_s": times,
        "q_rad": q,
        "dq_rad_s": dq,
        "tau_cmd_Nm": tau,
        "q_desired_rad": q_desired,
    }


def _reviewed_protocol(compiled) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load canonical metadata and prove it describes the motion being played."""
    protocol_root = repository_root() / "protocols" / compiled.spec.protocol_id
    manifest, desired = load_protocol_bundle(protocol_root)
    if protocol_family(manifest) != compiled.spec.family_id:
        raise RuntimeError(
            f"{compiled.spec.protocol_id}: compiled and reviewed families differ"
        )
    if protocol_role(manifest) != compiled.spec.role:
        raise RuntimeError(
            f"{compiled.spec.protocol_id}: compiled and reviewed roles differ"
        )
    for name, values in compiled.arrays().items():
        if not np.array_equal(values, desired[name]):
            raise RuntimeError(
                f"{compiled.spec.protocol_id}: compiled {name} differs from "
                "the reviewed canonical protocol"
            )
    return manifest, desired


def _mapped_row_metadata(
    protocol: dict[str, object],
    desired: dict[str, np.ndarray],
    recording_time_s: NDArray[np.float64],
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    alignment = alignment_from_start_time(
        protocol,
        start_time_s=0.0,
        speed_scale=1.0,
    )
    segments = map_protocol_segments(protocol, desired, alignment)
    windows = map_analysis_windows(
        protocol,
        desired,
        alignment,
        edge_guard_s=DEFAULT_ANALYSIS_EDGE_GUARD_S,
    )

    segment_index = np.full(len(recording_time_s), -1, dtype=np.int16)
    for index, segment in enumerate(segments):
        segment_index[segment.mask(recording_time_s)] = index
    if np.any(segment_index < 0):
        missing = int(np.count_nonzero(segment_index < 0))
        raise RuntimeError(
            f"{protocol['protocol_id']}: {missing} simulated rows are not bound "
            "to a protocol segment"
        )

    family = protocol_family(protocol)
    if family == FRICTION_FAMILY:
        analysis_eligible = friction_cruise_mask(
            protocol,
            desired,
            recording_time_s,
            alignment,
        )
    elif family == INERTIAL_FAMILY:
        analysis_eligible = inertial_excitation_mask(
            protocol,
            desired,
            recording_time_s,
            alignment,
        )
    else:  # pragma: no cover - protocol_family is fail-closed
        raise RuntimeError(f"unsupported protocol family {family!r}")
    if not np.any(analysis_eligible):
        raise RuntimeError(f"{protocol['protocol_id']}: no fit-eligible rows")

    return (
        [asdict(interval) for interval in segments],
        [asdict(interval) for interval in windows],
        segment_index,
        analysis_eligible,
    )


def _bound_recording(
    protocol: dict[str, object],
    desired: dict[str, np.ndarray],
    arrays: dict[str, NDArray[np.float64]],
    *,
    position_noise_rad: float,
    velocity_noise_rad_s: float,
    simulation_truth: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Bind complete in-process arrays to their reviewed protocol."""
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    mapped_segments, mapped_windows, segment_index, analysis_eligible = (
        _mapped_row_metadata(protocol, desired, time_s)
    )
    payload = {
        **arrays,
        "protocol_segment_index": segment_index,
        "analysis_eligible": analysis_eligible,
    }

    duration_s = float(protocol["playback"]["duration_s"])
    end_stamp_ns = SYNTHETIC_START_STAMP_NS + int(round(duration_s * 1e9))
    selected_last_stamp_ns = SYNTHETIC_START_STAMP_NS + int(
        round(float(time_s[-1]) * 1e9)
    )
    manifest: dict[str, object] = {
        "format": BOUND_RECORDING_FORMAT,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "mujoco",
        "speed_scale": 1.0,
        "torque_source": (
            "in-process trajectory-controller output request; physical "
            "post-limiter/applied status is not asserted"
        ),
        "torque_channels": {
            "tau_cmd_Nm": {
                "source_topic": "in-process MuJoCo controller loop",
                "message_field": "data.ctrl",
                "meaning": (
                    "PD trajectory-controller request after clipping at the "
                    "declared FER torque bounds"
                ),
                "clock": "MuJoCo model timestep",
                "rate_limiting": "software magnitude clip; no slew limiter",
                "application_status": "not_asserted",
            }
        },
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "family": protocol["family"],
            "role": protocol["role"],
            "content_sha256": protocol["content_sha256"],
            "duration_s": duration_s,
            "segments": protocol["segments"],
            "analysis_windows": protocol["analysis_windows"],
        },
        "protocol_timing": {
            "marker_format": SYNTHETIC_MARKER_FORMAT,
            "marker_topic": SYNTHETIC_MARKER_TOPIC,
            "start_stamp_ns": SYNTHETIC_START_STAMP_NS,
            "end_stamp_ns": end_stamp_ns,
            "selected_first_stamp_ns": SYNTHETIC_START_STAMP_NS,
            "selected_last_stamp_ns": selected_last_stamp_ns,
            "marked_duration_s": duration_s,
            "speed_scale": 1.0,
            "recording_time_origin": "synthetic protocol start marker",
        },
        "mapped_segments": mapped_segments,
        "mapped_analysis_windows": mapped_windows,
        "analysis_policy": {
            "edge_guard_s": DEFAULT_ANALYSIS_EDGE_GUARD_S,
            "protocol_rows_array": "analysis_eligible",
            "telemetry_rows_array": None,
            "reversal_guard_s": None,
            "rate_limiter_policy": (
                "in-process control has no unobserved hardware rate limiter"
            ),
        },
        "torque_limit_Nm": list(protocol["limits"]["torque_Nm"]),
        "source_model": protocol["source_model"],
        "simulation": {
            "control_period_s": float(np.median(np.diff(time_s))),
            "position_noise_rad": position_noise_rad,
            "velocity_noise_rad_s": velocity_noise_rad_s,
            "preprocessing_mode": (
                "raw_engine_exact"
                if position_noise_rad == 0.0 and velocity_noise_rad_s == 0.0
                else "hardware_like"
            ),
            "truth": simulation_truth,
        },
    }
    return manifest, payload


def run(
    output_dir: str | Path,
    truth: mujoco.MjModel | None = None,
    *,
    protocols=CAMPAIGN,
    seed: int = 7,
    position_noise_rad: float = 0.0,
    velocity_noise_rad_s: float = 0.0,
    simulation_truth_path: str | Path | None = None,
) -> list[Path]:
    """Play every protocol and write recordings under ``output_dir``."""
    output_dir = Path(output_dir)
    paths = resolve_model_paths().require()
    plan_model = _fitting_spec(paths.hydrax).compile()
    destinations = [
        output_dir / protocol.protocol_id / "recording" for protocol in protocols
    ]
    existing = [destination for destination in destinations if destination.exists()]
    if existing:
        raise FileExistsError(
            "recordings are immutable; destination already exists: "
            + ", ".join(str(path) for path in existing)
        )
    if truth is not None and simulation_truth_path is not None:
        raise ValueError("pass either a truth model or a truth manifest, not both")

    simulation_truth: dict[str, object] | None = None
    if truth is None:
        from fer_mujoco_sysid.simulation_truth import (
            load_truth_manifest,
            truth_model_from_manifest,
            write_truth_manifest,
        )

        truth_path = Path(
            simulation_truth_path or output_dir / "simulation_truth.json"
        ).resolve()
        if simulation_truth_path is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_truth_manifest(truth_path, paths.hydrax)
        full_truth = load_truth_manifest(truth_path, source_path=paths.hydrax)
        truth = truth_model_from_manifest(
            truth_path,
            paths.hydrax,
            gravity=False,
        )
        simulation_truth = {
            "format": full_truth["format"],
            "content_sha256": full_truth["content_sha256"],
            "source_model_sha256": full_truth["source_model"]["sha256"],
            "manifest": str(truth_path),
        }

    written: list[Path] = []
    for index, spec in enumerate(protocols):
        compiled = compile_protocol(spec, plan_model, paths.hydrax)
        protocol, desired = _reviewed_protocol(compiled)
        motion = play_protocol(
            truth,
            compiled,
            seed=seed + index,
            position_noise_rad=position_noise_rad,
            velocity_noise_rad_s=velocity_noise_rad_s,
        )
        manifest, arrays = _bound_recording(
            protocol,
            desired,
            motion,
            position_noise_rad=position_noise_rad,
            velocity_noise_rad_s=velocity_noise_rad_s,
            simulation_truth=simulation_truth,
        )
        destination = output_dir / spec.protocol_id / "recording"
        written.append(write_recording(destination, manifest, arrays))
        print(f"  {spec.protocol_id}: {len(arrays['time_s'])} samples")
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    from fer_mujoco_sysid.campaign import repository_root
    from fer_mujoco_sysid.dataset import OUTPUT_DIRECTORY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--family",
        choices=("all", "friction", "inertial"),
        default="all",
        help="which protocols to play",
    )
    parser.add_argument(
        "--hardware-like-noise",
        action="store_true",
        help="optional robustness run; the default is the exact numerical gate",
    )
    parser.add_argument(
        "--simulation-truth",
        default=None,
        help="verified shared simulation-truth manifest to use and lineage-bind",
    )
    arguments = parser.parse_args(argv)

    from fer_mujoco_sysid.campaign import FRICTION_CAMPAIGN, INERTIAL_CAMPAIGN

    protocols = {
        "all": CAMPAIGN,
        "friction": FRICTION_CAMPAIGN,
        "inertial": INERTIAL_CAMPAIGN,
    }[arguments.family]

    output = Path(arguments.output or repository_root() / OUTPUT_DIRECTORY / "mujoco")
    print(f"playing {len(protocols)} protocol(s) in-process into {output} ...")
    run(
        output,
        protocols=protocols,
        position_noise_rad=(
            HARDWARE_LIKE_POSITION_NOISE_RAD if arguments.hardware_like_noise else 0.0
        ),
        velocity_noise_rad_s=(
            HARDWARE_LIKE_VELOCITY_NOISE_RAD_S if arguments.hardware_like_noise else 0.0
        ),
        simulation_truth_path=arguments.simulation_truth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
