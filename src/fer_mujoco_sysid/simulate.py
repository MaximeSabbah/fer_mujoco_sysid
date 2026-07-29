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

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.campaign import CAMPAIGN, compile_protocol
from fer_mujoco_sysid.dataset import write_recording
from fer_mujoco_sysid.excitation import FER_TORQUE_LIMIT_NM
from fer_mujoco_sysid.identify import _fitting_spec
from fer_mujoco_sysid.model import resolve_model_paths
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

#: The trajectory controller's gains, so the simulated backend differs from
#: the ROS one only in the plumbing and not in the control law.
JTC_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
JTC_KD = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0])

#: Sensor noise, so preprocessing is exercised rather than bypassed.
POSITION_NOISE_RAD = 2e-5
VELOCITY_NOISE_RAD_S = 2e-3


def play_protocol(
    model: mujoco.MjModel,
    compiled,
    *,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    """Track a compiled protocol under the trajectory controller's law."""
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
        # The controller sees noisy measurements, as it would on hardware.
        q_measured = data.qpos[:7] + POSITION_NOISE_RAD * rng.standard_normal(7)
        dq_measured = data.qvel[:7] + VELOCITY_NOISE_RAD_S * rng.standard_normal(7)
        command = np.clip(
            JTC_KP * (q_desired[k] - q_measured) + JTC_KD * (dq_desired[k] - dq_measured),
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


def run(
    output_dir: str | Path,
    truth: mujoco.MjModel | None = None,
    *,
    protocols=CAMPAIGN,
    seed: int = 7,
) -> list[Path]:
    """Play every protocol and write recordings under ``output_dir``."""
    output_dir = Path(output_dir)
    paths = resolve_model_paths().require()
    plan_model = _fitting_spec(paths.hydrax).compile()
    if truth is None:
        from fer_mujoco_sysid.report import truth_model

        truth = truth_model()
        truth.opt.gravity[:] = 0.0  # the recording convention (D029)

    written: list[Path] = []
    for index, spec in enumerate(protocols):
        compiled = compile_protocol(spec, plan_model, paths.hydrax)
        arrays = play_protocol(truth, compiled, seed=seed + index)
        manifest = {
            "joint_order": list(ROS_ARM_JOINT_NAMES),
            "backend": "mujoco",
            "speed_scale": 1.0,
            "torque_source": "in-process trajectory-controller law",
            "protocol": {
                "protocol_id": spec.protocol_id,
                "duration_s": float(compiled.time_s[-1]),
            },
            "torque_limit_Nm": FER_TORQUE_LIMIT_NM.tolist(),
        }
        destination = output_dir / spec.protocol_id / "recording"
        if destination.exists():
            import shutil

            shutil.rmtree(destination)
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
    arguments = parser.parse_args(argv)

    from fer_mujoco_sysid.campaign import FRICTION_CAMPAIGN, INERTIAL_CAMPAIGN

    protocols = {
        "all": CAMPAIGN,
        "friction": FRICTION_CAMPAIGN,
        "inertial": INERTIAL_CAMPAIGN,
    }[arguments.family]

    output = Path(
        arguments.output or repository_root() / OUTPUT_DIRECTORY / "mujoco"
    )
    print(f"playing {len(protocols)} protocol(s) in-process into {output} ...")
    run(output, protocols=protocols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
