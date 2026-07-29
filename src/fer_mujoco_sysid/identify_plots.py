"""Plots for an identification run from recorded data.

Four figures, each answering one question a reader actually has:

* ``rollout_vs_measurement.png`` — does the simulator reproduce what the arm
  did? This is the acceptance metric made visible, and the one to look at
  first;
* ``error_vs_horizon.png`` — how fast does that agreement decay? A model can
  be excellent over 0.1 s and useless over 2 s;
* ``friction_curves.png`` — the classical picture: friction torque against
  velocity, with the dry-friction step at zero and the viscous slope;
* ``torque_tracking.png`` — measured joint torque against what each model
  reconstructs, which is where friction is directly visible.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fer_mujoco_sysid_mpl")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from fer_mujoco_sysid.classical import (  # noqa: E402
    SLIDING_THRESHOLD_RAD_S,
    LinearFrictionFit,
    rigid_body_torque,
)
from fer_mujoco_sysid.diagnostics import predicted_torque  # noqa: E402

_INK = "#374151"
_MEASURED = "#111827"
_NOMINAL = "#9ca3af"
_CLASSICAL = "#b45309"
_REFINED = "#2563eb"


def _style(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=9, color=_INK)
    axis.set_xlabel(xlabel, fontsize=8, color=_INK)
    axis.set_ylabel(ylabel, fontsize=8, color=_INK)
    axis.tick_params(labelsize=7, colors=_INK)
    axis.grid(alpha=0.25, linewidth=0.5)


def plot_rollout_vs_measurement(
    path: Path,
    models: dict[str, mujoco.MjModel],
    run,
    *,
    horizon_s: float = 2.0,
    windows: int = 3,
) -> Path:
    """Predicted against measured joint angles, over several rollout windows.

    The acceptance metric as a picture: each panel starts the simulator from a
    measured state, drives it with the recorded torque, and lets it run open
    loop. Where the coloured line leaves the black one, the model is wrong.
    """
    import mujoco.rollout

    reference = next(iter(models.values()))
    steps = int(round(horizon_s / reference.opt.timestep))
    starts = np.linspace(0, max(len(run.control) - steps - 1, 0), windows, dtype=int)

    figure, axes = plt.subplots(
        windows, 3, figsize=(13, 2.6 * windows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    joints = (1, 3, 5)  # a shoulder, an elbow and a wrist joint

    for row, start in enumerate(starts):
        stop = min(start + steps, len(run.control))
        measured = run.measured[start:stop]
        time = np.arange(stop - start) * reference.opt.timestep
        predictions = {}
        for name, model in models.items():
            data = mujoco.MjData(model)
            data.qpos[:] = run.measured[start, :7]
            data.qvel[:] = run.measured[start, 7:14]
            state = np.empty(
                mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
            )
            mujoco.mj_getState(
                model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
            )
            _, sensor = mujoco.rollout.rollout(
                model, data, state, run.control[start:stop]
            )
            predictions[name] = np.squeeze(sensor, axis=0)

        for column, joint in enumerate(joints):
            axis = axes[row, column]
            axis.plot(
                time,
                measured[:, joint],
                color=_MEASURED,
                linewidth=2.0,
                label="measured",
            )
            for name, colour in (
                ("nominal", _NOMINAL),
                ("classical", _CLASSICAL),
                ("identified", _REFINED),
            ):
                if name in predictions:
                    axis.plot(
                        time,
                        predictions[name][:, joint],
                        color=colour,
                        linewidth=1.2,
                        linestyle="--" if name == "identified" else "-",
                        label=name,
                    )
            _style(
                axis,
                f"joint {joint + 1}, window from "
                f"{start * reference.opt.timestep:.0f} s",
                "time in window [s]",
                "q [rad]",
            )
            if row == 0 and column == 0:
                axis.legend(fontsize=7, loc="best")

    figure.suptitle(
        f"Open-loop rollout against measurement, {horizon_s:g} s horizon "
        "(where the line separates, the model is wrong)",
        color=_INK,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_error_vs_horizon(path: Path, horizons: dict) -> Path:
    """How the agreement decays with rollout length, per model."""
    labels = list(horizons)
    values = [float(label.rstrip("s")) for label in labels]

    figure, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    for name, colour in (
        ("nominal", _NOMINAL),
        ("classical", _CLASSICAL),
        ("identified", _REFINED),
    ):
        axes[0].plot(
            values,
            [horizons[label][name]["gripper_rmse_mm"] for label in labels],
            marker="o",
            color=colour,
            label=name,
        )
        axes[1].plot(
            values,
            [horizons[label][name]["worst_q_rmse_rad"] for label in labels],
            marker="o",
            color=colour,
            label=name,
        )
        axes[2].plot(
            values,
            [horizons[label][name]["worst_dq_rmse_rad_s"] for label in labels],
            marker="o",
            color=colour,
            label=name,
        )
    for axis, ylabel in (
        (axes[0], "gripper RMSE [mm]"),
        (axes[1], "worst q RMSE [rad]"),
        (axes[2], "worst dq RMSE [rad/s]"),
    ):
        axis.set_yscale("log")
        _style(axis, "held-out prediction error", "rollout horizon [s]", ylabel)
    axes[0].legend(fontsize=7)
    figure.suptitle("Prediction error against rollout horizon", color=_INK)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_friction_curves(
    path: Path,
    nominal: mujoco.MjModel,
    linear: LinearFrictionFit,
    refined: tuple[NDArray[np.float64], NDArray[np.float64]],
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    *,
    stride: int = 25,
) -> Path:
    """Friction torque against velocity: the classical picture.

    Points are what the recording says friction was — commanded torque minus
    the frictionless rigid-body torque. The step at zero velocity is Coulomb
    friction; the slope away from it is viscous damping. The nominal model
    has no step at all, which is the whole problem this project exists for.
    """
    residual = np.asarray(tau_Nm) - rigid_body_torque(
        nominal, q_rad, dq_rad_s, ddq_rad_s2
    )
    refined_fl, refined_d = refined

    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for joint in range(7):
        axis = axes.flat[joint]
        sliding = np.abs(dq_rad_s[:, joint]) > SLIDING_THRESHOLD_RAD_S
        axis.scatter(
            dq_rad_s[sliding, joint][::stride],
            residual[sliding, joint][::stride],
            s=3,
            alpha=0.25,
            color=_MEASURED,
            label="measured",
        )
        speed = np.linspace(
            float(dq_rad_s[:, joint].min()), float(dq_rad_s[:, joint].max()), 400
        )
        speed = speed[np.abs(speed) > SLIDING_THRESHOLD_RAD_S]
        for values, colour, name, style in (
            (
                (linear.frictionloss[joint], linear.damping[joint]),
                _CLASSICAL,
                "classical",
                "-",
            ),
            ((refined_fl[joint], refined_d[joint]), _REFINED, "refined", "--"),
        ):
            axis.plot(
                speed,
                values[0] * np.sign(speed) + values[1] * speed,
                color=colour,
                linewidth=1.6,
                linestyle=style,
                label=name,
            )
        axis.plot(speed, 1.0 * speed, color=_NOMINAL, linewidth=1.2, label="nominal")
        _style(axis, f"joint {joint + 1}", "dq [rad/s]", "friction torque [Nm]")
        if joint == 0:
            axis.legend(fontsize=7)
    axes.flat[7].axis("off")
    figure.suptitle(
        "Friction torque against velocity — the step at zero is Coulomb "
        "friction, the slope is viscous",
        color=_INK,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_torque_tracking(
    path: Path,
    models: dict[str, mujoco.MjModel],
    run,
    ddq_rad_s2: NDArray[np.float64],
    recorded_torque_Nm: NDArray[np.float64],
    *,
    torque_label: str,
    seconds: float = 12.0,
) -> Path:
    """The exact evaluated torque channel against model reconstruction.

    **Read the spikes at the corners as an artifact, not a defect.** This is
    inverse dynamics, ``tau = M(q) ddq + C + friction``, and its ``ddq`` comes
    from differentiating zero-phase-filtered velocity. At a sharp velocity
    transition the Butterworth rings symmetrically; differentiating that
    ringing amplifies it, and the error multiplies the inertia matrix into a
    torque spike right at the corner.

    Measured on the held-out protocol: the reconstruction error rises from
    0.002 Nm in the low-jerk half of the run to 0.61 Nm in the top jerk
    percentile — a 280x increase — while correlating with the *jerk* (0.3-0.6)
    far more than with the acceleration (0.02-0.23). Where jerk is low the
    model reconstructs the torque essentially exactly, and the rollout
    prediction, which never uses ``ddq`` at all, is exact throughout. A wrong
    model would fail both of those.
    """
    reference = next(iter(models.values()))
    stop = min(int(seconds / reference.opt.timestep), len(run.control))
    time = np.arange(stop) * reference.opt.timestep
    q, dq = run.measured[:stop, :7], run.measured[:stop, 7:14]
    recorded = np.asarray(recorded_torque_Nm, dtype=np.float64)
    if recorded.shape != run.control.shape:
        raise ValueError(
            f"recorded torque shape {recorded.shape} does not match "
            f"run control shape {run.control.shape}"
        )

    reconstructions = {
        name: predicted_torque(model, q, dq, ddq_rad_s2[:stop])
        for name, model in models.items()
    }

    figure, axes = plt.subplots(
        7, 1, figsize=(11, 12), sharex=True, constrained_layout=True
    )
    for joint in range(7):
        axis = axes[joint]
        axis.plot(
            time,
            recorded[:stop, joint],
            color=_MEASURED,
            linewidth=1.4,
            label=f"recorded {torque_label}",
        )
        for name, colour in (
            ("nominal", _NOMINAL),
            ("classical", _CLASSICAL),
            ("identified", _REFINED),
        ):
            if name in reconstructions:
                axis.plot(
                    time,
                    reconstructions[name][:, joint],
                    color=colour,
                    linewidth=1.0,
                    linestyle="--" if name == "identified" else "-",
                    label=name,
                )
        axis.set_ylabel(f"j{joint + 1} [Nm]", fontsize=8, color=_INK)
        axis.tick_params(labelsize=7, colors=_INK)
        axis.grid(alpha=0.25, linewidth=0.5)
        if joint == 0:
            axis.legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("time [s]", fontsize=8, color=_INK)
    figure.suptitle(
        f"Evaluated joint torque ({torque_label}): recorded against model "
        "reconstruction",
        color=_INK,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


#: Offscreen framebuffer for replay videos. The MuJoCo default is 640x480 and
#: has to be enlarged in the spec before compiling; the height is divisible by
#: 16 so the video encoder does not have to pad.
RENDER_WIDTH = 800
RENDER_HEIGHT = 608


def render_recording_replay(
    path: Path,
    model_path: str | Path,
    q_rad: NDArray[np.float64],
    time_s: NDArray[np.float64],
    *,
    q_desired_rad: NDArray[np.float64] | None = None,
    fps: int = 25,
) -> Path:
    """Replay a recording as a video, with the commanded pose superimposed.

    Distinct from the protocol videos, which show the motion that was
    *designed*. This shows the motion that *happened* — so for a hardware run
    it is the real arm, played back through the model.

    When the recording carries the controller's reference, two arms are drawn
    in the same scene: the measured one solid, and a translucent ghost of
    where the controller was asking it to be. Tracking error stops being a
    number and becomes something you can see — and the places where the ghost
    pulls ahead are exactly the reversals where friction bites.
    """
    import imageio.v2 as imageio

    from fer_mujoco_sysid.model import build_hydrax_arm_spec

    spec = build_hydrax_arm_spec(model_path)
    ghost_prefix = "expected_"
    with_ghost = q_desired_rad is not None
    if with_ghost:
        spec.attach(
            build_hydrax_arm_spec(model_path),
            prefix=ghost_prefix,
            frame=spec.worldbody.add_frame(),
        )

    floor = spec.worldbody.add_geom()
    floor.name = "replay_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.05]
    floor.rgba = [0.86, 0.87, 0.89, 1.0]
    floor.contype = 0
    floor.conaffinity = 0
    spec.visual.global_.offwidth = RENDER_WIDTH
    spec.visual.global_.offheight = RENDER_HEIGHT
    model = spec.compile()

    if with_ghost:
        # Materials win over geom rgba at render time, so the material has to
        # be dropped before the ghost can be made translucent.
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            if name.startswith(ghost_prefix):
                model.geom_matid[index] = -1
                model.geom_rgba[index] = [0.15, 0.45, 0.85, 0.30]

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.05, 0.0, 0.45)
    camera.distance = 1.55
    camera.azimuth = 135.0
    camera.elevation = -15.0

    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
            for frame_time in np.arange(0.0, float(time_s[-1]), 1.0 / fps):
                row = min(int(np.searchsorted(time_s, frame_time)), len(time_s) - 1)
                data.qpos[:7] = q_rad[row]
                if with_ghost:
                    data.qpos[7:14] = q_desired_rad[row]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    return path


def plot_tracking(
    path: Path,
    time_s: NDArray[np.float64],
    q_rad: NDArray[np.float64],
    q_desired_rad: NDArray[np.float64],
    *,
    seconds: float = 45.0,
) -> Path:
    """What the controller asked for against what the arm did.

    A different question from every other plot here: those judge the *model*,
    this judges the *run*. If the arm did not follow the protocol, the data
    describes a motion nobody designed — still fittable, but no longer the
    trajectory whose limits and clearance were validated. The error peaks at
    the velocity reversals, which is friction refusing to let go.
    """
    stop = int(np.searchsorted(time_s, time_s[0] + seconds))
    time = time_s[:stop] - time_s[0]
    error = (q_desired_rad[:stop] - q_rad[:stop]) * 1e3

    figure, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    for joint in range(7):
        axes[0].plot(time, error[:, joint], linewidth=0.9, label=f"joint {joint + 1}")
    _style(
        axes[0],
        "tracking error (commanded minus measured)",
        "time [s]",
        "error [mrad]",
    )
    axes[0].legend(fontsize=7, ncol=7)

    worst = int(np.argmax(np.abs(error).max(axis=0)))
    axes[1].plot(
        time,
        q_desired_rad[:stop, worst],
        color=_CLASSICAL,
        linewidth=1.4,
        label="commanded",
    )
    axes[1].plot(
        time,
        q_rad[:stop, worst],
        color=_MEASURED,
        linewidth=1.0,
        linestyle="--",
        label="measured",
    )
    _style(
        axes[1],
        f"joint {worst + 1} — the worst-tracked joint",
        "time [s]",
        "q [rad]",
    )
    axes[1].legend(fontsize=7)

    figure.suptitle(
        "Did the arm follow the protocol? (a property of the run, not the model)",
        color=_INK,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_end_effector(
    path: Path,
    models: dict[str, mujoco.MjModel],
    run,
    *,
    horizon_s: float = 2.0,
    windows: int = 3,
) -> Path:
    """The same acceptance question, asked at the gripper instead of the joints.

    Joint-space error is what the fit minimizes; end-effector error is what
    the task cares about. They are not interchangeable — small joint errors
    can compound along the kinematic chain, and a model can look fine per
    joint while missing by centimetres where it matters.
    """
    import mujoco.rollout

    reference = next(iter(models.values()))
    site = int(reference.site("gripper").id)
    steps = int(round(horizon_s / reference.opt.timestep))
    starts = np.linspace(0, max(len(run.control) - steps - 1, 0), windows, dtype=int)

    def positions(
        model: mujoco.MjModel, rows: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        data = mujoco.MjData(model)
        out = np.empty((len(rows), 3))
        for index, value in enumerate(rows):
            data.qpos[:] = value
            mujoco.mj_kinematics(model, data)
            out[index] = data.site_xpos[site]
        return out

    figure, axes = plt.subplots(
        windows, 2, figsize=(11, 2.8 * windows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for row, start in enumerate(starts):
        stop = min(start + steps, len(run.control))
        time = np.arange(stop - start) * reference.opt.timestep
        measured = positions(reference, run.measured[start:stop, :7])

        for name, colour in (
            ("nominal", _NOMINAL),
            ("classical", _CLASSICAL),
            ("identified", _REFINED),
        ):
            if name not in models:
                continue
            model = models[name]
            data = mujoco.MjData(model)
            data.qpos[:] = run.measured[start, :7]
            data.qvel[:] = run.measured[start, 7:14]
            state = np.empty(
                mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
            )
            mujoco.mj_getState(
                model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
            )
            _, sensor = mujoco.rollout.rollout(
                model, data, state, run.control[start:stop]
            )
            predicted = positions(model, np.squeeze(sensor, axis=0)[:, :7])
            error = 1e3 * np.linalg.norm(predicted - measured, axis=1)
            axes[row, 0].plot(
                time,
                predicted[:, 0],
                color=colour,
                linewidth=1.1,
                linestyle="--" if name == "identified" else "-",
                label=name,
            )
            axes[row, 1].semilogy(
                time, np.maximum(error, 1e-4), color=colour, linewidth=1.2, label=name
            )

        axes[row, 0].plot(
            time,
            measured[:, 0],
            color=_MEASURED,
            linewidth=2.0,
            label="measured",
            zorder=0,
        )
        _style(
            axes[row, 0],
            f"gripper x, window from {start * reference.opt.timestep:.0f} s",
            "time in window [s]",
            "x [m]",
        )
        _style(
            axes[row, 1], "gripper position error", "time in window [s]", "error [mm]"
        )
        if row == 0:
            axes[row, 0].legend(fontsize=7)

    figure.suptitle(
        f"End-effector prediction over a {horizon_s:g} s open-loop rollout",
        color=_INK,
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
