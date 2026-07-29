"""Gates for turning a recording into something the fit can roll out.

The recording arrives on the controller's clock; the fit rolls the model out
at the model's timestep. Getting that hand-off wrong is silent — the fit just
explains the misalignment with wrong parameters — so it is gated here.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from fer_mujoco_sysid.dataset import RECORDING_FORMAT, write_recording
from fer_mujoco_sysid.identify import HOLDOUT_SUFFIX, prepare_recording
from fer_mujoco_sysid.model import ModelPaths, build_hydrax_arm_spec
from fer_mujoco_sysid.protocol import ROS_ARM_JOINT_NAMES

_RATE_HZ = 500.0
_DURATION_S = 3.0


@pytest.fixture(scope="module")
def model(model_paths: ModelPaths) -> mujoco.MjModel:
    return build_hydrax_arm_spec(model_paths.hydrax, joint_state_sensors=True).compile()


def _write(
    root: Path, *, protocol_id: str = "fer-friction-a", rate_hz: float = _RATE_HZ
) -> Path:
    time_s = np.arange(int(_DURATION_S * rate_hz)) / rate_hz
    phase = 2 * np.pi * 0.2 * time_s
    q = np.column_stack([0.2 * np.sin(phase + joint) for joint in range(7)])
    dq = np.column_stack(
        [0.2 * 2 * np.pi * 0.2 * np.cos(phase + joint) for joint in range(7)]
    )
    arrays = {
        "time_s": time_s,
        "q_rad": q,
        "dq_rad_s": dq,
        "tau_cmd_Nm": 1.1 * np.sign(dq) + 0.7 * dq,
    }
    manifest = {
        "format": RECORDING_FORMAT,
        "joint_order": list(ROS_ARM_JOINT_NAMES),
        "backend": "mujoco",
        "torque_limit_Nm": [87.0] * 4 + [12.0] * 3,
        "protocol": {"protocol_id": protocol_id, "duration_s": _DURATION_S},
    }
    return write_recording(root, manifest, arrays)


def test_recording_is_resampled_onto_the_model_timestep(
    tmp_path: Path, model: mujoco.MjModel
) -> None:
    """The fit rolls out at the model's step; the recording arrives on another."""
    prepared = prepare_recording(_write(tmp_path / "run"), model)
    times = prepared.run.control_times
    step = float(model.opt.timestep)

    np.testing.assert_allclose(np.diff(times), step, rtol=1e-9)
    assert prepared.run.control.shape[1] == model.nu
    assert prepared.run.measured.shape[1] == 14


def test_row_convention_matches_the_rollout_skew(
    tmp_path: Path, model: mujoco.MjModel
) -> None:
    """``measured`` carries the post-step stamp ``mujoco.rollout`` emits.

    Getting this wrong shifts the torque against the state by one step, which
    biases friction exactly at the velocity reversals it is read from.
    """
    prepared = prepare_recording(_write(tmp_path / "run"), model)
    step = float(model.opt.timestep)

    np.testing.assert_allclose(prepared.run.control_times[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(prepared.run.measured_times[0], step, rtol=1e-9)
    np.testing.assert_allclose(
        prepared.run.measured_times - prepared.run.control_times, step, rtol=1e-9
    )
    np.testing.assert_allclose(
        prepared.run.qpos0, prepared.run.measured[0, :7], atol=1e-12
    )
    np.testing.assert_allclose(
        prepared.run.qvel0, prepared.run.measured[0, 7:14], atol=1e-12
    )


def test_a_recording_at_a_different_rate_still_lands_on_the_grid(
    tmp_path: Path, model: mujoco.MjModel
) -> None:
    """Hardware will not publish at exactly the model's rate."""
    prepared = prepare_recording(_write(tmp_path / "run", rate_hz=333.0), model)
    step = float(model.opt.timestep)
    np.testing.assert_allclose(np.diff(prepared.run.control_times), step, rtol=1e-9)
    assert prepared.run.control.shape[0] > 0


def test_holdout_role_comes_from_the_protocol_name(
    tmp_path: Path, model: mujoco.MjModel
) -> None:
    """Roles are fixed by name so a held-out result cannot be chosen later."""
    train = prepare_recording(_write(tmp_path / "a", protocol_id="fer-friction-a"), model)
    held = prepare_recording(
        _write(tmp_path / "b", protocol_id=f"fer-friction{HOLDOUT_SUFFIX}"), model
    )
    assert not train.is_holdout
    assert held.is_holdout
