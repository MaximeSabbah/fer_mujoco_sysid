"""Classical identification diagnostics: regressor conditioning, sigma%, torques.

``mujoco.sysid`` fits by rollout matching and exposes no regressor, so the
standard robotics diagnostics are built here on top of it:

* **Regressor conditioning** — the excitation quality measure. For a
  parameter set whose torque contribution is linear, ``tau = Y(q,dq,ddq) theta``;
  ``cond(Y)`` says whether the motion can separate the parameters at all.
  Friction and armature are linear in exactly this sense with analytically
  known columns, so their regressor is exact rather than estimated.
* **Relative standard deviation** — ``sigma% = 100 * sigma_i / |theta_i|`` from
  the least-squares covariance. The classical rule of thumb is that a
  parameter with ``sigma%`` above ~10-20 % is not identified by this data and
  must be frozen, grouped, or dropped rather than reported.
* **Torque reconstruction** — friction acts on torques, so the decisive plot
  is measured joint torque against the torque a model predicts for the same
  measured motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

# A joint is "moving" above this speed; below it sign(dq) is ill-defined and
# the friction regressor row carries no information.
MOVING_THRESHOLD_RAD_S = 1e-3
# Classical acceptance rule for a reported parameter.
RELATIVE_STD_LIMIT_PERCENT = 20.0


@dataclass(frozen=True)
class RegressorReport:
    """Excitation quality of one linear parameter block, per joint."""

    parameter_names: tuple[str, ...]
    condition_number: NDArray[np.float64]
    sample_count: NDArray[np.int64]
    worst_condition_number: float
    note: str = ""

    def is_well_excited(self, limit: float = 100.0) -> bool:
        return bool(np.nanmax(self.condition_number) <= limit)


def friction_regressor(
    dq_rad_s: NDArray[np.float64], joint: int
) -> NDArray[np.float64]:
    """Columns ``[sign(dq), dq]`` for joint *j*, over its moving samples.

    ``tau_friction = frictionloss * sign(dq) + damping * dq`` is exactly
    linear in the two parameters, so this is the true regressor, not an
    approximation.
    """
    velocity = np.asarray(dq_rad_s, dtype=np.float64)[:, joint]
    moving = np.abs(velocity) > MOVING_THRESHOLD_RAD_S
    return np.column_stack([np.sign(velocity[moving]), velocity[moving]])


def armature_friction_regressor(
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
    joint: int,
) -> NDArray[np.float64]:
    """Columns ``[sign(dq), dq, ddq]``: friction plus rotor inertia.

    Armature adds ``armature * ddq`` to the same joint's torque, so the three
    parameters share one linear model — and this regressor is what says
    whether a protocol can separate them.
    """
    velocity = np.asarray(dq_rad_s, dtype=np.float64)[:, joint]
    acceleration = np.asarray(ddq_rad_s2, dtype=np.float64)[:, joint]
    moving = np.abs(velocity) > MOVING_THRESHOLD_RAD_S
    return np.column_stack(
        [np.sign(velocity[moving]), velocity[moving], acceleration[moving]]
    )


def _condition_number(regressor: NDArray[np.float64], normalize: bool) -> float:
    if len(regressor) < 2 * regressor.shape[1]:
        return float("nan")
    matrix = np.asarray(regressor, dtype=np.float64)
    if normalize:
        # Column scaling removes the arbitrary unit choice (Nm vs Nm.s/rad),
        # which otherwise dominates the condition number.
        norms = np.linalg.norm(matrix, axis=0)
        if np.any(norms == 0.0):
            return float("inf")
        matrix = matrix / norms
    return float(np.linalg.cond(matrix))


def friction_regressor_report(
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64] | None = None,
    *,
    normalize: bool = True,
) -> RegressorReport:
    """Per-joint conditioning of the friction (optionally + armature) block."""
    joints = np.asarray(dq_rad_s).shape[1]
    conditions = np.empty(joints)
    counts = np.zeros(joints, dtype=np.int64)
    for joint in range(joints):
        if ddq_rad_s2 is None:
            regressor = friction_regressor(dq_rad_s, joint)
        else:
            regressor = armature_friction_regressor(dq_rad_s, ddq_rad_s2, joint)
        conditions[joint] = _condition_number(regressor, normalize)
        counts[joint] = len(regressor)
    names = ("frictionloss", "damping") + (() if ddq_rad_s2 is None else ("armature",))
    return RegressorReport(
        parameter_names=names,
        condition_number=conditions,
        sample_count=counts,
        worst_condition_number=float(np.nanmax(conditions)),
        note=(
            "column-normalized cond(Y) per joint; 1 is ideal, >100 means the "
            "motion barely separates these parameters"
        ),
    )


@dataclass(frozen=True)
class ParameterQuality:
    """Relative standard deviations and the resulting keep/freeze verdict."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    std_dev: NDArray[np.float64]
    relative_percent: NDArray[np.float64]
    limit_percent: float

    @property
    def rejected(self) -> list[str]:
        """Parameters whose relative standard deviation exceeds the limit."""
        return [
            name
            for name, percent in zip(self.names, self.relative_percent, strict=True)
            if not np.isfinite(percent) or percent > self.limit_percent
        ]

    def table(self) -> str:
        rows = [
            "| parameter | value | std dev | sigma% | verdict |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, value, std, percent in zip(
            self.names, self.values, self.std_dev, self.relative_percent, strict=True
        ):
            ok = np.isfinite(percent) and percent <= self.limit_percent
            rows.append(
                f"| {name} | {value:.4f} | {std:.4f} "
                f"| {percent:.1f}% | {'keep' if ok else 'FREEZE'} |"
            )
        return "\n".join(rows)


def parameter_quality(
    names: tuple[str, ...],
    values: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    limit_percent: float = RELATIVE_STD_LIMIT_PERCENT,
) -> ParameterQuality:
    """Relative standard deviations from a parameter covariance matrix."""
    values = np.asarray(values, dtype=np.float64)
    variance = np.diag(np.asarray(covariance, dtype=np.float64))
    std_dev = np.sqrt(np.clip(variance, 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = 100.0 * std_dev / np.abs(values)
    relative = np.where(np.abs(values) > 0, relative, np.inf)
    return ParameterQuality(
        names=names,
        values=values,
        std_dev=std_dev,
        relative_percent=relative,
        limit_percent=limit_percent,
    )


def predicted_torque(
    model: mujoco.MjModel,
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Inverse-dynamics torque of *model* along a measured trajectory.

    This is the model's answer to "what torque would produce this motion?" —
    including its gravity, Coriolis, inertial, damping and friction terms.
    Comparing it with the recorded torque is the classical validation.
    """
    data = mujoco.MjData(model)
    torque = np.empty((len(q_rad), model.nv))
    for index in range(len(q_rad)):
        data.qpos[:] = q_rad[index]
        data.qvel[:] = dq_rad_s[index]
        data.qacc[:] = ddq_rad_s2[index]
        mujoco.mj_inverse(model, data)
        torque[index] = data.qfrc_inverse
    return torque


@dataclass(frozen=True)
class TorqueResidualReport:
    """Measured-minus-predicted joint torque, per joint."""

    rmse_Nm: NDArray[np.float64]
    max_abs_Nm: NDArray[np.float64]
    mean_Nm: NDArray[np.float64]

    @property
    def worst_rmse(self) -> float:
        return float(np.max(self.rmse_Nm))


def torque_residuals(
    measured_Nm: NDArray[np.float64], predicted_Nm: NDArray[np.float64]
) -> TorqueResidualReport:
    residual = np.asarray(measured_Nm) - np.asarray(predicted_Nm)
    return TorqueResidualReport(
        rmse_Nm=np.sqrt(np.mean(residual**2, axis=0)),
        max_abs_Nm=np.max(np.abs(residual), axis=0),
        mean_Nm=np.mean(residual, axis=0),
    )
