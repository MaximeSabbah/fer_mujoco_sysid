"""Classical regressor identification: tau = Y(q, dq, ddq) theta, solved directly.

The textbook method, and it belongs here for three reasons that have nothing
to do with it being fast.

* It answers a question the rollout fit does not: **can this data determine
  these parameters at all?** That is the condition number of ``Y`` and the
  relative standard deviations, both of which fall out of the linear solve.
* It is an excellent starting point. Seeded from it, the rollout fit converged
  in a single iteration on recorded data; started from nominal it spent
  twenty-four minutes travelling.
* Run alongside the rollout fit it is an independent check, and **the
  disagreement between the two is itself a measurement**. They minimize
  different things — this one balances the torque equation sample by sample,
  the rollout fit reproduces the trajectory — so they agree only when the
  model class is right. On simulated data with friction inside the class they
  agree to about 1%. A large gap on the real robot means the class is wrong
  (Stribeck, temperature, transmission), which is a reason to stop and think
  rather than a number to report.

What it cannot do is replace the rollout fit. It needs ``ddq``, obtained by
differentiating measured velocity, and it optimizes torque residual rather
than trajectory fidelity — which is the quantity a planner actually meets.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

#: Below this speed a joint is not reliably sliding: ``sign(dq)`` is
#: meaningless in the stiction band and drags the Coulomb estimate down.
#: Measured on a recorded campaign: including samples from 1e-3 rad/s biased
#: frictionloss low by 2-17%, and excluding them brought it to 0.1%.
SLIDING_THRESHOLD_RAD_S = 0.03

# Ten lags span 0.1 s at the intentional 100 Hz Franka telemetry rate. The
# differentiated/filtered torque residual is serially correlated over several
# samples, so treating every row as independent makes sigma percentages much
# too optimistic.
DEFAULT_HAC_LAGS = 10


@dataclass(frozen=True)
class LinearFrictionFit:
    """Per-joint Coulomb and viscous friction from a direct least squares."""

    frictionloss: NDArray[np.float64]
    damping: NDArray[np.float64]
    #: Column-normalized condition number of Y per joint. 1 is ideal.
    condition_number: NDArray[np.float64]
    #: Relative standard deviation per parameter, in percent.
    frictionloss_sigma_percent: NDArray[np.float64]
    damping_sigma_percent: NDArray[np.float64]
    #: Samples that survived the sliding threshold, per joint.
    samples: NDArray[np.intp]
    #: Uncertainty estimator and its maximum serial-correlation lag.
    covariance_method: str
    covariance_lags: int

    def table(self) -> str:
        lines = [
            "| joint | frictionloss [Nm] | sigma% | damping [Nm s/rad] "
            "| sigma% | cond(Y) | samples |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for joint in range(len(self.frictionloss)):
            lines.append(
                f"| {joint + 1} | {self.frictionloss[joint]:.4f} "
                f"| {self.frictionloss_sigma_percent[joint]:.2f}% "
                f"| {self.damping[joint]:.4f} "
                f"| {self.damping_sigma_percent[joint]:.2f}% "
                f"| {self.condition_number[joint]:.2f} "
                f"| {int(self.samples[joint])} |"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class CandidateClassicalFit:
    """Constrained direct seed matching one friction candidate contract.

    ``linear`` keeps the established reporting shape. ``fitted_damping``
    distinguishes an estimated viscous coefficient from a fixed candidate
    value; a fixed coordinate has zero conditional uncertainty and is never
    presented as an optimizer result.
    """

    linear: LinearFrictionFit
    fitted_damping: tuple[bool, ...]
    bound_hits: tuple[str, ...]


def newey_west_covariance(
    regressor: NDArray[np.float64],
    residual: NDArray[np.float64],
    *,
    max_lags: int = DEFAULT_HAC_LAGS,
) -> NDArray[np.float64]:
    """Heteroskedasticity/autocorrelation-robust OLS covariance.

    A Bartlett kernel gives nearby telemetry rows progressively less weight.
    ``max_lags=0`` reduces to heteroskedasticity-robust HC1 covariance; the
    default additionally accounts for 0.1 s of correlation at 100 Hz.
    """
    design = np.asarray(regressor, dtype=np.float64)
    errors = np.asarray(residual, dtype=np.float64)
    if design.ndim != 2 or errors.shape != (len(design),):
        raise ValueError("regressor must be 2-D and residual must match its rows")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(errors)):
        raise ValueError("regressor and residual must be finite")
    rows, parameters = design.shape
    if rows <= parameters:
        raise ValueError(
            f"need more rows than parameters for covariance, got {rows} and "
            f"{parameters}"
        )
    if not isinstance(max_lags, int) or not 0 <= max_lags < rows:
        raise ValueError(f"max_lags must be an integer in [0, {rows - 1}]")

    bread = np.linalg.inv(design.T @ design)
    scores = design * errors[:, None]
    meat = scores.T @ scores
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        lagged = scores[lag:].T @ scores[:-lag]
        meat += weight * (lagged + lagged.T)

    # HC1 small-sample correction. Symmetrization removes roundoff asymmetry
    # before callers inspect the diagonal.
    covariance = (rows / (rows - parameters)) * bread @ meat @ bread
    return 0.5 * (covariance + covariance.T)


def rigid_body_torque(
    model: mujoco.MjModel,
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Inverse dynamics with friction removed, sample by sample.

    The reference must be **frictionless**, and the nominal model is not: it
    carries ``damping=1`` on every joint. Leaving that in puts ``1.0 * dq``
    into the reference, so it cancels out of the residual the regressor sees
    and the viscous estimate comes back short by exactly 1.0 on every joint —
    a constant offset across all seven, which is how this was found.
    """
    scratch = mujoco.MjData(model)
    # Zero the friction terms for the reference, then put the caller's model
    # back exactly as it was: this is a query, not an edit.
    saved_damping = np.array(model.dof_damping, copy=True)
    saved_friction = np.array(model.dof_frictionloss, copy=True)
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    try:
        out = np.empty((len(q_rad), 7), dtype=np.float64)
        for index in range(len(q_rad)):
            scratch.qpos[:] = q_rad[index]
            scratch.qvel[:] = dq_rad_s[index]
            scratch.qacc[:] = ddq_rad_s2[index]
            mujoco.mj_inverse(model, scratch)
            out[index] = scratch.qfrc_inverse[:7]
    finally:
        model.dof_damping[:] = saved_damping
        model.dof_frictionloss[:] = saved_friction
    return out


def fit_friction(
    model: mujoco.MjModel,
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    *,
    sliding_threshold_rad_s: float = SLIDING_THRESHOLD_RAD_S,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> LinearFrictionFit:
    """Solve ``tau - tau_rigid = frictionloss*sign(dq) + damping*dq`` per joint.

    ``model`` must carry the same conventions as the recording — in
    particular gravity, which is disabled when the torque channel is a
    gravity-compensated commanded effort.
    """
    if not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer")
    residual = np.asarray(tau_Nm, dtype=np.float64) - rigid_body_torque(
        model, q_rad, dq_rad_s, ddq_rad_s2
    )
    dq = np.asarray(dq_rad_s, dtype=np.float64)

    joints = dq.shape[1]
    frictionloss = np.zeros(joints)
    damping = np.zeros(joints)
    condition = np.zeros(joints)
    sigma = np.zeros((joints, 2))
    counts = np.zeros(joints, dtype=np.intp)

    for joint in range(joints):
        sliding = np.abs(dq[:, joint]) > sliding_threshold_rad_s
        counts[joint] = int(sliding.sum())
        if counts[joint] < 3:
            raise ValueError(
                f"joint{joint + 1} slides in only {counts[joint]} samples above "
                f"{sliding_threshold_rad_s} rad/s; the protocol did not move it"
            )
        regressor = np.column_stack(
            [np.sign(dq[sliding, joint]), dq[sliding, joint]]
        )
        target = residual[sliding, joint]
        theta, *_ = np.linalg.lstsq(regressor, target, rcond=None)
        frictionloss[joint], damping[joint] = theta

        scale = np.linalg.norm(regressor, axis=0)
        condition[joint] = float(np.linalg.cond(regressor / scale))

        error = target - regressor @ theta
        effective_lags = min(hac_lags, len(target) - 1)
        covariance = newey_west_covariance(
            regressor, error, max_lags=effective_lags
        )
        sigma[joint] = 100.0 * np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        ) / np.maximum(
            np.abs(theta), 1e-12
        )

    return LinearFrictionFit(
        frictionloss=frictionloss,
        damping=damping,
        condition_number=condition,
        frictionloss_sigma_percent=sigma[:, 0],
        damping_sigma_percent=sigma[:, 1],
        samples=counts,
        covariance_method="Newey-West HAC" if hac_lags else "HC1",
        covariance_lags=hac_lags,
    )


def fit_friction_conditioned(
    model: mujoco.MjModel,
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    ddq_rad_s2: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    *,
    damping_reference_Nm_s: NDArray[np.float64],
    fit_damping: NDArray[np.bool_],
    sliding_threshold_rad_s: float = SLIDING_THRESHOLD_RAD_S,
    hac_lags: int = DEFAULT_HAC_LAGS,
    frictionloss_bounds_Nm: tuple[float, float] = (0.0, 3.0),
    damping_bounds_Nm_s: tuple[float, float] = (0.0, 8.0),
) -> CandidateClassicalFit:
    """Fit nonnegative Coulomb friction with explicitly conditioned damping.

    A fixed-damping candidate first subtracts ``damping_reference * dq`` and
    regresses only the Coulomb column. Selected ``C+`` joints retain both
    columns. This prevents a nominal or zero damping value from entering the
    optimizer vector merely because it appears in the compiled model.
    """
    q = np.asarray(q_rad, dtype=np.float64)
    dq = np.asarray(dq_rad_s, dtype=np.float64)
    ddq = np.asarray(ddq_rad_s2, dtype=np.float64)
    tau = np.asarray(tau_Nm, dtype=np.float64)
    damping_reference = np.asarray(damping_reference_Nm_s, dtype=np.float64)
    fitted_damping = np.asarray(fit_damping, dtype=np.bool_)

    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"q_rad must have shape (samples, 7), got {q.shape}")
    expected = q.shape
    for name, values in (("dq_rad_s", dq), ("ddq_rad_s2", ddq), ("tau_Nm", tau)):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if damping_reference.shape != (7,):
        raise ValueError("damping_reference_Nm_s must have shape (7,)")
    if fitted_damping.shape != (7,):
        raise ValueError("fit_damping must have shape (7,)")
    if not np.isfinite(sliding_threshold_rad_s) or sliding_threshold_rad_s < 0.0:
        raise ValueError("sliding_threshold_rad_s must be finite and nonnegative")
    if not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer")
    for name, bounds in (
        ("frictionloss_bounds_Nm", frictionloss_bounds_Nm),
        ("damping_bounds_Nm_s", damping_bounds_Nm_s),
    ):
        if (
            len(bounds) != 2
            or not np.isfinite(bounds).all()
            or bounds[0] < 0.0
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"{name} must be finite, nonnegative, and increasing")
    if not all(np.isfinite(values).all() for values in (q, dq, ddq, tau)):
        raise ValueError("friction regression inputs must be finite")
    if not np.isfinite(damping_reference).all() or (damping_reference < 0.0).any():
        raise ValueError("damping_reference_Nm_s must be finite and nonnegative")

    residual = tau - rigid_body_torque(model, q, dq, ddq)
    frictionloss = np.zeros(7, dtype=np.float64)
    damping = damping_reference.copy()
    condition = np.zeros(7, dtype=np.float64)
    friction_sigma = np.zeros(7, dtype=np.float64)
    damping_sigma = np.zeros(7, dtype=np.float64)
    counts = np.zeros(7, dtype=np.intp)
    bound_hits: list[str] = []

    for joint in range(7):
        sliding = np.abs(dq[:, joint]) > sliding_threshold_rad_s
        counts[joint] = int(sliding.sum())
        columns = 2 if fitted_damping[joint] else 1
        if counts[joint] <= columns:
            raise ValueError(
                f"joint{joint + 1} slides in only {counts[joint]} samples above "
                f"{sliding_threshold_rad_s} rad/s"
            )

        velocity = dq[sliding, joint]
        target = residual[sliding, joint]
        if fitted_damping[joint]:
            regressor = np.column_stack((np.sign(velocity), velocity))
            lower = np.array(
                (frictionloss_bounds_Nm[0], damping_bounds_Nm_s[0])
            )
            upper = np.array(
                (frictionloss_bounds_Nm[1], damping_bounds_Nm_s[1])
            )
        else:
            target = target - damping_reference[joint] * velocity
            regressor = np.sign(velocity)[:, None]
            lower = np.array((frictionloss_bounds_Nm[0],))
            upper = np.array((frictionloss_bounds_Nm[1],))

        scale = np.linalg.norm(regressor, axis=0)
        if (scale <= 0.0).any() or np.linalg.matrix_rank(regressor) != columns:
            raise ValueError(f"joint{joint + 1} friction regressor is rank deficient")
        solution = lsq_linear(regressor, target, bounds=(lower, upper))
        if not solution.success or not np.isfinite(solution.x).all():
            raise ValueError(
                f"joint{joint + 1} constrained friction solve failed: "
                f"{solution.message}"
            )
        frictionloss[joint] = solution.x[0]
        if fitted_damping[joint]:
            damping[joint] = solution.x[1]
        condition[joint] = float(np.linalg.cond(regressor / scale))

        error = target - regressor @ solution.x
        covariance = newey_west_covariance(
            regressor,
            error,
            max_lags=min(hac_lags, len(target) - 1),
        )
        relative_sigma = 100.0 * np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        ) / np.maximum(np.abs(solution.x), 1e-12)
        friction_sigma[joint] = relative_sigma[0]
        if fitted_damping[joint]:
            damping_sigma[joint] = relative_sigma[1]

        proximity = np.minimum(solution.x - lower, upper - solution.x)
        hit = proximity <= 1e-3 * (upper - lower)
        if hit[0]:
            bound_hits.append(f"joint{joint + 1}_frictionloss")
        if fitted_damping[joint] and hit[1]:
            bound_hits.append(f"joint{joint + 1}_damping")

    return CandidateClassicalFit(
        linear=LinearFrictionFit(
            frictionloss=frictionloss,
            damping=damping,
            condition_number=condition,
            frictionloss_sigma_percent=friction_sigma,
            damping_sigma_percent=damping_sigma,
            samples=counts,
            covariance_method="Newey-West HAC" if hac_lags else "HC1",
            covariance_lags=hac_lags,
        ),
        fitted_damping=tuple(bool(value) for value in fitted_damping),
        bound_hits=tuple(bound_hits),
    )
