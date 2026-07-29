"""Measurement preprocessing: zero-phase filtering, differentiation, decimation.

Classical identification (tau = Y(q, dq, ddq) theta) *must* filter, because it
differentiates measured positions twice and differentiation amplifies noise.
The rollout-matching fit used here never differentiates, so filtering is not
structurally required — but it is still needed against sensor noise, and the
diagnostics in :mod:`fer_mujoco_sysid.diagnostics` do use accelerations.

Filtering is therefore an explicit, recorded step: what was applied is stored
in :class:`FilterSettings` and reported, never applied silently inside a fit.

The filter is applied forward and backward (``filtfilt``), which squares the
magnitude response and gives **zero phase lag**. Phase lag matters here: a
causal filter shifts joint states in time relative to the recorded torque,
which biases friction (a sign-of-velocity effect) exactly around the velocity
reversals we care about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

DEFAULT_ORDER = 4
DEFAULT_CUTOFF_HZ = 8.0


@dataclass(frozen=True)
class FilterSettings:
    """A zero-phase Butterworth low-pass, recorded with its results.

    ``enabled=False`` is reserved for the deterministic simulation-oracle
    gate.  It proves that acquisition, fitting, and export close on exact
    engine data without preprocessing changing the system being identified.
    Hardware and robustness runs keep filtering enabled.

    ``order`` is the per-pass order; ``filtfilt`` applies it twice, so the
    effective magnitude response is that of a 2*order filter (its -3 dB point
    is below ``cutoff_hz``; the nominal cutoff is reported as configured,
    which is the convention in the identification literature).
    """

    enabled: bool = True
    order: int = DEFAULT_ORDER
    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    zero_phase: bool = True

    def describe(self, sample_rate_hz: float) -> dict[str, object]:
        if not self.enabled:
            return {
                "kind": "none",
                "enabled": False,
                "sample_rate_hz": float(sample_rate_hz),
                "reason": "deterministic simulation numerical-closure gate",
            }
        return {
            "kind": "butterworth_lowpass",
            "enabled": True,
            "order": self.order,
            "cutoff_hz": self.cutoff_hz,
            "zero_phase": self.zero_phase,
            "sample_rate_hz": float(sample_rate_hz),
            "normalized_cutoff": float(self.cutoff_hz / (0.5 * sample_rate_hz)),
        }


def sample_rate_hz(times: NDArray[np.float64]) -> float:
    """Mean sample rate of a monotonically increasing time vector."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a 1-D vector with at least two samples")
    steps = np.diff(times)
    if not np.all(steps > 0):
        raise ValueError("times must be strictly increasing")
    return float(1.0 / steps.mean())


def lowpass(
    values: NDArray[np.float64],
    sample_rate: float,
    settings: FilterSettings | None = None,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth low-pass along axis 0 (one column per joint)."""
    settings = settings or FilterSettings()
    if not settings.enabled:
        return np.asarray(values, dtype=np.float64).copy()
    nyquist = 0.5 * sample_rate
    if not 0.0 < settings.cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff {settings.cutoff_hz} Hz must lie in (0, {nyquist}) Hz "
            f"for a {sample_rate} Hz signal"
        )
    array = np.asarray(values, dtype=np.float64)
    sos = signal.butter(
        settings.order, settings.cutoff_hz, btype="low", fs=sample_rate, output="sos"
    )
    # Padding must be shorter than the signal; sosfiltfilt's default padlen can
    # exceed short windows.
    padlen = min(3 * (2 * settings.order + 1), max(len(array) - 1, 0))
    if settings.zero_phase:
        return signal.sosfiltfilt(sos, array, axis=0, padlen=padlen)
    return signal.sosfilt(sos, array, axis=0)


def central_difference(
    values: NDArray[np.float64], times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Differentiate along axis 0 with second-order accurate differences."""
    return np.gradient(
        np.asarray(values, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        axis=0,
        edge_order=2,
    )


@dataclass(frozen=True)
class PreparedSignals:
    """Filtered joint states, torques, and the derived acceleration.

    ``ddq_rad_s2`` is differentiated from the *filtered* velocity and is a
    diagnostic quantity only: it feeds the regressor-based excitation and
    torque-reconstruction diagnostics, never the rollout fit.
    """

    times: NDArray[np.float64]
    q_rad: NDArray[np.float64]
    dq_rad_s: NDArray[np.float64]
    ddq_rad_s2: NDArray[np.float64]
    tau_Nm: NDArray[np.float64]
    settings: dict[str, object]


def prepare(
    times: NDArray[np.float64],
    q_rad: NDArray[np.float64],
    dq_rad_s: NDArray[np.float64],
    tau_Nm: NDArray[np.float64],
    settings: FilterSettings | None = None,
) -> PreparedSignals:
    """Filter measured signals identically and derive acceleration.

    Positions, velocities and torques go through the *same* filter so that no
    relative phase or magnitude distortion is introduced between them — an
    asymmetric filter would corrupt the torque/velocity relationship that
    friction identification reads.
    """
    settings = settings or FilterSettings()
    rate = sample_rate_hz(times)
    q = lowpass(q_rad, rate, settings)
    dq = lowpass(dq_rad_s, rate, settings)
    tau = lowpass(tau_Nm, rate, settings)
    return PreparedSignals(
        times=np.asarray(times, dtype=np.float64),
        q_rad=q,
        dq_rad_s=dq,
        ddq_rad_s2=central_difference(dq, times),
        tau_Nm=tau,
        settings=settings.describe(rate),
    )
