"""Gates for measurement preprocessing (filtering and differentiation)."""

from __future__ import annotations

import numpy as np
import pytest

from fer_mujoco_sysid.preprocessing import (
    FilterSettings,
    central_difference,
    lowpass,
    prepare,
    sample_rate_hz,
)

_RATE_HZ = 1000.0
_DURATION_S = 2.0


def _time() -> np.ndarray:
    return np.arange(int(_RATE_HZ * _DURATION_S)) / _RATE_HZ


def test_sample_rate_and_monotonicity() -> None:
    assert sample_rate_hz(_time()) == pytest.approx(_RATE_HZ, rel=1e-9)
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_rate_hz(np.array([0.0, 0.002, 0.001]))


def test_lowpass_removes_noise_and_keeps_signal() -> None:
    """A 1 Hz signal survives an 8 Hz cutoff; 200 Hz noise does not."""
    time = _time()
    clean = np.column_stack([np.sin(2 * np.pi * 1.0 * time)] * 7)
    rng = np.random.default_rng(0)
    noisy = clean + 0.05 * rng.standard_normal(clean.shape)

    filtered = lowpass(noisy, _RATE_HZ, FilterSettings(order=4, cutoff_hz=8.0))

    edge = int(0.1 * _RATE_HZ)  # ignore filter edges
    noise_before = np.abs(noisy - clean)[edge:-edge].std()
    noise_after = np.abs(filtered - clean)[edge:-edge].std()
    assert noise_after < 0.25 * noise_before
    assert np.abs(filtered - clean)[edge:-edge].max() < 0.02


def test_lowpass_is_zero_phase() -> None:
    """filtfilt must not shift the signal in time; a causal filter does."""
    time = _time()
    signal = np.column_stack([np.sin(2 * np.pi * 3.0 * time)])
    settings = FilterSettings(cutoff_hz=8.0)

    zero_phase = lowpass(signal, _RATE_HZ, settings)
    causal = lowpass(signal, _RATE_HZ, FilterSettings(cutoff_hz=8.0, zero_phase=False))

    def lag_samples(filtered: np.ndarray) -> int:
        window = slice(int(0.2 * _RATE_HZ), int(1.8 * _RATE_HZ))
        reference = signal[window, 0]
        candidate = filtered[window, 0]
        shifts = np.arange(-60, 61)
        errors = [
            np.mean((np.roll(candidate, -shift) - reference) ** 2) for shift in shifts
        ]
        return int(shifts[int(np.argmin(errors))])

    assert lag_samples(zero_phase) == 0
    assert lag_samples(causal) != 0


def test_lowpass_rejects_cutoff_above_nyquist() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        lowpass(np.zeros((100, 7)), 100.0, FilterSettings(cutoff_hz=60.0))


def test_lowpass_handles_short_windows() -> None:
    """Short segments must not trip the filter's padding requirement."""
    short = np.random.default_rng(1).standard_normal((25, 7))
    assert lowpass(short, _RATE_HZ).shape == short.shape


def test_disabled_filter_preserves_exact_simulation_rows() -> None:
    values = np.random.default_rng(2).standard_normal((100, 7))
    settings = FilterSettings(enabled=False)
    unfiltered = lowpass(values, _RATE_HZ, settings)

    np.testing.assert_array_equal(unfiltered, values)
    assert unfiltered is not values
    description = settings.describe(_RATE_HZ)
    assert description["kind"] == "none"
    assert description["enabled"] is False


def test_central_difference_matches_analytic_derivative() -> None:
    time = _time()
    values = np.column_stack([np.sin(2 * np.pi * 1.0 * time)])
    expected = 2 * np.pi * np.cos(2 * np.pi * 1.0 * time)
    derivative = central_difference(values, time)[:, 0]
    assert np.abs(derivative - expected).max() < 1e-3


def test_prepare_reports_its_settings_and_derives_acceleration() -> None:
    time = _time()
    q = np.column_stack([np.sin(2 * np.pi * 0.5 * time)] * 7)
    dq = np.column_stack([np.pi * np.cos(2 * np.pi * 0.5 * time)] * 7)
    tau = np.ones((len(time), 7))

    prepared = prepare(time, q, dq, tau)

    assert prepared.settings["kind"] == "butterworth_lowpass"
    assert prepared.settings["order"] == 4
    assert prepared.settings["zero_phase"] is True
    assert prepared.settings["sample_rate_hz"] == pytest.approx(_RATE_HZ)
    # Interior only: filtfilt padding distorts both ends. Tolerance is 1% of
    # the pi^2 ~ 9.87 rad/s^2 amplitude.
    expected = -(np.pi**2) * np.sin(2 * np.pi * 0.5 * time)
    edge = int(0.2 * _RATE_HZ)
    error = np.abs(prepared.ddq_rad_s2[edge:-edge, 0] - expected[edge:-edge]).max()
    assert error < 0.1
