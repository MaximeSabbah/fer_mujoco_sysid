"""Turn explicit recording masks into causal, contiguous fitting runs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from fer_mujoco_sysid.fitting import MeasuredRun


def contiguous_regions(
    mask: NDArray[np.bool_],
    *,
    minimum_samples: int = 2,
) -> tuple[slice, ...]:
    """Return half-open slices for the sufficiently long true runs in *mask*."""
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.ndim != 1:
        raise ValueError("selection mask must be one-dimensional")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    padded = np.pad(selected.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_samples
    )


def select_run(
    run: MeasuredRun,
    rows: slice,
    *,
    label: str | None = None,
) -> MeasuredRun:
    """Extract one contiguous interval and rebase its timestamps.

    Arbitrary masked rows must never be concatenated into a rollout: doing so
    invents a state transition and asks the optimizer to explain it with
    dynamics. A selected interval therefore becomes its own run, initialized
    from its first measured state.
    """
    start = 0 if rows.start is None else int(rows.start)
    stop = len(run.control) if rows.stop is None else int(rows.stop)
    if start < 0 or stop > len(run.control) or stop - start < 2:
        raise ValueError(f"invalid run interval [{start}, {stop})")

    origin = float(run.control_times[start])
    control_times = run.control_times[start:stop] - origin
    measured_times = run.measured_times[start:stop] - origin
    measured = np.asarray(run.measured[start:stop], dtype=np.float64)
    return replace(
        run,
        label=label or f"{run.label}[{start}:{stop}]",
        qpos0=measured[0, :7].copy(),
        qvel0=measured[0, 7:14].copy(),
        control_times=control_times,
        control=np.asarray(run.control[start:stop], dtype=np.float64),
        measured_times=measured_times,
        measured=measured,
    )


def runs_from_mask(
    run: MeasuredRun,
    mask: NDArray[np.bool_],
    *,
    minimum_samples: int = 2,
) -> tuple[MeasuredRun, ...]:
    """Split every selected interval into an independent rollout."""
    if len(mask) != len(run.control):
        raise ValueError(
            f"mask has {len(mask)} rows for a run with {len(run.control)} samples"
        )
    regions = contiguous_regions(mask, minimum_samples=minimum_samples)
    if not regions:
        raise ValueError(f"{run.label} has no usable selected interval")
    return tuple(
        select_run(run, region, label=f"{run.label}:{index}")
        for index, region in enumerate(regions)
    )
