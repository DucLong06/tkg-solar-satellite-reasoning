"""Sliding-window index math (lazy — no window materialisation).

To avoid blowing up memory by replicating satellite frames k times per window,
windowing only computes the valid window START indices. The Dataset slices the
contiguous arrays on __getitem__.

A window starting at i uses input steps [i, i+k); the "current" time is i+k-1;
targets are PV at (i+k-1 + h) for each horizon step h in HORIZON_STEPS.
"""

from __future__ import annotations

import numpy as np

from src.common.shapes import HORIZON_STEPS


def valid_starts(n_steps: int, k: int, horizon_steps: tuple[int, ...] = HORIZON_STEPS) -> np.ndarray:
    """Start indices i such that input [i,i+k) and all horizon targets fit in [0,n)."""
    max_h = max(horizon_steps)
    last = n_steps - k - max_h
    if last < 0:
        return np.empty((0,), dtype=int)
    return np.arange(0, last + 1, dtype=int)


def target_indices(start: int, k: int, horizon_steps: tuple[int, ...] = HORIZON_STEPS) -> list[int]:
    """Absolute indices of the horizon targets for a window starting at ``start``."""
    current = start + k - 1
    return [current + h for h in horizon_steps]
