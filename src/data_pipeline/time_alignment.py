"""Align the 3 sources to a common 10-min UTC grid + enforce the overlap gate.

The sources are non-co-located (Europe PV vs Asia meteo/sat), so "aligned" means
clock-aligned only, not physically paired — see docs/assumptions.md. We still
require a genuine non-empty timestamp INTERSECTION so windowing has real samples
(never silently proceed on a near-empty dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.shapes import SAT_IMG_SIZE


class EmptyOverlapError(RuntimeError):
    """Raised when the 3-source timestamp intersection is too thin to train on."""


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of [T,C,H,W] frames to [T,C,size,size]."""
    t, c, h, w = frames.shape
    if (h, w) == (size, size):
        return frames
    yi = (np.linspace(0, h - 1, size)).round().astype(int)
    xi = (np.linspace(0, w - 1, size)).round().astype(int)
    return frames[:, :, yi][:, :, :, xi]


def build_common_grid(
    pv: pd.Series,
    meteo: pd.DataFrame,
    sat_frames: np.ndarray,
    sat_ts: pd.DatetimeIndex,
    cadence_min: int = 10,
    min_steps: int = 200,
    img_size: int = SAT_IMG_SIZE,
) -> dict:
    """Return aligned arrays on the common 10-min grid.

    Steps: build a 10-min grid over the overlapping date range; interpolate the
    15-min OPSD PV onto it; reindex meteo + satellite; keep only timestamps where
    ALL three have data; assert >= ``min_steps``.
    """
    # Overlapping date span across the 3 sources.
    start = max(pv.index.min(), meteo.index.min(), sat_ts.min())
    end = min(pv.index.max(), meteo.index.max(), sat_ts.max())
    if start >= end:
        raise EmptyOverlapError(
            "No overlapping date range across sources.\n"
            f"  OPSD : {pv.index.min()} .. {pv.index.max()}\n"
            f"  NSRDB: {meteo.index.min()} .. {meteo.index.max()}\n"
            f"  Himawari: {sat_ts.min()} .. {sat_ts.max()}"
        )

    grid = pd.date_range(start, end, freq=f"{cadence_min}min", tz="UTC")

    # PV: 15-min -> 10-min via time interpolation (paper: OPSD interpolated).
    pv_g = pv.reindex(pv.index.union(grid)).interpolate(method="time").reindex(grid)
    meteo_g = meteo.reindex(meteo.index.union(grid)).interpolate(method="time").reindex(grid)

    sat_series = pd.Series(range(len(sat_ts)), index=sat_ts)
    sat_idx = sat_series.reindex(grid, method="nearest", tolerance=pd.Timedelta(minutes=cadence_min))

    # Intersection mask: every source present at this grid timestamp.
    mask = pv_g.notna() & meteo_g.notna().all(axis=1) & sat_idx.notna()
    n = int(mask.sum())
    if n < min_steps:
        raise EmptyOverlapError(
            f"Common 3-source intersection too thin: {n} steps < min_steps={min_steps}.\n"
            f"  overlap span {start} .. {end}\n"
            "  Widen the data window or lower min_steps for a smoke run."
        )

    grid = grid[mask.to_numpy()]
    sat_sel = sat_idx[mask.to_numpy()].to_numpy().astype(int)
    frames = _resize_frames(sat_frames[sat_sel], img_size)

    return {
        "timestamps": grid,
        "pv": pv_g[mask.to_numpy()].to_numpy(dtype="float32"),       # [T]
        "meteo": meteo_g[mask.to_numpy()].to_numpy(dtype="float32"),  # [T, n_feat]
        "sat": frames,                                                # [T, C, H, W]
    }
