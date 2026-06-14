"""Align co-located DKASC PV+meteo + Himawari frames to a common 5-min UTC grid.

DKASC PV, meteo, and the Himawari ROI are physically co-located (Alice Springs),
so "aligned" is a genuine same-site timestamp join — the old cross-continent
intersection gate is gone. We still require a non-empty join (>= min_steps) so
windowing never silently proceeds on a near-empty dataset, and we drop night
samples (GHI below a small threshold) since PV is ~0 at night.

``build_common_grid`` (legacy 3-source, 10-min) is retained below for reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.shapes import BASE_CADENCE_MIN, SAT_IMG_SIZE


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


def align_colocated(
    pv: pd.Series,
    meteo: pd.DataFrame,
    sat_frames: np.ndarray,
    sat_ts: pd.DatetimeIndex,
    *,
    cadence_min: int = BASE_CADENCE_MIN,
    min_steps: int = 200,
    img_size: int = SAT_IMG_SIZE,
    night_ghi_thresh: float = 5.0,
    max_interp_gap: int = 3,
) -> dict:
    """Align co-located DKASC + Himawari to the common 5-min grid (daytime only).

    Steps: build a ``cadence_min`` grid over the overlapping span; interpolate PV
    and meteo onto it (only across gaps < ``max_interp_gap`` samples); pick the
    nearest satellite frame within one cadence step; keep timestamps where all
    sources are present AND ``GHI >= night_ghi_thresh`` (daytime); assert
    >= ``min_steps``.

    Note: dropping night rows makes the kept array temporally non-contiguous across
    day boundaries; sliding windows therefore span day breaks (documented
    approximation, see docs/assumptions.md).
    """
    start = max(pv.index.min(), meteo.index.min(), sat_ts.min())
    end = min(pv.index.max(), meteo.index.max(), sat_ts.max())
    if start >= end:
        raise EmptyOverlapError(
            "No overlapping date range across DKASC PV/meteo + Himawari.\n"
            f"  PV   : {pv.index.min()} .. {pv.index.max()}\n"
            f"  meteo: {meteo.index.min()} .. {meteo.index.max()}\n"
            f"  Himawari: {sat_ts.min()} .. {sat_ts.max()}"
        )

    grid = pd.date_range(start, end, freq=f"{cadence_min}min", tz="UTC")
    pv_g = (
        pv.reindex(pv.index.union(grid))
        .interpolate(method="time", limit=max_interp_gap)
        .reindex(grid)
    )
    meteo_g = (
        meteo.reindex(meteo.index.union(grid))
        .interpolate(method="time", limit=max_interp_gap)
        .reindex(grid)
    )

    sat_series = pd.Series(range(len(sat_ts)), index=sat_ts)
    sat_idx = sat_series.reindex(grid, method="nearest", tolerance=pd.Timedelta(minutes=cadence_min))

    present = pv_g.notna() & meteo_g.notna().all(axis=1) & sat_idx.notna()
    daytime = meteo_g["ghi"] >= night_ghi_thresh  # PV ~0 at night -> drop
    mask = (present & daytime).to_numpy()
    n = int(mask.sum())
    if n < min_steps:
        raise EmptyOverlapError(
            f"Co-located daytime join too thin: {n} steps < min_steps={min_steps}.\n"
            f"  overlap span {start} .. {end}\n"
            "  Widen the data window, lower min_steps, or check GHI/night threshold."
        )

    sat_sel = sat_idx[mask].to_numpy().astype(int)
    frames = _resize_frames(sat_frames[sat_sel], img_size)
    return {
        "timestamps": grid[mask],
        "pv": pv_g[mask].to_numpy(dtype="float32"),        # [T]
        "meteo": meteo_g[mask].to_numpy(dtype="float32"),  # [T, n_feat]
        "sat": frames,                                     # [T, C, H, W]
    }


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
