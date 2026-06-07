"""Missing-data fill + outlier clipping for the aligned arrays.

Short gaps were already interpolated in alignment; here we backstop any residual
NaNs and clip extreme outliers on continuous meteo/PV channels (z-score based).
Satellite frames are clamped to a valid reflectance range.
"""

from __future__ import annotations

import numpy as np


def fill_residual_nans(arr: np.ndarray) -> np.ndarray:
    """Forward/back-fill residual NaNs along time (axis 0); zero any leftover."""
    out = arr.copy()
    if out.ndim == 1:
        out = out[:, None]
        squeeze = True
    else:
        squeeze = False
    # Forward fill then backward fill along time.
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = np.where(~np.isnan(col))[0]
        if idx.size == 0:
            col[:] = 0.0
            continue
        col[: idx[0]] = col[idx[0]]            # back-fill leading
        last = idx[0]
        for i in range(out.shape[0]):
            if np.isnan(col[i]):
                col[i] = col[last]
            else:
                last = i
        out[:, j] = col
    return out[:, 0] if squeeze else out


def fit_clip_bounds(train_arr: np.ndarray, z_thresh: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ±z_thresh-sigma clip bounds from the TRAIN split only (no leakage)."""
    flat = train_arr.reshape(train_arr.shape[0], -1) if train_arr.ndim > 1 else train_arr[:, None]
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (mean - z_thresh * std), (mean + z_thresh * std)


def apply_clip(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Apply precomputed clip bounds to a (sub)array, preserving its shape."""
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
    clipped = np.clip(flat, lo, hi)
    return clipped.reshape(arr.shape) if arr.ndim > 1 else clipped[:, 0]


def clean_arrays(data: dict) -> dict:
    """NaN fill on pv/meteo + satellite clamp to [0,1].

    Outlier clipping is deliberately NOT done here: it requires train-only stats,
    so it is applied later in the pipeline AFTER the chronological split (see
    fit_clip_bounds / apply_clip). Keeping it out of the pre-split cache avoids the
    full-series-statistics leakage the otherwise-strict pipeline would violate.
    """
    data = dict(data)
    data["pv"] = fill_residual_nans(data["pv"])
    data["meteo"] = fill_residual_nans(data["meteo"])
    sat = np.nan_to_num(data["sat"], nan=0.0)
    data["sat"] = np.clip(sat, 0.0, 1.0).astype("float32")
    return data
