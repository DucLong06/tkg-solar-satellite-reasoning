"""MAE / RMSE / MAPE on inverse-scaled (original-unit) PV values.

Night-time MAPE convention: solar PV is ~0 at night, so MAPE is
undefined/explosive for ~half the samples. We compute MAPE ONLY over samples
where |y_true| >= ``mape_min_value`` (a small absolute daytime floor) and report
the share excluded. The paper's masking convention is unknown, so MAPE is the
least comparable of the three headline metrics — documented, not hidden.
"""

from __future__ import annotations

import numpy as np

ArrayLike = "np.ndarray | list | float"


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype="float64")


def mae(y_true, y_pred) -> float:
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, mape_min_value: float = 1e-3) -> float:
    """Percent error over daytime samples only (|y_true| >= mape_min_value)."""
    yt, yp = _to_np(y_true), _to_np(y_pred)
    mask = np.abs(yt) >= mape_min_value
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def compute_all(y_true, y_pred, mape_min_value: float = 1e-3) -> dict[str, float]:
    """All three metrics + the fraction of samples excluded from MAPE."""
    yt = _to_np(y_true)
    excluded = float(np.mean(np.abs(yt) < mape_min_value))
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred, mape_min_value),
        "mape_excluded_frac": excluded,
    }


def compute_per_horizon(
    y_true, y_pred, horizon_labels=(10, 30, 60), mape_min_value: float = 1e-3
) -> dict:
    """Per-horizon + overall metrics for [N, H] arrays."""
    yt, yp = _to_np(y_true), _to_np(y_pred)
    if yt.ndim != 2:
        raise ValueError(f"expected [N,H] arrays, got {yt.shape}")
    out = {"overall": compute_all(yt.ravel(), yp.ravel(), mape_min_value)}
    for j, label in enumerate(horizon_labels):
        out[f"{label}min"] = compute_all(yt[:, j], yp[:, j], mape_min_value)
    return out
