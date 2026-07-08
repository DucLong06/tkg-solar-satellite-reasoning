"""Evaluation harness: model + test loader -> per-horizon metrics (original units)."""

from __future__ import annotations

import torch

from src.common.shapes import HORIZON_MINUTES
from src.data_pipeline.scaling import Scalers
from src.metrics.regression_metrics import compute_per_horizon
from src.training.train_loop import predict_loader


@torch.no_grad()
def evaluate_model(
    model, loader, scalers: Scalers, device: str, mape_min_value: float = 1e-3,
    max_windows: int | None = None, progress: bool = False,
) -> dict:
    """Return per-horizon + overall MAE/RMSE/MAPE, inverse-scaled to original units.

    ``max_windows`` evenly subsamples the test windows before evaluating — used for
    the per-window ARIMA baseline, which fits one model per window and is otherwise
    far too slow on a multi-year test set (the subsample stays representative; note
    it in the report). ``progress`` shows a tqdm bar (see predict_loader).
    """
    if max_windows is not None and len(loader.dataset) > max_windows:
        from torch.utils.data import DataLoader, Subset
        step = len(loader.dataset) // max_windows
        idx = list(range(0, len(loader.dataset), step))[:max_windows]
        loader = DataLoader(Subset(loader.dataset, idx), batch_size=64)
    yt, yp = predict_loader(model, loader, device, progress=progress)
    return compute_per_horizon(
        scalers.inverse_pv(yt.numpy()),
        scalers.inverse_pv(yp.numpy()),
        HORIZON_MINUTES,
        mape_min_value,
    )
