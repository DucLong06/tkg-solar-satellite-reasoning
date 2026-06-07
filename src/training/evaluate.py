"""Evaluation harness: model + test loader -> per-horizon metrics (original units)."""

from __future__ import annotations

import torch

from src.common.shapes import HORIZON_MINUTES
from src.data_pipeline.scaling import Scalers
from src.metrics.regression_metrics import compute_per_horizon
from src.training.train_loop import predict_loader


@torch.no_grad()
def evaluate_model(model, loader, scalers: Scalers, device: str, mape_min_value: float = 1e-3) -> dict:
    """Return per-horizon + overall MAE/RMSE/MAPE, inverse-scaled to original units."""
    yt, yp = predict_loader(model, loader, device)
    return compute_per_horizon(
        scalers.inverse_pv(yt.numpy()),
        scalers.inverse_pv(yp.numpy()),
        HORIZON_MINUTES,
        mape_min_value,
    )
