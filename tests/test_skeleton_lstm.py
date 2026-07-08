"""Phase 2 skeleton test: LSTM baseline trains end-to-end and produces finite metrics."""

from __future__ import annotations

import math

import torch

from src.common.seeding import seed_everything
from src.training.config import Config
from src.training.train_loop import fit, predict_loader
from src.lstm_baseline.lstm_forecaster import LSTMForecaster


def test_lstm_baseline_trains(pipeline_splits, tmp_path):
    seed_everything(42)
    cfg = Config(model="lstm", epochs=2, embed_dim=128, batch_size=16,
                 checkpoint_dir=str(tmp_path), early_stop_patience=5)
    model = LSTMForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)

    history = fit(model, pipeline_splits, cfg, verbose=False)
    assert math.isfinite(history["best_val_mae"])
    assert (tmp_path / "best_lstm.pt").exists()

    yt, yp = predict_loader(model, pipeline_splits.test_loader, cfg.device)
    assert yp.shape == yt.shape
    assert not torch.isnan(yp).any()
