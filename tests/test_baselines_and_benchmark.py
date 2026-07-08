"""Phase 6 tests: baseline models forward [B,3], benchmark table + ordering."""

from __future__ import annotations

import torch

from src.common.shapes import N_HORIZONS, N_METEO_FEATURES
from src.evaluation.baseline_models import build_baseline, is_eval_only
from src.evaluation.benchmark_table import build_benchmark, relative_ordering
from src.training.config import Config


def _dummy_batch():
    return {
        "meteo_seq": torch.randn(2, 12, N_METEO_FEATURES),
        "pv_hist": torch.randn(2, 12, 1),
        "sat_seq": torch.randn(2, 12, 1, 64, 64),
        "target": torch.rand(2, 3),
    }


def test_baselines_forward_shape():
    cfg = Config()
    batch = _dummy_batch()
    for name in ("gru", "transformer", "temporal-gnn"):
        out = build_baseline(name, cfg)(batch)
        assert out.shape == (2, N_HORIZONS), name
        assert not torch.isnan(out).any()


def test_persistence_eval_only():
    cfg = Config()
    model = build_baseline("persistence", cfg)
    assert is_eval_only(model) and not list(model.parameters())  # no trainable params
    batch = _dummy_batch()
    out = model(batch)
    assert out.shape == (2, N_HORIZONS)
    # forecast == last observed PV repeated across horizons
    last = batch["pv_hist"][:, -1, :]
    assert torch.allclose(out, last.expand(-1, N_HORIZONS))


def test_benchmark_table_and_ordering():
    res = {"LSTM": {"mae": 0.5, "rmse": 0.7, "mape": 12.0},
           "Proposed": {"mae": 0.3, "rmse": 0.5, "mape": 8.0}}
    table = build_benchmark(res)
    assert "MAE (paper)" in table and "0.128" in table  # paper LSTM target present
    assert relative_ordering(res) == ["Proposed", "LSTM"]
