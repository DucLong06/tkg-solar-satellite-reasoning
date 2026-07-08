"""Baseline forecasters for the benchmark comparison rows.

All share the model interface ``forward(batch) -> [B, N_HORIZONS]``:
  - Persistence          : P-hat(t+dt) = P(t)  (parameter-free, EVAL_ONLY)
  - ARIMAForecaster      : per-window ARIMA(5,1,2) via statsmodels (EVAL_ONLY)
  - GRUForecaster        : PV+meteo -> GRU -> head
  - TransformerForecaster: PV+meteo -> TransformerEncoder -> head
  - TemporalGNNForecaster: graph-only (tkg_builder + graph_learner) -> head
    (proposed model minus the satellite+meteo fusion)

EVAL_ONLY models have no trainable params (or fit outside autograd) and MUST
bypass the Adam training loop — the runner routes them through an eval-only path
(see scripts/run_baselines.py). The LSTM baseline lives in src/lstm_baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import (
    EMBED_DIM,
    HORIZON_STEPS,
    N_HORIZONS,
    N_METEO_FEATURES,
    N_PV_FEATURES,
)
from src.graph_learner.graph_learner import GraphLearner

_IN = N_METEO_FEATURES + N_PV_FEATURES


def _seq_input(batch: dict) -> torch.Tensor:
    return torch.cat([batch["meteo_seq"], batch["pv_hist"]], dim=-1)  # [B,k,_IN]


class Persistence(nn.Module):
    """Naive persistence: forecast = last observed PV, repeated across horizons.

    Parameter-free -> EVAL_ONLY (Adam would choke on an empty parameter list).
    """

    EVAL_ONLY = True

    def forward(self, batch: dict) -> torch.Tensor:
        last = batch["pv_hist"][:, -1, :]          # [B,1] last observed PV (scaled)
        return last.expand(-1, N_HORIZONS).contiguous()


class ARIMAForecaster(nn.Module):
    """Per-window ARIMA(p,d,q) forecast via statsmodels (EVAL_ONLY).

    For each sample, fit ARIMA on its k-step PV history and forecast the
    max-horizon ahead, then select the horizon steps. Fixed order (default 5,1,2);
    AIC tuning per window is too slow over a multi-year test set. statsmodels is an
    optional dependency (``pip install statsmodels``); imported lazily so the core
    install stays light.
    """

    EVAL_ONLY = True

    def __init__(self, order: tuple[int, int, int] = (5, 1, 2)) -> None:
        super().__init__()
        self.order = order
        self._max_h = max(HORIZON_STEPS)
        self._hz = [h - 1 for h in HORIZON_STEPS]  # 0-based index into the forecast

    def forward(self, batch: dict) -> torch.Tensor:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "ARIMA baseline needs statsmodels: pip install statsmodels"
            ) from e
        import warnings

        hist = batch["pv_hist"][:, :, 0].detach().cpu().numpy()  # [B,k]
        preds = []
        for series in hist:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(series, order=self.order).fit()
                    fc = fit.forecast(steps=self._max_h)
                preds.append([float(fc[i]) for i in self._hz])
            except Exception:  # singular / non-convergent window -> persistence fallback
                preds.append([float(series[-1])] * N_HORIZONS)
        return torch.tensor(preds, dtype=torch.float32)


class GRUForecaster(nn.Module):
    def __init__(self, hidden_dim: int = EMBED_DIM, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(_IN, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, N_HORIZONS)

    def forward(self, batch: dict) -> torch.Tensor:
        out, _ = self.gru(_seq_input(batch))
        return self.head(out[:, -1, :])


class TransformerForecaster(nn.Module):
    def __init__(self, hidden_dim: int = EMBED_DIM, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Linear(_IN, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 2,
                                           batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(hidden_dim, N_HORIZONS)

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.encoder(self.embed(_seq_input(batch)))
        return self.head(x.mean(dim=1))


class TemporalGNNForecaster(nn.Module):
    def __init__(self, n_nodes: int = 8, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.graph = GraphLearner(out_dim=EMBED_DIM, n_nodes=n_nodes, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM // 2), nn.ReLU(),
                                  nn.Linear(EMBED_DIM // 2, N_HORIZONS))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.graph(batch["meteo_seq"], batch["pv_hist"]))


def build_baseline(name: str, cfg) -> nn.Module:
    name = name.lower()
    if name == "persistence":
        return Persistence()
    if name == "arima":
        return ARIMAForecaster()
    if name == "gru":
        return GRUForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name == "transformer":
        return TransformerForecaster(hidden_dim=cfg.embed_dim, n_heads=cfg.n_heads, dropout=cfg.dropout)
    if name in ("temporal-gnn", "temporal_gnn", "tgnn"):
        return TemporalGNNForecaster(n_nodes=cfg.n_graph_nodes, n_heads=cfg.n_heads, dropout=cfg.dropout)
    raise ValueError(f"unknown baseline {name!r}")


def is_eval_only(model: nn.Module) -> bool:
    """True for parameter-free / non-autograd baselines that must skip Adam."""
    return getattr(model, "EVAL_ONLY", False)
