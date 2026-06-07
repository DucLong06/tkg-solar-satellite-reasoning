"""Baseline forecasters for the Table 6 comparison rows.

All share the model interface ``forward(batch) -> [B, N_HORIZONS]``:
  - GRUForecaster        : PV+meteo -> GRU -> head
  - TransformerForecaster: PV+meteo -> TransformerEncoder -> head
  - TemporalGNNForecaster: graph-only (tkg_builder + graph_learner) -> head
    (proposed model minus the satellite+meteo fusion)

The LSTM baseline lives in src/lstm_baseline. Kept minimal (don't
let baselines explode scope).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM, N_HORIZONS, N_METEO_FEATURES, N_PV_FEATURES
from src.graph_learner.graph_learner import GraphLearner

_IN = N_METEO_FEATURES + N_PV_FEATURES


def _seq_input(batch: dict) -> torch.Tensor:
    return torch.cat([batch["meteo_seq"], batch["pv_hist"]], dim=-1)  # [B,k,8]


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
    if name == "gru":
        return GRUForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name == "transformer":
        return TransformerForecaster(hidden_dim=cfg.embed_dim, n_heads=cfg.n_heads, dropout=cfg.dropout)
    if name in ("temporal-gnn", "temporal_gnn", "tgnn"):
        return TemporalGNNForecaster(n_nodes=cfg.n_graph_nodes, n_heads=cfg.n_heads, dropout=cfg.dropout)
    raise ValueError(f"unknown baseline {name!r}")
