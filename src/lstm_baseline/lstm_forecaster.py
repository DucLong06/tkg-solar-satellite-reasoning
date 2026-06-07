"""LSTM baseline forecaster (paper Table 6 LSTM row ~0.128 MAE).

Uses PV history + meteo only (satellite ignored in the baseline). Shares the
common model interface ``forward(batch) -> [B, N_HORIZONS]`` so the training loop
is model-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import N_HORIZONS, N_METEO_FEATURES, N_PV_FEATURES


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_horizons: int = N_HORIZONS,
    ) -> None:
        super().__init__()
        in_dim = N_METEO_FEATURES + N_PV_FEATURES
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["meteo_seq"], batch["pv_hist"]], dim=-1)  # [B,k,8]
        out, _ = self.lstm(x)
        last = out[:, -1, :]               # [B, hidden]
        return self.head(last)            # [B, n_horizons]
