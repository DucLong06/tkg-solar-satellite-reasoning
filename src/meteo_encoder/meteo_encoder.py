"""Meteo sequence encoder -> H_met [B, 128].

GRU (short-term dynamics) -> TransformerEncoder (long-range attention) -> mean-pool
over time -> projection to the locked 128-dim embedding (paper Table 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM, N_METEO_FEATURES, assert_embedding


class MeteoEncoder(nn.Module):
    def __init__(
        self,
        n_features: int = N_METEO_FEATURES,
        hidden_dim: int = EMBED_DIM,
        out_dim: int = EMBED_DIM,
        n_heads: int = 4,
        n_gru_layers: int = 2,
        n_transformer_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_embedding = nn.Linear(n_features, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_gru_layers,
            batch_first=True, dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, meteo_seq: torch.Tensor) -> torch.Tensor:
        # meteo_seq: [B, T, n_features]
        x = self.feature_embedding(meteo_seq)   # [B, T, hidden]
        x, _ = self.gru(x)                       # [B, T, hidden]
        x = self.transformer(x)                  # [B, T, hidden]
        pooled = x.mean(dim=1)                    # [B, hidden]
        out = self.out_proj(pooled)              # [B, out_dim]
        assert_embedding(out, "MeteoEncoder")
        return out
