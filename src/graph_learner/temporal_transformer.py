"""Temporal Transformer over per-timestep graph embeddings -> sequence context.

Input [B, k, dim] (one graph embedding per window step) -> [B, k, dim].
The GraphLearner mean-pools the output over time to produce H_graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            batch_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
