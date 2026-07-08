"""Fuse the 3 modality embeddings -> multi-horizon PV prediction.

Z = Concat(F_sat, H_met, H_graph)  (3 x 128 = 384) -> MLP -> per-horizon heads [B,3].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM, FUSION_DIM, N_HORIZONS


class FusionPredictor(nn.Module):
    def __init__(self, hidden_dim: int = EMBED_DIM * 2, dropout: float = 0.2, n_horizons: int = N_HORIZONS) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim // 2, 1) for _ in range(n_horizons)])

    def forward(self, f_sat: torch.Tensor, h_met: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        if not (f_sat.shape[-1] == h_met.shape[-1] == h_graph.shape[-1] == EMBED_DIM):
            raise ValueError(
                f"each modality must be {EMBED_DIM}-dim, got "
                f"{f_sat.shape[-1]}/{h_met.shape[-1]}/{h_graph.shape[-1]}"
            )
        z = torch.cat([f_sat, h_met, h_graph], dim=-1)        # [B, 384]
        hidden = self.fusion(z)
        return torch.cat([head(hidden) for head in self.heads], dim=-1)  # [B, n_horizons]
