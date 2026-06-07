"""Graph representation learner -> H_graph [B, 128].

Pipeline: the TKG builder builds per-timestep node features + adjacency -> append Fourier
positional features -> batched GAT per timestep -> mean-pool nodes -> temporal
Transformer over the k step-embeddings -> mean-pool time -> project to 128.

Returns ONE [B, 128] embedding per batch sample (per-sample graph design — fixes
the batch<->node correspondence; see docs/assumptions.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM, assert_embedding
from src.tkg_builder.tkg_builder import TKGBuilder
from src.graph_learner.fourier_features import build_fourier_features
from src.graph_learner.gat_layer import BatchedGAT
from src.graph_learner.temporal_transformer import TemporalTransformer


class GraphLearner(nn.Module):
    def __init__(
        self,
        node_dim: int = 64,
        fourier_dim: int = 16,
        gat_dim: int = EMBED_DIM,
        out_dim: int = EMBED_DIM,
        n_nodes: int = 8,
        n_heads: int = 4,
        dropout: float = 0.2,
        edge_weights: dict | None = None,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.builder = TKGBuilder(node_dim=node_dim, n_nodes=n_nodes, edge_weights=edge_weights)
        self.fourier_dim = fourier_dim
        self.use_temporal = use_temporal
        self.gat = BatchedGAT(node_dim + fourier_dim, gat_dim, gat_dim, n_heads=n_heads, dropout=dropout)
        self.temporal = TemporalTransformer(gat_dim, n_heads=n_heads, dropout=dropout)
        self.out_proj = nn.Linear(gat_dim, out_dim)

    def forward(self, meteo_seq: torch.Tensor, pv_hist: torch.Tensor) -> torch.Tensor:
        node_feats, adjacency = self.builder(meteo_seq, pv_hist)   # [B,k,N,nd], [B,k,N,N]
        b, k, n, _ = node_feats.shape

        fourier = build_fourier_features(k, self.fourier_dim, device=node_feats.device)  # [k, fd]
        fourier = fourier.view(1, k, 1, self.fourier_dim).expand(b, k, n, self.fourier_dim)
        node_feats = torch.cat([node_feats, fourier], dim=-1)      # [B,k,N,nd+fd]

        g = b * k
        gat_out = self.gat(node_feats.reshape(g, n, -1), adjacency.reshape(g, n, n))  # [G,N,gat]
        graph_emb = gat_out.mean(dim=1).reshape(b, k, -1)          # pool nodes -> [B,k,gat]

        if self.use_temporal:
            graph_emb = self.temporal(graph_emb)                  # [B,k,gat]
        h = graph_emb.mean(dim=1)                                  # pool time -> [B,gat]
        out = self.out_proj(h)
        assert_embedding(out, "GraphLearner")
        return out
