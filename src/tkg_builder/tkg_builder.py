"""Build the per-sample, per-timestep TKG (node features + dynamic adjacency).

Produces, for an input window:
  node_feats_seq [B, k, N, node_dim]   typed nodes carrying the sample's features
  adjacency_seq  [B, k, N, N]          symmetric-normalized dynamic adjacency A_t

N = 8 (one node per entity type). See docs/assumptions.md for the per-sample graph
rationale and the edge-weight formula.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.shapes import N_METEO_FEATURES, N_PV_FEATURES
from src.tkg_builder.adjacency_builder import build_adjacency, geo_weight_matrix
from src.tkg_builder.entities import NODE_COORDS, N_ENTITY_TYPES

# Nodes beyond the 8 entity types repeat types via (% N_ENTITY_TYPES); n_nodes < 8
# drops the trailing entity types. Relation typing (the r in (h,r,o,t)) is deferred
# in the MVP — relation embeddings are not wired into the adjacency (see assumptions A6).


class TKGBuilder(nn.Module):
    def __init__(
        self,
        node_dim: int = 64,
        n_nodes: int = N_ENTITY_TYPES,
        edge_weights: dict | None = None,
    ) -> None:
        super().__init__()
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.edge_weights = edge_weights

        self.entity_type_embedding = nn.Embedding(N_ENTITY_TYPES, node_dim)
        self.feature_proj = nn.Linear(N_METEO_FEATURES + N_PV_FEATURES, node_dim)

        type_ids = torch.arange(n_nodes) % N_ENTITY_TYPES
        self.register_buffer("type_ids", type_ids, persistent=False)
        coords = torch.tensor(
            [NODE_COORDS[i % N_ENTITY_TYPES] for i in range(n_nodes)], dtype=torch.float32
        )
        self.register_buffer("geo", geo_weight_matrix(coords), persistent=False)

    def forward(self, meteo_seq: torch.Tensor, pv_hist: torch.Tensor):
        # meteo_seq [B,k,N_METEO_FEATURES], pv_hist [B,k,1]
        b, k, _ = meteo_seq.shape
        x = torch.cat([meteo_seq, pv_hist], dim=-1)          # [B,k,N_METEO+N_PV]
        proj = self.feature_proj(x)                          # [B,k,node_dim]

        type_emb = self.entity_type_embedding(self.type_ids)  # [N, node_dim]
        node_feats = proj.unsqueeze(2) + type_emb.view(1, 1, self.n_nodes, -1)  # [B,k,N,node_dim]

        adjacency = build_adjacency(node_feats, self.geo, self.edge_weights)     # [B,k,N,N]
        return node_feats, adjacency
