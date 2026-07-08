"""Batched dense-graph GAT: 2 GATConv layers over many small graphs at once.

The dynamic adjacency [G, N, N] is dense (every entry > 0 after similarity +
self-loops + symmetric normalization), so the graph is fully connected. We build
a constant fully-connected, block-offset ``edge_index`` from ``arange`` (no grad,
no in-place hazard) and take ``edge_attr`` live from the adjacency so gradients
still flow to the TKG builder. All G graphs run through GATConv in one pass — no
Python loop over graphs/edges (kept simple; batch sizes are small).

(We avoid ``dense_to_sparse`` here: when the adjacency requires grad it returns an
edge_index that is later bumped in-place internally, breaking the edge_attr
gather's backward.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def fully_connected_edge_index(n_graphs: int, n_nodes: int, device) -> torch.Tensor:
    """Block-offset edge_index [2, G*N*N] for G fully-connected N-node graphs.

    Edge order is (g, i, j) with i outer, j inner — matches adjacency.reshape(G, N*N).
    """
    i = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)   # [N*N]
    j = torch.arange(n_nodes, device=device).repeat(n_nodes)              # [N*N]
    base = torch.stack([i, j], dim=0)                                     # [2, N*N]
    offsets = (torch.arange(n_graphs, device=device) * n_nodes).repeat_interleave(n_nodes * n_nodes)
    return base.repeat(1, n_graphs) + offsets.unsqueeze(0)                # [2, G*N*N]


class BatchedGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        # Self-loops already in A (i==j entries); disable GATConv's own injection.
        self.gat1 = GATConv(in_dim, hidden_dim, heads=n_heads, concat=True,
                            edge_dim=1, dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim * n_heads, out_dim, heads=1, concat=False,
                            edge_dim=1, dropout=dropout, add_self_loops=False)

    def forward(self, node_feats: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # node_feats [G, N, F], adjacency [G, N, N] -> [G, N, out_dim]
        g, n, f = node_feats.shape
        edge_index = fully_connected_edge_index(g, n, node_feats.device)  # constant
        edge_attr = adjacency.reshape(g * n * n, 1)                       # live grad
        x = node_feats.reshape(g * n, f)
        # GATConv mutates edge_index in place; pass clones to keep the constant pristine.
        x = F.relu(self.gat1(x, edge_index.clone(), edge_attr=edge_attr))
        x = self.gat2(x, edge_index.clone(), edge_attr=edge_attr)
        return x.reshape(g, n, -1)
