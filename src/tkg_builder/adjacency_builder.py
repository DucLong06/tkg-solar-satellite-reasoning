"""Dynamic adjacency construction + symmetric normalization for the TKG.

Edge-weight formula (documented assumption — paper gives no explicit formula):
    A = w_geo * geo + w_meteo * feat_sim + w_cloud * 0 + w_pv * 0
The paper lists 4 edge components (geo distance, meteo similarity, cloud motion,
PV-power correlation). In this MVP only geo + per-sample feature similarity are
available; cloud-motion and PV-correlation have no per-node signal yet, so their
terms are 0 (kept in the formula for fidelity). Default weights follow the
research default [0.3, 0.3, 0.2, 0.2].

Geo caveat: with the faithful-to-paper mismatched data the geo term
is not physically meaningful — see docs/assumptions.md.
"""

from __future__ import annotations

import torch

DEFAULT_EDGE_WEIGHTS = {"geo": 0.3, "meteo": 0.3, "cloud": 0.2, "pv": 0.2}


def geo_weight_matrix(coords: torch.Tensor, length_scale: float = 0.1) -> torch.Tensor:
    """Gaussian-of-distance affinity [N, N] in (0, 1] from node (lat, lon)."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)        # [N, N, 2]
    dist = torch.linalg.norm(diff, dim=-1)                  # [N, N]
    return torch.exp(-(dist ** 2) / (2 * length_scale ** 2))


def feature_similarity(node_feats: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity in [0, 1] -> [..., N, N]."""
    x = torch.nn.functional.normalize(node_feats, dim=-1)   # [..., N, F]
    cos = torch.matmul(x, x.transpose(-1, -2))              # [..., N, N] in [-1,1]
    return (cos + 1.0) / 2.0


def symmetric_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute D^{-1/2} (A + I) D^{-1/2} for a (batched) [..., N, N] adjacency."""
    n = adj.shape[-1]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    a_hat = adj + eye
    deg = a_hat.sum(dim=-1).clamp(min=eps)                  # [..., N]
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * a_hat * d_inv_sqrt.unsqueeze(-2)


def build_adjacency(
    node_feats: torch.Tensor,
    geo: torch.Tensor,
    edge_weights: dict | None = None,
) -> torch.Tensor:
    """Build symmetric-normalized A from node features + fixed geo affinity.

    node_feats: [..., N, F]; geo: [N, N]; returns [..., N, N].
    """
    w = edge_weights or DEFAULT_EDGE_WEIGHTS
    sim = feature_similarity(node_feats)                    # [..., N, N]
    # Broadcast geo over leading (batch/time) dims.
    a = w["geo"] * geo + w["meteo"] * sim                   # cloud/pv terms = 0 (MVP)
    a = (a + a.transpose(-1, -2)) / 2.0                     # enforce symmetry
    return symmetric_normalize(a)
