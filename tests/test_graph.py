"""M7/M8 tests: symmetric-normalization correctness, GraphLearner shape/grad/no-NaN."""

from __future__ import annotations

import torch

from src.common.shapes import EMBED_DIM, N_METEO_FEATURES
from src.tkg_builder.adjacency_builder import symmetric_normalize
from src.graph_learner.graph_learner import GraphLearner


def test_symmetric_normalize_matches_manual_formula():
    a = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    norm = symmetric_normalize(a)
    a_hat = a + torch.eye(3)
    deg = a_hat.sum(dim=1)
    expected = a_hat / torch.sqrt(deg).unsqueeze(0) / torch.sqrt(deg).unsqueeze(1)
    assert torch.allclose(norm, expected, atol=1e-5)
    assert torch.allclose(norm, norm.t(), atol=1e-6)   # symmetric
    assert not torch.isnan(norm).any()


def test_graph_learner_shape_and_grad():
    torch.manual_seed(0)
    gl = GraphLearner(node_dim=64, fourier_dim=16, n_nodes=8)
    meteo = torch.randn(3, 12, N_METEO_FEATURES, requires_grad=True)
    pv = torch.randn(3, 12, 1)
    out = gl(meteo, pv)
    assert out.shape == (3, EMBED_DIM)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert meteo.grad is not None
