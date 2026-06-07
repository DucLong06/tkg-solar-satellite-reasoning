"""Fourier (sinusoidal) positional features for periodicity awareness.

These are Fourier *features* — sin/cos of the timestep position at multiple
frequencies — NOT an FFT over a signal axis (the two were conflated in the source
notes; we deliberately use sinusoidal features, which is the standard, stable
choice for injecting temporal periodicity into node features).
"""

from __future__ import annotations

import math

import torch


def build_fourier_features(n_steps: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return [n_steps, dim] sinusoidal features over normalized position t/n_steps.

    Half the dims are sin, half cos, across geometrically-spaced frequencies.
    """
    if dim % 2 != 0:
        raise ValueError("fourier dim must be even")
    pos = torch.arange(n_steps, device=device, dtype=dtype).unsqueeze(1) / max(n_steps, 1)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = pos * freqs.unsqueeze(0) * (2 * math.pi)      # [n_steps, half]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [n_steps, dim]
