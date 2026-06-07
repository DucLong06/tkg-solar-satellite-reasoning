"""Lazy windowed Dataset over one split's contiguous scaled arrays.

Yields per sample:
  sat_seq   [k, C, H, W]
  meteo_seq [k, n_feat]
  pv_hist   [k, 1]
  target    [N_HORIZONS]   (scaled PV at +10/+30/+60 min)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.shapes import HORIZON_STEPS
from src.data_pipeline.windowing import target_indices, valid_starts


class SolarWindowDataset(Dataset):
    def __init__(
        self,
        sat: np.ndarray,      # [T, C, H, W] scaled
        meteo: np.ndarray,    # [T, n_feat] scaled
        pv: np.ndarray,       # [T] scaled
        k: int,
        horizon_steps: tuple[int, ...] = HORIZON_STEPS,
    ) -> None:
        assert len(sat) == len(meteo) == len(pv), "source length mismatch"
        self.sat = sat
        self.meteo = meteo
        self.pv = pv
        self.k = k
        self.horizon_steps = horizon_steps
        self.starts = valid_starts(len(pv), k, horizon_steps)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.starts[idx])
        sl = slice(i, i + self.k)
        tgt_idx = target_indices(i, self.k, self.horizon_steps)
        return {
            "sat_seq": torch.from_numpy(self.sat[sl].copy()),
            "meteo_seq": torch.from_numpy(self.meteo[sl].copy()),
            "pv_hist": torch.from_numpy(self.pv[sl].copy()).unsqueeze(-1),
            "target": torch.from_numpy(self.pv[tgt_idx].copy()),
        }
