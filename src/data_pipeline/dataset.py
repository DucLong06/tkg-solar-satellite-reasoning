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

from src.common.shapes import HORIZON_STEPS, SAT_CHANNELS, SAT_IMG_SIZE
from src.data_pipeline.windowing import target_indices, valid_starts


class SolarWindowDataset(Dataset):
    def __init__(
        self,
        sat: np.ndarray | None,  # [T, C, H, W] scaled, or None (satellite-free mode)
        meteo: np.ndarray,       # [T, n_feat] scaled
        pv: np.ndarray,          # [T] scaled
        k: int,
        horizon_steps: tuple[int, ...] = HORIZON_STEPS,
        img_size: int = SAT_IMG_SIZE,
    ) -> None:
        assert (sat is None or len(sat) == len(meteo)) and len(meteo) == len(pv), "source length mismatch"
        self.sat = sat
        self.meteo = meteo
        self.pv = pv
        self.k = k
        self.horizon_steps = horizon_steps
        self.img_size = img_size
        self.starts = valid_starts(len(pv), k, horizon_steps)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.starts[idx])
        sl = slice(i, i + self.k)
        tgt_idx = target_indices(i, self.k, self.horizon_steps)
        if self.sat is not None:
            sat_seq = torch.from_numpy(self.sat[sl].copy())
        else:  # baseline-only mode: zero frames (baselines ignore them; keeps batch shape)
            sat_seq = torch.zeros(self.k, SAT_CHANNELS, self.img_size, self.img_size, dtype=torch.float32)
        return {
            "sat_seq": sat_seq,
            "meteo_seq": torch.from_numpy(self.meteo[sl].copy()),
            "pv_hist": torch.from_numpy(self.pv[sl].copy()).unsqueeze(-1),
            "target": torch.from_numpy(self.pv[tgt_idx].copy()),
        }
