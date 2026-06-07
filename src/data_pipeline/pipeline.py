"""Data-pipeline orchestrator: raw sources -> aligned/clean/scaled/windowed DataLoaders.

Public entrypoint ``DataPipeline.load(...)`` matches the main.py call signature.
The expensive align+clean+resize step is cached (keyed by inputs + a code version);
the cheap split+scale+window runs every call so split fractions stay live.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.shapes import HORIZON_STEPS, N_METEO_FEATURES, SAT_CHANNELS
from src.data_pipeline.cleaning import apply_clip, clean_arrays, fit_clip_bounds
from src.data_pipeline.dataset import SolarWindowDataset
from src.data_pipeline.loaders import find_himawari_file, load_himawari, load_nsrdb, load_opsd
from src.data_pipeline.scaling import Scalers, fit_scalers
from src.data_pipeline.splits import chronological_bounds
from src.data_pipeline.time_alignment import build_common_grid

# Bump when align/clean logic changes so stale caches are invalidated.
# v2: outlier clip moved out of the pre-split cache (now train-only, post-split).
CACHE_VERSION = "v2"


@dataclass
class Splits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: Scalers
    meta: dict


def _cache_key(paths: list[Path], img_size: int, min_steps: int) -> str:
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(f"{img_size}-{min_steps}".encode())
    for p in paths:
        stat = p.stat()
        h.update(f"{p}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _load_aligned(opsd, nsrdb, him_file, img_size, min_steps, cache_dir, use_cache) -> dict:
    key = _cache_key([Path(opsd), Path(nsrdb), him_file], img_size, min_steps)
    cache_path = Path(cache_dir) / f"aligned_{key}.npz"
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return {"timestamps": None, "pv": z["pv"], "meteo": z["meteo"], "sat": z["sat"]}

    pv = load_opsd(opsd)
    meteo = load_nsrdb(nsrdb)
    frames, sat_ts = load_himawari(him_file)
    grid = build_common_grid(pv, meteo, frames, sat_ts, min_steps=min_steps, img_size=img_size)
    data = clean_arrays(grid)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, pv=data["pv"], meteo=data["meteo"], sat=data["sat"])
    return data


class DataPipeline:
    @staticmethod
    def load(
        opsd_path: str,
        nsrdb_path: str,
        himawari_dir: str,
        *,
        k: int = 12,
        batch_size: int = 64,
        img_size: int = 64,
        min_steps: int = 200,
        min_train_windows: int = 10,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        cache_dir: str = "data/cache",
        num_workers: int = 0,
        use_cache: bool = True,
        scaler_out: str | None = None,
    ) -> Splits:
        him_file = find_himawari_file(himawari_dir)
        data = _load_aligned(opsd_path, nsrdb_path, him_file, img_size, min_steps, cache_dir, use_cache)
        sat, meteo, pv = data["sat"], data["meteo"], data["pv"]
        n = len(pv)
        b = chronological_bounds(n, train_frac, val_frac)

        # Outlier clip with TRAIN-only stats (no leakage), applied to all splits.
        m_lo, m_hi = fit_clip_bounds(meteo[b.train])
        p_lo, p_hi = fit_clip_bounds(pv[b.train])
        meteo = apply_clip(meteo, m_lo, m_hi)
        pv = apply_clip(pv, p_lo, p_hi)

        scalers = fit_scalers(meteo[b.train], pv[b.train], sat[b.train])
        if scaler_out:
            scalers.save(scaler_out)

        meteo_s = scalers.transform_meteo(meteo)
        pv_s = scalers.transform_pv(pv)
        sat_s = scalers.transform_sat(sat)

        def make(sl) -> SolarWindowDataset:
            return SolarWindowDataset(sat_s[sl], meteo_s[sl], pv_s[sl], k, HORIZON_STEPS)

        train_ds, val_ds, test_ds = make(b.train), make(b.val), make(b.test)
        if len(train_ds) < min_train_windows:
            raise RuntimeError(
                f"Only {len(train_ds)} train windows < min_train_windows={min_train_windows}. "
                f"Common steps={n}, k={k}. Provide a longer data window."
            )

        gen = torch.Generator().manual_seed(0)  # pin shuffle order for reproducibility

        def _seed_worker(worker_id: int) -> None:
            import numpy as _np
            import random as _random
            wseed = torch.initial_seed() % 2 ** 32
            _np.random.seed(wseed)
            _random.seed(wseed)

        def loader(ds, shuffle) -> DataLoader:
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                generator=gen if shuffle else None,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
            )

        meta = {
            "n_meteo_features": N_METEO_FEATURES,
            "sat_channels": SAT_CHANNELS,
            "img_size": img_size,
            "k": k,
            "n_steps": n,
            "n_train_windows": len(train_ds),
            "n_val_windows": len(val_ds),
            "n_test_windows": len(test_ds),
        }
        return Splits(
            train_loader=loader(train_ds, True),
            val_loader=loader(val_ds, False),
            test_loader=loader(test_ds, False),
            scalers=scalers,
            meta=meta,
        )
