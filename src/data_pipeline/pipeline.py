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

import pandas as pd

from src.common.shapes import BASE_CADENCE_MIN, HORIZON_STEPS, N_METEO_FEATURES, SAT_CHANNELS
from src.data_pipeline.cleaning import apply_clip, clean_arrays, fit_clip_bounds
from src.data_pipeline.dataset import SolarWindowDataset
from src.data_pipeline.loaders import find_himawari_file, load_dkasc, load_himawari_alice
from src.data_pipeline.scaling import Scalers, fit_scalers
from src.data_pipeline.splits import bounds_from_dates, chronological_bounds
from src.data_pipeline.time_alignment import align_colocated

# Bump when align/clean logic changes so stale caches are invalidated.
# v3: DKASC co-located source (single CSV + Himawari Alice), 5-min grid, night
#     filter, timestamps cached for fixed-date splits.
# v4: satellite clamp fixed to percent-reflectance range (was [0,1], which killed
#     the real satellite signal).
CACHE_VERSION = "v4"


@dataclass
class Splits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: Scalers
    meta: dict


def _cache_key(
    paths: list[Path], img_size: int, min_steps: int, cadence_min: int, night_ghi_thresh: float
) -> str:
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(f"{img_size}-{min_steps}-{cadence_min}-{night_ghi_thresh}".encode())
    for p in paths:
        stat = p.stat()
        h.update(f"{p}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _load_aligned(
    dkasc_csv, himawari_dir, img_size, min_steps, cadence_min, night_ghi_thresh,
    cache_dir, use_cache, use_satellite,
) -> dict:
    frames = sat_ts = None
    key_paths = [Path(dkasc_csv)]
    if use_satellite:
        frames, sat_ts = load_himawari_alice(himawari_dir)
        # Key on the RESOLVED frames file (not the dir) so overwriting frames.h5 busts
        # the cache; include night_ghi_thresh since it selects which rows survive.
        key_paths.append(find_himawari_file(himawari_dir))
    key = _cache_key(key_paths, img_size, min_steps, cadence_min, night_ghi_thresh)
    tag = "sat" if use_satellite else "nosat"
    cache_path = Path(cache_dir) / f"aligned_{tag}_{key}.npz"
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        ts = pd.to_datetime(z["timestamps"], utc=True)
        sat = z["sat"] if "sat" in z.files else None
        return {"timestamps": ts, "pv": z["pv"], "meteo": z["meteo"], "sat": sat}

    pv, meteo = load_dkasc(dkasc_csv)
    grid = align_colocated(
        pv, meteo, frames, sat_ts, cadence_min=cadence_min, min_steps=min_steps,
        img_size=img_size, night_ghi_thresh=night_ghi_thresh,
    )
    ts = grid["timestamps"]
    data = clean_arrays(grid)
    data["timestamps"] = ts
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # as_unit('ns'): asi8 must be nanoseconds — pandas 2.x indexes parsed from ISO
        # strings can be datetime64[us], whose asi8 (microseconds) would be misread as
        # ns on reload, shifting every timestamp to 1970 and breaking date splits.
        arrays = {"pv": data["pv"], "meteo": data["meteo"],
                  "timestamps": ts.as_unit("ns").asi8}
        if data["sat"] is not None:
            arrays["sat"] = data["sat"]
        np.savez(cache_path, **arrays)
    return data


class DataPipeline:
    @staticmethod
    def load(
        dkasc_csv: str,
        himawari_dir: str,
        *,
        k: int = 12,
        batch_size: int = 64,
        img_size: int = 64,
        min_steps: int = 200,
        min_train_windows: int = 10,
        train_end: str | None = None,
        val_end: str | None = None,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        cadence_min: int = BASE_CADENCE_MIN,
        night_ghi_thresh: float = 5.0,
        cache_dir: str = "data/cache",
        num_workers: int = 0,
        use_cache: bool = True,
        use_satellite: bool = True,
        scaler_out: str | None = None,
    ) -> Splits:
        # use_satellite=False -> PV+meteo only over the FULL DKASC span (train the
        # baselines without waiting for the Himawari download); sat_seq is zeros.
        data = _load_aligned(
            dkasc_csv, himawari_dir, img_size, min_steps, cadence_min,
            night_ghi_thresh, cache_dir, use_cache, use_satellite,
        )
        sat, meteo, pv = data["sat"], data["meteo"], data["pv"]
        n = len(pv)
        # Fixed-date split when boundaries given (DKASC reproduction); else fractional.
        if train_end and val_end:
            b = bounds_from_dates(data["timestamps"], train_end, val_end)
        else:
            b = chronological_bounds(n, train_frac, val_frac)

        # Outlier clip with TRAIN-only stats (no leakage), applied to all splits.
        m_lo, m_hi = fit_clip_bounds(meteo[b.train])
        p_lo, p_hi = fit_clip_bounds(pv[b.train])
        meteo = apply_clip(meteo, m_lo, m_hi)
        pv = apply_clip(pv, p_lo, p_hi)

        scalers = fit_scalers(meteo[b.train], pv[b.train], sat[b.train] if sat is not None else None)
        if scaler_out:
            scalers.save(scaler_out)

        meteo_s = scalers.transform_meteo(meteo)
        pv_s = scalers.transform_pv(pv)
        sat_s = scalers.transform_sat(sat) if sat is not None else None

        def make(sl) -> SolarWindowDataset:
            return SolarWindowDataset(
                sat_s[sl] if sat_s is not None else None,
                meteo_s[sl], pv_s[sl], k, HORIZON_STEPS, img_size=img_size,
            )

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
            "source": "dkasc_alice",
            "pv_units": "kW",
            "use_satellite": use_satellite,
            "n_meteo_features": N_METEO_FEATURES,
            "sat_channels": SAT_CHANNELS,
            "img_size": img_size,
            "cadence_min": cadence_min,
            "k": k,
            "n_steps": n,
            "split_mode": "dates" if (train_end and val_end) else "fractional",
            "train_end": train_end,
            "val_end": val_end,
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
