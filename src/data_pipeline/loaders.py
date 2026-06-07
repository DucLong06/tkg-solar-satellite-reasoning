"""Raw-source readers: OPSD CSV, NSRDB h5, Himawari h5.

Each returns data on its native cadence with a UTC DatetimeIndex / timestamp array.
Time alignment to the common grid happens in ``time_alignment.py``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.common.shapes import METEO_FEATURES


def load_opsd(csv_path: str | Path, target_col: str | None = None) -> pd.Series:
    """Read OPSD 15-min CSV -> PV power Series indexed by UTC timestamp.

    Picks the first ``*_solar_generation_actual`` column unless ``target_col`` given.
    """
    df = pd.read_csv(csv_path)
    ts_col = "utc_timestamp" if "utc_timestamp" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[ts_col], utc=True)

    if target_col is None:
        cands = [c for c in df.columns if c.endswith("_solar_generation_actual")]
        if not cands:
            raise KeyError("No *_solar_generation_actual column found in OPSD CSV")
        target_col = cands[0]
    series = pd.Series(df[target_col].to_numpy(dtype="float32"), index=idx, name="pv")
    return series[~series.index.duplicated(keep="first")].sort_index()


def load_nsrdb(h5_path: str | Path) -> pd.DataFrame:
    """Read NSRDB h5 -> meteo DataFrame [time, METEO_FEATURES] indexed by UTC."""
    with h5py.File(h5_path, "r") as f:
        ts = pd.to_datetime([t.decode() for t in f["time_index"][:]], utc=True)
        data = {feat: f[feat][:].astype("float32") for feat in METEO_FEATURES}
    df = pd.DataFrame(data, index=ts)[list(METEO_FEATURES)]
    return df[~df.index.duplicated(keep="first")].sort_index()


def load_himawari(h5_path: str | Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Read Himawari h5 -> (frames [T,C,H,W] float32, UTC timestamps)."""
    with h5py.File(h5_path, "r") as f:
        frames = f["frames"][:].astype("float32")
        ts = pd.to_datetime([t.decode() for t in f["timestamps"][:]], utc=True)
    if frames.ndim != 4:
        raise ValueError(f"Himawari frames must be [T,C,H,W], got {frames.shape}")
    order = np.argsort(ts.values)
    return frames[order], ts[order]


def find_himawari_file(himawari_dir: str | Path) -> Path:
    """Locate the frames h5 (real 'frames.h5' preferred, else synthetic)."""
    d = Path(himawari_dir)
    for name in ("frames.h5", "synthetic_frames.h5"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(
        f"No Himawari frames h5 in {d} (expected frames.h5 or synthetic_frames.h5). "
        "Run scripts/download_himawari.py or scripts/generate_synthetic_data.py."
    )
