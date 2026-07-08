"""Raw-source readers: DKASC CSV (co-located PV+meteo), Himawari h5.

Each returns data on its native cadence with a UTC DatetimeIndex / timestamp array.
Time alignment to the common grid happens in ``time_alignment.py``.

The legacy 3-source readers (OPSD/NSRDB) are retained below for reference but are
no longer wired into the DKASC pipeline (the cross-continent fusion was retired).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.common.shapes import METEO_FEATURES

# Canonical DKASC clean-CSV columns produced by scripts/download_dkasc.py.
# timestamp -> UTC index; Pac -> PV target (kW); the four meteo columns map to
# METEO_FEATURES in contract order.
DKASC_METEO_COLUMNS: dict[str, str] = {
    "GHI": "ghi",
    "Tamb": "air_temperature",
    "RH": "relative_humidity",
    "WS": "wind_speed",
}


def load_dkasc(csv_path: str | Path) -> tuple[pd.Series, pd.DataFrame]:
    """Read one DKASC Alice Springs array CSV -> (pv [kW] Series, meteo DataFrame).

    Expected clean schema: ``timestamp, Pac, GHI, Tamb, RH, WS`` (5-min cadence,
    UTC). ``Pac`` is PV active power in kW (the forecast target); the four meteo
    columns are reordered/renamed to the contract's ``METEO_FEATURES``.
    """
    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[ts_col], utc=True)

    if "Pac" not in df.columns:
        raise KeyError(f"DKASC CSV missing 'Pac' (PV kW) column; has {list(df.columns)}")
    missing = [c for c in DKASC_METEO_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"DKASC CSV missing meteo columns {missing}; has {list(df.columns)}")

    pv = pd.Series(df["Pac"].to_numpy(dtype="float32"), index=idx, name="pv")
    pv = pv[~pv.index.duplicated(keep="first")].sort_index()

    meteo = pd.DataFrame(
        {dst: df[src].to_numpy(dtype="float32") for src, dst in DKASC_METEO_COLUMNS.items()},
        index=idx,
    )[list(METEO_FEATURES)]
    meteo = meteo[~meteo.index.duplicated(keep="first")].sort_index()
    return pv, meteo


def load_himawari_alice(himawari_dir: str | Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Locate + read the co-located Himawari Alice Springs frames h5."""
    return load_himawari(find_himawari_file(himawari_dir))


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
