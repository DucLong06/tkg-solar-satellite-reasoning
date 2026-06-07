"""Generate small synthetic OPSD / NSRDB / Himawari files in the REAL on-disk formats.

Purpose: exercise the actual M1 loaders end-to-end without the multi-GB real
downloads (which need an NREL key + AWS). The synthetic series carry a diurnal
solar shape so metrics / night-time MAPE masking behave realistically.

This is a TEST FIXTURE generator, not part of the reproduction's data path.
Run:  python scripts/generate_synthetic_data.py [--days 30 --out data]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Allow running as a plain script (python scripts/x.py): put project root on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.shapes import (
    METEO_FEATURES,
    SAT_CHANNELS,
    SAT_IMG_SIZE,
)

START = datetime(2016, 6, 1, tzinfo=timezone.utc)
CADENCE_MIN = 10


def _diurnal(t_index: np.ndarray) -> np.ndarray:
    """Daily solar bell curve in [0, 1], zero at night."""
    minutes_of_day = (t_index % (24 * 60))
    # Sun up ~06:00-18:00; cosine bell peaking at noon.
    frac = (minutes_of_day - 6 * 60) / (12 * 60)
    bell = np.sin(np.clip(frac, 0, 1) * np.pi)
    return np.clip(bell, 0.0, 1.0)


def _timestamps(n: int, cadence_min: int) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=n, freq=f"{cadence_min}min", tz="UTC")


def gen_opsd(out: Path, days: int, rng: np.random.Generator) -> None:
    """OPSD: 15-min CSV with a *_solar_generation_actual column."""
    n = days * 24 * 4  # 15-min steps
    ts = _timestamps(n, 15)
    minutes = np.arange(n) * 15
    bell = _diurnal(minutes)
    gen = 9000 * bell * (0.8 + 0.4 * rng.random(n))  # MW, with cloud noise
    df = pd.DataFrame(
        {"utc_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "DE_solar_generation_actual": np.round(gen, 2)}
    )
    (out / "opsd").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "opsd" / "time_series_15min_singleindex.csv", index=False)


def gen_nsrdb(out: Path, days: int, rng: np.random.Generator) -> None:
    """NSRDB: h5 with one dataset per meteo variable + a time_index, 10-min."""
    n = days * 24 * 6  # 10-min steps
    ts = _timestamps(n, CADENCE_MIN)
    minutes = np.arange(n) * CADENCE_MIN
    bell = _diurnal(minutes)
    fields = {
        "ghi": 1000 * bell * (0.85 + 0.3 * rng.random(n)),
        "dni": 900 * bell * (0.8 + 0.4 * rng.random(n)),
        "dhi": 200 * bell * (0.7 + 0.6 * rng.random(n)),
        "air_temperature": 25 + 8 * bell + rng.normal(0, 1, n),
        "relative_humidity": np.clip(70 - 25 * bell + rng.normal(0, 3, n), 0, 100),
        "wind_speed": np.abs(3 + rng.normal(0, 1.5, n)),
        "surface_pressure": 1010 + rng.normal(0, 2, n),
    }
    assert set(fields) == set(METEO_FEATURES), "meteo field mismatch with shape contract"
    (out / "nsrdb").mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "nsrdb" / "vietnam_2016.h5", "w") as f:
        f.create_dataset(
            "time_index",
            data=np.array(ts.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(), dtype="S20"),
        )
        for name, arr in fields.items():
            f.create_dataset(name, data=arr.astype("float32"))


def gen_himawari(out: Path, days: int, rng: np.random.Generator) -> None:
    """Himawari: h5 of frames [T, C, H, W] + timestamps, 10-min visible band."""
    n = days * 24 * 6
    ts = _timestamps(n, CADENCE_MIN)
    minutes = np.arange(n) * CADENCE_MIN
    bell = _diurnal(minutes)
    h = w = SAT_IMG_SIZE
    # Brightness ~ daylight; add moving "cloud" blobs for spatial structure.
    base = bell[:, None, None, None] * np.ones((n, SAT_CHANNELS, h, w), dtype="float32")
    clouds = rng.random((n, SAT_CHANNELS, h, w)).astype("float32") * 0.3
    frames = np.clip(base * 0.7 + clouds, 0.0, 1.0).astype("float32")
    (out / "himawari").mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "himawari" / "synthetic_frames.h5", "w") as f:
        f.create_dataset("frames", data=frames, compression="gzip")
        f.create_dataset(
            "timestamps",
            data=np.array(ts.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(), dtype="S20"),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30, help="days of overlapping data")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    gen_opsd(args.out, args.days, rng)
    gen_nsrdb(args.out, args.days, rng)
    gen_himawari(args.out, args.days, rng)
    print(f"Synthetic data written to {args.out}/ ({args.days} days, start {START.date()}).")


if __name__ == "__main__":
    main()
