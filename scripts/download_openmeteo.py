"""Download real Vietnam meteo from Open-Meteo archive -> NSRDB-style h5.

Open-Meteo historical archive (ERA5-based) is free, no API key, global coverage.
Used as a drop-in real-data source for the meteo branch when NSRDB/NREL is
unreachable. Writes the exact h5 layout the M1 loader expects
(one dataset per METEO_FEATURES name + a byte-string ``time_index``).

Run (default = HCM, June 2016, hourly):
    python scripts/download_openmeteo.py \
        --lat 10.82 --lon 106.63 --start 2016-06-01 --end 2016-06-30 \
        [--out data/nsrdb/vietnam_2016.h5]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.shapes import METEO_FEATURES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo hourly var -> our fixed NSRDB feature name.
OPENMETEO_MAP = {
    "ghi": "shortwave_radiation",
    "dni": "direct_normal_irradiance",
    "dhi": "diffuse_radiation",
    "air_temperature": "temperature_2m",
    "relative_humidity": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m",
    "surface_pressure": "surface_pressure",
}


def fetch(lat: float, lon: float, start: str, end: str) -> dict:
    hourly = ",".join(OPENMETEO_MAP[f] for f in METEO_FEATURES)
    url = (
        f"{ARCHIVE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly={hourly}&timezone=UTC"
    )
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (trusted host)
        return json.load(r)["hourly"]


def to_h5(hourly: dict, out: Path) -> int:
    times = hourly["time"]  # e.g. "2016-06-01T00:00"
    n = len(times)
    ts_bytes = np.array([f"{t}:00Z".encode() for t in times], dtype="S")

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("time_index", data=ts_bytes)
        for feat in METEO_FEATURES:
            col = np.asarray(hourly[OPENMETEO_MAP[feat]], dtype="float32")
            # ERA5 gaps come back as None -> NaN; M1 cleaning interpolates them.
            col = np.where(col == None, np.nan, col).astype("float32")  # noqa: E711
            f.create_dataset(feat, data=col)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=10.82)
    ap.add_argument("--lon", type=float, default=106.63)
    ap.add_argument("--start", default="2016-06-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2016-06-30", help="YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=Path("data/nsrdb/vietnam_2016.h5"))
    args = ap.parse_args()

    print(f"Open-Meteo archive: ({args.lat},{args.lon}) {args.start}..{args.end} hourly")
    hourly = fetch(args.lat, args.lon, args.start, args.end)
    n = to_h5(hourly, args.out)
    print(f"Wrote {n} hourly steps -> {args.out} ({args.out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
