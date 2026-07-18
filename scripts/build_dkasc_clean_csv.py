"""Build the canonical clean DKASC CSV from a manually-downloaded master-meter export.

The DKASC portal (https://dkasc.solar) is reCAPTCHA-gated, so the raw per-site CSV
must be downloaded by hand. This script turns that raw export into the canonical
schema ``load_dkasc`` expects — ``timestamp, Pac, GHI, Tamb, RH, WS`` — doing:

  1. column map (Active_Power->Pac, Global_Horizontal_Radiation->GHI,
     Weather_Temperature_Celsius->Tamb, Weather_Relative_Humidity->RH)
  2. filter to the paper window (2020-01-01..2022-12-31)
  3. timestamp ACST (UTC+9:30, Alice local) -> UTC  [the portal exports local time;
     Himawari frames are UTC, so this MUST be converted or the two desync by 9.5h]
  4. sentinel/outlier cleaning (bad-sensor flags like Temp=-40, RH=131, GHI=1524)
  5. Wind Speed: the site's anemometer died after 2016 (100% NaN for 2020-2022), so
     WS is sourced from ERA5 reanalysis via the Open-Meteo archive (free, no key),
     hourly, interpolated to the 5-min grid.

Run:
    python scripts/build_dkasc_clean_csv.py \
        --raw data/dkasc/96-Site_DKA-MasterMeter1.csv \
        --out data/dkasc/alice_2020_2023_clean.csv \
        --lat -23.76 --lon 133.87
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ACST_OFFSET = pd.Timedelta(hours=9, minutes=30)  # Alice Springs local = UTC+9:30 (no DST)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Raw DKASC column -> canonical clean column.
COLUMN_MAP = {
    "Active_Power": "Pac",
    "Global_Horizontal_Radiation": "GHI",
    "Weather_Temperature_Celsius": "Tamb",
    "Weather_Relative_Humidity": "RH",
}
# Physically plausible ranges for Alice Springs; values outside -> NaN -> interpolated.
VALID = {
    "Pac": (-5.0, 10_000.0),     # kW; small negatives at night are real
    "GHI": (0.0, 1400.0),        # W/m^2
    "Tamb": (-5.0, 55.0),        # deg C
    "RH": (0.0, 100.0),          # %
}


def fetch_era5_wind(lat: float, lon: float, start: str, end: str) -> pd.Series:
    """ERA5 10 m wind speed (m/s) via Open-Meteo archive, hourly UTC. Fetched per year."""
    pieces = []
    for yr in range(int(start[:4]), int(end[:4]) + 1):
        y0 = max(start, f"{yr}-01-01")
        y1 = min(end, f"{yr}-12-31")
        url = (
            f"{ARCHIVE_URL}?latitude={lat}&longitude={lon}"
            f"&start_date={y0}&end_date={y1}"
            "&hourly=wind_speed_10m&timezone=UTC&wind_speed_unit=ms"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        h = d["hourly"]
        s = pd.Series(h["wind_speed_10m"], index=pd.to_datetime(h["time"], utc=True), name="WS")
        pieces.append(s)
        print(f"  ERA5 wind {yr}: {len(s)} hours", flush=True)
    return pd.concat(pieces).sort_index()


def clean_dkasc(raw_path: str, start: str, end: str) -> pd.DataFrame:
    """Load raw export -> canonical Pac/GHI/Tamb/RH on a 5-min UTC grid (no WS yet)."""
    df = pd.read_csv(raw_path, usecols=["timestamp", *COLUMN_MAP])
    df = df.rename(columns=COLUMN_MAP)
    # Local ACST -> UTC.
    ts = pd.to_datetime(df["timestamp"]) - ACST_OFFSET
    df.index = ts.dt.tz_localize("UTC")
    df = df.drop(columns="timestamp").sort_index()
    df = df[(df.index >= f"{start}T00:00:00Z") & (df.index <= f"{end}T23:59:59Z")]

    for col, (lo, hi) in VALID.items():
        bad = (df[col] < lo) | (df[col] > hi)
        df.loc[bad, col] = np.nan

    grid = pd.date_range(f"{start}T00:00:00Z", f"{end}T23:55:00Z", freq="5min", tz="UTC")
    df = df.reindex(df.index.union(grid)).interpolate(method="time", limit=6).reindex(grid)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Raw DKASC master-meter CSV path.")
    ap.add_argument("--out", required=True, help="Output canonical clean CSV path.")
    ap.add_argument("--lat", type=float, default=-23.76)
    ap.add_argument("--lon", type=float, default=133.87)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    print(f"Cleaning DKASC {args.raw} -> {args.start}..{args.end} (ACST->UTC)...", flush=True)
    df = clean_dkasc(args.raw, args.start, args.end)

    print("Fetching ERA5 wind (Open-Meteo, hourly UTC)...", flush=True)
    wind_h = fetch_era5_wind(args.lat, args.lon, args.start, args.end)
    # Hourly -> 5-min on the same grid via time interpolation.
    df["WS"] = wind_h.reindex(wind_h.index.union(df.index)).interpolate(method="time").reindex(df.index)

    df = df[["Pac", "GHI", "Tamb", "RH", "WS"]].round(4)
    out = df.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    nulls = df.isna().mean() * 100
    print(f"\nWrote {len(out):,} rows -> {args.out}")
    print("null% after fill:", {c: round(float(nulls[c]), 3) for c in df.columns})
    print("ranges:", {c: (round(float(df[c].min()), 2), round(float(df[c].max()), 2)) for c in df.columns})


if __name__ == "__main__":
    main()
