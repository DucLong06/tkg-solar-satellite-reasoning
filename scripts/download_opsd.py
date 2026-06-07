"""Download OPSD 15-min time series (solar generation + load). No auth required.

Easiest of the three sources (~107 MB CSV, European). Run:
    python scripts/download_opsd.py [--out data/opsd]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_15min_singleindex.csv"
)
FILENAME = "time_series_15min_singleindex.csv"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "tkg-solar-repro/0.1"})
    with urlopen(req) as resp, open(dest, "wb") as fh:  # noqa: S310 (trusted host)
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            got += len(block)
            if total:
                pct = 100 * got / total
                print(f"\r  {got/1e6:7.1f}/{total/1e6:.1f} MB ({pct:4.1f}%)", end="")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/opsd"))
    args = ap.parse_args()

    dest = args.out / FILENAME
    if dest.exists():
        print(f"Already present: {dest} ({dest.stat().st_size/1e6:.1f} MB). Skipping.")
        return
    print(f"Downloading OPSD -> {dest}")
    try:
        download(OPSD_URL, dest)
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nFAILED: {exc}\n"
            f"Manual fallback: download {FILENAME} from\n"
            f"  https://data.open-power-system-data.org/time_series/\n"
            f"and place it at {dest}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Done: {dest} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
