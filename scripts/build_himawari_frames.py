"""Build real Himawari-8 frames.h5 from the NOAA AWS open bucket.

Completes the acquisition the repo's download_himawari.py left as a TODO: for each
UTC timestep it downloads the B03 visible segment (R20 = 2 km, ~22 MB), decompresses,
reads with satpy (reader='ahi_hsd'), crops to the Vietnam ROI, resamples to a fixed
HxW lat/lon grid, and stacks into data/himawari/frames.h5 with the layout the M1
satellite loader expects (frames [T,1,H,W] float32 + byte-string timestamps).

Processing is incremental (download -> decompress -> resample -> delete raw) to keep
peak disk low. R20 chosen because the M1 grid downsamples to 64x64 anyway.

Run (default = 2016-06-01..06-07, daytime UTC 00..09 hourly):
    python scripts/build_himawari_frames.py --start 2016-06-01 --end 2016-06-07
"""

from __future__ import annotations

import argparse
import bz2
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BUCKET = "noaa-himawari8"
PREFIX = "AHI-L1b-FLDK"
VN_BBOX = (102.0, 8.0, 110.0, 24.0)  # lon_min, lat_min, lon_max, lat_max


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def seg_key(day: datetime, hour: int, minute: int = 0) -> str:
    hm = f"{hour:02d}{minute:02d}"
    s = f"{day:%Y/%m/%d}/{hm}"
    name = f"HS_H08_{day:%Y%m%d}_{hm}_B03_FLDK_R20_S0101.DAT.bz2"
    return f"{PREFIX}/{s}/{name}"


def make_area(h: int, w: int):
    from pyresample.geometry import AreaDefinition

    return AreaDefinition(
        "vn", "Vietnam ROI", "vn",
        {"proj": "longlat", "datum": "WGS84"},
        w, h, VN_BBOX,
    )


def frame_from_dat(dat: Path, area) -> np.ndarray:
    from satpy import Scene

    scn = Scene(reader="ahi_hsd", filenames=[str(dat)])
    scn.load(["B03"])
    out = scn.resample(area, resampler="nearest", radius_of_influence=20000)
    return np.asarray(out["B03"].values, dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2016-06-07")
    ap.add_argument("--hours", default="0,1,2,3,4,5,6,7,8,9", help="UTC hours (VN daytime)")
    ap.add_argument("--step-min", type=int, default=10, help="intra-hour cadence (min)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("data/himawari/frames.h5"))
    args = ap.parse_args()

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    raw = args.out.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    area = make_area(args.size, args.size)
    hours = [int(h) for h in args.hours.split(",")]

    minutes = list(range(0, 60, args.step_min))
    frames, stamps = [], []
    for day in daterange(args.start, args.end):
        for hour in hours:
            for minute in minutes:
                key = seg_key(day, hour, minute)
                bz = raw / Path(key).name
                try:
                    if not bz.exists():
                        s3.download_file(BUCKET, key, str(bz))
                except Exception as exc:  # noqa: BLE001 (missing timestep -> skip)
                    print(f"  skip {day:%Y-%m-%d} {hour:02d}:{minute:02d} ({type(exc).__name__})")
                    continue
                dat = raw / bz.stem
                with bz2.open(bz, "rb") as fi, open(dat, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                try:
                    arr = frame_from_dat(dat, area)
                finally:
                    dat.unlink(missing_ok=True)
                    bz.unlink(missing_ok=True)
                frames.append(arr[None, :, :])  # [1,H,W]
                stamps.append(f"{day:%Y-%m-%d}T{hour:02d}:{minute:02d}:00Z")
                print(f"  ok {stamps[-1]}  mean={np.nanmean(arr):.1f}")

    if not frames:
        print("No frames built — check date coverage.", file=sys.stderr)
        sys.exit(1)

    data = np.stack(frames).astype("float32")  # [T,1,H,W]
    data = np.nan_to_num(data, nan=0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("frames", data=data)
        f.create_dataset("timestamps", data=np.array([s.encode() for s in stamps], dtype="S"))
    print(f"WROTE {args.out}  frames={data.shape}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
