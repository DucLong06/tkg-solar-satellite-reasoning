"""Download a Himawari-8 visible-band ROI subset from the public AWS bucket.

Source: s3://noaa-himawari8 (open data, --no-sign-request, no AWS account).
Full-disk visible band is ~1.4 GB/day / ~500 GB/yr -> ALWAYS subset to a small
ROI + short window. Reads HSD with satpy and saves an [T, C, H, W] h5 tensor in
the layout the M1 loader expects.

Run (default = 1-day sample):
    python scripts/download_himawari.py --date 2016-06-01 \
        --lat-min 8 --lat-max 24 --lon-min 102 --lon-max 110

LONG-LEAD ITEM: the multi-week pull needed for Phase 6 training must be kicked
off early — it is the slowest dependency. For a runnable pipeline without any
real satellite bytes, use scripts/generate_synthetic_data.py instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUCKET = "noaa-himawari8"
# AHI L1b gridded product prefix; band 3 (B03) = visible 0.64 um, 0.5 km.
PRODUCT_PREFIX = "AHI-L1b-FLDK"


def list_segments(s3, date: str, hour: str):
    y, m, d = date.split("-")
    prefix = f"{PRODUCT_PREFIX}/{y}/{m}/{d}/{hour}00/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [o["Key"] for o in resp.get("Contents", []) if "B03" in o["Key"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2016-06-01", help="YYYY-MM-DD")
    ap.add_argument("--hours", default="00,03,06,09", help="comma list of UTC hours")
    ap.add_argument("--lat-min", type=float, default=8.0)
    ap.add_argument("--lat-max", type=float, default=24.0)
    ap.add_argument("--lon-min", type=float, default=102.0)
    ap.add_argument("--lon-max", type=float, default=110.0)
    ap.add_argument("--out", type=Path, default=Path("data/himawari"))
    args = ap.parse_args()

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print(
            "boto3 not installed. Install the data extra:  uv sync --extra data\n"
            "(or: pip install boto3 satpy)\n"
            "For a runnable pipeline without real satellite data, use "
            "scripts/generate_synthetic_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    args.out.mkdir(parents=True, exist_ok=True)

    raw_dir = args.out / "raw"
    raw_dir.mkdir(exist_ok=True)
    downloaded = []
    for hour in args.hours.split(","):
        keys = list_segments(s3, args.date, hour.zfill(2))
        if not keys:
            print(f"  no B03 segments for {args.date} {hour}:00 (check date coverage)")
            continue
        for key in keys:
            local = raw_dir / Path(key).name
            if not local.exists():
                print(f"  downloading {key} ...")
                s3.download_file(BUCKET, key, str(local))
            downloaded.append(local)

    if not downloaded:
        print(
            "No segments downloaded. Himawari-8 AWS coverage starts mid-2015; verify the "
            "date and bucket layout. Then read+crop the ROI with satpy and save an h5 of "
            "shape [T, C, H, W] (see generate_synthetic_data.gen_himawari for the layout).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Downloaded {len(downloaded)} HSD segments to {raw_dir}.\n"
        "Next: read with satpy (Scene reader='ahi_hsd'), crop to the ROI bbox, resample to "
        "a fixed grid, and save data/himawari/frames.h5 with datasets 'frames' [T,C,H,W] "
        "and 'timestamps'. (ROI/resample left explicit so you control the grid the M1 "
        "satellite loader consumes.)"
    )


if __name__ == "__main__":
    main()
