"""H8→H9 bucket/filename routing around the 2022-12-13 operational handover."""

from __future__ import annotations

from datetime import datetime

from scripts.build_himawari_frames import bucket_for, seg_key


def test_bucket_for_cutover():
    assert bucket_for(datetime(2022, 12, 12)) == ("noaa-himawari8", "H08")
    assert bucket_for(datetime(2022, 12, 13)) == ("noaa-himawari9", "H09")
    assert bucket_for(datetime(2020, 1, 1)) == ("noaa-himawari8", "H08")
    assert bucket_for(datetime(2023, 12, 31)) == ("noaa-himawari9", "H09")


def test_seg_key_filename_per_satellite():
    k8 = seg_key(datetime(2022, 6, 15), 2, 30, "R05", "S0810", "H08")
    k9 = seg_key(datetime(2023, 6, 15), 2, 30, "R05", "S0810", "H09")
    assert k8 == "AHI-L1b-FLDK/2022/06/15/0230/HS_H08_20220615_0230_B03_FLDK_R05_S0810.DAT.bz2"
    assert k9 == "AHI-L1b-FLDK/2023/06/15/0230/HS_H09_20230615_0230_B03_FLDK_R05_S0810.DAT.bz2"
