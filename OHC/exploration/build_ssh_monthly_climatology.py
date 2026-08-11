"""Monthly-mean RTOFS SSH climatology on the native grid.

Groundwork for Dr. Jacobs' suggestion (2026-08): use the deviation of surface
height from a (lat, lon, month) mean as the feature, instead of raw SSH.
Streams every cached per-date diagnostic file and accumulates per-month means.

Output: /data/suramya/rtofs_ssh_climatology/monthly_mean_ssh.nc with
mean_ssh(month, Y, X) and count(month).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE = Path("/data/suramya/rtofs_global_cache")
OUT_DIR = Path("/data/suramya/rtofs_ssh_climatology")


def main() -> None:
    files = sorted(CACHE.glob("rtofs.*/rtofs_glo_2ds_f006_diag.nc"))
    logger.info("%d diagnostic files", len(files))
    sums = counts = lat = lon = None
    done = 0
    for f in files:
        date = f.parent.name.split(".", 1)[1]
        month = int(date[4:6]) - 1
        try:
            with xr.open_dataset(f) as ds:
                ssh = np.asarray(ds["ssh"].isel(MT=0).values, dtype=np.float64)
                if sums is None:
                    sums = np.zeros((12,) + ssh.shape)
                    counts = np.zeros(12, dtype=np.int32)
                    lat = np.asarray(ds["Latitude"].values, dtype=np.float32)
                    lon = np.asarray(ds["Longitude"].values, dtype=np.float32)
            sums[month] += np.where(np.isfinite(ssh), ssh, 0.0)
            counts[month] += 1
        except Exception:
            logger.exception("failed on %s", f)
        done += 1
        if done % 50 == 0:
            logger.info("%d/%d", done, len(files))

    mean = np.full_like(sums, np.nan, dtype=np.float32)
    for m in range(12):
        if counts[m] > 0:
            mean[m] = (sums[m] / counts[m]).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={
            "mean_ssh": (("month", "Y", "X"), mean),
            "count": (("month",), counts),
            "Latitude": (("Y", "X"), lat),
            "Longitude": (("Y", "X"), lon),
        },
        coords={"month": np.arange(1, 13)},
        attrs={"description": "Monthly mean RTOFS SSH from per-date diagnostic files, 2024-2025."},
    ).to_netcdf(OUT_DIR / "monthly_mean_ssh.nc")
    logger.info("wrote %s (months coverage: %s)", OUT_DIR / "monthly_mean_ssh.nc", counts.tolist())


if __name__ == "__main__":
    main()
