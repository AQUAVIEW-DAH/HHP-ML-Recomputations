"""RTOFS 3dz downloader for HHP.

Pulls `rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc` files from the NOAA public S3
bucket (`noaa-nws-rtofs-pds`) for a supplied list of YYYYMMDD dates, with a
local cache layout compatible with the MLD project's existing cache at
/data/suramya/rtofs_time_matched so we don't duplicate storage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

RTOFS_S3_BASE = "https://noaa-nws-rtofs-pds.s3.amazonaws.com"
RTOFS_FILE_NAME = "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc"


def rtofs_url_for_date(yyyymmdd: str) -> str:
    return f"{RTOFS_S3_BASE}/rtofs.{yyyymmdd}/{RTOFS_FILE_NAME}"


def cached_rtofs_path(yyyymmdd: str, cache_dir: Path) -> Path:
    return cache_dir / f"rtofs.{yyyymmdd}" / RTOFS_FILE_NAME


def download_rtofs_date(
    yyyymmdd: str,
    cache_dir: Path,
    timeout: float = 600.0,
    force: bool = False,
) -> Path:
    """Download the RTOFS 3dz f006 file for one date. Returns the local path.

    Uses atomic rename via .tmp file so a partial download never overwrites a
    valid cached file.
    """
    local_path = cached_rtofs_path(yyyymmdd, cache_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        logger.info("Cached RTOFS %s -> %s", yyyymmdd, local_path)
        return local_path

    url = rtofs_url_for_date(yyyymmdd)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    logger.info("Downloading RTOFS %s from %s", yyyymmdd, url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_bytes = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    total_bytes += len(chunk)
        tmp_path.replace(local_path)
    logger.info("Wrote %s (%.1f MB)", local_path, total_bytes / (1024 * 1024))
    return local_path


def download_rtofs_date_range(
    dates: list[str],
    cache_dir: Path,
    timeout: float = 600.0,
) -> dict[str, Path | None]:
    """Download RTOFS for many dates; returns {yyyymmdd: path or None on failure}."""
    results: dict[str, Path | None] = {}
    for idx, d in enumerate(dates, start=1):
        try:
            results[d] = download_rtofs_date(d, cache_dir, timeout=timeout)
        except Exception as exc:
            logger.warning("RTOFS %s fetch failed (%d/%d): %s", d, idx, len(dates), exc)
            results[d] = None
    return results
