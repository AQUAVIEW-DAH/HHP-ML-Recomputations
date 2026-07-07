"""Fetch missing per-date RTOFS 2D diagnostic files (rtofs_glo_2ds_f006_diag.nc).

The global-physics feature builder samples SSH / mixed-layer thickness /
surface-boundary-layer thickness from the per-date diagnostic product. The
reduced-field backfill does not fetch these, so after a backfill this script
fills the gaps for every date that has a reduced daily field. ~185 MB per date.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.sources.rtofs_global_source import download_global_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("/data/suramya/rtofs_global_cache")
FIELD_DIRS = [
    Path("/data/suramya/rtofs_global_ohc_fields_2024"),
    Path("/data/suramya/rtofs_global_ohc_fields_2025"),
]
WORKERS = 3


def main() -> None:
    dates = sorted(
        p.stem.split("_")[-1]
        for d in FIELD_DIRS
        for p in d.glob("rtofs_tchp_*.nc")
    )
    missing = [
        d for d in dates
        if not (CACHE_DIR / f"rtofs.{d}" / "rtofs_glo_2ds_f006_diag.nc").exists()
    ]
    logger.info("%d field dates, %d missing diag files", len(dates), len(missing))
    ok = failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_global_grid, d, CACHE_DIR): d for d in missing}
        for fut in as_completed(futures):
            date = futures[fut]
            try:
                fut.result()
                ok += 1
            except Exception:
                logger.exception("diag download failed for %s", date)
                failed += 1
            if (ok + failed) % 25 == 0:
                logger.info("progress: %d ok, %d failed, %d remaining", ok, failed, len(missing) - ok - failed)
    print({"dates": len(dates), "downloaded": ok, "failed": failed})


if __name__ == "__main__":
    main()
