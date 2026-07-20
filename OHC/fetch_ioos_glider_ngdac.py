"""Fetch IOOS NGDAC glider deployment files from the NCEI THREDDS server.

Mentor direction (Dr. Hill): bring underwater-glider profiles into the
truth/label set alongside Argo. This script only handles the raw download;
profile parsing, QC, and TEOS-10 TCHP/D26 computation are a follow-up step.

Catalog layout (one CF file per deployment):
  https://www.ncei.noaa.gov/thredds-ocean/catalog/ioos/ngdac/<year>/catalog.xml
  file download via the fileServer service base.

Resumable: files already present with the advertised size are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_URL = "https://www.ncei.noaa.gov/thredds-ocean/catalog/ioos/ngdac/{year}/catalog.xml"
FILESERVER_BASE = "https://www.ncei.noaa.gov/thredds-ocean/fileServer/"
DEFAULT_OUT_ROOT = Path("/data/suramya/glider_cache_ngdac")
THREDDS_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"


def _list_deployments(year: int) -> list[dict]:
    with urllib.request.urlopen(CATALOG_URL.format(year=year), timeout=120) as resp:
        tree = ET.fromstring(resp.read())
    out = []
    for ds in tree.iter(f"{THREDDS_NS}dataset"):
        url_path = ds.attrib.get("urlPath")
        if not url_path or not url_path.endswith(".nc"):
            continue
        size_el = ds.find(f"{THREDDS_NS}dataSize")
        size_mb = float(size_el.text) if size_el is not None else None
        out.append({"name": ds.attrib["name"], "url_path": url_path, "size_mb": size_mb})
    return out


def _download(entry: dict, out_dir: Path) -> str:
    dest = out_dir / entry["name"]
    if dest.exists() and entry["size_mb"] and abs(dest.stat().st_size / 1e6 - entry["size_mb"]) < max(1.0, 0.02 * entry["size_mb"]):
        return "cached"
    url = FILESERVER_BASE + entry["url_path"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    summary = {}
    for year in args.years:
        out_dir = args.out_root / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        deployments = _list_deployments(year)
        logger.info("%d: %d deployment files (%.1f GB advertised)", year, len(deployments),
                    sum(d["size_mb"] or 0 for d in deployments) / 1024.0)
        counts = {"downloaded": 0, "cached": 0, "failed": 0}
        for i, entry in enumerate(deployments, 1):
            t0 = time.perf_counter()
            try:
                status = _download(entry, out_dir)
                counts[status] += 1
                if status == "downloaded":
                    logger.info("[%d/%d] %s (%.0f MB, %.1fs)", i, len(deployments), entry["name"],
                                entry["size_mb"] or -1, time.perf_counter() - t0)
            except Exception:
                logger.exception("failed: %s", entry["name"])
                counts["failed"] += 1
        summary[year] = counts
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
