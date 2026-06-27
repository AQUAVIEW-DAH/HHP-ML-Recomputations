"""Inspect global RTOFS archive scope and storage cost for seasonal windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import RTOFS_CACHE_DIR, RTOFS_GLOBAL_CACHE_DIR
from ml.sources.rtofs_global_source import (
    download_global_archv_b,
    head_global_product_sizes,
    parse_archv_b,
    summarize_archv_fields,
)

DEFAULT_WINTER_MONTHS = {1, 2, 3}
DEFAULT_SUMMER_MONTHS = {7, 8, 9}


def cached_regional_dates(cache_dir: Path) -> list[str]:
    dates: list[str] = []
    for p in sorted(cache_dir.glob("rtofs.*/rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc")):
        name = p.parent.name
        if name.startswith("rtofs.") and len(name) == 14:
            dates.append(name.split(".", 1)[1])
    return dates


def select_dates(dates: list[str], years: list[int] | None, seasons: list[str] | None) -> list[str]:
    season_months: set[int] | None = None
    if seasons:
        season_months = set()
        for season in seasons:
            s = season.lower()
            if s in {"winter", "jfm", "winter_jfm"}:
                season_months |= DEFAULT_WINTER_MONTHS
            elif s in {"summer", "jas", "summer_jas"}:
                season_months |= DEFAULT_SUMMER_MONTHS
            else:
                raise ValueError(f"Unsupported season {season!r}")
    out: list[str] = []
    for d in dates:
        year = int(d[:4])
        month = int(d[4:6])
        if years and year not in years:
            continue
        if season_months and month not in season_months:
            continue
        out.append(d)
    return out


def fmt_gb(n: int | None) -> float | None:
    if n is None:
        return None
    return round(n / (1024 ** 3), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--seasons", nargs="+", default=["winter", "summer"])
    parser.add_argument("--sample-date", default="20240131")
    parser.add_argument("--global-cache-dir", type=Path, default=RTOFS_GLOBAL_CACHE_DIR)
    args = parser.parse_args()

    dates = select_dates(cached_regional_dates(RTOFS_CACHE_DIR), args.years, args.seasons)
    sizes = head_global_product_sizes(args.sample_date)
    header_path = download_global_archv_b(args.sample_date, args.global_cache_dir)
    header = parse_archv_b(header_path)
    counts = summarize_archv_fields(header)

    per_date_archv_bytes = sizes["archv_a_tgz_bytes"] or 0
    total_archv_bytes = per_date_archv_bytes * len(dates)
    summary = {
        "dates_requested": len(dates),
        "date_range": [dates[0], dates[-1]] if dates else [],
        "sample_date": args.sample_date,
        "sample_sizes_bytes": sizes,
        "sample_sizes_gb": {k: fmt_gb(v) for k, v in sizes.items()},
        "estimated_total_archv_a_tgz_gb": fmt_gb(total_archv_bytes),
        "archv_header": {
            "idm": header.idm,
            "jdm": header.jdm,
            "record_words": header.record_words,
            "n_records": len(header.fields),
            "field_counts": counts,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
