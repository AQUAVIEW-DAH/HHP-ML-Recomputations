"""Build sampled monthly Argo-GDAC vs RTOFS OHC/TCHP collocations.

The goal is a tractable monthly comparison study inside the Gulf of Mexico.
Instead of downloading every RTOFS day in a month, we select the top-N dates
with the highest Argo profile counts, then pair all profiles from those dates
to same-day RTOFS columns at the profile coordinates.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import ARGO_CACHE_DIR, RTOFS_CACHE_DIR
from ml.processing.build_milton_pairs import _load_argo_profile, _nearest_rtofs_column
from ml.sources.argo_gdac_source import (
    ARGO_GDAC_BASE,
    _download_to_cache,
    matching_index_rows,
)
from ml.sources.rtofs_source import cached_rtofs_path, download_rtofs_date_range
from OHC.teos_ohc import compute_ohc_teos10

logger = logging.getLogger(__name__)

DEFAULT_BBOX = [-98.0, 18.0, -80.0, 31.0]
DEFAULT_MONTHS = [1, 2, 3, 8, 9, 10]


def month_range(year: int, month: int) -> tuple[str, str]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def select_top_dates(rows: list[dict], dates_per_month: int) -> list[str]:
    counts = Counter(r["date"] for r in rows)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in ordered[:dates_per_month]]


def ensure_argo_files(rows: list[dict]) -> None:
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        source_file = row.get("file") or row.get("source_file") or row.get("cast_id")
        url = f"{ARGO_GDAC_BASE}/{source_file}"
        local = ARGO_CACHE_DIR / source_file
        if idx == 1 or idx % 25 == 0 or idx == total:
            logger.info("Argo fetch %d/%d: %s", idx, total, source_file)
        _download_to_cache(url, local, timeout_seconds=180)


def build_pairs(
    *,
    year: int,
    bbox: list[float],
    months: list[int],
    dates_per_month: int,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dates_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for month in months:
        start, end = month_range(year, month)
        rows = matching_index_rows(
            bbox=bbox,
            start_yyyymmdd=start,
            end_yyyymmdd=end,
            max_profiles=None,
            max_per_platform=None,
        )
        chosen_dates = select_top_dates(rows, dates_per_month)
        logger.info(
            "Month %02d %d: %d matched profiles across %d dates; selected %d dates %s",
            month,
            year,
            len(rows),
            len({r['date'] for r in rows}),
            len(chosen_dates),
            chosen_dates,
        )
        for d in chosen_dates:
            selected_dates_rows.append({
                "year": year,
                "month": month,
                "date": d,
                "profile_count": sum(1 for r in rows if r["date"] == d),
            })
        filtered = [r for r in rows if r["date"] in set(chosen_dates)]
        manifest_rows.extend([
            {
                "cast_id": r["file"],
                "source_file": r["file"],
                "platform": r["platform"],
                "lat": r["lat"],
                "lon": r["lon"],
                "obs_date": r["date"],
                "month": month,
                "year": year,
            }
            for r in filtered
        ])

    selected_dates_df = pd.DataFrame(selected_dates_rows).sort_values(["month", "date"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["obs_date", "platform", "cast_id"]).reset_index(drop=True)

    selected_dates_csv = out_dir / "selected_dates.csv"
    manifest_csv = out_dir / "argo_manifest.csv"
    selected_dates_df.to_csv(selected_dates_csv, index=False)
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info("Wrote %s and %s", selected_dates_csv, manifest_csv)

    ensure_argo_files(manifest_rows)

    unique_dates = sorted(manifest_df["obs_date"].unique().tolist())
    logger.info("Ensuring %d RTOFS dates are cached", len(unique_dates))
    download_rtofs_date_range(unique_dates, RTOFS_CACHE_DIR)

    pairs_rows: list[dict] = []
    for obs_date, group in manifest_df.groupby("obs_date"):
        rtofs_path = cached_rtofs_path(obs_date, RTOFS_CACHE_DIR)
        if not rtofs_path.exists():
            logger.warning("Skipping %s; no RTOFS file at %s", obs_date, rtofs_path)
            continue
        logger.info("Pairing %s with %d Argo casts", obs_date, len(group))
        ds = xr.open_dataset(rtofs_path)
        try:
            for _, row in group.iterrows():
                argo = _load_argo_profile(row["source_file"])
                if not argo:
                    continue
                obs = compute_ohc_teos10(
                    vertical=argo["pressure_dbar"],
                    temp_c=argo["temperature_c"],
                    salinity_psu=argo["salinity_psu"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    vertical_axis="pressure",
                )
                col = _nearest_rtofs_column(ds, float(row["lat"]), float(row["lon"]))
                if col is None:
                    continue
                mod = compute_ohc_teos10(
                    vertical=col["depth_m"],
                    temp_c=col["temperature_c"],
                    salinity_psu=col["salinity_psu"],
                    lat=float(col["grid_lat"]),
                    lon=float(col["grid_lon"]),
                    vertical_axis="depth",
                )
                pairs_rows.append({
                    "cast_id": row["cast_id"],
                    "platform": row["platform"],
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "obs_date": row["obs_date"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "model_grid_lat": float(col["grid_lat"]),
                    "model_grid_lon": float(col["grid_lon"]),
                    "model_grid_distance_km": float(col["distance_km"]),
                    "obs_ohc_j_m2": obs.ohc_j_per_m2,
                    "obs_tchp_kj_cm2": obs.tchp_kj_per_cm2,
                    "obs_d26_m": obs.d26_m,
                    "model_ohc_j_m2": mod.ohc_j_per_m2,
                    "model_tchp_kj_cm2": mod.tchp_kj_per_cm2,
                    "model_d26_m": mod.d26_m,
                    "delta_ohc_j_m2": None if obs.ohc_j_per_m2 is None or mod.ohc_j_per_m2 is None else obs.ohc_j_per_m2 - mod.ohc_j_per_m2,
                    "delta_tchp_kj_cm2": None if obs.tchp_kj_per_cm2 is None or mod.tchp_kj_per_cm2 is None else obs.tchp_kj_per_cm2 - mod.tchp_kj_per_cm2,
                })
        finally:
            ds.close()

    pairs_df = pd.DataFrame(pairs_rows).sort_values(["month", "obs_date", "platform"]).reset_index(drop=True)
    pairs_csv = out_dir / "monthly_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    logger.info("Wrote %s (%d rows)", pairs_csv, len(pairs_df))
    return {
        "selected_dates_csv": selected_dates_csv,
        "manifest_csv": manifest_csv,
        "pairs_csv": pairs_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sampled monthly Argo/RTOFS OHC collocations.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--dates-per-month", type=int, default=5)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--out-dir", type=Path, default=Path("OHC/output/gom_2024_top5days"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    paths = build_pairs(
        year=args.year,
        bbox=list(args.bbox),
        months=args.months,
        dates_per_month=args.dates_per_month,
        out_dir=args.out_dir,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
