"""Inventory Argo/RTOFS overlap for ML correction work.

This is the "step 1" data-scope script for the RTOFS -> Argo correction
direction. It answers:

1. What Argo-derived TCHP/D26 data do we already have on disk?
2. What RTOFS-derived fields do we already have on disk?
3. What date overlap exists today for global collocation-style ML?
4. What is available only as raw/regional cache and would need more work
   before it can participate in the same training pipeline?

Outputs:
- JSON summary
- Markdown note
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ARGO_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
GLOBAL_RTOFS_FIELDS_DIR = Path("/data/suramya/rtofs_global_ohc_fields_2024")
GLOBAL_RTOFS_CACHE_DIR = Path("/data/suramya/rtofs_global_cache")
TIME_MATCHED_RTOFS_DIR = Path("/data/suramya/rtofs_time_matched")
COLLOC_2024_SUMMARY = Path(
    "/home/suramya/HHP-Prediction/OHC/output/rtofs_at_argo_points_2024_surface_diff/data/summary_2024_winter_summer.json"
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "ml_data_inventory"
OUT_JSON = OUT_DIR / "rtofs_argo_ml_scope_summary.json"
OUT_MD = OUT_DIR / "rtofs_argo_ml_scope_summary.md"

RTOFS_FIELD_RE = re.compile(r"rtofs_tchp_(\d{8})\.nc$")
DATE_RE = re.compile(r"(\d{8})")


def _extract_dates_from_names(paths: list[Path], *, match_re: re.Pattern[str] | None = None) -> list[str]:
    dates: list[str] = []
    for path in paths:
        if match_re is not None:
            match = match_re.search(path.name)
            if match:
                dates.append(match.group(1))
            continue
        match = DATE_RE.search(path.name)
        if match:
            dates.append(match.group(1))
    return sorted(set(dates))


def _counter_by_year(dates: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(d[:4] for d in dates).items()))


def _month_name(month: int) -> str:
    names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    return names[month]


def _season_label(month: int) -> str:
    if month in {1, 2, 3}:
        return "winter_jfm"
    if month in {7, 8, 9}:
        return "summer_jas"
    return "other"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    argo = pd.read_parquet(ARGO_PATH, columns=["date", "year", "month", "error"])
    argo = argo[argo["error"].isna()].copy()
    argo["date"] = argo["date"].astype(str)
    argo["year"] = argo["year"].astype("Int64")
    argo["month"] = argo["month"].astype("Int64")

    argo_rows_by_year = {
        str(int(year)): int(count)
        for year, count in argo.groupby("year").size().items()
    }
    argo_dates_by_year = {
        str(int(year)): int(count)
        for year, count in argo.groupby("year")["date"].nunique().items()
    }

    global_field_files = sorted(GLOBAL_RTOFS_FIELDS_DIR.glob("rtofs_tchp_*.nc"))
    global_field_dates = _extract_dates_from_names(global_field_files, match_re=RTOFS_FIELD_RE)

    global_cache_dates = _extract_dates_from_names(sorted(GLOBAL_RTOFS_CACHE_DIR.iterdir()))
    time_matched_dates = _extract_dates_from_names(sorted(TIME_MATCHED_RTOFS_DIR.iterdir()))

    overlap_global = argo[argo["date"].isin(global_field_dates)].copy()
    overlap_time_matched = argo[argo["date"].isin(time_matched_dates)].copy()

    overlap_global["season"] = overlap_global["month"].astype(int).map(_season_label)
    overlap_time_matched["season"] = overlap_time_matched["month"].astype(int).map(_season_label)

    colloc_2024_summary = {}
    if COLLOC_2024_SUMMARY.exists():
        colloc_2024_summary = json.loads(COLLOC_2024_SUMMARY.read_text())

    summary = {
        "argo_dataset": {
            "path": str(ARGO_PATH),
            "rows_total": int(len(argo)),
            "rows_by_year": argo_rows_by_year,
            "unique_dates_by_year": argo_dates_by_year,
        },
        "rtofs_assets": {
            "global_daily_fields": {
                "path": str(GLOBAL_RTOFS_FIELDS_DIR),
                "date_count": len(global_field_dates),
                "dates_by_year": _counter_by_year(global_field_dates),
                "earliest_date": min(global_field_dates) if global_field_dates else None,
                "latest_date": max(global_field_dates) if global_field_dates else None,
            },
            "global_cache_raw": {
                "path": str(GLOBAL_RTOFS_CACHE_DIR),
                "date_count": len(global_cache_dates),
                "dates_by_year": _counter_by_year(global_cache_dates),
                "earliest_date": min(global_cache_dates) if global_cache_dates else None,
                "latest_date": max(global_cache_dates) if global_cache_dates else None,
            },
            "time_matched_regional_cache": {
                "path": str(TIME_MATCHED_RTOFS_DIR),
                "date_count": len(time_matched_dates),
                "dates_by_year": _counter_by_year(time_matched_dates),
                "earliest_date": min(time_matched_dates) if time_matched_dates else None,
                "latest_date": max(time_matched_dates) if time_matched_dates else None,
                "product": "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc",
            },
        },
        "overlap_ready_for_global_ml_now": {
            "dates": len(global_field_dates),
            "rows": int(len(overlap_global)),
            "rows_by_year": {
                str(int(year)): int(count)
                for year, count in overlap_global.groupby("year").size().items()
            },
            "rows_by_month": {
                str(int(month)): int(count)
                for month, count in overlap_global.groupby("month").size().items()
            },
            "rows_by_month_named": {
                _month_name(int(month)): int(count)
                for month, count in overlap_global.groupby("month").size().items()
            },
            "dates_by_month": {
                str(int(month)): int(count)
                for month, count in overlap_global.groupby("month")["date"].nunique().items()
            },
            "season_counts": {
                season: int(count)
                for season, count in overlap_global.groupby("season").size().items()
            },
        },
        "overlap_if_we_use_regional_time_matched_cache": {
            "dates": int(overlap_time_matched["date"].nunique()),
            "rows": int(len(overlap_time_matched)),
            "rows_by_year": {
                str(int(year)): int(count)
                for year, count in overlap_time_matched.groupby("year").size().items()
            },
            "season_counts": {
                season: int(count)
                for season, count in overlap_time_matched.groupby("season").size().items()
            },
        },
        "existing_2024_collocation_product": colloc_2024_summary,
        "key_constraints": [
            "The current Argo-derived TCHP/D26 master table ends at 2024.",
            "The current global RTOFS daily TCHP/D26 field set also only covers 2024 dates, concentrated in JFM and JAS.",
            "The broader /data/suramya/rtofs_time_matched cache reaches into 2025, but it is the US-east regional 3D product, not the global reduced daily-field product used in the current Argo-vs-RTOFS comparison workflow.",
            "A true multi-year global ML training table will require either extending the Argo master table beyond 2024 and/or building additional global RTOFS daily TCHP/D26 fields for more years.",
        ],
        "recommended_next_actions_after_step_1": [
            "Use the current 2024 global collocated dataset as the first training pilot if we want to start model prototyping immediately.",
            "If we want genuine multi-year global training, extend the Argo build to 2025 and generate matching global RTOFS daily fields for the overlapping dates.",
            "If we want a larger-but-regional experiment sooner, we can build a separate US-east/regional collocation path from /data/suramya/rtofs_time_matched, but that should be treated as a different experiment from the current global comparison pipeline.",
        ],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    lines = [
        "# RTOFS/Argo ML Data Scope",
        "",
        "This note documents the current on-disk overlap available for the `RTOFS -> Argo` correction modeling work.",
        "",
        "## Argo master table",
        f"- Path: `{ARGO_PATH}`",
        f"- Total usable rows: `{summary['argo_dataset']['rows_total']:,}`",
        f"- Years covered: `{', '.join(summary['argo_dataset']['rows_by_year'].keys())}`",
        "",
        "Rows by year:",
    ]
    for year, count in summary["argo_dataset"]["rows_by_year"].items():
        lines.append(f"- `{year}`: `{count:,}` rows across `{summary['argo_dataset']['unique_dates_by_year'][year]}` unique dates")

    lines += [
        "",
        "## RTOFS assets on disk",
        f"- Global reduced daily TCHP/D26 fields: `{summary['rtofs_assets']['global_daily_fields']['date_count']}` dates",
        f"  - Range: `{summary['rtofs_assets']['global_daily_fields']['earliest_date']}` -> `{summary['rtofs_assets']['global_daily_fields']['latest_date']}`",
        f"- Global raw cache dirs: `{summary['rtofs_assets']['global_cache_raw']['date_count']}` dates",
        f"  - Range: `{summary['rtofs_assets']['global_cache_raw']['earliest_date']}` -> `{summary['rtofs_assets']['global_cache_raw']['latest_date']}`",
        f"- Regional time-matched cache (`US_east` product): `{summary['rtofs_assets']['time_matched_regional_cache']['date_count']}` dates",
        f"  - Range: `{summary['rtofs_assets']['time_matched_regional_cache']['earliest_date']}` -> `{summary['rtofs_assets']['time_matched_regional_cache']['latest_date']}`",
        "",
        "## Global overlap available for ML right now",
        f"- Overlapping global RTOFS-field dates with the current Argo master table: `{summary['overlap_ready_for_global_ml_now']['dates']}`",
        f"- Argo rows on those dates: `{summary['overlap_ready_for_global_ml_now']['rows']:,}`",
        f"- Year breakdown: `{summary['overlap_ready_for_global_ml_now']['rows_by_year']}`",
        f"- Month breakdown: `{summary['overlap_ready_for_global_ml_now']['rows_by_month_named']}`",
        f"- Season counts: `{summary['overlap_ready_for_global_ml_now']['season_counts']}`",
        "",
        "This is the honest pool that is immediately compatible with the existing global collocation scripts.",
        "",
        "## Existing 2024 collocation artifact",
    ]

    if colloc_2024_summary:
        lines += [
            f"- Collocated rows already built: `{colloc_2024_summary.get('collocated_rows', 0):,}`",
            f"- Collocated dates already built: `{colloc_2024_summary.get('collocated_dates', 0)}`",
            f"- Interpolation methods already rendered: `{', '.join(colloc_2024_summary.get('render_methods', []))}`",
        ]
    else:
        lines.append("- No 2024 summary artifact found.")

    lines += [
        "",
        "## Important constraint",
        "Even though the raw `rtofs_time_matched` cache reaches into 2025, the current global Argo-vs-RTOFS ML-ready path is still effectively a 2024 experiment because:",
        "- the Argo-derived TCHP/D26 master table currently ends at 2024",
        "- the global reduced RTOFS daily-field set currently covers only 2024",
        "- the broader 2025 cache is regional (`US_east`) and should be treated as a different experiment unless we intentionally pivot to a regional study",
        "",
        "## Recommendation after step 1",
        "If we want to start model training immediately, the cleanest first pilot is the existing 2024 global collocated dataset.",
        "",
        "If we want a true multi-year global training table, the next data-building work should be one of:",
        "1. extend the Argo TCHP/D26 build into 2025 and create matching global RTOFS daily fields for those dates",
        "2. backfill additional global RTOFS daily fields for earlier years that overlap the 2020-2024 Argo table",
        "3. explicitly run a separate regional experiment using the `US_east` RTOFS cache rather than mixing it with the current global pipeline",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
