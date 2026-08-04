"""Generate the feature and pipeline inventory (docs/feature_inventory.md + .csv).

One authoritative, regenerable document listing every feature considered across
the pipelines with its support (valid rows), date coverage, value range, and
whether the current recommended recipes use it, plus the pipeline-level
constants (tables, protocols, data sources). Rerun after any table rebuild:

    ./hhp-env/bin/python OHC/build_feature_inventory.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    BASE_FEATURES,
    FEATURE_SETS_BY_TARGET,
    NEIGHBORHOOD_CORE,
)

DATA = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data")
DOCS = Path("/home/suramya/HHP-Prediction/docs")
TABLES = {
    "base": DATA / "argo_rtofs_collocated_2024_2025.parquet",
    "global_physics": DATA / "argo_rtofs_collocated_2024_2025_physics.parquet",
    "profile_physics": DATA / "argo_rtofs_collocated_2024_2025_profile_physics.parquet",
    "neighborhood": DATA / "argo_rtofs_collocated_2024_2025_neighborhood.parquet",
    "woa_climatology (side exploration)": DATA / "argo_rtofs_collocated_2024_2025_woa_clim.parquet",
}
NON_FEATURE = {
    "date", "year", "month", "lat_r", "lon_r", "_dup_idx", "season", "error",
    "platform", "cast_id", "profile_index", "profile_key", "source_file",
    "target_tchp_available", "target_d26_available",
}
TARGET_COLS = [
    "argo_tchp_kj_per_cm2", "argo_d26_m",
    "model_interp_tchp_kj_per_cm2", "model_interp_d26_m",
    "delta_tchp_kj_per_cm2", "delta_d26_m",
]
BEST_RECIPES = {
    "tchp": "global_pruned_plus_neighborhood",
    "d26": "drop_both_lat_interactions_plus_neighborhood",
}


def _units(col: str) -> str:
    if col.endswith("_kj_per_cm2"):
        return "kJ/cm²"
    if col.endswith("_s2"):
        return "s⁻²"
    if col.endswith("_km"):
        return "km"
    if col.endswith("_m") or col == "model_ssh_m":
        return "m"
    if "temp" in col and col.endswith("_c") or col.endswith("_26c"):
        return "°C"
    if "ratio" in col or "sin" in col or "cos" in col or col.startswith("is_") or "flag" in col:
        return "–"
    if col in ("lat", "lon", "abs_lat"):
        return "°"
    if col in ("year", "month_int"):
        return "–"
    if "grad_mag" in col:
        return "per 100 km"
    return "–"


def _feature_row(df: pd.DataFrame, col: str, family: str, in_best: set[str]) -> dict:
    v = pd.to_numeric(df[col], errors="coerce")
    finite = np.isfinite(v.to_numpy(float))
    dates = df.loc[finite, "date"].astype(str)
    q = v[finite]
    return {
        "feature": col,
        "family": family,
        "units": _units(col),
        "valid_rows": int(finite.sum()),
        "valid_pct": round(100.0 * finite.mean(), 1),
        "first_valid_date": dates.min() if finite.any() else "–",
        "last_valid_date": dates.max() if finite.any() else "–",
        "valid_dates": int(dates.nunique()),
        "p5": float(q.quantile(0.05)) if finite.any() else np.nan,
        "median": float(q.median()) if finite.any() else np.nan,
        "p95": float(q.quantile(0.95)) if finite.any() else np.nan,
        "in_best_recipe": "yes" if col in in_best else "",
    }


def main() -> None:
    tables = {}
    for name, path in TABLES.items():
        df = pd.read_parquet(path).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        tables[name] = df
    base = tables["base"]
    n = len(base)

    in_best = set()
    for tname, recipe in BEST_RECIPES.items():
        in_best |= set(FEATURE_SETS_BY_TARGET[tname][recipe])

    seen: set[str] = set(NON_FEATURE)
    rows: list[dict] = []
    for col in TARGET_COLS:
        rows.append({**_feature_row(base, col, "targets_and_raw_model", in_best), "family": "targets_and_raw_model"})
        seen.add(col)
    for fname, df in tables.items():
        for col in df.columns:
            if col in seen or col in NON_FEATURE or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].dtype == bool:
                continue
            rows.append(_feature_row(df, col, fname, in_best))
            seen.add(col)
    # Calendar/geometry features derived at train time (not stored in tables).
    from OHC.benchmark_rtofs_argo_tabular_models import _prepare_features

    derived_base = pd.read_parquet(TABLES["base"]).reset_index(drop=True)
    derived_base = _prepare_features(derived_base)
    derived_base["date"] = pd.to_datetime(derived_base["date"]).dt.strftime("%Y%m%d")
    for col in BASE_FEATURES:
        if col in seen or col not in derived_base.columns:
            continue
        rows.append(_feature_row(derived_base, col, "base (derived at train time)", in_best))
        seen.add(col)

    inv = pd.DataFrame(rows)
    DOCS.mkdir(exist_ok=True)
    inv.to_csv(DOCS / "feature_inventory.csv", index=False)

    date_min = base["date"].min()
    date_max = base["date"].max()
    tchp_finite = int(np.isfinite(base["delta_tchp_kj_per_cm2"]).sum())
    d26_finite = int(np.isfinite(base["delta_d26_m"]).sum())
    fold_note = json.loads(Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/tabular_benchmark_folds.json").read_text())

    def count_files(path: str, pattern: str) -> int:
        return len(list(Path(path).glob(pattern))) if Path(path).exists() else 0

    lines = [
        "# Feature and pipeline inventory",
        "",
        f"Auto-generated by `OHC/build_feature_inventory.py` on {datetime.now():%Y-%m-%d}. "
        "Regenerate after any table rebuild; machine-readable copy in `feature_inventory.csv`.",
        "",
        "## Pipeline-level statistics",
        "",
        f"- Collocation base table: **{n:,} rows** over **{base['date'].nunique()} days**, {date_min} → {date_max}",
        f"- Valid residual rows: TCHP **{tchp_finite:,}**, D26 **{d26_finite:,}** "
        "(targets only exist where the upper ocean crosses 26 °C)",
        "- Collocation rule: 8 nearest native RTOFS cells, inverse-distance-squared weights",
        f"- Locked protocol: {fold_note['n_folds']} blocked-forward folds, {fold_note['embargo_dates']}-date embargo, "
        "out-of-fold evaluation on the observed scale",
        "- Model: XGBoost, 300 trees, depth 4, learning rate 0.03, subsample 0.8, "
        "column subsample 0.8, L2 lambda 1.0; training-fold median imputation",
        "",
        "### Source data on disk",
        "",
        f"- RTOFS reduced daily fields: {count_files('/data/suramya/rtofs_global_ohc_fields_2024', 'rtofs_tchp_*.nc')} days (2024), "
        f"{count_files('/data/suramya/rtofs_global_ohc_fields_2025', 'rtofs_tchp_*.nc')} days (2025)",
        "- Argo caches: full-calendar 2024 and 2025 (plus a 2015 cache for the GOFS pilot dates)",
        f"- GOFS 3.1 reanalysis pilot fields (side exploration): {count_files('/data/suramya/gofs31_ohc_fields_2015', 'gofs31_tchp_*.nc')} days of 2015",
        f"- IOOS glider deployment files (truth-set expansion, not yet processed): "
        f"{count_files('/data/suramya/glider_cache_ngdac/2024', '*.nc')} (2024) + {count_files('/data/suramya/glider_cache_ngdac/2025', '*.nc')} (2025)",
        "- WOA23 monthly climatology: 12 months, 1° pilot resolution (side exploration)",
        "",
        "### Reading the feature tables",
        "",
        "- **valid rows / %**: rows where the feature has a usable value, out of "
        f"{n:,}. Profile-physics features are low because only 76 calendar-spread "
        "days have the 3-D archive processed; D26-derived features are low for "
        "physical reasons (no 26 °C crossing in cold water).",
        "- **date range / valid dates**: calendar span and number of days with any valid value.",
        "- **p5 / median / p95**: value distribution over valid rows.",
        "- **in best recipe**: used by the current recommended models "
        f"(TCHP: `{BEST_RECIPES['tchp']}`, D26: `{BEST_RECIPES['d26']}`).",
        "",
    ]

    for family in inv["family"].unique():
        sub = inv[inv["family"] == family]
        lines.append(f"## {family} ({len(sub)} columns)")
        lines.append("")
        lines.append("| feature | units | valid rows | valid % | date range | valid dates | p5 | median | p95 | in best recipe |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            rng = f"{r['first_valid_date']}–{r['last_valid_date']}" if r["valid_rows"] else "–"
            fmt = lambda x: ("–" if not np.isfinite(x) else f"{x:.3g}")
            lines.append(
                f"| `{r['feature']}` | {r['units']} | {r['valid_rows']:,} | {r['valid_pct']} | {rng} | "
                f"{r['valid_dates']} | {fmt(r['p5'])} | {fmt(r['median'])} | {fmt(r['p95'])} | {r['in_best_recipe']} |"
            )
        lines.append("")

    (DOCS / "feature_inventory.md").write_text("\n".join(lines))
    print(f"{len(inv)} features documented -> docs/feature_inventory.md")


if __name__ == "__main__":
    main()
