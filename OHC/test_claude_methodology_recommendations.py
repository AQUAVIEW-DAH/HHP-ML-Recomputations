"""Run the decisive global methodology checks raised in the research-hub review.

This script focuses on the active OHC/TCHP/D26 correction pipeline and tests:

1. Threshold-censoring / 26 C crossing breakdown
2. Skill decay with distance to 2024 training support
3. Platform carryover / leakage audit
4. Geography-vs-state ablations
5. Spatial and platform-grouped "honest split" evaluations
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.analyze_locked_xgb_results import _assign_region, _season_from_month  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402


COLLOC_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
ARGO_2024_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
ARGO_2025_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2025")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/claude_methodology_tests")
EARTH_R_KM = 6371.0
DIST_BINS_KM = [0, 50, 100, 250, 500, np.inf]
DIST_BIN_LABELS = ["0-50", "50-100", "100-250", "250-500", ">500"]

GEO_STATE_FEATURES = [
    "year",
    "month_int",
    "lat",
    "lon",
    "abs_lat",
    "nearest_rtofs_grid_distance_km",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_winter_jfm",
    "is_summer_jas",
    "is_other",
    "model_interp_tchp_kj_per_cm2",
    "model_interp_d26_m",
    "model_ssh_m",
    "model_mixed_layer_thickness_m",
    "model_surface_boundary_layer_thickness_m",
    "model_temp_excess_26c",
    "d26_minus_mlt_m",
    "d26_to_sblt_ratio",
    "model_ssh_x_abs_lat",
    "model_mlt_x_abs_lat",
    "model_temp_excess_x_abs_lat",
]

STATE_ONLY_FEATURES = [
    "year",
    "month_int",
    "nearest_rtofs_grid_distance_km",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_winter_jfm",
    "is_summer_jas",
    "is_other",
    "model_interp_tchp_kj_per_cm2",
    "model_interp_d26_m",
    "model_ssh_m",
    "model_mixed_layer_thickness_m",
    "model_surface_boundary_layer_thickness_m",
    "model_temp_excess_26c",
    "d26_minus_mlt_m",
    "d26_to_sblt_ratio",
]

BEST_PARAMS = {
    "tchp": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "d26": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
}


def _rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((y_true - y_pred) ** 2))


def _preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)
    ])


def _make_model(target_name: str) -> XGBRegressor:
    return XGBRegressor(
        random_state=0,
        objective="reg:squarederror",
        n_jobs=8,
        **BEST_PARAMS[target_name],
    )


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, *, rows: int, dates: int, extra: dict | None = None) -> dict:
    payload = {
        "rows": int(rows),
        "dates": int(dates),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(_rmse(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(y_true) > 1 else np.nan,
    }
    if extra:
        payload.update(extra)
    return payload


def _load_enriched_collocation() -> pd.DataFrame:
    colloc = pd.read_parquet(COLLOC_PATH).copy()

    def prep_source(path: Path, year: int) -> pd.DataFrame:
        src = pd.read_parquet(path)
        src = src[(src["error"].isna()) & (src["year"] == year)].copy()
        src = src[np.isfinite(src["lat"]) & np.isfinite(src["lon"])].copy().reset_index(drop=True)
        cols = ["cast_id", "platform", "date", "lat", "lon", "surface_t_c", "tchp_kj_per_cm2", "d26_m"]
        return src[cols].copy()

    src_2024 = prep_source(ARGO_2024_PATH, 2024)
    src_2025 = prep_source(ARGO_2025_PATH, 2025)
    src = pd.concat([src_2024, src_2025], ignore_index=True)

    # The collocation table preserves row order after the same filtering steps, but we merge
    # via a running key to make the linkage explicit and robust to repeated lat/lon on a date.
    for df in (src, colloc):
        df["date"] = df["date"].astype(str)
        df["lat_r"] = df["lat"].round(5)
        df["lon_r"] = df["lon"].round(5)
        df["_dup_idx"] = df.groupby(["date", "lat_r", "lon_r"]).cumcount()

    merged = colloc.merge(
        src.rename(
            columns={
                "surface_t_c": "argo_surface_temp_c",
                "tchp_kj_per_cm2": "argo_tchp_source_kj_per_cm2",
                "d26_m": "argo_d26_source_m",
            }
        )[
            [
                "cast_id",
                "platform",
                "date",
                "lat_r",
                "lon_r",
                "_dup_idx",
                "argo_surface_temp_c",
                "argo_tchp_source_kj_per_cm2",
                "argo_d26_source_m",
            ]
        ],
        on=["date", "lat_r", "lon_r", "_dup_idx"],
        how="left",
        validate="1:1",
    )
    merged.drop(columns=["lat_r", "lon_r", "_dup_idx"], inplace=True)
    return merged


def _threshold_breakdown(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["argo_crosses_26"] = np.isfinite(work["argo_tchp_kj_per_cm2"])
    work["rtofs_crosses_26"] = np.isfinite(work["model_interp_tchp_kj_per_cm2"])
    work["threshold_cell"] = np.select(
        [
            work["argo_crosses_26"] & work["rtofs_crosses_26"],
            (~work["argo_crosses_26"]) & (~work["rtofs_crosses_26"]),
            work["argo_crosses_26"] & (~work["rtofs_crosses_26"]),
            (~work["argo_crosses_26"]) & work["rtofs_crosses_26"],
        ],
        ["both_cross", "neither_cross", "argo_only", "rtofs_only"],
        default="unknown",
    )

    rows = []
    for cell, gdf in work.groupby("threshold_cell"):
        row = {
            "cell": cell,
            "rows": int(len(gdf)),
            "dates": int(gdf["date"].nunique()),
            "rows_pct": float(len(gdf) / len(work)),
            "argo_surface_temp_mean_c": float(gdf["argo_surface_temp_c"].mean()) if "argo_surface_temp_c" in gdf else np.nan,
            "model_surface_temp_mean_c": float(gdf["model_surface_temp_c"].mean()),
            "surface_temp_bias_model_minus_argo_c": float((gdf["model_surface_temp_c"] - gdf["argo_surface_temp_c"]).mean()),
            "argo_tchp_mean": float(gdf["argo_tchp_kj_per_cm2"].mean()),
            "model_tchp_mean": float(gdf["model_interp_tchp_kj_per_cm2"].mean()),
            "argo_d26_mean": float(gdf["argo_d26_m"].mean()),
            "model_d26_mean": float(gdf["model_interp_d26_m"].mean()),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("rows", ascending=False)
    out.to_csv(OUT_DIR / "threshold_breakdown_2x2.csv", index=False)
    season = work.pivot_table(index="threshold_cell", columns="season", values="date", aggfunc="count", fill_value=0)
    season.to_csv(OUT_DIR / "threshold_breakdown_by_season.csv")
    return {
        "summary_csv": str(OUT_DIR / "threshold_breakdown_2x2.csv"),
        "season_csv": str(OUT_DIR / "threshold_breakdown_by_season.csv"),
        "rows": out.to_dict(orient="records"),
    }


def _nearest_train_support_km(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train_xyz = latlon_to_xyz(train_df["lat"].to_numpy(float), train_df["lon"].to_numpy(float)).astype(np.float32)
    test_xyz = latlon_to_xyz(test_df["lat"].to_numpy(float), test_df["lon"].to_numpy(float)).astype(np.float32)
    tree = cKDTree(train_xyz)
    chord, _ = tree.query(test_xyz, k=1, workers=-1)
    return EARTH_R_KM * (2.0 * np.arcsin(np.clip(chord.astype(np.float64) / 2.0, 0.0, 1.0)))


def _train_year_holdout_predictions(df: pd.DataFrame, target, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)
    train_df = work[work["year"] == 2024].copy()
    test_df = work[work["year"] == 2025].copy()
    pre = _preprocessor(feature_cols)
    x_train = pre.fit_transform(train_df[feature_cols])
    x_test = pre.transform(test_df[feature_cols])
    model = _make_model(target.name)
    model.fit(x_train, train_df[target.delta_col].to_numpy(float))
    pred = test_df[target.model_col].to_numpy(float) + model.predict(x_test)
    out = test_df.copy()
    out["prediction"] = pred
    out["model_name"] = model_name
    return out


def _support_distance_decay(df: pd.DataFrame) -> dict:
    rows = []
    for target in TARGETS:
        pred_df = _train_year_holdout_predictions(df, target, GEO_STATE_FEATURES, "geo_state_year_holdout")
        train_df = _prepare_features(df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy())
        train_df = train_df[train_df["year"] == 2024].copy()
        d_km = _nearest_train_support_km(train_df, pred_df)
        pred_df["nearest_train_support_km"] = d_km
        pred_df["support_bin"] = pd.cut(d_km, bins=DIST_BINS_KM, labels=DIST_BIN_LABELS, include_lowest=True, right=False)
        for key, gdf in pred_df.groupby("support_bin", observed=False):
            if len(gdf) == 0:
                continue
            y_true = gdf[target.obs_col].to_numpy(float)
            rows.append(
                {
                    "target": target.name,
                    "model": "raw_rtofs",
                    "support_bin": str(key),
                    **_metric_dict(y_true, gdf[target.model_col].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()),
                }
            )
            rows.append(
                {
                    "target": target.name,
                    "model": "geo_state_year_holdout",
                    "support_bin": str(key),
                    **_metric_dict(y_true, gdf["prediction"].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()),
                }
            )
        pred_df.to_parquet(OUT_DIR / f"support_distance_predictions_{target.name}.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "support_distance_decay_metrics.csv", index=False)
    return {
        "metrics_csv": str(OUT_DIR / "support_distance_decay_metrics.csv"),
    }


def _platform_leakage_audit(df: pd.DataFrame) -> dict:
    work = df[pd.notna(df["platform"])].copy()
    train = work[work["year"] == 2024].copy()
    test = work[work["year"] == 2025].copy()
    train_platforms = set(train["platform"].astype(str))
    test_platforms = set(test["platform"].astype(str))
    overlap = train_platforms & test_platforms
    test["platform_seen_in_train"] = test["platform"].astype(str).isin(overlap)

    same_platform_rows = int(test["platform_seen_in_train"].sum())
    same_platform_share = float(same_platform_rows / len(test)) if len(test) else np.nan

    # Nearest same-platform training support for overlapping platforms.
    same_platform_dist_rows = []
    for plat, gdf in test[test["platform_seen_in_train"]].groupby("platform"):
        tr = train[train["platform"] == plat]
        if tr.empty:
            continue
        d = _nearest_train_support_km(tr, gdf)
        same_platform_dist_rows.extend(d.tolist())

    platform_counts = work.groupby(["year", "platform"]).size().reset_index(name="rows")
    platform_counts.to_csv(OUT_DIR / "platform_counts_by_year.csv", index=False)

    summary = {
        "train_2024_unique_platforms": int(len(train_platforms)),
        "test_2025_unique_platforms": int(len(test_platforms)),
        "platform_overlap_count": int(len(overlap)),
        "platform_overlap_share_of_test_platforms": float(len(overlap) / len(test_platforms)) if test_platforms else np.nan,
        "test_rows_on_seen_platforms": same_platform_rows,
        "test_row_share_on_seen_platforms": same_platform_share,
        "median_same_platform_train_distance_km": float(np.median(same_platform_dist_rows)) if same_platform_dist_rows else np.nan,
        "p90_same_platform_train_distance_km": float(np.percentile(same_platform_dist_rows, 90)) if same_platform_dist_rows else np.nan,
        "platform_counts_csv": str(OUT_DIR / "platform_counts_by_year.csv"),
    }
    (OUT_DIR / "platform_leakage_audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def _tile_id(lat: float, lon: float, tile_deg: int = 15) -> str:
    lat_bin = int(np.floor((lat + 90.0) / tile_deg))
    lon_bin = int(np.floor((lon + 180.0) / tile_deg))
    return f"tile_{lat_bin:02d}_{lon_bin:02d}"


def _grouped_oof_eval(df: pd.DataFrame, target, feature_cols: list[str], group_col: str, model_name: str, n_splits: int = 5) -> dict:
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)
    groups = work[group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.full(len(work), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(work, groups=groups), 1):
        tr = work.iloc[tr_idx]
        va = work.iloc[va_idx]
        pre = _preprocessor(feature_cols)
        x_tr = pre.fit_transform(tr[feature_cols])
        x_va = pre.transform(va[feature_cols])
        model = _make_model(target.name)
        model.fit(x_tr, tr[target.delta_col].to_numpy(float))
        preds[va_idx] = va[target.model_col].to_numpy(float) + model.predict(x_va)

    valid = np.isfinite(preds)
    scored = work.iloc[valid].copy()
    scored["prediction"] = preds[valid]
    scored.to_parquet(OUT_DIR / f"grouped_oof_predictions_{target.name}_{model_name}.parquet", index=False)
    return _metric_dict(
        scored[target.obs_col].to_numpy(float),
        scored["prediction"].to_numpy(float),
        rows=len(scored),
        dates=scored["date"].nunique(),
    )


def _ablation_and_honest_splits(df: pd.DataFrame) -> dict:
    rows = []
    work = df.copy()
    work["tile_group"] = [_tile_id(lat, lon) for lat, lon in zip(work["lat"], work["lon"])]
    work["platform_group"] = work["platform"].fillna("unknown").astype(str)

    for target in TARGETS:
        # A: current direction with geo+state, year-holdout
        pred_geo = _train_year_holdout_predictions(work, target, GEO_STATE_FEATURES, "geo_state_year_holdout")
        rows.append({
            "target": target.name,
            "evaluation": "year_holdout",
            "model": "geo_state_year_holdout",
            **_metric_dict(
                pred_geo[target.obs_col].to_numpy(float),
                pred_geo["prediction"].to_numpy(float),
                rows=len(pred_geo),
                dates=pred_geo["date"].nunique(),
            ),
        })

        # C: state-only year-holdout
        pred_state = _train_year_holdout_predictions(work, target, STATE_ONLY_FEATURES, "state_only_year_holdout")
        rows.append({
            "target": target.name,
            "evaluation": "year_holdout",
            "model": "state_only_year_holdout",
            **_metric_dict(
                pred_state[target.obs_col].to_numpy(float),
                pred_state["prediction"].to_numpy(float),
                rows=len(pred_state),
                dates=pred_state["date"].nunique(),
            ),
        })

        # Raw year-holdout reference
        rows.append({
            "target": target.name,
            "evaluation": "year_holdout",
            "model": "raw_rtofs",
            **_metric_dict(
                pred_geo[target.obs_col].to_numpy(float),
                pred_geo[target.model_col].to_numpy(float),
                rows=len(pred_geo),
                dates=pred_geo["date"].nunique(),
            ),
        })

        # B: tile-group spatial OOF with geo+state
        rows.append({
            "target": target.name,
            "evaluation": "tile_group_oof",
            "model": "geo_state_tile_group",
            **_grouped_oof_eval(work, target, GEO_STATE_FEATURES, "tile_group", "geo_state_tile_group"),
        })
        rows.append({
            "target": target.name,
            "evaluation": "tile_group_oof",
            "model": "state_only_tile_group",
            **_grouped_oof_eval(work, target, STATE_ONLY_FEATURES, "tile_group", "state_only_tile_group"),
        })

        # Platform-grouped OOF as honest leakage guard
        known = work[pd.notna(work["platform"])].copy()
        rows.append({
            "target": target.name,
            "evaluation": "platform_group_oof",
            "model": "geo_state_platform_group",
            **_grouped_oof_eval(known, target, GEO_STATE_FEATURES, "platform_group", "geo_state_platform_group"),
        })
        rows.append({
            "target": target.name,
            "evaluation": "platform_group_oof",
            "model": "state_only_platform_group",
            **_grouped_oof_eval(known, target, STATE_ONLY_FEATURES, "platform_group", "state_only_platform_group"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "global_methodology_ablation_summary.csv", index=False)
    return {"summary_csv": str(OUT_DIR / "global_methodology_ablation_summary.csv")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_enriched_collocation()

    summary = {
        "input_path": str(COLLOC_PATH),
        "output_dir": str(OUT_DIR),
        "threshold_breakdown": _threshold_breakdown(df),
        "support_distance_decay": _support_distance_decay(df),
        "platform_leakage_audit": _platform_leakage_audit(df),
        "ablation_and_honest_splits": _ablation_and_honest_splits(df),
    }
    out_json = OUT_DIR / "global_methodology_test_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
