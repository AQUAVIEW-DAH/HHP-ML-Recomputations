"""Compare base/global/profile physics feature sets for 2024-train / 2025-test XGBoost."""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.analyze_locked_xgb_results import _assign_region, _season_from_month  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _prepare_features  # noqa: E402


DATA_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")

BASE_FEATURES = [
    "year", "month_int", "lat", "lon", "abs_lat", "nearest_rtofs_grid_distance_km",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "is_winter_jfm", "is_summer_jas", "is_other",
    "model_interp_tchp_kj_per_cm2", "model_interp_d26_m",
]

GLOBAL_PHYSICS_FEATURES = [
    "model_surface_temp_c", "model_ssh_m", "model_mixed_layer_thickness_m",
    "model_surface_boundary_layer_thickness_m", "model_temp_excess_26c",
    "d26_minus_mlt_m", "d26_minus_sblt_m", "d26_to_mlt_ratio", "d26_to_sblt_ratio",
    "warm_layer_thickness_positive_m", "model_ssh_x_abs_lat",
    "model_mlt_x_abs_lat", "model_temp_excess_x_abs_lat",
]

PROFILE_PHYSICS_FEATURES = [
    "model_steric_0_1000_m",
    "model_steric_0_2000_m",
    "model_steric_1000_ref2000_m",
    "model_n2_mean_upper200_s2",
    "model_n2_max_upper200_s2",
    "model_n2_mean_to_d26_s2",
    "model_n2_max_to_d26_s2",
]

FEATURE_SETS = {
    "base": BASE_FEATURES,
    "base_plus_global_physics": BASE_FEATURES + GLOBAL_PHYSICS_FEATURES,
    "base_plus_global_plus_profile": BASE_FEATURES + GLOBAL_PHYSICS_FEATURES + PROFILE_PHYSICS_FEATURES,
    "base_plus_profile_only": BASE_FEATURES + PROFILE_PHYSICS_FEATURES,
}

BEST_PARAMS = {
    "tchp": dict(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    "d26": dict(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
}


def _rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((y_true - y_pred) ** 2))


def _metric_row(*, target: str, model: str, split: str, y_true, y_pred, rows: int, dates: int) -> dict:
    return {
        "target": target,
        "model": model,
        "split": split,
        "rows": int(rows),
        "dates": int(dates),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(_rmse(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(y_true) > 1 else np.nan,
    }


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)])


def _make_model(params: dict) -> XGBRegressor:
    return XGBRegressor(random_state=0, objective="reg:squarederror", n_jobs=8, **params)


def _add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_group"] = [_assign_region(lat, lon) for lat, lon in zip(out["lat"], out["lon"])]
    out["season_group"] = out["month"].astype(int).map(_season_from_month)
    return out


def _safe_feature_importance(feature_cols: list[str], model: XGBRegressor) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for i, feature in enumerate(feature_cols):
        rows.append({"feature": feature, "gain": float(gain.get(f"f{i}", 0.0)), "weight": float(weight.get(f"f{i}", 0.0))})
    return pd.DataFrame(rows).sort_values(["gain", "weight"], ascending=False)


def run_target(df: pd.DataFrame, target, feature_set_name: str, feature_cols: list[str]):
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)
    train_df = work[work["year"] == 2024].copy()
    test_df = work[work["year"] == 2025].copy()

    pre = _make_preprocessor(feature_cols)
    x_train = pre.fit_transform(train_df[feature_cols])
    x_test = pre.transform(test_df[feature_cols])
    y_train_delta = train_df[target.delta_col].to_numpy(float)
    y_test_obs = test_df[target.obs_col].to_numpy(float)
    raw_test = test_df[target.model_col].to_numpy(float)

    model = _make_model(BEST_PARAMS[target.name])
    model.fit(x_train, y_train_delta)
    pred_test = raw_test + model.predict(x_test)

    pred_df = test_df[["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]].copy()
    pred_df = _add_group_cols(pred_df)
    pred_df["pred_obs__raw_rtofs"] = raw_test
    pred_df[f"pred_obs__{feature_set_name}"] = pred_test

    rows = [
        _metric_row(target=target.name, model="raw_rtofs", split="2025_all", y_true=y_test_obs, y_pred=raw_test, rows=len(pred_df), dates=pred_df["date"].nunique()),
        _metric_row(target=target.name, model=feature_set_name, split="2025_all", y_true=y_test_obs, y_pred=pred_test, rows=len(pred_df), dates=pred_df["date"].nunique()),
    ]
    fi_df = _safe_feature_importance(feature_cols, model)
    fi_df["target"] = target.name
    fi_df["feature_set"] = feature_set_name
    return pd.DataFrame(rows), pred_df, fi_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    all_metrics, all_importance, summary_rows = [], [], []
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        for target in TARGETS:
            metrics_df, pred_df, fi_df = run_target(df, target, feature_set_name, feature_cols)
            metrics_df.to_csv(OUT_DIR / f"year_holdout_profile_physics_metrics_{target.name}_{feature_set_name}.csv", index=False)
            pred_df.to_parquet(OUT_DIR / f"year_holdout_profile_physics_predictions_{target.name}_{feature_set_name}.parquet", index=False)
            fi_df.to_csv(OUT_DIR / f"year_holdout_profile_physics_importance_{target.name}_{feature_set_name}.csv", index=False)
            all_metrics.append(metrics_df)
            all_importance.append(fi_df)
            summary_rows.append(metrics_df[metrics_df["split"] == "2025_all"].copy())

    metrics = pd.concat(all_metrics, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    summary = pd.concat(summary_rows, ignore_index=True).sort_values(["target", "model"])
    metrics.to_csv(OUT_DIR / "year_holdout_profile_physics_metrics_all.csv", index=False)
    importance.to_csv(OUT_DIR / "year_holdout_profile_physics_importance_all.csv", index=False)
    summary.to_csv(OUT_DIR / "year_holdout_profile_physics_summary.csv", index=False)
    payload = {
        "scheme": "train_2024_test_2025",
        "data_path": str(DATA_PATH),
        "feature_sets": FEATURE_SETS,
        "summary_csv": str(OUT_DIR / "year_holdout_profile_physics_summary.csv"),
        "importance_csv": str(OUT_DIR / "year_holdout_profile_physics_importance_all.csv"),
    }
    (OUT_DIR / "year_holdout_profile_physics_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nYear-holdout summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
