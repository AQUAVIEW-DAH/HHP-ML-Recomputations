"""Hyperparameter and feature-set sweep for 2024-train / 2025-test XGBoost."""
from __future__ import annotations

import itertools
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
from OHC.benchmark_rtofs_argo_tabular_models import DATA_PATH, TARGETS, _prepare_features  # noqa: E402


OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")

BASE_FEATURES = [
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
]

ENRICHED_FEATURES = BASE_FEATURES + [
    "region_atlantic",
    "region_indian",
    "region_pacific",
    "region_northern_high_lat",
    "region_southern_high_lat",
    "warm_tchp_flag",
    "warm_d26_flag",
    "model_tchp_x_abs_lat",
    "model_d26_x_abs_lat",
]


def _rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((y_true - y_pred) ** 2))


def _prepare_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    x = _prepare_features(df)
    regions = [_assign_region(lat, lon) for lat, lon in zip(x["lat"], x["lon"])]
    for region in ["atlantic", "indian", "pacific", "northern_high_lat", "southern_high_lat"]:
        x[f"region_{region}"] = np.array([1 if r == region else 0 for r in regions], dtype=int)
    x["warm_tchp_flag"] = np.isfinite(x["model_interp_tchp_kj_per_cm2"]).astype(int)
    x["warm_d26_flag"] = np.isfinite(x["model_interp_d26_m"]).astype(int)
    x["model_tchp_x_abs_lat"] = x["model_interp_tchp_kj_per_cm2"] * x["abs_lat"]
    x["model_d26_x_abs_lat"] = x["model_interp_d26_m"] * x["abs_lat"]
    return x


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)
    ])


def _metric(y_true, y_pred) -> dict:
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(_rmse(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(y_true) > 1 else np.nan,
    }


def _build_inner_folds(train_dates: list[str], n_folds: int) -> list[dict]:
    n = len(train_dates)
    initial = max(15, int(round(n * 0.4)))
    val_block = max(8, int(round((n - initial - 1) / n_folds)))
    folds = []
    train_end = initial
    for i in range(n_folds):
        embargo_end = min(n, train_end + 1)
        val_start = embargo_end
        if val_start >= n:
            break
        val_end = n if i == n_folds - 1 else min(n, val_start + val_block)
        val_dates = train_dates[val_start:val_end]
        if not val_dates:
            break
        folds.append({"fold": i + 1, "train_dates": train_dates[:train_end], "val_dates": val_dates})
        train_end = val_end
    return folds


def _xgb_model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        random_state=0,
        objective="reg:squarederror",
        n_jobs=8,
        **params,
    )


def _evaluate_config(train_df: pd.DataFrame, target, feature_cols: list[str], params: dict, inner_folds: list[dict]) -> dict:
    pre = _make_preprocessor(feature_cols)
    rows = []
    date_str = train_df["date"].dt.strftime("%Y%m%d")
    for fold in inner_folds:
        tr = train_df[date_str.isin(set(fold["train_dates"]))].copy()
        va = train_df[date_str.isin(set(fold["val_dates"]))].copy()
        if tr.empty or va.empty:
            continue
        x_tr = pre.fit_transform(tr[feature_cols])
        x_va = pre.transform(va[feature_cols])
        y_tr = tr[target.delta_col].to_numpy(float)
        y_va_obs = va[target.obs_col].to_numpy(float)
        raw_va = va[target.model_col].to_numpy(float)
        model = _xgb_model(params)
        model.fit(x_tr, y_tr)
        pred = raw_va + model.predict(x_va)
        rows.append(_metric(y_va_obs, pred))
    if not rows:
        return {"cv_mae": np.nan, "cv_rmse": np.nan}
    return {
        "cv_mae": float(np.mean([r["mae"] for r in rows])),
        "cv_rmse": float(np.mean([r["rmse"] for r in rows])),
        "cv_bias": float(np.mean([r["bias"] for r in rows])),
        "cv_corr": float(np.mean([r["corr"] for r in rows])),
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = _prepare_feature_sets(df)
    train = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()
    train_dates = sorted(train["date"].dt.strftime("%Y%m%d").unique().tolist())

    feature_sets = {
        "base": BASE_FEATURES,
        "enriched": ENRICHED_FEATURES,
    }
    param_grid = list(
        itertools.product(
            [300, 500],
            [4, 6],
            [0.03, 0.05],
            [0.8, 1.0],
            [0.8, 1.0],
        )
    )
    inner_fold_options = [3, 4]

    all_results = []
    best_summary = []

    for target in TARGETS:
        eligible_train = train[pd.notna(train[target.obs_col]) & pd.notna(train[target.model_col]) & pd.notna(train[target.delta_col])].copy()
        eligible_test = test[pd.notna(test[target.obs_col]) & pd.notna(test[target.model_col]) & pd.notna(test[target.delta_col])].copy()
        if eligible_train.empty or eligible_test.empty:
            continue

        target_results = []
        for feature_name, feature_cols in feature_sets.items():
            for inner_k in inner_fold_options:
                inner_folds = _build_inner_folds(train_dates, inner_k)
                for n_estimators, max_depth, learning_rate, subsample, colsample in param_grid:
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                        "reg_lambda": 1.0,
                    }
                    cv_metrics = _evaluate_config(eligible_train, target, feature_cols, params, inner_folds)

                    pre = _make_preprocessor(feature_cols)
                    x_tr = pre.fit_transform(eligible_train[feature_cols])
                    x_te = pre.transform(eligible_test[feature_cols])
                    y_tr = eligible_train[target.delta_col].to_numpy(float)
                    y_te_obs = eligible_test[target.obs_col].to_numpy(float)
                    raw_te = eligible_test[target.model_col].to_numpy(float)

                    model = _xgb_model(params)
                    model.fit(x_tr, y_tr)
                    pred = raw_te + model.predict(x_te)
                    test_metrics = _metric(y_te_obs, pred)
                    target_results.append(
                        {
                            "target": target.name,
                            "feature_set": feature_name,
                            "inner_folds": inner_k,
                            **params,
                            **cv_metrics,
                            "test_mae_2025": test_metrics["mae"],
                            "test_rmse_2025": test_metrics["rmse"],
                            "test_bias_2025": test_metrics["bias"],
                            "test_corr_2025": test_metrics["corr"],
                            "train_rows": int(len(eligible_train)),
                            "test_rows": int(len(eligible_test)),
                            "test_dates": int(eligible_test["date"].nunique()),
                        }
                    )

        result_df = pd.DataFrame(target_results).sort_values(["test_mae_2025", "test_rmse_2025"]).reset_index(drop=True)
        result_df.to_csv(OUT_DIR / f"year_holdout_xgb_sweep_{target.name}.csv", index=False)
        best_summary.append(result_df.iloc[0].to_dict())
        all_results.append(result_df)

    summary_df = pd.DataFrame(best_summary)
    summary_df.to_csv(OUT_DIR / "year_holdout_xgb_sweep_best_summary.csv", index=False)
    payload = {
        "feature_sets": list(feature_sets.keys()),
        "param_grid_size_per_feature_fold": len(param_grid),
        "inner_fold_options": inner_fold_options,
        "best_summary_csv": str(OUT_DIR / "year_holdout_xgb_sweep_best_summary.csv"),
    }
    (OUT_DIR / "year_holdout_xgb_sweep_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nBest configs:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
