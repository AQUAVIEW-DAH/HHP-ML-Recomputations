"""Train on 2024, test on 2025 for locked XGBoost correction models."""
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
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)
    ])


def _make_model() -> XGBRegressor:
    return XGBRegressor(
        random_state=0,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=8,
    )


def _add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_group"] = [_assign_region(lat, lon) for lat, lon in zip(out["lat"], out["lon"])]
    out["season_group"] = out["month"].astype(int).map(_season_from_month)
    return out


def run_target(df: pd.DataFrame, target, feature_cols: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)
    train_df = work[work["year"] == 2024].copy()
    test_df = work[work["year"] == 2025].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Missing train/test rows for target {target.name}")

    pre = _make_preprocessor(feature_cols)
    x_train = pre.fit_transform(train_df[feature_cols])
    x_test = pre.transform(test_df[feature_cols])
    y_train_delta = train_df[target.delta_col].to_numpy(float)
    y_test_obs = test_df[target.obs_col].to_numpy(float)
    raw_test = test_df[target.model_col].to_numpy(float)

    model = _make_model()
    model.fit(x_train, y_train_delta)
    pred_test = raw_test + model.predict(x_test)

    pred_df = test_df[["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]].copy()
    pred_df = _add_group_cols(pred_df)
    pred_df["pred_obs__raw_rtofs"] = raw_test
    pred_df["pred_obs__xgb_2024_train_2025_test"] = pred_test

    rows: list[dict] = []
    rows.append(
        _metric_row(
            target=target.name,
            model="raw_rtofs",
            split="2025_all",
            y_true=y_test_obs,
            y_pred=raw_test,
            rows=len(pred_df),
            dates=pred_df["date"].nunique(),
        )
    )
    rows.append(
        _metric_row(
            target=target.name,
            model=model_name,
            split="2025_all",
            y_true=y_test_obs,
            y_pred=pred_test,
            rows=len(pred_df),
            dates=pred_df["date"].nunique(),
        )
    )

    for group_name, col in [("season", "season_group"), ("region", "region_group")]:
        for key, gdf in pred_df.groupby(col):
            y_true = gdf[target.obs_col].to_numpy(float)
            rows.append(_metric_row(target=target.name, model="raw_rtofs", split=f"2025_{group_name}:{key}", y_true=y_true, y_pred=gdf["pred_obs__raw_rtofs"].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()))
            rows.append(_metric_row(target=target.name, model=model_name, split=f"2025_{group_name}:{key}", y_true=y_true, y_pred=gdf["pred_obs__xgb_2024_train_2025_test"].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()))

    return pd.DataFrame(rows), pred_df


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    metrics_all = []
    preds_all = []
    for target in TARGETS:
        metrics_df, pred_df = run_target(df, target, BASE_FEATURES, "xgb_2024_train_2025_test")
        metrics_df.to_csv(OUT_DIR / f"year_holdout_metrics_{target.name}.csv", index=False)
        pred_df.to_parquet(OUT_DIR / f"year_holdout_predictions_{target.name}.parquet", index=False)
        metrics_all.append(metrics_df)
        preds_all.append(pred_df)

    metrics = pd.concat(metrics_all, ignore_index=True)
    metrics.to_csv(OUT_DIR / "year_holdout_metrics_all.csv", index=False)

    payload = {
        "scheme": "train_2024_test_2025",
        "data_path": str(DATA_PATH),
        "feature_set": BASE_FEATURES,
        "summary_csv": str(OUT_DIR / "year_holdout_metrics_all.csv"),
    }
    (OUT_DIR / "year_holdout_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nYear-holdout metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
