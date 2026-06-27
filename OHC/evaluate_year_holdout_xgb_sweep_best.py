"""Evaluate best 2024->2025 sweep configs with grouped 2025 metrics."""
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
from OHC.run_year_holdout_xgb_sweep_2024_2025 import BASE_FEATURES, ENRICHED_FEATURES  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")
BEST_CSV = OUT_DIR / "year_holdout_xgb_sweep_best_summary.csv"


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


def _add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_group"] = [_assign_region(lat, lon) for lat, lon in zip(out["lat"], out["lon"])]
    out["season_group"] = out["month"].astype(int).map(_season_from_month)
    return out


def _xgb_model(params: dict) -> XGBRegressor:
    return XGBRegressor(random_state=0, objective="reg:squarederror", n_jobs=8, **params)


def main() -> None:
    best = pd.read_csv(BEST_CSV)
    df = _prepare_feature_sets(pd.read_parquet(DATA_PATH))
    feature_sets = {"base": BASE_FEATURES, "enriched": ENRICHED_FEATURES}

    all_metrics = []
    all_preds = []
    summary_payload = {"source_best_summary": str(BEST_CSV), "targets": {}}

    target_lookup = {t.name: t for t in TARGETS}
    for _, row in best.iterrows():
        target = target_lookup[row["target"]]
        feature_cols = feature_sets[row["feature_set"]]
        params = {
            "n_estimators": int(row["n_estimators"]),
            "max_depth": int(row["max_depth"]),
            "learning_rate": float(row["learning_rate"]),
            "subsample": float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
            "reg_lambda": float(row["reg_lambda"]),
        }

        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        train_df = work[work["year"] == 2024].copy()
        test_df = work[work["year"] == 2025].copy()

        pre = _make_preprocessor(feature_cols)
        x_train = pre.fit_transform(train_df[feature_cols])
        x_test = pre.transform(test_df[feature_cols])
        y_train_delta = train_df[target.delta_col].to_numpy(float)
        y_test_obs = test_df[target.obs_col].to_numpy(float)
        raw_test = test_df[target.model_col].to_numpy(float)

        model = _xgb_model(params)
        model.fit(x_train, y_train_delta)
        pred_test = raw_test + model.predict(x_test)

        pred_df = test_df[["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]].copy()
        pred_df = _add_group_cols(pred_df)
        pred_df["pred_obs__raw_rtofs"] = raw_test
        pred_df["pred_obs__xgb_sweep_best_2024_train_2025_test"] = pred_test
        all_preds.append(pred_df)

        metrics = []
        metrics.append(_metric_row(target=target.name, model="raw_rtofs", split="2025_all", y_true=y_test_obs, y_pred=raw_test, rows=len(pred_df), dates=pred_df["date"].nunique()))
        metrics.append(_metric_row(target=target.name, model="xgb_sweep_best_2024_train_2025_test", split="2025_all", y_true=y_test_obs, y_pred=pred_test, rows=len(pred_df), dates=pred_df["date"].nunique()))

        for group_name, col in [("season", "season_group"), ("region", "region_group")]:
            for key, gdf in pred_df.groupby(col):
                y_true = gdf[target.obs_col].to_numpy(float)
                metrics.append(_metric_row(target=target.name, model="raw_rtofs", split=f"2025_{group_name}:{key}", y_true=y_true, y_pred=gdf["pred_obs__raw_rtofs"].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()))
                metrics.append(_metric_row(target=target.name, model="xgb_sweep_best_2024_train_2025_test", split=f"2025_{group_name}:{key}", y_true=y_true, y_pred=gdf["pred_obs__xgb_sweep_best_2024_train_2025_test"].to_numpy(float), rows=len(gdf), dates=gdf["date"].nunique()))

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(OUT_DIR / f"year_holdout_sweep_best_grouped_metrics_{target.name}.csv", index=False)
        pred_df.to_parquet(OUT_DIR / f"year_holdout_sweep_best_predictions_{target.name}.parquet", index=False)
        all_metrics.append(metrics_df)
        summary_payload["targets"][target.name] = {
            "feature_set": row["feature_set"],
            "inner_folds": int(row["inner_folds"]),
            "params": params,
            "metrics_csv": str(OUT_DIR / f"year_holdout_sweep_best_grouped_metrics_{target.name}.csv"),
        }

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(OUT_DIR / "year_holdout_sweep_best_grouped_metrics_all.csv", index=False)
    (OUT_DIR / "year_holdout_sweep_best_grouped_run_summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
