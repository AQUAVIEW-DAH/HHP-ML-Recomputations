"""Benchmark tabular ML models for RTOFS -> Argo correction.

This is the first ML benchmarking phase on the point-collocated table.
It uses a conservative, date-grouped forward-chaining evaluation to reduce
leakage from same-day spatial correlation.

Models benchmarked:
- raw RTOFS baseline
- mean-delta baseline
- Ridge
- Random Forest
- Gradient Boosting
- sklearn MLP
- XGBoost (if installed)
- simple mean-ensemble fusion of nonlinear models
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


DATA_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")

EMBARGO_DATES = 1
N_FOLDS = 3


@dataclass
class TargetSpec:
    name: str
    obs_col: str
    model_col: str
    delta_col: str


TARGETS = [
    TargetSpec(
        name="tchp",
        obs_col="argo_tchp_kj_per_cm2",
        model_col="model_interp_tchp_kj_per_cm2",
        delta_col="delta_tchp_kj_per_cm2",
    ),
    TargetSpec(
        name="d26",
        obs_col="argo_d26_m",
        model_col="model_interp_d26_m",
        delta_col="delta_d26_m",
    ),
]


def _rmse(y_true, y_pred) -> float:
    return sqrt(mean_squared_error(y_true, y_pred))


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], format="%Y%m%d")
    x["dayofyear"] = x["date"].dt.dayofyear.astype(int)
    x["month_int"] = x["month"].astype(int)
    x["month_sin"] = np.sin(2 * np.pi * x["month_int"] / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * x["month_int"] / 12.0)
    x["doy_sin"] = np.sin(2 * np.pi * x["dayofyear"] / 366.0)
    x["doy_cos"] = np.cos(2 * np.pi * x["dayofyear"] / 366.0)
    x["abs_lat"] = np.abs(x["lat"])
    x["is_winter_jfm"] = (x["season"] == "winter_jfm").astype(int)
    x["is_summer_jas"] = (x["season"] == "summer_jas").astype(int)
    x["is_other"] = (x["season"] == "other").astype(int)
    return x


FEATURE_COLUMNS = [
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


def _scaled_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            FEATURE_COLUMNS,
        )
    ])


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            FEATURE_COLUMNS,
        )
    ])


def _model_defs() -> dict[str, Callable[[], Pipeline] | None]:
    scaled = _scaled_preprocessor()
    tree = _tree_preprocessor()

    defs: dict[str, Callable[[], Pipeline] | None] = {
        "raw_rtofs": None,
        "mean_delta": None,
        "ridge_delta": lambda: Pipeline([
            ("pre", scaled),
            ("model", Ridge(alpha=1.0)),
        ]),
        "rf_delta": lambda: Pipeline([
            ("pre", tree),
            ("model", RandomForestRegressor(
                random_state=0,
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]),
        "gbr_delta": lambda: Pipeline([
            ("pre", tree),
            ("model", GradientBoostingRegressor(
                random_state=0,
                n_estimators=200,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
            )),
        ]),
        "mlp_delta": lambda: Pipeline([
            ("pre", scaled),
            ("model", MLPRegressor(
                random_state=0,
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
            )),
        ]),
    }

    if XGBRegressor is not None:
        defs["xgb_delta"] = lambda: Pipeline([
            ("pre", tree),
            ("model", XGBRegressor(
                random_state=0,
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                n_jobs=8,
            )),
        ])
    return defs


def _build_forward_folds(unique_dates: list[str], *, n_folds: int = N_FOLDS, embargo_dates: int = EMBARGO_DATES) -> list[dict]:
    n_dates = len(unique_dates)
    if n_dates < 20:
        raise ValueError(f"Too few dates for blocked forward folds: {n_dates}")

    initial_train = max(20, int(round(n_dates * 0.33)))
    val_block = max(10, int(round((n_dates - initial_train) / n_folds)))
    folds: list[dict] = []
    train_end = initial_train
    for fold_idx in range(n_folds):
        val_start = train_end + embargo_dates
        if val_start >= n_dates:
            break
        if fold_idx == n_folds - 1:
            val_end = n_dates
        else:
            val_end = min(n_dates, val_start + val_block)
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[val_start:val_end]
        if not val_dates:
            break
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_dates": train_dates,
                "val_dates": val_dates,
                "embargo_dates": unique_dates[train_end:val_start],
            }
        )
        train_end = val_end
    return folds


def _metric_row(*, target: str, model: str, evaluation: str, y_true: np.ndarray, y_pred: np.ndarray, rows: int, dates: int) -> dict:
    return {
        "target": target,
        "model": model,
        "evaluation": evaluation,
        "rows": int(rows),
        "dates": int(dates),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(_rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan,
    }


def benchmark_target(df: pd.DataFrame, target: TargetSpec, folds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[np.isfinite(df[target.obs_col]) & np.isfinite(df[target.model_col]) & np.isfinite(df[target.delta_col])].copy()
    work = _prepare_features(work)
    model_defs = _model_defs()

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    nonlinear_for_fusion = [m for m in ["rf_delta", "gbr_delta", "mlp_delta", "xgb_delta"] if m in model_defs]

    for fold in folds:
        train_mask = work["date"].dt.strftime("%Y%m%d").isin(fold["train_dates"])
        val_mask = work["date"].dt.strftime("%Y%m%d").isin(fold["val_dates"])
        train_df = work[train_mask].copy()
        val_df = work[val_mask].copy()
        if train_df.empty or val_df.empty:
            continue

        x_train = train_df[FEATURE_COLUMNS]
        x_val = val_df[FEATURE_COLUMNS]
        y_train_obs = train_df[target.obs_col].to_numpy(float)
        y_val_obs = val_df[target.obs_col].to_numpy(float)
        y_train_delta = train_df[target.delta_col].to_numpy(float)
        raw_val = val_df[target.model_col].to_numpy(float)

        fold_preds: dict[str, np.ndarray] = {}

        fold_preds["raw_rtofs"] = raw_val.copy()
        metric_rows.append(
            _metric_row(
                target=target.name,
                model="raw_rtofs",
                evaluation=f"fold_{fold['fold']}",
                y_true=y_val_obs,
                y_pred=fold_preds["raw_rtofs"],
                rows=len(val_df),
                dates=val_df["date"].nunique(),
            )
        )

        mean_delta = float(np.mean(y_train_delta))
        fold_preds["mean_delta"] = raw_val + mean_delta
        metric_rows.append(
            _metric_row(
                target=target.name,
                model="mean_delta",
                evaluation=f"fold_{fold['fold']}",
                y_true=y_val_obs,
                y_pred=fold_preds["mean_delta"],
                rows=len(val_df),
                dates=val_df["date"].nunique(),
            )
        )

        for name, factory in model_defs.items():
            if name in {"raw_rtofs", "mean_delta"}:
                continue
            model = factory()
            model.fit(x_train, y_train_delta)
            pred = raw_val + model.predict(x_val)
            fold_preds[name] = pred
            metric_rows.append(
                _metric_row(
                    target=target.name,
                    model=name,
                    evaluation=f"fold_{fold['fold']}",
                    y_true=y_val_obs,
                    y_pred=pred,
                    rows=len(val_df),
                    dates=val_df["date"].nunique(),
                )
            )

        if nonlinear_for_fusion:
            stacked = np.column_stack([fold_preds[m] for m in nonlinear_for_fusion if m in fold_preds])
            fold_preds["mean_nonlinear_ensemble"] = np.mean(stacked, axis=1)
            metric_rows.append(
                _metric_row(
                    target=target.name,
                    model="mean_nonlinear_ensemble",
                    evaluation=f"fold_{fold['fold']}",
                    y_true=y_val_obs,
                    y_pred=fold_preds["mean_nonlinear_ensemble"],
                    rows=len(val_df),
                    dates=val_df["date"].nunique(),
                )
            )

        pred_df = val_df[["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]].copy()
        pred_df["target"] = target.name
        pred_df["fold"] = fold["fold"]
        for name, pred in fold_preds.items():
            pred_df[f"pred_obs__{name}"] = pred
        prediction_rows.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    summary_rows: list[dict] = []
    if not preds_df.empty:
        pred_cols = [c for c in preds_df.columns if c.startswith("pred_obs__")]
        y_true = preds_df[target.obs_col].to_numpy(float)
        for col in pred_cols:
            name = col.replace("pred_obs__", "")
            summary_rows.append(
                _metric_row(
                    target=target.name,
                    model=name,
                    evaluation="oof_summary",
                    y_true=y_true,
                    y_pred=preds_df[col].to_numpy(float),
                    rows=len(preds_df),
                    dates=preds_df["date"].nunique(),
                )
            )
        metrics_df = pd.concat([metrics_df, pd.DataFrame(summary_rows)], ignore_index=True)

    return metrics_df, preds_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    unique_dates = sorted(pd.Series(df["date"].astype(str).unique()).tolist())
    folds = _build_forward_folds(unique_dates)

    fold_note = {
        "n_dates_total": len(unique_dates),
        "embargo_dates": EMBARGO_DATES,
        "n_folds": len(folds),
        "folds": [
            {
                "fold": f["fold"],
                "train_date_count": len(f["train_dates"]),
                "val_date_count": len(f["val_dates"]),
                "embargo_date_count": len(f["embargo_dates"]),
                "train_start": f["train_dates"][0],
                "train_end": f["train_dates"][-1],
                "val_start": f["val_dates"][0],
                "val_end": f["val_dates"][-1],
            }
            for f in folds
        ],
    }
    (OUT_DIR / "tabular_benchmark_folds.json").write_text(json.dumps(fold_note, indent=2))

    all_metrics: list[pd.DataFrame] = []
    all_preds: list[pd.DataFrame] = []
    for target in TARGETS:
        metrics_df, preds_df = benchmark_target(df, target, folds)
        metrics_df.to_csv(OUT_DIR / f"tabular_metrics_{target.name}.csv", index=False)
        preds_df.to_parquet(OUT_DIR / f"tabular_oof_predictions_{target.name}.parquet", index=False)
        all_metrics.append(metrics_df)
        all_preds.append(preds_df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "tabular_metrics_all.csv", index=False)
    summary = (
        metrics[metrics["evaluation"] == "oof_summary"]
        .sort_values(["target", "mae", "rmse"])
        .reset_index(drop=True)
    )
    summary.to_csv(OUT_DIR / "tabular_metrics_oof_summary.csv", index=False)

    payload = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUT_DIR),
        "models_attempted": list(_model_defs().keys()) + ["mean_nonlinear_ensemble"],
        "targets": [t.name for t in TARGETS],
        "fold_note_path": str(OUT_DIR / "tabular_benchmark_folds.json"),
        "summary_csv": str(OUT_DIR / "tabular_metrics_oof_summary.csv"),
        "xgboost_available": XGBRegressor is not None,
    }
    (OUT_DIR / "tabular_benchmark_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nOOF summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
