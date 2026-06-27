"""Run locked-protocol XGBoost evaluation for selected semi-ablation feature sets.

This is the stricter follow-up to the 2024->2025 year-holdout pass. It reuses
the date-blocked forward folds from the locked protocol and compares a compact
set of candidate feature recipes on the merged global-physics + profile-physics
table.

The goal is robustness checking, not a fresh hyperparameter search.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.analyze_locked_xgb_results import _assign_region, _season_from_month  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402


BASE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
GLOBAL_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
PROFILE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")
FOLD_PATH = OUT_DIR / "tabular_benchmark_folds.json"

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

GLOBAL_PRUNED = [
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

FEATURE_SETS_BY_TARGET = {
    "tchp": {
        "base": BASE_FEATURES,
        "global_pruned": BASE_FEATURES + GLOBAL_PRUNED,
        "drop_temp_lat_interaction": BASE_FEATURES + [
            "model_ssh_m",
            "model_mixed_layer_thickness_m",
            "model_surface_boundary_layer_thickness_m",
            "model_temp_excess_26c",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_ssh_x_abs_lat",
            "model_mlt_x_abs_lat",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
        "drop_ssh_lat_interaction": BASE_FEATURES + [
            "model_ssh_m",
            "model_mixed_layer_thickness_m",
            "model_surface_boundary_layer_thickness_m",
            "model_temp_excess_26c",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_mlt_x_abs_lat",
            "model_temp_excess_x_abs_lat",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
        "surface_temp_swap": BASE_FEATURES + [
            "model_ssh_m",
            "model_mixed_layer_thickness_m",
            "model_surface_boundary_layer_thickness_m",
            "model_surface_temp_c",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_ssh_x_abs_lat",
            "model_mlt_x_abs_lat",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
    },
    "d26": {
        "base": BASE_FEATURES,
        "global_pruned": BASE_FEATURES + GLOBAL_PRUNED,
        "drop_both_lat_interactions": BASE_FEATURES + [
            "model_ssh_m",
            "model_mixed_layer_thickness_m",
            "model_surface_boundary_layer_thickness_m",
            "model_temp_excess_26c",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_mlt_x_abs_lat",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
        "drop_temp_lat_interaction": BASE_FEATURES + [
            "model_ssh_m",
            "model_mixed_layer_thickness_m",
            "model_surface_boundary_layer_thickness_m",
            "model_temp_excess_26c",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_ssh_x_abs_lat",
            "model_mlt_x_abs_lat",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
        "global_pruned_plus_profile_core": BASE_FEATURES + GLOBAL_PRUNED + [
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ],
    },
}

BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def _rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)
    ])


def _xgb_model() -> XGBRegressor:
    return XGBRegressor(
        random_state=0,
        objective="reg:squarederror",
        n_jobs=8,
        **BEST_PARAMS,
    )


def _metric_row(*, target: str, model: str, evaluation: str, y_true, y_pred, rows: int, dates: int) -> dict:
    return {
        "target": target,
        "model": model,
        "evaluation": evaluation,
        "rows": int(rows),
        "dates": int(dates),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(_rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float((y_pred - y_true).mean()),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(y_true) > 1 else float("nan"),
    }


def _merge_feature_tables() -> pd.DataFrame:
    base = pd.read_parquet(BASE_PATH).reset_index(drop=True)
    global_df = pd.read_parquet(GLOBAL_PATH).reset_index(drop=True)
    profile_df = pd.read_parquet(PROFILE_PATH).reset_index(drop=True)

    key_cols = [
        "date",
        "year",
        "month",
        "lat",
        "lon",
        "nearest_rtofs_grid_distance_km",
        "argo_tchp_kj_per_cm2",
        "argo_d26_m",
        "model_interp_tchp_kj_per_cm2",
        "model_interp_d26_m",
        "delta_tchp_kj_per_cm2",
        "delta_d26_m",
    ]
    for name, df in [("global", global_df), ("profile", profile_df)]:
        if len(df) != len(base):
            raise RuntimeError(f"{name} table length {len(df)} does not match base length {len(base)}")
        for col in key_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(base[col]) and pd.api.types.is_numeric_dtype(df[col]):
                equal = np.allclose(np.asarray(base[col], dtype=float), np.asarray(df[col], dtype=float), equal_nan=True)
            else:
                equal = base[col].astype(str).equals(df[col].astype(str))
            if not equal:
                raise RuntimeError(f"{name} feature table is not aligned with base rows at column {col}")

    extra_global = [c for c in global_df.columns if c not in base.columns]
    extra_profile = [c for c in profile_df.columns if c not in base.columns and c not in extra_global]
    return pd.concat([base, global_df[extra_global], profile_df[extra_profile]], axis=1)


def _add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region_group"] = [_assign_region(lat, lon) for lat, lon in zip(out["lat"], out["lon"])]
    out["season_group"] = out["month"].astype(int).map(_season_from_month)
    return out


def run_target(df: pd.DataFrame, target) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)

    fold_note = json.loads(FOLD_PATH.read_text())
    unique_dates = sorted(pd.Series(work["date"].dt.strftime("%Y%m%d").unique()).tolist())
    full_folds = _build_forward_folds(
        unique_dates,
        n_folds=fold_note["n_folds"],
        embargo_dates=fold_note["embargo_dates"],
    )

    feature_sets = FEATURE_SETS_BY_TARGET[target.name]
    for name, cols in feature_sets.items():
        missing = [c for c in cols if c not in work.columns]
        if missing:
            raise RuntimeError(f"Feature set {target.name}:{name} is missing columns: {missing}")

    metrics: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for fold in full_folds:
        train_dates = set(fold["train_dates"])
        val_dates = set(fold["val_dates"])
        date_str = work["date"].dt.strftime("%Y%m%d")
        train_df = work[date_str.isin(train_dates)].copy()
        val_df = work[date_str.isin(val_dates)].copy()
        if train_df.empty or val_df.empty:
            continue

        y_train_delta = train_df[target.delta_col].to_numpy(float)
        y_val_obs = val_df[target.obs_col].to_numpy(float)
        raw_val = val_df[target.model_col].to_numpy(float)

        metrics.append(
            _metric_row(
                target=target.name,
                model="raw_rtofs",
                evaluation=f"fold_{fold['fold']}",
                y_true=y_val_obs,
                y_pred=raw_val,
                rows=len(val_df),
                dates=val_df["date"].nunique(),
            )
        )

        pred_df = val_df[
            ["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]
        ].copy()
        pred_df["target"] = target.name
        pred_df["fold"] = fold["fold"]
        pred_df = _add_group_cols(pred_df)
        pred_df["pred_obs__raw_rtofs"] = raw_val

        for feature_set_name, feature_cols in feature_sets.items():
            pre = _make_preprocessor(feature_cols)
            x_train = pre.fit_transform(train_df[feature_cols])
            x_val = pre.transform(val_df[feature_cols])

            model = _xgb_model()
            model.fit(x_train, y_train_delta)
            pred = raw_val + model.predict(x_val)

            metrics.append(
                _metric_row(
                    target=target.name,
                    model=feature_set_name,
                    evaluation=f"fold_{fold['fold']}",
                    y_true=y_val_obs,
                    y_pred=pred,
                    rows=len(val_df),
                    dates=val_df["date"].nunique(),
                )
            )
            pred_df[f"pred_obs__{feature_set_name}"] = pred

        pred_frames.append(pred_df)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    y_true = pred_all[target.obs_col].to_numpy(float)
    pred_cols = [c for c in pred_all.columns if c.startswith("pred_obs__")]
    for pred_col in pred_cols:
        metrics.append(
            _metric_row(
                target=target.name,
                model=pred_col.replace("pred_obs__", ""),
                evaluation="oof_summary",
                y_true=y_true,
                y_pred=pred_all[pred_col].to_numpy(float),
                rows=len(pred_all),
                dates=pred_all["date"].nunique(),
            )
        )

    return pd.DataFrame(metrics), pred_all


def _grouped_metrics(pred_df: pd.DataFrame, *, target: str, obs_col: str) -> pd.DataFrame:
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_obs__")]
    rows: list[dict] = []
    for grouping, col in [("season", "season_group"), ("region", "region_group"), ("fold", "fold")]:
        for key, gdf in pred_df.groupby(col):
            y_true = gdf[obs_col].to_numpy(float)
            for pred_col in pred_cols:
                model = pred_col.replace("pred_obs__", "")
                y_pred = gdf[pred_col].to_numpy(float)
                rows.append(
                    {
                        "target": target,
                        "grouping": grouping,
                        "group": str(key),
                        "model": model,
                        "rows": int(len(gdf)),
                        "dates": int(gdf["date"].nunique()),
                        "mae": float(np.mean(np.abs(y_true - y_pred))),
                        "rmse": float(_rmse(y_true, y_pred)),
                        "bias": float(np.mean(y_pred - y_true)),
                        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(gdf) > 1 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()

    all_metrics = []
    all_preds = []
    grouped = []
    for target in TARGETS:
        metrics_df, preds_df = run_target(df, target)
        metrics_df.to_csv(OUT_DIR / f"locked_physics_semi_ablation_metrics_{target.name}.csv", index=False)
        preds_df.to_parquet(OUT_DIR / f"locked_physics_semi_ablation_predictions_{target.name}.parquet", index=False)
        grouped_df = _grouped_metrics(preds_df, target=target.name, obs_col=target.obs_col)
        grouped_df.to_csv(OUT_DIR / f"locked_physics_semi_ablation_grouped_{target.name}.csv", index=False)
        all_metrics.append(metrics_df)
        all_preds.append(preds_df)
        grouped.append(grouped_df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    grouped_df = pd.concat(grouped, ignore_index=True)
    metrics.to_csv(OUT_DIR / "locked_physics_semi_ablation_metrics_all.csv", index=False)
    grouped_df.to_csv(OUT_DIR / "locked_physics_semi_ablation_grouped_all.csv", index=False)

    summary = (
        metrics[metrics["evaluation"] == "oof_summary"]
        .sort_values(["target", "mae", "rmse", "model"])
        .reset_index(drop=True)
    )
    summary.to_csv(OUT_DIR / "locked_physics_semi_ablation_oof_summary.csv", index=False)

    payload = {
        "base_path": str(BASE_PATH),
        "global_path": str(GLOBAL_PATH),
        "profile_path": str(PROFILE_PATH),
        "fold_path": str(FOLD_PATH),
        "best_params": BEST_PARAMS,
        "feature_sets_by_target": FEATURE_SETS_BY_TARGET,
        "summary_csv": str(OUT_DIR / "locked_physics_semi_ablation_oof_summary.csv"),
        "grouped_csv": str(OUT_DIR / "locked_physics_semi_ablation_grouped_all.csv"),
    }
    (OUT_DIR / "locked_physics_semi_ablation_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nLocked semi-ablation OOF summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
