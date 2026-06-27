"""Run semi-ablation XGBoost experiments on added physics features.

This script keeps the existing year-holdout protocol:

- train on collocated 2024 rows
- test on collocated 2025 rows

It focuses on a small set of targeted feature-set variants motivated by:

- Pearson redundancy checks across the added physics features
- XGBoost gain importance from the earlier global-physics benchmark

The goal is not an exhaustive feature search. It is a decision-oriented
"semi-ablation" pass that tests which correlated features should be kept,
swapped, or dropped before we widen the data pool.
"""
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

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _prepare_features  # noqa: E402


BASE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
GLOBAL_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
PROFILE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet")

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

PROFILE_FEATURES = [
    "model_steric_0_1000_m",
    "model_steric_0_2000_m",
    "model_steric_1000_ref2000_m",
    "model_n2_mean_upper200_s2",
    "model_n2_max_upper200_s2",
    "model_n2_mean_to_d26_s2",
    "model_n2_max_to_d26_s2",
]

BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

FEATURE_SETS = {
    # Re-baseline from the best earlier global-physics run.
    "global_pruned": BASE_FEATURES + GLOBAL_PRUNED,
    # Add the strongest low-count profile summaries without bringing in all
    # redundant variants at once.
    "global_pruned_plus_profile_core": BASE_FEATURES + GLOBAL_PRUNED + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Alternate profile view using absolute 0/1000 steric and upper-200 mean N2.
    "global_pruned_plus_profile_alt": BASE_FEATURES + GLOBAL_PRUNED + [
        "model_steric_0_1000_m",
        "model_n2_mean_upper200_s2",
        "model_n2_max_upper200_s2",
    ],
    # Remove the high-correlation SSH interaction to test whether raw SSH is enough.
    "drop_ssh_lat_interaction": BASE_FEATURES + [
        f for f in GLOBAL_PRUNED if f != "model_ssh_x_abs_lat"
    ] + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Remove the high-correlation temperature interaction to see how much it
    # really buys beyond temperature excess itself.
    "drop_temp_lat_interaction": BASE_FEATURES + [
        f for f in GLOBAL_PRUNED if f != "model_temp_excess_x_abs_lat"
    ] + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Stronger prune: remove both interaction terms and keep the profile core.
    "drop_both_lat_interactions": BASE_FEATURES + [
        f
        for f in GLOBAL_PRUNED
        if f not in {"model_ssh_x_abs_lat", "model_temp_excess_x_abs_lat"}
    ] + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Swap temperature excess for raw surface temperature from the correlated pair.
    "surface_temp_swap": BASE_FEATURES + [
        "model_ssh_m",
        "model_mixed_layer_thickness_m",
        "model_surface_boundary_layer_thickness_m",
        "model_surface_temp_c",
        "d26_minus_mlt_m",
        "d26_to_sblt_ratio",
        "model_ssh_x_abs_lat",
        "model_mlt_x_abs_lat",
    ] + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Swap warm-layer feature choice inside the near-duplicate pair.
    "warm_layer_swap": BASE_FEATURES + [
        "model_ssh_m",
        "model_mixed_layer_thickness_m",
        "model_surface_boundary_layer_thickness_m",
        "model_temp_excess_26c",
        "warm_layer_thickness_positive_m",
        "d26_to_sblt_ratio",
        "model_ssh_x_abs_lat",
        "model_mlt_x_abs_lat",
        "model_temp_excess_x_abs_lat",
    ] + [
        "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
    # Keep a broader but still non-exhaustive profile set to see if extra N2
    # summaries and one absolute steric level help once redundancy is trimmed.
    "profile_expanded_nonredundant": BASE_FEATURES + GLOBAL_PRUNED + [
        "model_steric_0_1000_m",
        "model_steric_1000_ref2000_m",
        "model_n2_mean_upper200_s2",
        "model_n2_max_upper200_s2",
        "model_n2_mean_to_d26_s2",
    ],
}


def _rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((y_true - y_pred) ** 2))


def _metric_row(*, target: str, model: str, y_true, y_pred, raw_pred, dates: int) -> dict:
    return {
        "target": target,
        "model": model,
        "rows": int(len(y_true)),
        "dates": int(dates),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(_rmse(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(y_true) > 1 else np.nan,
        "raw_mae": float(np.mean(np.abs(y_true - raw_pred))),
        "mae_gain_vs_raw": float(np.mean(np.abs(y_true - raw_pred)) - np.mean(np.abs(y_true - y_pred))),
    }


def _make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols)
    ])


def _make_model() -> XGBRegressor:
    return XGBRegressor(
        random_state=0,
        objective="reg:squarederror",
        n_jobs=8,
        **BEST_PARAMS,
    )


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
        mismatch = False
        for col in key_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(base[col]) and pd.api.types.is_numeric_dtype(df[col]):
                equal = np.allclose(
                    np.asarray(base[col], dtype=float),
                    np.asarray(df[col], dtype=float),
                    equal_nan=True,
                )
            else:
                equal = base[col].astype(str).equals(df[col].astype(str))
            if not equal:
                mismatch = True
                break
        if mismatch:
            raise RuntimeError(f"{name} feature table is not aligned with base rows; aborting merge.")

    extra_global = [c for c in global_df.columns if c not in base.columns]
    extra_profile = [c for c in profile_df.columns if c not in base.columns and c not in extra_global]
    merged = pd.concat([base, global_df[extra_global], profile_df[extra_profile]], axis=1)
    return merged


def _run_feature_set(df: pd.DataFrame, target, feature_set_name: str, feature_cols: list[str]) -> tuple[dict, pd.DataFrame]:
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

    metrics = _metric_row(
        target=target.name,
        model=feature_set_name,
        y_true=y_test_obs,
        y_pred=pred_test,
        raw_pred=raw_test,
        dates=test_df["date"].nunique(),
    )

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    imp_rows = []
    for i, feature in enumerate(feature_cols):
        key = f"num__{feature}"
        alt_key = f"f{i}"
        imp_rows.append(
            {
                "target": target.name,
                "feature_set": feature_set_name,
                "feature": feature,
                "gain": float(gain.get(key, gain.get(alt_key, 0.0))),
                "weight": float(weight.get(key, weight.get(alt_key, 0.0))),
            }
        )
    imp_df = pd.DataFrame(imp_rows).sort_values(["gain", "weight"], ascending=False)
    return metrics, imp_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = _merge_feature_tables()
    prepared = _prepare_features(merged)

    metrics_rows: list[dict] = []
    importance_frames: list[pd.DataFrame] = []

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        missing = [c for c in feature_cols if c not in prepared.columns]
        if missing:
            raise RuntimeError(f"Feature set {feature_set_name} is missing columns: {missing}")
        for target in TARGETS:
            metrics, imp_df = _run_feature_set(merged, target, feature_set_name, feature_cols)
            metrics_rows.append(metrics)
            importance_frames.append(imp_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "mae", "model"]).reset_index(drop=True)
    importance_df = pd.concat(importance_frames, ignore_index=True)

    metrics_path = OUT_DIR / "year_holdout_physics_semi_ablation_summary.csv"
    importance_path = OUT_DIR / "year_holdout_physics_semi_ablation_importance.csv"
    json_path = OUT_DIR / "year_holdout_physics_semi_ablation_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    best_rows = (
        metrics_df.sort_values(["target", "mae"])
        .groupby("target", as_index=False)
        .first()
        .to_dict(orient="records")
    )
    payload = {
        "scheme": "train_2024_test_2025",
        "base_path": str(BASE_PATH),
        "global_path": str(GLOBAL_PATH),
        "profile_path": str(PROFILE_PATH),
        "best_params": BEST_PARAMS,
        "feature_sets": FEATURE_SETS,
        "best_by_target": best_rows,
        "summary_csv": str(metrics_path),
        "importance_csv": str(importance_path),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print("\nSemi-ablation summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
