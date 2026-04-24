"""Train a first-pass HHP correction model on the Milton pair table.

This script benchmarks a few simple tabular regressors under grouped
cross-validation and freezes the best simple baseline as a correction model.
For the prototype we learn the TCHP delta:

    obs_tchp_kj_cm2 - model_tchp_kj_cm2

so the corrected estimate remains physically interpretable:

    corrected_tchp = raw_rtofs_tchp + learned_delta
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.features import FEATURE_COLUMNS, build_training_feature_frame
from ml.modeling import DEFAULT_MODEL_PATH, save_model_bundle
from ml.paths import BENCHMARK_REPORTS_DIR, DATASETS_DIR

PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
OOF_CSV = DATASETS_DIR / "milton_pairs_with_ml_recomputed.csv"
BENCHMARK_CSV = BENCHMARK_REPORTS_DIR / "milton_model_benchmark.csv"
BENCHMARK_MD = BENCHMARK_REPORTS_DIR / "milton_model_benchmark.md"


def _rmse(y_true, y_pred) -> float:
    return sqrt(mean_squared_error(y_true, y_pred))


def _numeric_preprocessor() -> ColumnTransformer:
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


def _benchmark_models(pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    X = build_training_feature_frame(pairs_df)
    y_obs = pairs_df["obs_tchp_kj_cm2"]
    y_delta = pairs_df["tchp_delta_kj_cm2"]
    groups = pairs_df["platform"]
    cv = GroupKFold(n_splits=5)
    pre = _numeric_preprocessor()

    model_defs = {
        "raw_rtofs": None,
        "ridge_delta": Pipeline([
            ("pre", pre),
            ("model", Ridge(alpha=1.0)),
        ]),
        "gbr_delta": Pipeline([
            ("pre", pre),
            ("model", GradientBoostingRegressor(
                random_state=0,
                n_estimators=120,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
            )),
        ]),
        "rf_delta": Pipeline([
            ("pre", pre),
            ("model", RandomForestRegressor(
                random_state=0,
                n_estimators=300,
                min_samples_leaf=2,
            )),
        ]),
    }

    benchmark_rows: list[dict[str, float | str]] = []
    predictions: dict[str, list[float]] = {}

    raw_pred = pairs_df["model_tchp_kj_cm2"].to_numpy()
    predictions["raw_rtofs"] = raw_pred.tolist()
    benchmark_rows.append({
        "model": "raw_rtofs",
        "target": "obs_tchp",
        "mae": mean_absolute_error(y_obs, raw_pred),
        "rmse": _rmse(y_obs, raw_pred),
        "r2": r2_score(y_obs, raw_pred),
        "bias": float((raw_pred - y_obs).mean()),
    })

    for name, model in model_defs.items():
        if name == "raw_rtofs":
            continue
        pred_delta = cross_val_predict(model, X, y_delta, cv=cv, groups=groups)
        pred_obs = raw_pred + pred_delta
        predictions[name] = pred_obs.tolist()
        benchmark_rows.append({
            "model": name,
            "target": "delta_tchp",
            "mae": mean_absolute_error(y_obs, pred_obs),
            "rmse": _rmse(y_obs, pred_obs),
            "r2": r2_score(y_obs, pred_obs),
            "bias": float((pred_obs - y_obs).mean()),
        })

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return benchmark_df, predictions


def _freeze_best_model(pairs_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    best_name = benchmark_df.iloc[0]["model"]
    if best_name != "gbr_delta":
        raise RuntimeError(f"Expected gbr_delta to win for the first Milton baseline, got {best_name}")

    X = build_training_feature_frame(pairs_df)
    y_delta = pairs_df["tchp_delta_kj_cm2"]

    final_model = Pipeline([
        ("pre", _numeric_preprocessor()),
        ("model", GradientBoostingRegressor(
            random_state=0,
            n_estimators=120,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
        )),
    ])
    final_model.fit(X, y_delta)

    bundle = {
        "model_name": "gbr_delta",
        "target_name": "tchp_delta_kj_cm2",
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(pairs_df)),
        "train_platforms": int(pairs_df["platform"].nunique()),
        "train_event": "MILTON_2024",
        "train_date_min": str(pairs_df["obs_time"].min()),
        "train_date_max": str(pairs_df["obs_time"].max()),
        "model": final_model,
    }
    save_model_bundle(bundle, DEFAULT_MODEL_PATH)
    return bundle


def _write_report(benchmark_df: pd.DataFrame, bundle: dict) -> None:
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(BENCHMARK_CSV, index=False)

    best = benchmark_df.iloc[0]
    raw = benchmark_df[benchmark_df["model"] == "raw_rtofs"].iloc[0]
    md = f"""# Milton HHP Baseline Benchmark

This benchmark compares simple grouped-cross-validation baselines on the Milton
2024 Argo/RTOFS pair table. Groups are Argo `platform` IDs so the model does
not train and test on the same float.

## Best simple model

- Model: `{best['model']}`
- MAE: `{best['mae']:.3f}` kJ/cm²
- RMSE: `{best['rmse']:.3f}` kJ/cm²
- R²: `{best['r2']:.3f}`
- Bias: `{best['bias']:.3f}` kJ/cm²

## Raw RTOFS baseline

- MAE: `{raw['mae']:.3f}` kJ/cm²
- RMSE: `{raw['rmse']:.3f}` kJ/cm²
- R²: `{raw['r2']:.3f}`
- Bias: `{raw['bias']:.3f}` kJ/cm²

## Improvement over raw RTOFS

- MAE improvement: `{raw['mae'] - best['mae']:.3f}` kJ/cm²
- RMSE improvement: `{raw['rmse'] - best['rmse']:.3f}` kJ/cm²

## Frozen prototype artifact

```json
{json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2)}
```
"""
    BENCHMARK_MD.write_text(md, encoding="utf-8")


def main() -> None:
    pairs_df = pd.read_csv(PAIRS_CSV)
    benchmark_df, predictions = _benchmark_models(pairs_df)

    out = pairs_df.copy()
    for name, pred in predictions.items():
        out[f"{name}_pred_tchp_kj_cm2"] = pred
        out[f"{name}_pred_delta_kj_cm2"] = out[f"{name}_pred_tchp_kj_cm2"] - out["model_tchp_kj_cm2"]
        out[f"{name}_abs_error_kj_cm2"] = (out[f"{name}_pred_tchp_kj_cm2"] - out["obs_tchp_kj_cm2"]).abs()
    out.to_csv(OOF_CSV, index=False)

    bundle = _freeze_best_model(pairs_df, benchmark_df)
    _write_report(benchmark_df, bundle)

    print(benchmark_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote OOF recomputed pairs: {OOF_CSV}")
    print(f"Wrote frozen model: {DEFAULT_MODEL_PATH}")
    print(f"Wrote benchmark report: {BENCHMARK_MD}")


if __name__ == "__main__":
    main()
