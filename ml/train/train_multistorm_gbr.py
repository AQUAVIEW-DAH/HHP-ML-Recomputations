"""Train and benchmark simple HHP correction models across multiple storms.

This keeps the current Milton replay app stable while giving us a cleaner
multi-storm training/evaluation path. The default workflow:

1. Load pair tables for one or more storms.
2. Benchmark simple grouped-CV baselines on the combined table.
3. Run leave-one-storm-out evaluations to test event transfer.
4. Freeze a gradient-boosted correction model on all provided storms.
"""
from __future__ import annotations

import argparse
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
from ml.modeling import save_model_bundle
from ml.paths import BENCHMARK_REPORTS_DIR, DATASETS_DIR, MODELS_DIR


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


def _model_defs() -> dict[str, Pipeline | None]:
    pre = _numeric_preprocessor()
    return {
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


def _canonical_event_name(storm_name: str, storm_season: int | str) -> str:
    return f"{str(storm_name).upper()}_{int(storm_season)}"


def _load_event_table(event_name: str, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "storm_name" not in df.columns or "storm_season" not in df.columns:
        storm_name, storm_season = event_name.rsplit("_", 1)
        df["storm_name"] = storm_name
        df["storm_season"] = int(storm_season)
    df["event_name"] = [
        _canonical_event_name(name, season)
        for name, season in zip(df["storm_name"], df["storm_season"])
    ]
    return df


def _filter_complete_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "obs_tchp_kj_cm2",
        "model_tchp_kj_cm2",
        "tchp_delta_kj_cm2",
    ]
    keep = ~pairs_df[required].isna().any(axis=1)
    return pairs_df.loc[keep].copy()


def _combined_grouped_cv(pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    X = build_training_feature_frame(pairs_df)
    y_obs = pairs_df["obs_tchp_kj_cm2"]
    y_delta = pairs_df["tchp_delta_kj_cm2"]
    groups = pairs_df["platform"]
    n_splits = min(5, int(groups.nunique()))
    if n_splits < 2:
        raise RuntimeError("Need at least two unique platforms for grouped cross-validation")

    benchmark_rows: list[dict[str, float | str]] = []
    predictions: dict[str, list[float]] = {}

    raw_pred = pairs_df["model_tchp_kj_cm2"].to_numpy()
    predictions["raw_rtofs"] = raw_pred.tolist()
    benchmark_rows.append({
        "evaluation": "combined_grouped_cv",
        "model": "raw_rtofs",
        "mae": mean_absolute_error(y_obs, raw_pred),
        "rmse": _rmse(y_obs, raw_pred),
        "r2": r2_score(y_obs, raw_pred),
        "bias": float((raw_pred - y_obs).mean()),
        "rows": int(len(pairs_df)),
        "events": int(pairs_df["event_name"].nunique()),
    })

    cv = GroupKFold(n_splits=n_splits)
    for name, model in _model_defs().items():
        if name == "raw_rtofs":
            continue
        pred_delta = cross_val_predict(model, X, y_delta, cv=cv, groups=groups)
        pred_obs = raw_pred + pred_delta
        predictions[name] = pred_obs.tolist()
        benchmark_rows.append({
            "evaluation": "combined_grouped_cv",
            "model": name,
            "mae": mean_absolute_error(y_obs, pred_obs),
            "rmse": _rmse(y_obs, pred_obs),
            "r2": r2_score(y_obs, pred_obs),
            "bias": float((pred_obs - y_obs).mean()),
            "rows": int(len(pairs_df)),
            "events": int(pairs_df["event_name"].nunique()),
        })

    return pd.DataFrame(benchmark_rows).sort_values(["mae", "rmse"]).reset_index(drop=True), predictions


def _leave_one_storm_out(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for holdout_event in sorted(pairs_df["event_name"].unique()):
        train_df = pairs_df[pairs_df["event_name"] != holdout_event].copy()
        test_df = pairs_df[pairs_df["event_name"] == holdout_event].copy()
        if train_df.empty or test_df.empty:
            continue

        y_test = test_df["obs_tchp_kj_cm2"].to_numpy()
        raw_pred = test_df["model_tchp_kj_cm2"].to_numpy()
        rows.append({
            "evaluation": "leave_one_storm_out",
            "holdout_event": holdout_event,
            "train_events": ",".join(sorted(train_df["event_name"].unique())),
            "model": "raw_rtofs",
            "mae": mean_absolute_error(y_test, raw_pred),
            "rmse": _rmse(y_test, raw_pred),
            "r2": r2_score(y_test, raw_pred),
            "bias": float((raw_pred - y_test).mean()),
            "rows": int(len(test_df)),
        })

        X_train = build_training_feature_frame(train_df)
        y_train = train_df["tchp_delta_kj_cm2"].to_numpy()
        X_test = build_training_feature_frame(test_df)

        for name, model in _model_defs().items():
            if name == "raw_rtofs":
                continue
            model.fit(X_train, y_train)
            pred_obs = raw_pred + model.predict(X_test)
            rows.append({
                "evaluation": "leave_one_storm_out",
                "holdout_event": holdout_event,
                "train_events": ",".join(sorted(train_df["event_name"].unique())),
                "model": name,
                "mae": mean_absolute_error(y_test, pred_obs),
                "rmse": _rmse(y_test, pred_obs),
                "r2": r2_score(y_test, pred_obs),
                "bias": float((pred_obs - y_test).mean()),
                "rows": int(len(test_df)),
            })

    return pd.DataFrame(rows).sort_values(["holdout_event", "mae", "rmse"]).reset_index(drop=True)


def _freeze_full_model(pairs_df: pd.DataFrame, model_name: str, model_path: Path) -> dict:
    model = _model_defs()[model_name]
    if model is None:
        raise ValueError(f"Cannot freeze baseline without fitted model: {model_name}")
    X = build_training_feature_frame(pairs_df)
    y = pairs_df["tchp_delta_kj_cm2"].to_numpy()
    model.fit(X, y)
    bundle = {
        "model_name": model_name,
        "target_name": "tchp_delta_kj_cm2",
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(pairs_df)),
        "train_platforms": int(pairs_df["platform"].nunique()),
        "train_events": sorted(pairs_df["event_name"].unique().tolist()),
        "train_date_min": str(pairs_df["obs_time"].min()),
        "train_date_max": str(pairs_df["obs_time"].max()),
        "model": model,
    }
    save_model_bundle(bundle, model_path)
    return bundle


def _write_summary(
    *,
    combined_df: pd.DataFrame,
    loso_df: pd.DataFrame,
    bundle: dict,
    report_md: Path,
) -> None:
    best_combined = combined_df.iloc[0]
    raw_combined = combined_df[combined_df["model"] == "raw_rtofs"].iloc[0]

    holdout_sections: list[str] = []
    for event in sorted(loso_df["holdout_event"].dropna().unique()):
        event_df = loso_df[loso_df["holdout_event"] == event]
        best = event_df.iloc[0]
        raw = event_df[event_df["model"] == "raw_rtofs"].iloc[0]
        holdout_sections.append(
            "\n".join([
                f"### Holdout: `{event}`",
                f"- Best model: `{best['model']}`",
                f"- Best MAE: `{best['mae']:.3f}` kJ/cm²",
                f"- Raw RTOFS MAE: `{raw['mae']:.3f}` kJ/cm²",
                f"- MAE improvement: `{raw['mae'] - best['mae']:.3f}` kJ/cm²",
                f"- Best RMSE: `{best['rmse']:.3f}` kJ/cm²",
                f"- Raw RTOFS RMSE: `{raw['rmse']:.3f}` kJ/cm²",
                f"- RMSE improvement: `{raw['rmse'] - best['rmse']:.3f}` kJ/cm²",
            ])
        )

    md = f"""# Multi-Storm HHP Benchmark

This report benchmarks simple correction baselines across the supplied storm
pair tables and then runs leave-one-storm-out tests to estimate event transfer.

## Combined grouped cross-validation

- Best model: `{best_combined['model']}`
- MAE: `{best_combined['mae']:.3f}` kJ/cm²
- RMSE: `{best_combined['rmse']:.3f}` kJ/cm²
- R²: `{best_combined['r2']:.3f}`
- Bias: `{best_combined['bias']:.3f}` kJ/cm²

## Combined raw RTOFS baseline

- MAE: `{raw_combined['mae']:.3f}` kJ/cm²
- RMSE: `{raw_combined['rmse']:.3f}` kJ/cm²
- R²: `{raw_combined['r2']:.3f}`
- Bias: `{raw_combined['bias']:.3f}` kJ/cm²

## Combined improvement over raw RTOFS

- MAE improvement: `{raw_combined['mae'] - best_combined['mae']:.3f}` kJ/cm²
- RMSE improvement: `{raw_combined['rmse'] - best_combined['rmse']:.3f}` kJ/cm²

## Leave-One-Storm-Out

{chr(10).join(holdout_sections)}

## Frozen artifact

```json
{json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2)}
```
"""
    report_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        action="append",
        metavar="NAME=CSV",
        help="Event mapping like MILTON_2024=/path/to/pairs.csv. Repeat for multiple events.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="multistorm_hhp",
        help="Prefix for model and report artifact filenames.",
    )
    parser.add_argument(
        "--freeze-model",
        choices=["gbr_delta", "rf_delta", "ridge_delta"],
        default="gbr_delta",
        help="Which trained delta model to freeze as the deployable artifact.",
    )
    args = parser.parse_args()

    event_args = args.event or [
        f"MILTON_2024={DATASETS_DIR / 'milton_pairs.csv'}",
        f"HELENE_2024={DATASETS_DIR / 'helene_2024_pairs.csv'}",
    ]

    frames: list[pd.DataFrame] = []
    for mapping in event_args:
        event_name, csv_value = mapping.split("=", 1)
        frames.append(_load_event_table(event_name.strip(), Path(csv_value).expanduser().resolve()))

    combined_pairs = pd.concat(frames, ignore_index=True)
    combined_pairs = combined_pairs.sort_values(["event_name", "obs_time"]).reset_index(drop=True)
    combined_pairs = _filter_complete_pairs(combined_pairs)

    combined_df, predictions = _combined_grouped_cv(combined_pairs)
    loso_df = _leave_one_storm_out(combined_pairs)

    oof_out = combined_pairs.copy()
    for name, pred in predictions.items():
        oof_out[f"{name}_pred_tchp_kj_cm2"] = pred
        oof_out[f"{name}_pred_delta_kj_cm2"] = oof_out[f"{name}_pred_tchp_kj_cm2"] - oof_out["model_tchp_kj_cm2"]
        oof_out[f"{name}_abs_error_kj_cm2"] = (oof_out[f"{name}_pred_tchp_kj_cm2"] - oof_out["obs_tchp_kj_cm2"]).abs()

    model_path = MODELS_DIR / f"{args.artifact_prefix}_{args.freeze_model}_model.pkl"
    combined_csv = BENCHMARK_REPORTS_DIR / f"{args.artifact_prefix}_combined_grouped_cv.csv"
    loso_csv = BENCHMARK_REPORTS_DIR / f"{args.artifact_prefix}_leave_one_storm_out.csv"
    oof_csv = DATASETS_DIR / f"{args.artifact_prefix}_pairs_with_ml_recomputed.csv"
    report_md = BENCHMARK_REPORTS_DIR / f"{args.artifact_prefix}_benchmark.md"

    bundle = _freeze_full_model(combined_pairs, args.freeze_model, model_path)
    oof_out.to_csv(oof_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)
    loso_df.to_csv(loso_csv, index=False)
    _write_summary(combined_df=combined_df, loso_df=loso_df, bundle=bundle, report_md=report_md)

    print("Combined grouped CV:")
    print(combined_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nLeave-one-storm-out:")
    print(loso_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote combined OOF table: {oof_csv}")
    print(f"Wrote frozen model: {model_path}")
    print(f"Wrote benchmark report: {report_md}")


if __name__ == "__main__":
    main()
