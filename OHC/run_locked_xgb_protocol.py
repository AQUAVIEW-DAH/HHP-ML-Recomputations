"""Run the locked-protocol XGBoost rerun for the RTOFS->Argo correction task."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.benchmark_rtofs_argo_tabular_models import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGETS,
    _build_forward_folds,
    _prepare_features,
    _tree_preprocessor,
)


OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")
FOLD_PATH = OUT_DIR / "tabular_benchmark_folds.json"
FULL_FOLD_PATH = OUT_DIR / "locked_protocol_full_folds.json"


def _rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def _xgb_model():
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


def run_target(df: pd.DataFrame, target) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)
    fold_note = json.loads(FOLD_PATH.read_text())
    unique_dates = sorted(pd.Series(df["date"].astype(str).unique()).tolist())
    full_folds = _build_forward_folds(
        unique_dates,
        n_folds=fold_note["n_folds"],
        embargo_dates=fold_note["embargo_dates"],
    )
    FULL_FOLD_PATH.write_text(json.dumps({"folds": full_folds}, indent=2))

    metrics: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    pre = _tree_preprocessor()

    for fold in full_folds:
        train_dates = set(fold["train_dates"])
        val_dates = set(fold["val_dates"])
        date_str = work["date"].dt.strftime("%Y%m%d")
        train_df = work[date_str.isin(train_dates)].copy()
        val_df = work[date_str.isin(val_dates)].copy()
        if train_df.empty or val_df.empty:
            continue

        x_train = pre.fit_transform(train_df[FEATURE_COLUMNS])
        x_val = pre.transform(val_df[FEATURE_COLUMNS])
        y_train_delta = train_df[target.delta_col].to_numpy(float)
        y_val_obs = val_df[target.obs_col].to_numpy(float)
        raw_val = val_df[target.model_col].to_numpy(float)

        raw_pred = raw_val.copy()
        metrics.append(
            _metric_row(
                target=target.name,
                model="raw_rtofs",
                evaluation=f"fold_{fold['fold']}",
                y_true=y_val_obs,
                y_pred=raw_pred,
                rows=len(val_df),
                dates=val_df["date"].nunique(),
            )
        )

        model = _xgb_model()
        model.fit(x_train, y_train_delta)
        xgb_pred = raw_val + model.predict(x_val)
        metrics.append(
            _metric_row(
                target=target.name,
                model="xgb_delta_locked",
                evaluation=f"fold_{fold['fold']}",
                y_true=y_val_obs,
                y_pred=xgb_pred,
                rows=len(val_df),
                dates=val_df["date"].nunique(),
            )
        )

        pred_df = val_df[
            ["date", "year", "month", "lat", "lon", target.obs_col, target.model_col, target.delta_col]
        ].copy()
        pred_df["target"] = target.name
        pred_df["fold"] = fold["fold"]
        pred_df["pred_obs__raw_rtofs"] = raw_pred
        pred_df["pred_obs__xgb_delta_locked"] = xgb_pred
        pred_frames.append(pred_df)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    y_true = pred_all[target.obs_col].to_numpy(float)
    metrics.append(
        _metric_row(
            target=target.name,
            model="raw_rtofs",
            evaluation="oof_summary",
            y_true=y_true,
            y_pred=pred_all["pred_obs__raw_rtofs"].to_numpy(float),
            rows=len(pred_all),
            dates=pred_all["date"].nunique(),
        )
    )
    metrics.append(
        _metric_row(
            target=target.name,
            model="xgb_delta_locked",
            evaluation="oof_summary",
            y_true=y_true,
            y_pred=pred_all["pred_obs__xgb_delta_locked"].to_numpy(float),
            rows=len(pred_all),
            dates=pred_all["date"].nunique(),
        )
    )
    return pd.DataFrame(metrics), pred_all


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    all_metrics = []
    all_preds = []
    for target in TARGETS:
        metrics_df, preds_df = run_target(df, target)
        metrics_df.to_csv(OUT_DIR / f"locked_xgb_metrics_{target.name}.csv", index=False)
        preds_df.to_parquet(OUT_DIR / f"locked_xgb_oof_predictions_{target.name}.parquet", index=False)
        all_metrics.append(metrics_df)
        all_preds.append(preds_df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "locked_xgb_metrics_all.csv", index=False)
    summary = metrics[metrics["evaluation"] == "oof_summary"].sort_values(["target", "mae", "rmse"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "locked_xgb_oof_summary.csv", index=False)

    payload = {
        "data_path": str(DATA_PATH),
        "fold_path": str(FOLD_PATH),
        "summary_csv": str(OUT_DIR / "locked_xgb_oof_summary.csv"),
        "note_path": str(OUT_DIR / "locked_protocol_note.md"),
    }
    (OUT_DIR / "locked_xgb_run_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("\nLocked OOF summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
