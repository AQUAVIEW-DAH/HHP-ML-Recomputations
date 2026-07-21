"""Mentor experiment: does a Gulf-of-Mexico-only model beat the global model there?

Trains, per target, the best+neighborhood recipe two ways under the same
blocked-forward date protocol, evaluating both only on Gulf of Mexico rows
(lat 18..31, lon -98..-80):

- ML_global: trained on all rows (evaluation subset taken from the standard
  locked OOF predictions, so it is exactly the production model)
- ML_gom:    trained only on Gulf rows from the same training dates

If ML_gom wins, the per-fold feature-importance comparison hints at what the
global model fails to exploit locally.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/gom_local_vs_global")
PRED_PATHS = {
    "tchp": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet"),
    "d26": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet"),
}
RECIPE = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
PRED_COL = {"tchp": "pred_obs__global_pruned_plus_neighborhood", "d26": "pred_obs__drop_both_lat_interactions_plus_neighborhood"}
GOM = {"lat_min": 18.0, "lat_max": 31.0, "lon_min": -98.0, "lon_max": -80.0}


def _in_gom(df: pd.DataFrame) -> pd.Series:
    return (
        (df["lat"] >= GOM["lat_min"]) & (df["lat"] <= GOM["lat_max"])
        & (df["lon"] >= GOM["lon_min"]) & (df["lon"] <= GOM["lon_max"])
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    err = y_pred[ok] - y_true[ok]
    return {
        "rows": int(ok.sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "corr": float(pd.Series(y_true[ok]).corr(pd.Series(y_pred[ok]))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    results = []
    importances = []

    for target in TARGETS:
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        gom_mask = _in_gom(work)
        gom = work[gom_mask].copy()
        cols = FEATURE_SETS_BY_TARGET[target.name][RECIPE[target.name]]
        cols = [c for c in cols if c in work.columns]

        # ML_global evaluated in the Gulf: reuse the standard locked OOF predictions.
        pred = pd.read_parquet(PRED_PATHS[target.name])
        pred_gom = pred[_in_gom(pred) & np.isfinite(pred[PRED_COL[target.name]])]
        results.append({
            "target": target.name, "model": "raw_rtofs_in_gom",
            **_metrics(pred_gom[target.obs_col].to_numpy(float), pred_gom["pred_obs__raw_rtofs"].to_numpy(float)),
        })
        results.append({
            "target": target.name, "model": "ml_global_in_gom",
            **_metrics(pred_gom[target.obs_col].to_numpy(float), pred_gom[PRED_COL[target.name]].to_numpy(float)),
        })

        # ML_gom: same blocked-forward protocol on Gulf rows only.
        gom_dates = sorted(gom["date"].dt.strftime("%Y%m%d").unique().tolist())
        folds = _build_forward_folds(gom_dates, n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        oof = np.full(len(gom), np.nan)
        gom_date_str = gom["date"].dt.strftime("%Y%m%d")
        for fold in folds:
            tr = gom[gom_date_str.isin(set(fold["train_dates"]))]
            va_idx = np.where(gom_date_str.isin(set(fold["val_dates"])).to_numpy())[0]
            if tr.empty or va_idx.size == 0:
                continue
            pre = _make_preprocessor(cols)
            X_tr = pre.fit_transform(tr[cols])
            X_va = pre.transform(gom.iloc[va_idx][cols])
            model = _xgb_model()
            model.fit(X_tr, tr[target.delta_col].to_numpy(float))
            oof[va_idx] = gom.iloc[va_idx][target.model_col].to_numpy(float) + model.predict(X_va)
            gains = model.feature_importances_
            for c, g in zip(cols, gains):
                importances.append({"target": target.name, "feature": c, "gain": float(g)})
        ok = np.isfinite(oof)
        results.append({
            "target": target.name, "model": "ml_gom_local",
            **_metrics(gom[target.obs_col].to_numpy(float)[ok], oof[ok]),
        })
        print(f"{target.name}: gom rows={len(gom)}, dates={len(gom_dates)}")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "gom_local_vs_global_summary.csv", index=False)
    imp = pd.DataFrame(importances).groupby(["target", "feature"], as_index=False)["gain"].mean()
    imp.sort_values(["target", "gain"], ascending=[True, False]).to_csv(OUT_DIR / "gom_local_feature_importance.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
