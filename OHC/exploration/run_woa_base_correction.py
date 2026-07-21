"""Mentor curiosity: can ML-corrected WOA climatology rival ML-corrected RTOFS?

Under the same blocked-forward date protocol, on identical rows (valid Argo
observation + valid WOA climatology value + valid RTOFS value), compare:

- raw WOA climatology vs Argo
- raw RTOFS vs Argo
- ML-corrected WOA (inputs: calendar, location, and the WOA climatology
  values only; no RTOFS information at all)
- ML-corrected RTOFS (the standard best+neighborhood recipe)

If corrected WOA comes close to corrected RTOFS, the dynamical model adds
little beyond climatology at these points; a wide gap means RTOFS's day-to-day
state information genuinely matters.
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
    BASE_FEATURES,
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    NEIGHBORHOOD_CORE,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)

WOA_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_woa_clim.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/woa_base_correction")
WOA_CLIM_COL = {"tchp": "woa_tchp_clim_kj_per_cm2", "d26": "woa_d26_clim_m"}
WOA_FEATURES = ["woa_tchp_clim_kj_per_cm2", "woa_d26_clim_m", "woa_sst_clim_c"]
RTOFS_RECIPE = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
CALENDAR_LOCATION = [
    c for c in BASE_FEATURES
    if c not in ("model_interp_tchp_kj_per_cm2", "model_interp_d26_m", "nearest_rtofs_grid_distance_km")
]


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


def _oof(work: pd.DataFrame, folds: list[dict], cols: list[str], y_delta: np.ndarray, base_pred: np.ndarray, target_delta_name: str) -> np.ndarray:
    oof = np.full(len(work), np.nan)
    date_str = work["date"].dt.strftime("%Y%m%d")
    for fold in folds:
        tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
        va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
        if not tr_mask.any() or not va_mask.any():
            continue
        pre = _make_preprocessor(cols)
        X_tr = pre.fit_transform(work.loc[tr_mask, cols])
        X_va = pre.transform(work.loc[va_mask, cols])
        model = _xgb_model()
        model.fit(X_tr, y_delta[tr_mask])
        oof[va_mask] = base_pred[va_mask] + model.predict(X_va)
    return oof


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    woa = pd.read_parquet(WOA_PATH).reset_index(drop=True)
    if len(woa) != len(df):
        raise RuntimeError("WOA table not aligned with merged base table")
    for col in WOA_FEATURES:
        df[col] = woa[col].to_numpy()

    fold_note = json.loads(FOLD_PATH.read_text())
    results = []
    for target in TARGETS:
        clim_col = WOA_CLIM_COL[target.name]
        work = df[
            pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])
            & np.isfinite(pd.to_numeric(df[clim_col], errors="coerce"))
        ].copy()
        work = _prepare_features(work).reset_index(drop=True)
        y = work[target.obs_col].to_numpy(float)
        rtofs_raw = work[target.model_col].to_numpy(float)
        woa_raw = work[clim_col].to_numpy(float)

        dates = sorted(work["date"].dt.strftime("%Y%m%d").unique().tolist())
        folds = _build_forward_folds(dates, n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

        rtofs_cols = [c for c in FEATURE_SETS_BY_TARGET[target.name][RTOFS_RECIPE[target.name]] if c in work.columns]
        woa_cols = CALENDAR_LOCATION + WOA_FEATURES

        oof_rtofs = _oof(work, folds, rtofs_cols, work[target.delta_col].to_numpy(float), rtofs_raw, "rtofs")
        oof_woa = _oof(work, folds, woa_cols, y - woa_raw, woa_raw, "woa")

        results.append({"target": target.name, "model": "raw_woa_climatology", **_metrics(y, woa_raw)})
        results.append({"target": target.name, "model": "raw_rtofs", **_metrics(y, rtofs_raw)})
        results.append({"target": target.name, "model": "corrected_woa", **_metrics(y, oof_woa)})
        results.append({"target": target.name, "model": "corrected_rtofs", **_metrics(y, oof_rtofs)})
        print(f"{target.name}: {len(work)} rows, {len(dates)} dates")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "woa_vs_rtofs_corrected_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
