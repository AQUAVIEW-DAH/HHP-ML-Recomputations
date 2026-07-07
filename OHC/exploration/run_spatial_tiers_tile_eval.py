"""SIDE EXPLORATION harness: tile-holdout evaluation of the spatial-feature tiers.

Evaluates, on identical 15-degree-tile GroupKFold folds (the memorization
detector from SPATIAL_FEATURES_DIRECTIONS.md), for each target:

- raw_rtofs                      : no correction baseline
- xgb_best                       : current recommended recipe
- xgb_best_plus_neighborhood     : Tier 0 (RTOFS-only stencils)
- xgb_best_plus_nbhd_plus_woa    : Tier 1 (adds WOA23 climatology priors; skipped
                                   if the WOA table has not been built)
- gpboost_state_gp               : Tier 2 (trees on state features without raw
                                   coordinates + Matern GP over lat/lon, Vecchia)

Training is on the residual (delta) target with predictions reconstructed to
the observed scale, mirroring run_locked_xgb_physics_semi_ablation.py. The
tile split means every validation row sits in a 15-degree tile never seen in
training, so geography lookup cannot help.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    BEST_PARAMS,
    FEATURE_SETS_BY_TARGET,
    NEIGHBORHOOD_CORE,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)

WOA_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_woa_clim.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/spatial_tiers")
TILE_DEG = 15
N_SPLITS = 5
COORD_COLS = {"lat", "lon", "abs_lat"}
WOA_FEATURES = [
    "woa_tchp_clim_kj_per_cm2",
    "woa_d26_clim_m",
    "woa_sst_clim_c",
    "model_minus_woa_tchp",
    "model_minus_woa_d26",
]
BEST_RECIPE = {"tchp": "global_pruned", "d26": "drop_both_lat_interactions"}


def _tile_id(lat: float, lon: float) -> str:
    return f"tile_{int(np.floor((lat + 90.0) / TILE_DEG)):02d}_{int(np.floor((lon + 180.0) / TILE_DEG)):02d}"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "rows": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))),
    }


def _fit_predict_xgb(train_df, val_df, feature_cols, delta_col, model_col):
    pre = _make_preprocessor(feature_cols)
    X_tr = pre.fit_transform(train_df[feature_cols])
    X_va = pre.transform(val_df[feature_cols])
    model = _xgb_model()
    model.fit(X_tr, train_df[delta_col].to_numpy(float))
    return val_df[model_col].to_numpy(float) + model.predict(X_va)


def _fit_predict_gpboost(train_df, val_df, feature_cols, delta_col, model_col):
    import gpboost as gpb

    state_cols = [c for c in feature_cols if c not in COORD_COLS]
    pre = _make_preprocessor(state_cols)
    X_tr = pre.fit_transform(train_df[state_cols])
    X_va = pre.transform(val_df[state_cols])
    gp_model = gpb.GPModel(
        gp_coords=train_df[["lat", "lon"]].to_numpy(float),
        cov_function="matern",
        cov_fct_shape=1.5,
        gp_approx="vecchia",
        num_neighbors=30,
        likelihood="gaussian",
    )
    data = gpb.Dataset(X_tr, train_df[delta_col].to_numpy(float))
    params = {
        "objective": "regression",
        "learning_rate": BEST_PARAMS["learning_rate"],
        "max_depth": BEST_PARAMS["max_depth"],
        "bagging_fraction": BEST_PARAMS["subsample"],
        "feature_fraction": BEST_PARAMS["colsample_bytree"],
        "lambda_l2": BEST_PARAMS["reg_lambda"],
        "verbose": -1,
    }
    bst = gpb.train(params=params, train_set=data, gp_model=gp_model, num_boost_round=BEST_PARAMS["n_estimators"])
    pred = bst.predict(
        data=X_va,
        gp_coords_pred=val_df[["lat", "lon"]].to_numpy(float),
        predict_var=False,
        pred_latent=False,
    )
    return val_df[model_col].to_numpy(float) + np.asarray(pred["response_mean"], dtype=float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()

    woa_available = WOA_PATH.exists()
    if woa_available:
        woa = pd.read_parquet(WOA_PATH).reset_index(drop=True)
        if len(woa) != len(df):
            raise RuntimeError("WOA table is not aligned with the merged base table")
        for col in WOA_FEATURES:
            df[col] = woa[col].to_numpy()

    rows: list[dict] = []
    for target in TARGETS:
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work)
        work["tile_group"] = [_tile_id(la, lo) for la, lo in zip(work["lat"], work["lon"])]

        best_cols = FEATURE_SETS_BY_TARGET[target.name][BEST_RECIPE[target.name]]
        variants: dict[str, tuple[str, list[str]]] = {
            "xgb_best": ("xgb", best_cols),
            "xgb_best_plus_neighborhood": ("xgb", best_cols + NEIGHBORHOOD_CORE),
            "gpboost_state_gp": ("gpboost", best_cols + NEIGHBORHOOD_CORE),
        }
        if woa_available:
            variants["xgb_best_plus_nbhd_plus_woa"] = ("xgb", best_cols + NEIGHBORHOOD_CORE + WOA_FEATURES)

        gkf = GroupKFold(n_splits=N_SPLITS)
        groups = work["tile_group"].to_numpy()
        oof: dict[str, np.ndarray] = {name: np.full(len(work), np.nan) for name in variants}
        for tr_idx, va_idx in gkf.split(work, groups=groups):
            train_df, val_df = work.iloc[tr_idx], work.iloc[va_idx]
            for name, (kind, cols) in variants.items():
                cols = [c for c in cols if c in work.columns]
                if kind == "xgb":
                    pred = _fit_predict_xgb(train_df, val_df, cols, target.delta_col, target.model_col)
                else:
                    pred = _fit_predict_gpboost(train_df, val_df, cols, target.delta_col, target.model_col)
                oof[name][va_idx] = pred

        y_true = work[target.obs_col].to_numpy(float)
        rows.append({
            "target": target.name, "model": "raw_rtofs", "evaluation": f"tile{TILE_DEG}_group_oof",
            **_metrics(y_true, work[target.model_col].to_numpy(float)),
        })
        for name in variants:
            mask = np.isfinite(oof[name])
            rows.append({
                "target": target.name, "model": name, "evaluation": f"tile{TILE_DEG}_group_oof",
                **_metrics(y_true[mask], oof[name][mask]),
            })
        print(f"{target.name}: done ({work['tile_group'].nunique()} tiles, {len(work)} rows)")

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "spatial_tiers_tile_oof_summary.csv"
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
