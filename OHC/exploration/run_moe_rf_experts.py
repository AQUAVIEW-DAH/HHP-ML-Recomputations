"""MoE blend with random-forest experts instead of XGBoost (comparison only).

Same architecture and winning configuration as the recommended model
(TCHP: alpha=0.75, K=6, w=0.05; D26: alpha=0.5, K=12, w=0.05), with every
expert swapped from the locked XGBoost to a random forest (300 trees,
min_samples_leaf=50 — the config used in the model-families comparison).
Diagnostic only; not part of the recommendation or the Drive bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.exploration.run_moe_v2_tuning as v2  # noqa: E402
import OHC.exploration.run_moe_regions as mr  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, _in_gom  # noqa: E402
from OHC.exploration.run_moe_regions import _metrics, _region_of  # noqa: E402

WINNING = {"tchp": {"alpha": 0.75, "k": 6, "w": 0.05},
           "d26": {"alpha": 0.5, "k": 12, "w": 0.05}}
XGB_REF = {"tchp": {"global": 11.189, "gulf": 12.41}, "d26": {"global": 10.553, "gulf": 11.52}}


def _fit_expert_rf(train_df, cols, delta_col, weights):
    pre = _make_preprocessor(cols)
    X = pre.fit_transform(train_df[cols])
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=50, n_jobs=16, random_state=42)
    model.fit(X, train_df[delta_col].to_numpy(float), sample_weight=weights)
    return pre, model


v2._fit_expert = _fit_expert_rf
mr._fit_expert = _fit_expert_rf


def main() -> None:
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    for target in TARGETS:
        tname = target.name
        cfg = WINNING[tname]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col])
                  & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        y = work[target.obs_col].to_numpy(float)
        gm = _in_gom(work).to_numpy()
        folds = _build_forward_folds(sorted(work["date"].dt.strftime("%Y%m%d").unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        geo = v2._run_geographic(work, target, cols, folds, cfg["w"])
        print(tname, "geo branch done", flush=True)
        reg = v2._run_regime(work, target, cols, folds, cfg["w"], cfg["k"])
        print(tname, "regime branch done", flush=True)
        blend = cfg["alpha"] * geo + (1 - cfg["alpha"]) * reg
        for name, oof in (("geo (RF experts)", geo), ("regime (RF experts)", reg), ("blend (RF experts)", blend)):
            g, gg = _metrics(y, oof), _metrics(y[gm], oof[gm])
            print(f"{tname} {name}: global MAE {g['mae']:.3f} (bias {g['bias']:+.2f}) | gulf {gg['mae']:.3f}", flush=True)
        print(f"{tname} XGB reference blend: global {XGB_REF[tname]['global']} | gulf {XGB_REF[tname]['gulf']}", flush=True)


if __name__ == "__main__":
    main()
