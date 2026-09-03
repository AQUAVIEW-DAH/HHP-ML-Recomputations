"""All model families from the lat/lon test, now on the full feature recipes.

Follow-up to run_latlon_only_models.py: does the learner ranking found on
(lat, lon) alone (RF and anisotropic GP best, XGBoost worst) survive when the
full 34/35-feature recipes are used? Same locked blocked-forward protocol and
the same warm-subset rows as the semi-ablation, so the XGBoost row reproduces
the known reference (TCHP 11.40, D26 10.76).

Models: XGBoost (locked params), random forest (as in the lat/lon run),
SVR-RBF (20k train subsample, standardized), GPR (3k subsample, RBF+White).
Output: OHC/output/latlon_only_20260903/full_features_model_families.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/latlon_only_20260903")
RECIPES = {"tchp": "global_pruned_plus_neighborhood",
           "d26": "drop_both_lat_interactions_plus_neighborhood"}
SEED = 42
SVR_MAX, GPR_MAX = 20000, 3000


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = abl._merge_feature_tables()
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    rows = []
    for target in TARGETS:
        cols = abl.FEATURE_SETS_BY_TARGET[target.name][RECIPES[target.name]]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col])
                  & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work)
        unique_dates = sorted(pd.Series(work["date"].dt.strftime("%Y%m%d").unique()).tolist())
        folds = _build_forward_folds(unique_dates, n_folds=fold_note["n_folds"],
                                     embargo_dates=fold_note["embargo_dates"])
        date_str = work["date"].dt.strftime("%Y%m%d")
        preds = {m: np.full(len(work), np.nan) for m in ("xgb", "rf", "svr_rbf", "gpr")}
        for fold in folds:
            tr = date_str.isin(set(fold["train_dates"])).to_numpy()
            va = date_str.isin(set(fold["val_dates"])).to_numpy()
            imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
            Xtr, Xva = imp.transform(work.loc[tr, cols]), imp.transform(work.loc[va, cols])
            ytr = work.loc[tr, target.delta_col].to_numpy(float)

            xgb = abl._xgb_model(); xgb.fit(Xtr, ytr)
            preds["xgb"][va] = xgb.predict(Xva)

            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=50, n_jobs=16, random_state=SEED)
            rf.fit(Xtr, ytr)
            preds["rf"][va] = rf.predict(Xva)

            sc = StandardScaler().fit(Xtr)
            sidx = rng.choice(len(ytr), size=min(SVR_MAX, len(ytr)), replace=False)
            svr = SVR(kernel="rbf", C=50.0, epsilon=1.0, gamma="scale", cache_size=1000)
            svr.fit(sc.transform(Xtr)[sidx], ytr[sidx])
            preds["svr_rbf"][va] = svr.predict(sc.transform(Xva))

            gidx = rng.choice(len(ytr), size=min(GPR_MAX, len(ytr)), replace=False)
            kernel = (ConstantKernel(10.0, (1e-2, 1e4)) * RBF(length_scale=3.0, length_scale_bounds=(0.1, 100.0))
                      + WhiteKernel(noise_level=50.0, noise_level_bounds=(1e-1, 1e4)))
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=SEED)
            gpr.fit(sc.transform(Xtr)[gidx], ytr[gidx])
            preds["gpr"][va] = gpr.predict(sc.transform(Xva))
            print(target.name, "fold", fold["fold"], "done", flush=True)

        y_obs = work[target.obs_col].to_numpy(float)
        y_mod = work[target.model_col].to_numpy(float)
        v = np.isfinite(preds["xgb"])
        rows.append({"target": target.name, "model": "raw RTOFS",
                     "mae": float(np.abs(y_mod - y_obs)[v].mean()), "rows": int(v.sum())})
        for m, p in preds.items():
            rows.append({"target": target.name, "model": f"{m} full features",
                         "mae": float(np.abs(y_mod + p - y_obs)[v].mean()), "rows": int(v.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "full_features_model_families.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
