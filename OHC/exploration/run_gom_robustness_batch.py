"""Robustness batch (meeting items B1-B3): do the Gulf conclusions survive a
different model family, and does a leaner, deeper model sharpen them?

All variants use the same blocked-forward Gulf folds and residual target as the
attribution analysis, evaluated out-of-fold on the observed scale:

B1  rf_full        random forest, full recipe             (different algorithm)
    rf_reduced     random forest, elimination-optimal set
B2  xgb_reduced    boosted trees, reduced set, sweep over depth {4,6,8}
                   x estimators {300,800}  ("how finely can it split")
    + histogram of the SSH split thresholds used by the best deep model
B3  mlp_reduced    small neural network (2x64, standardised inputs)

Also reports the random forest's permutation importances so the feature
ranking can be compared with the boosted model's SHAP ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    _make_preprocessor,
    _merge_feature_tables,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom, _short_name  # noqa: E402

RUN_DATE = "2026-08-11"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/gom_robustness_{RUN_DATE.replace('-', '')}")
LADDER_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/gom_attribution_20260722")
SWEEP = [(4, 300), (6, 300), (8, 300), (4, 800), (6, 800), (8, 800)]


def _reduced_features(tname: str, full_cols: list[str]) -> list[str]:
    """Feature set at the backward-elimination optimum (min OOF MAE step)."""
    ladder = pd.read_csv(LADDER_DIR / f"{tname}_gom_backward_elimination.csv")
    k = int(ladder["mae"].idxmin())
    removed = set(ladder.loc[1:k, "removed"])
    return [c for c in full_cols if c not in removed]


def _oof_run(gom, folds, cols, target, fit_predict) -> np.ndarray:
    """Generic OOF loop; fit_predict(train_X, train_y, val_X) -> predictions."""
    date_gom = gom["date"].dt.strftime("%Y%m%d")
    oof = np.full(len(gom), np.nan)
    for fold in folds:
        tr = gom[date_gom.isin(set(fold["train_dates"]))]
        va_idx = np.where(date_gom.isin(set(fold["val_dates"])).to_numpy())[0]
        if tr.empty or va_idx.size == 0:
            continue
        pre = _make_preprocessor(cols)
        X_tr = pre.fit_transform(tr[cols])
        X_va = pre.transform(gom.iloc[va_idx][cols])
        y_tr = tr[target.delta_col].to_numpy(float)
        oof[va_idx] = gom.iloc[va_idx][target.model_col].to_numpy(float) + fit_predict(X_tr, y_tr, X_va)
    return oof


def _metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    e = p[ok] - y[ok]
    return {"rows": int(ok.sum()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}


def _rf_factory():
    def fit_predict(X_tr, y_tr, X_va):
        m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=16, random_state=0)
        m.fit(X_tr, y_tr)
        fit_predict.last_model = m
        return m.predict(X_va)
    return fit_predict


def _xgb_factory(depth, n_est):
    def fit_predict(X_tr, y_tr, X_va):
        m = XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         objective="reg:squarederror", random_state=0, n_jobs=16)
        m.fit(X_tr, y_tr)
        fit_predict.last_model = m
        return m.predict(X_va)
    return fit_predict


def _mlp_factory():
    def fit_predict(X_tr, y_tr, X_va):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        m = MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True,
                         validation_fraction=0.15, max_iter=2000, random_state=0)
        m.fit(X_tr_s, y_tr)
        return m.predict(sc.transform(X_va))
    return fit_predict


def _ssh_split_hist(model: XGBRegressor, cols: list[str], tname: str, out_path: Path) -> int:
    j = cols.index("model_ssh_m")
    trees = model.get_booster().trees_to_dataframe()
    ssh_splits = trees[trees["Feature"] == f"f{j}"]["Split"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    ax.hist(ssh_splits, bins=45, color="#2563eb")
    ax.set_xlabel("SSH split threshold (m)")
    ax.set_ylabel("number of tree splits")
    ax.set_title(
        f"{SHORT[tname]}: where the deep reduced-feature model splits on SSH\n"
        f"{len(ssh_splits)} splits — finer spacing = finer resolution of the SSH response"
    )
    ax.grid(True, alpha=0.15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return int(len(ssh_splits))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    results, importances = [], []

    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        gom = work[_in_gom(work)].copy().reset_index(drop=True)
        y = gom[target.obs_col].to_numpy(float)
        full_cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        red_cols = _reduced_features(tname, full_cols)
        folds = _build_forward_folds(
            sorted(gom["date"].dt.strftime("%Y%m%d").unique().tolist()),
            n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"],
        )
        print(f"{tname}: reduced set ({len(red_cols)}): {red_cols}")

        # B1: random forest, full and reduced
        for label, cols in [("rf_full", full_cols), ("rf_reduced", red_cols)]:
            fp = _rf_factory()
            oof = _oof_run(gom, folds, cols, target, fp)
            results.append({"target": tname, "model": label, "features": len(cols), **_metrics(y, oof)})
            r = permutation_importance(fp.last_model, _make_preprocessor(cols).fit_transform(gom[cols]),
                                       gom[target.delta_col].to_numpy(float), n_repeats=5, random_state=0, n_jobs=16)
            for c, imp in zip(cols, r.importances_mean):
                importances.append({"target": tname, "model": label, "feature": c, "perm_importance": float(imp)})

        # B2: capacity sweep on the reduced set
        best = (None, np.inf, None)
        for depth, n_est in SWEEP:
            fp = _xgb_factory(depth, n_est)
            oof = _oof_run(gom, folds, red_cols, target, fp)
            m = _metrics(y, oof)
            results.append({"target": tname, "model": f"xgb_reduced_d{depth}_n{n_est}", "features": len(red_cols), **m})
            if m["mae"] < best[1]:
                best = (f"d{depth}_n{n_est}", m["mae"], fp.last_model)
        n_splits = _ssh_split_hist(best[2], red_cols, tname, OUT_DIR / f"{tname}_ssh_split_thresholds.png")
        print(f"{tname}: best sweep {best[0]} (MAE {best[1]:.2f}), SSH splits in final model: {n_splits}")

        # B3: small neural network on the reduced set
        oof = _oof_run(gom, folds, red_cols, target, _mlp_factory())
        results.append({"target": tname, "model": "mlp_reduced_64x64", "features": len(red_cols), **_metrics(y, oof)})

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "gom_robustness_summary.csv", index=False)
    imp = pd.DataFrame(importances)
    imp.to_csv(OUT_DIR / "rf_permutation_importance.csv", index=False)
    print(res.to_string(index=False))
    for tname in ("tchp", "d26"):
        top = imp[(imp.target == tname) & (imp.model == "rf_full")].nlargest(5, "perm_importance")
        print(f"\n{tname} RF top-5:", [_short_name(c) for c in top["feature"]])


if __name__ == "__main__":
    main()
