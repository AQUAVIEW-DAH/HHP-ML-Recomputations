"""Mixture-of-experts v2 tuning (follow-up to run_moe_regions.py).

Axes, all on the locked blocked-forward folds and the best+neighborhood recipe:

1. Global-prior weight sweep for the geographic experts:
   w in {0.05, 0.10, 0.15, 0.25}. Reported globally and per region, so a
   per-region prior can be assembled (flagged as selection-biased, since the
   per-region choice is made on the same OOF rows it is scored on).
2. Regime-count sweep for the learned-regime experts: K in {4, 6, 8, 12}
   at the best prior from axis 1.
3. Combined gate: convex blends of the best geographic-MoE and best
   regime-MoE out-of-fold predictions, alpha in {0.25, 0.5, 0.75}.

Outputs: per-config summary CSV, sweep figures, and a final comparison table
against the v1 numbers and the global baseline.
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    _merge_feature_tables,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom  # noqa: E402
from OHC.exploration.run_moe_regions import (  # noqa: E402
    GATE_TEMPERATURE,
    REGIME_FEATURES,
    _fit_expert,
    _metrics,
    _predict,
    _region_of,
)

RUN_DATE = "2026-08-11"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/moe_v2_tuning_{RUN_DATE.replace('-', '')}")
PRIOR_SWEEP = [0.05, 0.10, 0.15, 0.25]
K_SWEEP = [4, 6, 8, 12]
BLEND_ALPHAS = [0.25, 0.5, 0.75]


def _run_geographic(work, target, cols, folds, prior_w):
    r = work[target.model_col].to_numpy(float)
    date_str = work["date"].dt.strftime("%Y%m%d")
    oof = np.full(len(work), np.nan)
    for fold in folds:
        tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
        va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
        if not tr_mask.any() or not va_mask.any():
            continue
        train_df, val_df = work[tr_mask], work[va_mask]
        va_idx = np.where(va_mask)[0]
        pred = np.full(len(val_df), np.nan)
        for region in pd.unique(work["region"]):
            w = np.where(train_df["region"].to_numpy() == region, 1.0, prior_w)
            pre_e, model_e = _fit_expert(train_df, cols, target.delta_col, w)
            sel = val_df["region"].to_numpy() == region
            if sel.any():
                pred[sel] = _predict(pre_e, model_e, val_df[sel], cols)
        oof[va_idx] = r[va_idx] + pred
    return oof


def _run_regime(work, target, cols, folds, prior_w, k):
    r = work[target.model_col].to_numpy(float)
    date_str = work["date"].dt.strftime("%Y%m%d")
    oof = np.full(len(work), np.nan)
    for fold in folds:
        tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
        va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
        if not tr_mask.any() or not va_mask.any():
            continue
        train_df, val_df = work[tr_mask], work[va_mask]
        va_idx = np.where(va_mask)[0]
        imput = train_df[REGIME_FEATURES].apply(pd.to_numeric, errors="coerce")
        med = imput.median()
        scaler = StandardScaler().fit(imput.fillna(med))
        km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(scaler.transform(imput.fillna(med)))
        lab_tr = km.labels_
        Z_va = scaler.transform(val_df[REGIME_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(med))
        d_va = np.linalg.norm(Z_va[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        T = GATE_TEMPERATURE * np.median(d_va)
        gate = np.exp(-d_va / max(T, 1e-9))
        gate = gate / gate.sum(axis=1, keepdims=True)
        preds = np.zeros((len(val_df), k))
        for kk in range(k):
            w = np.where(lab_tr == kk, 1.0, prior_w)
            pre_k, model_k = _fit_expert(train_df, cols, target.delta_col, w)
            preds[:, kk] = _predict(pre_k, model_k, val_df, cols)
        oof[va_idx] = r[va_idx] + (gate * preds).sum(axis=1)
    return oof


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    rows = []
    winners = {}

    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        y = work[target.obs_col].to_numpy(float)
        gm = _in_gom(work).to_numpy()
        folds = _build_forward_folds(sorted(work["date"].dt.strftime("%Y%m%d").unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

        def record(variant, oof, extra=None):
            entry = {"target": tname, "variant": variant, **(extra or {})}
            rows.append({**entry, "scope": "global", **_metrics(y, oof)})
            rows.append({**entry, "scope": "gulf_of_mexico", **_metrics(y[gm], oof[gm])})
            for region in pd.unique(work["region"]):
                sel = (work["region"] == region).to_numpy()
                rows.append({**entry, "scope": f"region:{region}", **_metrics(y[sel], oof[sel])})

        # axis 1: prior sweep, geographic experts
        geo_oofs = {}
        for w in PRIOR_SWEEP:
            oof = _run_geographic(work, target, cols, folds, w)
            geo_oofs[w] = oof
            record("moe_region", oof, {"prior_w": w})
            print(f"{tname} moe_region w={w}: global {_metrics(y, oof)['mae']:.3f}, gom {_metrics(y[gm], oof[gm])['mae']:.3f}")

        best_w = min(geo_oofs, key=lambda w: _metrics(y, geo_oofs[w])["mae"])

        # per-region prior composite (selection-biased; exploratory upper bound)
        composite = np.full(len(work), np.nan)
        for region in pd.unique(work["region"]):
            sel = (work["region"] == region).to_numpy()
            w_star = min(geo_oofs, key=lambda w: _metrics(y[sel], geo_oofs[w][sel])["mae"])
            composite[sel] = geo_oofs[w_star][sel]
        record("moe_region_per_region_prior", composite, {"prior_w": np.nan})

        # axis 2: K sweep, regime experts at best prior
        reg_oofs = {}
        for k in K_SWEEP:
            oof = _run_regime(work, target, cols, folds, best_w, k)
            reg_oofs[k] = oof
            record("moe_regime", oof, {"prior_w": best_w, "k": k})
            print(f"{tname} moe_regime k={k} w={best_w}: global {_metrics(y, oof)['mae']:.3f}, gom {_metrics(y[gm], oof[gm])['mae']:.3f}")

        best_k = min(reg_oofs, key=lambda k: _metrics(y, reg_oofs[k])["mae"])

        # axis 3: combined gate (convex blend of the two best MoEs)
        for a in BLEND_ALPHAS:
            blend = a * geo_oofs[best_w] + (1 - a) * reg_oofs[best_k]
            record("moe_blend", blend, {"prior_w": best_w, "k": best_k, "alpha": a})

        winners[tname] = {"best_prior_w": best_w, "best_k": int(best_k)}

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "moe_v2_summary.csv", index=False)
    (OUT_DIR / "winners.json").write_text(json.dumps(winners, indent=2))

    # sweep figures
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for row_i, tname in enumerate(("tchp", "d26")):
        sub = res[(res.target == tname) & (res.variant == "moe_region")]
        for col_i, scope, title in [(0, "global", "global"), (1, "gulf_of_mexico", "Gulf of Mexico")]:
            ax = axes[row_i, col_i]
            s = sub[sub.scope == scope].sort_values("prior_w")
            ax.plot(s["prior_w"], s["mae"], marker="o", color="#2563eb", linewidth=2)
            ax.set_xlabel("global-prior weight")
            ax.set_ylabel(f"MAE ({UNITS[tname]})")
            ax.set_title(f"{SHORT[tname]} — geographic MoE, {title}")
            ax.grid(True, alpha=0.15)
    fig.suptitle("Prior-weight sweep: how much global data should each regional expert see?")
    fig.savefig(OUT_DIR / "prior_weight_sweep.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, tname in zip(axes, ("tchp", "d26")):
        sub = res[(res.target == tname) & (res.variant == "moe_regime") & (res.scope == "global")].sort_values("k")
        ax.plot(sub["k"], sub["mae"], marker="o", color="#16a34a", linewidth=2)
        ax.set_xlabel("number of learned regimes K")
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]} — regime MoE, global")
        ax.grid(True, alpha=0.15)
    fig.suptitle("Regime-count sweep")
    fig.savefig(OUT_DIR / "k_sweep.png", dpi=180)
    plt.close(fig)

    show = res[res.scope.isin(["global", "gulf_of_mexico"])]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
