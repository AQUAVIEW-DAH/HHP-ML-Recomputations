"""Interventional tests: is the Gulf strategy difference caused by the prior,
or is it correlated-feature credit shuffling? (TCHP, the puzzle target.)

Panel 1  Prediction-space agreement: expert vs dedicated out-of-fold Gulf
         predictions. Near-identical functions => cosmetic; structured
         disagreement => substantive.
Panel 2  2x2 substitutability ablation: expert and dedicated retrained
         without the SSH family and without the anomaly/context family.
         Asymmetric skill drops => genuine reliance differences.
Panel 3  Seed stability: SSH-family attribution share under 5 seeds per
         model. Stable separation => the prior, not noise, sets the strategy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.titlesize": 15})
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, _in_gom  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/expert_cross_eval_20260824")
PRIOR_W = 0.05
SEEDS = [0, 1, 2, 3, 4]
SSH_FAMILY = {"model_ssh_m", "model_ssh_x_abs_lat"}
CONTEXT_FAMILY_KEYS = ("anom_from", "local_std", "grad_mag")


def _model(seed=0):
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                        random_state=seed, n_jobs=16)


def main() -> None:
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    target = [t for t in TARGETS if t.name == "tchp"][0]
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work).reset_index(drop=True)
    work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
    cols_full = [c for c in FEATURE_SETS_BY_TARGET["tchp"][RECIPE["tchp"]] if c in work.columns]
    context = [c for c in cols_full if any(k in c for k in CONTEXT_FAMILY_KEYS)]
    sets = {"full": cols_full,
            "no SSH family": [c for c in cols_full if c not in SSH_FAMILY],
            "no context family": [c for c in cols_full if c not in context]}
    y = work[target.obs_col].to_numpy(float)
    r = work[target.model_col].to_numpy(float)
    gm = _in_gom(work).to_numpy()
    date_str = work["date"].dt.strftime("%Y%m%d")
    folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                 n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

    def run(kind, cols, seed=0, want_shap=False):
        oof = np.full(len(work), np.nan)
        shap_ssh_share = []
        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            train_df = work[tr_mask]
            va_idx = np.where(va_mask & gm)[0]
            if kind == "expert":
                tdf, w = train_df, np.where(train_df["region"].to_numpy() == "gulf_of_mexico", 1.0, PRIOR_W)
            else:
                tdf, w = train_df[train_df["region"] == "gulf_of_mexico"], None
            pre = _make_preprocessor(cols)
            m = _model(seed)
            m.fit(pre.fit_transform(tdf[cols]), tdf[target.delta_col].to_numpy(float), sample_weight=w)
            Xva = pre.transform(work.iloc[va_idx][cols])
            oof[va_idx] = r[va_idx] + m.predict(Xva)
            if want_shap:
                contrib = np.abs(m.get_booster().predict(xgb.DMatrix(Xva), pred_contribs=True)[:, :len(cols)])
                idx = [i for i, c in enumerate(cols) if c in SSH_FAMILY]
                shap_ssh_share.append(contrib[:, idx].sum() / max(contrib.sum(), 1e-9))
        ok = gm & np.isfinite(oof)
        mae = float(np.abs(oof[ok] - y[ok]).mean())
        return oof, mae, (float(np.mean(shap_ssh_share)) if shap_ssh_share else np.nan)

    # panel 1 + 2 base runs
    oof_e, mae_e_full, _ = run("expert", cols_full)
    oof_d, mae_d_full, _ = run("dedicated", cols_full)
    abl = {("expert", "full"): mae_e_full, ("dedicated", "full"): mae_d_full}
    for label in ("no SSH family", "no context family"):
        abl[("expert", label)] = run("expert", sets[label])[1]
        abl[("dedicated", label)] = run("dedicated", sets[label])[1]
    # panel 3 seeds
    shares = {"expert": [], "dedicated": []}
    for s in SEEDS:
        shares["expert"].append(run("expert", cols_full, seed=s, want_shap=True)[2])
        shares["dedicated"].append(run("dedicated", cols_full, seed=s, want_shap=True)[2])

    ok = gm & np.isfinite(oof_e) & np.isfinite(oof_d)
    pe, pd_, yy = oof_e[ok], oof_d[ok], y[ok]
    agree_r = float(np.corrcoef(pe, pd_)[0, 1])
    mad = float(np.mean(np.abs(pe - pd_)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6), constrained_layout=True)
    axes[0].hexbin(pd_, pe, gridsize=45, cmap="viridis", bins="log")
    lo, hi = np.percentile(np.concatenate([pe, pd_]), [0.5, 99.5])
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_xlabel("dedicated model prediction (kJ/cm²)")
    axes[0].set_ylabel("MoE expert prediction (kJ/cm²)")
    axes[0].set_title(f"1. Same function? r = {agree_r:.3f}, mean |diff| = {mad:.2f} kJ/cm²\n"
                      f"(vs mean |error| ≈ {mae_e_full:.2f})")

    x = np.arange(3)
    width = 0.35
    labels = ["full", "no SSH family", "no context family"]
    for i, kind, colr in ((0, "expert", "#2563eb"), (1, "dedicated", "#16a34a")):
        vals = [abl[(kind, l)] for l in labels]
        bars = axes[1].bar(x + (i - 0.5) * width, vals, width, color=colr, label=kind)
        for b, v in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Gulf OOF MAE (kJ/cm²)")
    axes[1].set_title("2. Substitutability ablation")
    axes[1].legend()
    axes[1].set_ylim(bottom=min(abl.values()) * 0.97)
    axes[1].grid(True, axis="y", alpha=0.15)

    for i, kind, colr in ((0, "expert", "#2563eb"), (1, "dedicated", "#16a34a")):
        vals = np.array(shares[kind]) * 100
        axes[2].scatter(np.full(len(vals), i) + np.linspace(-0.06, 0.06, len(vals)), vals, s=70, color=colr)
        axes[2].hlines(vals.mean(), i - 0.18, i + 0.18, color=colr, linewidth=2)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["MoE expert\n(w=0.05)", "dedicated\nGulf-only"])
    axes[2].set_ylabel("SSH-family share of total attribution (%)")
    axes[2].set_title("3. Seed stability (5 seeds each)")
    axes[2].grid(True, axis="y", alpha=0.15)

    fig.suptitle("TCHP Gulf: is the strategy difference caused by the prior or by feature correlation?")
    fig.savefig(OUT_DIR / "tchp_prior_vs_correlation_tests.png", dpi=180)
    plt.close(fig)
    out = {"agreement_r": agree_r, "mean_abs_diff": mad,
           "ablation": {f"{k[0]}|{k[1]}": v for k, v in abl.items()},
           "ssh_share_seeds": shares}
    (OUT_DIR / "tchp_prior_vs_correlation_tests.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
