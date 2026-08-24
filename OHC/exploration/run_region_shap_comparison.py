"""SHAP strategy comparison, MoE expert vs dedicated model, in EVERY region.

Extends the Gulf figure: for each of the five regions and each target, the
mean |SHAP| profile of the region's MoE expert (all rows, own-region 1.0 /
else 0.05) is compared against the dedicated region-only model on the same
out-of-fold rows. A summary panel shows the expert/dedicated attribution
ratio per feature FAMILY per region (families group near-duplicate features:
ssh with its |lat| interaction, the anomaly/context stencils, seasonal terms,
location terms), so the question "does the prior's strategy suppression
translate across regions" is answered in one view.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.titlesize": 16})
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _short_name  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/expert_cross_eval_20260824")
PRIOR_W = 0.05
REGIONS = ["gulf_of_mexico", "atlantic", "indian", "west_pacific", "epac_other"]
DISPLAY = {"gulf_of_mexico": "Gulf of Mexico", "atlantic": "Atlantic", "indian": "Indian",
           "west_pacific": "West Pacific", "epac_other": "E/C Pacific & rest"}
TOP_N = 8


def _families(cols):
    fam = {}
    for c in cols:
        if c in ("model_ssh_m", "model_ssh_x_abs_lat"):
            fam[c] = "SSH family"
        elif "anom_from" in c or "local_std" in c or "grad_mag" in c:
            fam[c] = "context/anomaly"
        elif c in ("month_int", "month_sin", "month_cos", "doy_sin", "doy_cos",
                   "is_winter_jfm", "is_summer_jas", "is_other", "year"):
            fam[c] = "calendar"
        elif c in ("lat", "lon", "abs_lat", "nearest_rtofs_grid_distance_km"):
            fam[c] = "location"
        elif c in ("model_interp_tchp_kj_per_cm2", "model_interp_d26_m"):
            fam[c] = "raw model value"
        else:
            fam[c] = "other physics"
    return fam


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        n_feat = len(cols)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        shap_e = {rg: np.full((len(work), n_feat), np.nan) for rg in REGIONS}
        shap_d = {rg: np.full((len(work), n_feat), np.nan) for rg in REGIONS}

        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            train_df = work[tr_mask]
            va_idx = np.where(va_mask)[0]
            val_df = work.iloc[va_idx]
            for rg in REGIONS:
                rg_va_idx = va_idx[val_df["region"].to_numpy() == rg]
                if rg_va_idx.size == 0:
                    continue
                Xrows = work.iloc[rg_va_idx]
                # expert
                pre = _make_preprocessor(cols)
                Xtr = pre.fit_transform(train_df[cols])
                w = np.where(train_df["region"].to_numpy() == rg, 1.0, PRIOR_W)
                m = _xgb_model()
                m.fit(Xtr, train_df[target.delta_col].to_numpy(float), sample_weight=w)
                shap_e[rg][rg_va_idx] = m.get_booster().predict(
                    xgb.DMatrix(pre.transform(Xrows[cols])), pred_contribs=True)[:, :n_feat]
                # dedicated
                tr_rg = train_df[train_df["region"] == rg]
                if tr_rg.empty:
                    continue
                pre2 = _make_preprocessor(cols)
                m2 = _xgb_model()
                m2.fit(pre2.fit_transform(tr_rg[cols]), tr_rg[target.delta_col].to_numpy(float))
                shap_d[rg][rg_va_idx] = m2.get_booster().predict(
                    xgb.DMatrix(pre2.transform(Xrows[cols])), pred_contribs=True)[:, :n_feat]

        fam = _families(cols)
        fam_names = ["SSH family", "context/anomaly", "calendar", "location", "raw model value", "other physics"]
        ratio = np.full((len(REGIONS), len(fam_names)), np.nan)

        fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
        for p, rg in enumerate(REGIONS):
            ax = axes.ravel()[p]
            ok = np.isfinite(shap_d[rg]).all(axis=1) & np.isfinite(shap_e[rg]).all(axis=1)
            me = np.abs(shap_e[rg][ok]).mean(axis=0)
            md = np.abs(shap_d[rg][ok]).mean(axis=0)
            order = np.argsort(md)[::-1][:TOP_N]
            yy = np.arange(len(order))
            ax.barh(yy - 0.2, me[order], 0.4, color="#2563eb", label="MoE expert (w=0.05)")
            ax.barh(yy + 0.2, md[order], 0.4, color="#16a34a", label="dedicated region-only")
            ax.set_yticks(yy)
            ax.set_yticklabels([_short_name(cols[i]) for i in order], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel(f"mean |SHAP| ({UNITS[tname]})")
            ax.set_title(f"{DISPLAY[rg]} (n={int(ok.sum()):,})")
            ax.grid(True, axis="x", alpha=0.15)
            if p == 0:
                ax.legend(fontsize=10)
            for fi, fn in enumerate(fam_names):
                idx = [i for i, c in enumerate(cols) if fam[c] == fn]
                if not idx:
                    continue
                de = float(np.abs(shap_d[rg][ok][:, idx]).sum(axis=1).mean())
                ex = float(np.abs(shap_e[rg][ok][:, idx]).sum(axis=1).mean())
                if de > 1e-9:
                    ratio[p, fi] = ex / de

        ax = axes.ravel()[5]
        norm = TwoSlopeNorm(vcenter=1.0, vmin=0.25, vmax=1.75)
        im = ax.imshow(ratio, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_xticks(range(len(fam_names)))
        ax.set_xticklabels(fam_names, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(REGIONS)))
        ax.set_yticklabels([DISPLAY[r] for r in REGIONS], fontsize=10)
        for i in range(len(REGIONS)):
            for j in range(len(fam_names)):
                if np.isfinite(ratio[i, j]):
                    ax.text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if abs(ratio[i, j] - 1) > 0.45 else "black")
        plt.colorbar(im, ax=ax, shrink=0.85).set_label("expert ÷ dedicated attribution\n(<1 = prior suppresses, >1 = prior amplifies)")
        ax.set_title("Does the prior's strategy shift translate?\n(family-grouped, robust to twin-feature credit shuffling)")
        fig.suptitle(f"{SHORT[tname]}: expert-vs-dedicated SHAP strategy in every region")
        fig.savefig(OUT_DIR / f"{tname}_region_shap_comparison.png", dpi=170)
        plt.close(fig)
        pd.DataFrame(ratio, index=REGIONS, columns=fam_names).to_csv(OUT_DIR / f"{tname}_region_family_suppression.csv")
        print(f"{tname} done")


if __name__ == "__main__":
    main()
