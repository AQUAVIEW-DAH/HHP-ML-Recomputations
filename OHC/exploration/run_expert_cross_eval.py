"""Expert cross-evaluation (meeting items, 2026-08).

1. Cross-region matrix: every geographic expert (trained with its own-region
   weight 1.0 / elsewhere 0.05) evaluated out-of-fold on EVERY region's rows —
   a 5x5 skill matrix per target showing specialisation vs transfer.
2. Gulf head-to-head: the MoE Gulf expert vs the dedicated Gulf-only model
   (trained on Gulf rows alone) on identical Gulf folds, with mean-|SHAP|
   profiles of both on the same rows, to explain the remaining D26 gap.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.titlesize": 16})
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom, _short_name  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/expert_cross_eval_20260824")
PRIOR_W = 0.05
REGIONS = ["gulf_of_mexico", "atlantic", "indian", "west_pacific", "epac_other"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    matrix_rows, shap_rows, headtohead = [], [], []

    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        y = work[target.obs_col].to_numpy(float)
        r = work[target.model_col].to_numpy(float)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        gm = _in_gom(work).to_numpy()

        # OOF predictions of every expert on ALL rows, plus dedicated-local and SHAP on Gulf rows
        oof = {e: np.full(len(work), np.nan) for e in REGIONS}
        oof_dedicated = np.full(len(work), np.nan)
        n_feat = len(cols)
        shap_expert = np.full((len(work), n_feat), np.nan)
        shap_dedicated = np.full((len(work), n_feat), np.nan)

        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            train_df = work[tr_mask]
            va_idx = np.where(va_mask)[0]
            va_gom_idx = np.where(va_mask & gm)[0]
            for e in REGIONS:
                w = np.where(train_df["region"].to_numpy() == e, 1.0, PRIOR_W)
                pre = _make_preprocessor(cols)
                X_tr = pre.fit_transform(train_df[cols])
                model = _xgb_model()
                model.fit(X_tr, train_df[target.delta_col].to_numpy(float), sample_weight=w)
                X_va = pre.transform(work.iloc[va_idx][cols])
                oof[e][va_idx] = r[va_idx] + model.predict(X_va)
                if e == "gulf_of_mexico" and va_gom_idx.size:
                    Xg = pre.transform(work.iloc[va_gom_idx][cols])
                    shap_expert[va_gom_idx] = model.get_booster().predict(xgb.DMatrix(Xg), pred_contribs=True)[:, :n_feat]
            # dedicated Gulf-only model
            tr_g = train_df[train_df["region"] == "gulf_of_mexico"]
            if not tr_g.empty and va_gom_idx.size:
                pre = _make_preprocessor(cols)
                Xg_tr = pre.fit_transform(tr_g[cols])
                model = _xgb_model()
                model.fit(Xg_tr, tr_g[target.delta_col].to_numpy(float))
                Xg_va = pre.transform(work.iloc[va_gom_idx][cols])
                oof_dedicated[va_gom_idx] = r[va_gom_idx] + model.predict(Xg_va)
                shap_dedicated[va_gom_idx] = model.get_booster().predict(xgb.DMatrix(Xg_va), pred_contribs=True)[:, :n_feat]

        # 1 — cross-region matrix
        mat = np.full((len(REGIONS), len(REGIONS)), np.nan)
        for i, e in enumerate(REGIONS):
            for j, reg in enumerate(REGIONS):
                sel = (work["region"] == reg).to_numpy() & np.isfinite(oof[e])
                if sel.sum() < 50:
                    continue
                mat[i, j] = float(np.abs(oof[e][sel] - y[sel]).mean())
                matrix_rows.append({"target": tname, "expert": e, "eval_region": reg,
                                    "rows": int(sel.sum()), "mae": mat[i, j]})
        fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
        im = ax.imshow(mat, cmap="viridis_r")
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels([x.replace("_", "\n") for x in REGIONS], fontsize=10)
        ax.set_yticks(range(len(REGIONS)))
        ax.set_yticklabels([x.replace("_", "\n") for x in REGIONS], fontsize=10)
        ax.set_xlabel("evaluated on region")
        ax.set_ylabel("expert (trained for region)")
        for i in range(len(REGIONS)):
            for j in range(len(REGIONS)):
                if np.isfinite(mat[i, j]):
                    best = np.nanargmin(mat[:, j]) == i
                    ax.text(j, i, f"{mat[i, j]:.2f}" + ("*" if best else ""), ha="center", va="center",
                            color="white" if mat[i, j] > np.nanmean(mat) else "black",
                            fontweight="bold" if best else "normal", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.85).set_label(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: each expert scored on every region (out-of-fold)\n* = best expert for that column")
        fig.savefig(OUT_DIR / f"{tname}_expert_cross_matrix.png", dpi=180)
        plt.close(fig)

        # 2 — Gulf head-to-head + SHAP comparison
        sel = gm & np.isfinite(oof["gulf_of_mexico"]) & np.isfinite(oof_dedicated)
        for name, pred in [("moe_gulf_expert_w0.05", oof["gulf_of_mexico"]), ("dedicated_gulf_only", oof_dedicated)]:
            e = pred[sel] - y[sel]
            headtohead.append({"target": tname, "model": name, "rows": int(sel.sum()),
                               "mae": float(np.abs(e).mean()), "bias": float(e.mean())})
        me = np.nanmean(np.abs(shap_expert[sel]), axis=0)
        md = np.nanmean(np.abs(shap_dedicated[sel]), axis=0)
        order = np.argsort(md)[::-1][:12]
        yy = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
        ax.barh(yy - 0.2, me[order], 0.4, color="#2563eb", label="MoE Gulf expert (w=0.05 prior)")
        ax.barh(yy + 0.2, md[order], 0.4, color="#16a34a", label="dedicated Gulf-only model")
        ax.set_yticks(yy)
        ax.set_yticklabels([_short_name(cols[i]) for i in order])
        ax.invert_yaxis()
        ax.set_xlabel(f"mean |SHAP contribution| ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: what the prior is still suppressing in the Gulf expert")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.15)
        fig.savefig(OUT_DIR / f"{tname}_gulf_expert_vs_dedicated_shap.png", dpi=180)
        plt.close(fig)
        for i in range(n_feat):
            shap_rows.append({"target": tname, "feature": cols[i],
                              "mean_abs_shap_moe_expert": float(me[i]), "mean_abs_shap_dedicated": float(md[i])})
        print(f"{tname} done")

    pd.DataFrame(matrix_rows).to_csv(OUT_DIR / "expert_cross_matrix.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(OUT_DIR / "gulf_expert_vs_dedicated_shap.csv", index=False)
    hh = pd.DataFrame(headtohead)
    hh.to_csv(OUT_DIR / "gulf_head_to_head.csv", index=False)
    print(hh.to_string(index=False))


if __name__ == "__main__":
    main()
