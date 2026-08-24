"""Head-to-head in every region (extension of the Gulf comparison).

For each of the five geographic regions and each target, three models are
scored out-of-fold on that region's rows under the same locked folds:

- single global model (one fit per fold, no weighting)
- the MoE expert for the region (all rows, own-region weight 1.0 / else 0.05)
- a dedicated region-only model (trained on the region's rows alone)

Output: grouped-bar figure (one row per target) + CSV.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/expert_cross_eval_20260824")
PRIOR_W = 0.05
REGIONS = ["gulf_of_mexico", "atlantic", "indian", "west_pacific", "epac_other"]
DISPLAY = {"gulf_of_mexico": "Gulf of\nMexico", "atlantic": "Atlantic", "indian": "Indian",
           "west_pacific": "West\nPacific", "epac_other": "E/C Pacific\n& rest"}
COLORS = {"global": "#94a3b8", "moe_expert": "#2563eb", "dedicated": "#16a34a"}
LABELS = {"global": "single global model", "moe_expert": "MoE expert (w=0.05 prior)",
          "dedicated": "dedicated region-only model"}


def _fit_predict(train_df, cols, delta_col, weights, val_df):
    pre = _make_preprocessor(cols)
    model = _xgb_model()
    model.fit(pre.fit_transform(train_df[cols]), train_df[delta_col].to_numpy(float), sample_weight=weights)
    return model.predict(pre.transform(val_df[cols]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    rows = []
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
        oof = {("global", None): np.full(len(work), np.nan)}
        for reg in REGIONS:
            oof[("moe_expert", reg)] = np.full(len(work), np.nan)
            oof[("dedicated", reg)] = np.full(len(work), np.nan)

        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            train_df = work[tr_mask]
            va_idx = np.where(va_mask)[0]
            val_df = work.iloc[va_idx]
            oof[("global", None)][va_idx] = r[va_idx] + _fit_predict(train_df, cols, target.delta_col, None, val_df)
            for reg in REGIONS:
                reg_va_idx = va_idx[val_df["region"].to_numpy() == reg]
                if reg_va_idx.size == 0:
                    continue
                reg_val = work.iloc[reg_va_idx]
                w = np.where(train_df["region"].to_numpy() == reg, 1.0, PRIOR_W)
                oof[("moe_expert", reg)][reg_va_idx] = r[reg_va_idx] + _fit_predict(train_df, cols, target.delta_col, w, reg_val)
                tr_reg = train_df[train_df["region"] == reg]
                if not tr_reg.empty:
                    oof[("dedicated", reg)][reg_va_idx] = r[reg_va_idx] + _fit_predict(tr_reg, cols, target.delta_col, None, reg_val)

        for reg in REGIONS:
            sel = (work["region"] == reg).to_numpy()
            for kind in ("global", "moe_expert", "dedicated"):
                pred = oof[("global", None)] if kind == "global" else oof[(kind, reg)]
                ok = sel & np.isfinite(pred)
                e = pred[ok] - y[ok]
                rows.append({"target": tname, "region": reg, "model": kind, "rows": int(ok.sum()),
                             "mae": float(np.abs(e).mean()), "bias": float(e.mean())})
        print(f"{tname} done")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "region_head_to_head.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
    x = np.arange(len(REGIONS))
    width = 0.26
    for ax, tname in zip(axes, ("tchp", "d26")):
        sub = res[res.target == tname]
        for i, kind in enumerate(("global", "moe_expert", "dedicated")):
            vals = [float(sub[(sub.region == rg) & (sub.model == kind)]["mae"].iloc[0]) for rg in REGIONS]
            bars = ax.bar(x + (i - 1) * width, vals, width, color=COLORS[kind], label=LABELS[kind])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        # winner stars
        for j, rg in enumerate(REGIONS):
            vals = {k: float(sub[(sub.region == rg) & (sub.model == k)]["mae"].iloc[0]) for k in ("global", "moe_expert", "dedicated")}
            win = min(vals, key=vals.get)
            i = ("global", "moe_expert", "dedicated").index(win)
            ax.text(j + (i - 1) * width, vals[win] * 0.985, "★", ha="center", va="top", fontsize=13, color="white")
        n_by_reg = [int(sub[(sub.region == rg) & (sub.model == "global")]["rows"].iloc[0]) for rg in REGIONS]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{DISPLAY[rg]}\n(n={n:,})" for rg, n in zip(REGIONS, n_by_reg)], fontsize=11)
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}")
        ax.grid(True, axis="y", alpha=0.15)
        lo = min(float(sub["mae"].min()) for _ in [0]) * 0.9
        ax.set_ylim(bottom=lo)
    axes[0].legend(loc="upper right")
    fig.suptitle("Head-to-head in every region: global model vs its MoE expert vs a dedicated region-only model\n"
                 "★ = best in region (out-of-fold, locked folds)")
    fig.savefig(OUT_DIR / "region_head_to_head.png", dpi=180)
    plt.close(fig)
    print(res.pivot_table(index=["target", "region"], columns="model", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
