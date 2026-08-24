"""Tree diagrams for Dr. Jacobs (meeting item, 2026-08).

Renders actual decision trees with feature names, split thresholds, and leaf
values, at the sizes actually used in the project:

1. One depth-3 and one depth-4 random-forest tree on the Gulf D26 reduced
   feature set (most legible; the model family he asked about).
2. The first tree of the depth-4 boosted (XGBoost) model on the same data,
   drawn from its exported structure, to show boosting's residual-fitting
   splits side by side with the forest's.

Trained on the first locked fold's training block so the diagrams correspond
to a real model from the evaluation, not a toy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, _in_gom, _short_name  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/tree_diagrams_20260824")
LADDER_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/gom_attribution_20260722")


def _reduced_features(tname: str, full_cols: list[str]) -> list[str]:
    ladder = pd.read_csv(LADDER_DIR / f"{tname}_gom_backward_elimination.csv")
    k = int(ladder["mae"].idxmin())
    removed = set(ladder.loc[1:k, "removed"])
    return [c for c in full_cols if c not in removed]


def _draw_xgb_tree(model, cols, out_path, title):
    """Manual layout of tree 0 from trees_to_dataframe (no graphviz needed)."""
    t = model.get_booster().trees_to_dataframe()
    t = t[t.Tree == 0].set_index("ID")
    # positions by depth-first layout
    xpos = {}
    counter = [0.0]

    def layout(node_id, depth):
        row = t.loc[node_id]
        if pd.isna(row["Split"]):
            xpos[node_id] = (counter[0], depth)
            counter[0] += 1.0
            return xpos[node_id][0]
        xl = layout(row["Yes"], depth + 1)
        xr = layout(row["No"], depth + 1)
        xpos[node_id] = ((xl + xr) / 2.0, depth)
        return xpos[node_id][0]

    root = t.index[0]
    layout(root, 0)
    fig, ax = plt.subplots(figsize=(22, 9), constrained_layout=True)
    for node_id, (x, d) in xpos.items():
        row = t.loc[node_id]
        if pd.isna(row["Split"]):
            label = f"Δ = {row['Gain']:+.2f}"
            fc = "#dbeafe"
        else:
            fname = _short_name(cols[int(row["Feature"][1:])]) if row["Feature"].startswith("f") else row["Feature"]
            label = f"{fname}\n< {row['Split']:.3g} ?"
            fc = "#fef3c7"
            for child, side in [(row["Yes"], "yes"), (row["No"], "no")]:
                cx, cd = xpos[child]
                ax.plot([x, cx], [-d, -cd], color="gray", linewidth=1.0, zorder=1)
                ax.text((x + cx) / 2, -(d + cd) / 2, side, fontsize=8, color="gray", ha="center")
        ax.text(x, -d, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": fc, "edgecolor": "gray"}, zorder=2)
    ax.axis("off")
    ax.set_title(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    target = [t for t in TARGETS if t.name == "d26"][0]
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work).reset_index(drop=True)
    gom = work[_in_gom(work)].copy().reset_index(drop=True)
    full_cols = [c for c in FEATURE_SETS_BY_TARGET["d26"][RECIPE["d26"]] if c in work.columns]
    cols = _reduced_features("d26", full_cols)
    date_str = gom["date"].dt.strftime("%Y%m%d")
    folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                 n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
    tr = gom[date_str.isin(set(folds[0]["train_dates"]))]
    pre = _make_preprocessor(cols)
    X = pre.fit_transform(tr[cols])
    y = tr[target.delta_col].to_numpy(float)
    names = [_short_name(c) for c in cols]

    for depth in (3, 4):
        rf = RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=5,
                                   max_features=0.5, random_state=0, n_jobs=16)
        rf.fit(X, y)
        fig, ax = plt.subplots(figsize=(26 if depth == 4 else 18, 10), constrained_layout=True)
        plot_tree(rf.estimators_[0], feature_names=names, filled=True, rounded=True,
                  precision=2, fontsize=9 if depth == 4 else 11, ax=ax)
        ax.set_title(f"D26 Gulf residual, one random-forest tree (depth {depth}); "
                     "values are the residual correction (m) each leaf predicts, n = training rows in the node")
        fig.savefig(OUT_DIR / f"rf_tree_depth{depth}_d26_gulf.png", dpi=170)
        plt.close(fig)

    xm = _xgb_model()
    xm.fit(X, y)
    _draw_xgb_tree(xm, cols, OUT_DIR / "xgb_tree0_d26_gulf.png",
                   "D26 Gulf residual: first tree of the boosted model (depth 4)\n"
                   "Leaves show this tree's increment Δ; the model sums 300 such trees × learning rate 0.03")
    print("tree diagrams written")


if __name__ == "__main__":
    main()
