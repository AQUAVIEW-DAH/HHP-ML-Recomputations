"""Ensemble coverage/support statistics (follow-up to the tree diagrams).

A tree partitions the whole feature space, so every case is always covered;
what varies is the evidence behind each prediction. For the same Gulf D26
setup as the tree diagrams (reduced features, locked fold 1), computes:

1. Leaf-support distribution across the full ensembles (all 100 RF trees /
   all 300 boosted trees).
2. Per-profile effective support: mean training-rows-per-leaf that each
   validation profile lands in, across all trees; histogram + Gulf map.
3. Extrapolation share: validation profiles with >=1 feature outside the
   training range (where tree predictions saturate).
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
import xgboost
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import GOM, RECIPE, _in_gom  # noqa: E402
from OHC.exploration.build_tree_diagrams import _reduced_features  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/tree_diagrams_20260824")


def main() -> None:
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
    va = gom[date_str.isin(set(folds[0]["val_dates"]))].reset_index(drop=True)
    pre = _make_preprocessor(cols)
    X_tr = pre.fit_transform(tr[cols])
    X_va = pre.transform(va[cols])
    y_tr = tr[target.delta_col].to_numpy(float)

    # --- Random forest (same settings as the diagram trees, depth 4)
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=5,
                               max_features=0.5, random_state=0, n_jobs=16)
    rf.fit(X_tr, y_tr)
    rf_leaf_sizes_all = np.concatenate([
        est.tree_.n_node_samples[est.tree_.children_left == -1] for est in rf.estimators_
    ])
    leaves_va = rf.apply(X_va)  # (n_va, n_trees) leaf index per tree
    rf_support_va = np.zeros(leaves_va.shape, dtype=float)
    for t_i, est in enumerate(rf.estimators_):
        rf_support_va[:, t_i] = est.tree_.n_node_samples[leaves_va[:, t_i]]
    rf_mean_support = rf_support_va.mean(axis=1)

    # --- XGBoost (locked settings): cover per leaf
    xm = _xgb_model()
    xm.fit(X_tr, y_tr)
    tdf = xm.get_booster().trees_to_dataframe()
    leaves = tdf[tdf.Feature == "Leaf"]
    xgb_leaf_cover_all = leaves["Cover"].to_numpy(float)
    cover_lut = {(int(r.Tree), int(r.ID.split("-")[1])): float(r.Cover) for r in leaves.itertuples()}
    leaf_ids = xm.get_booster().predict(xgboost.DMatrix(X_va), pred_leaf=True).astype(int)
    xgb_support_va = np.array([[cover_lut[(t_i, leaf_ids[i, t_i])] for t_i in range(leaf_ids.shape[1])]
                               for i in range(leaf_ids.shape[0])])
    xgb_mean_support = xgb_support_va.mean(axis=1)

    # --- extrapolation share. NOTE: with a forward-in-time validation
    # block, calendar features (month, year, day-of-year encodings) are
    # guaranteed to leave the training range, so report the physics-only
    # share as the meaningful number.
    tr_min, tr_max = X_tr.min(axis=0), X_tr.max(axis=0)
    out_matrix = (X_va < tr_min) | (X_va > tr_max)
    out_of_range = out_matrix.any(axis=1)
    CALENDAR = {"year", "month_int", "doy_sin", "doy_cos", "month_sin", "month_cos",
                "is_winter_jfm", "is_summer_jas", "is_other"}
    phys_idx = [i for i, c in enumerate(cols) if c not in CALENDAR]
    out_of_range_physics = out_matrix[:, phys_idx].any(axis=1)

    stats = {
        "train_rows": int(len(tr)), "val_rows": int(len(va)), "features": len(cols),
        "rf_leaves_total": int(rf_leaf_sizes_all.size),
        "rf_leaf_size_median": float(np.median(rf_leaf_sizes_all)),
        "rf_leaf_size_p10": float(np.percentile(rf_leaf_sizes_all, 10)),
        "rf_share_leaves_lt10": float((rf_leaf_sizes_all < 10).mean()),
        "xgb_leaves_total": int(xgb_leaf_cover_all.size),
        "xgb_leaf_cover_median": float(np.median(xgb_leaf_cover_all)),
        "rf_val_mean_support_median": float(np.median(rf_mean_support)),
        "rf_val_mean_support_p5": float(np.percentile(rf_mean_support, 5)),
        "xgb_val_mean_support_median": float(np.median(xgb_mean_support)),
        "extrapolation_share_val_any_feature": float(out_of_range.mean()),
        "extrapolation_share_val_physics_only": float(out_of_range_physics.mean()),
        "note": "any-feature share is dominated by calendar features, which necessarily "
                "leave the training range under a forward-in-time split",
    }
    (OUT_DIR / "ensemble_support_stats.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), constrained_layout=True)
    axes[0].hist(rf_leaf_sizes_all, bins=40, color="#2563eb", alpha=0.75, label=f"RF ({rf_leaf_sizes_all.size:,} leaves)")
    axes[0].hist(xgb_leaf_cover_all, bins=40, color="#16a34a", alpha=0.55, label=f"XGB cover ({xgb_leaf_cover_all.size:,} leaves)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training rows per leaf")
    axes[0].set_ylabel("number of leaves (log)")
    axes[0].set_title("Leaf support across the FULL ensembles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15)

    axes[1].hist(rf_mean_support, bins=40, color="#2563eb", alpha=0.75, label="RF")
    axes[1].hist(xgb_mean_support, bins=40, color="#16a34a", alpha=0.55, label="XGB")
    axes[1].set_xlabel("mean rows per leaf the profile lands in (across all trees)")
    axes[1].set_ylabel("validation profiles")
    axes[1].set_title(f"Per-profile effective support (median RF {stats['rf_val_mean_support_median']:.0f})\n"
                      f"out-of-training-range profiles: {100*stats['extrapolation_share_val_physics_only']:.1f}% on physics features "
                      f"({100*stats['extrapolation_share_val_any_feature']:.0f}% incl. calendar, inherent to forward validation)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15)

    sc = axes[2].scatter(va["lon"], va["lat"], c=rf_mean_support, s=14, cmap="magma", rasterized=True)
    add_land_overlay(axes[2], zorder=2)
    axes[2].set_xlim(GOM["lon_min"] - 1, GOM["lon_max"] + 1)
    axes[2].set_ylim(GOM["lat_min"] - 1, GOM["lat_max"] + 1)
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title("Where the forest's evidence is thin (RF mean support)")
    plt.colorbar(sc, ax=axes[2], shrink=0.85).set_label("mean rows per leaf")
    fig.suptitle("D26 Gulf, locked fold 1: every profile is covered by every tree — this is how much evidence backs each prediction")
    fig.savefig(OUT_DIR / "ensemble_support_stats.png", dpi=180)
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
