"""Showcase figures for the recommended MoE blend vs the previous best model.

Recomputes the winning v2 configurations (per docs/math/moe_recommended_model.md),
saves the out-of-fold blend predictions, and renders the mentor-familiar
figure families with the MoE as a third column:

1. `{t}_moe_density_comparison.png`  raw | previous best global | MoE blend,
   observed-vs-model log10(PDF) density panels, shared axes and colour scale.
2. `{t}_moe_named_box_mae.png`       per named 20-degree box MAE bars.
3. `{t}_moe_box_improvement_map.png` 20-degree boxes coloured by how much the
   MoE improves on the single global model.
4. `{t}_moe_conditional_skill.png`   MAE vs observed value for all three.

Predictions are saved to `moe_predictions_{t}.parquet` so later bundles can
be regenerated on the recommended model without retraining.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import FEATURE_SETS_BY_TARGET, FOLD_PATH, _merge_feature_tables  # noqa: E402
from OHC.build_hhp_density_scatter_diagnostics import (  # noqa: E402
    NAMED_BOXES,
    _common_axis_limits,
    _density_color_limits,
    _density_image,
    _plot_density_panel,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom  # noqa: E402
from OHC.exploration.run_moe_regions import _metrics, _region_of  # noqa: E402
from OHC.exploration.run_moe_v2_tuning import _run_geographic, _run_regime  # noqa: E402

RUN_DATE = "2026-08-11"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/moe_showcase_{RUN_DATE.replace('-', '')}")
WINNERS = {"tchp": {"alpha": 0.75, "k": 6, "w": 0.05}, "d26": {"alpha": 0.50, "k": 12, "w": 0.05}}
PRED_PATHS = {
    "tchp": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet"),
    "d26": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet"),
}
GLOBAL_COL = {
    "tchp": "pred_obs__global_pruned_plus_neighborhood",
    "d26": "pred_obs__drop_both_lat_interactions_plus_neighborhood",
}
BOX_DEG = 20
MIN_BOX_ROWS = 25
N_BINS = 12
LABELS = ["raw RTOFS", "previous best (single global model)", "recommended MoE blend"]
COLORS = ["#dc2626", "#94a3b8", "#2563eb"]


def _box_ids(lat, lon):
    la = np.floor(lat / BOX_DEG) * BOX_DEG
    lo = np.floor(((lon + 180.0) % 360.0 - 180.0) / BOX_DEG) * BOX_DEG
    return la, lo


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())

    for target in TARGETS:
        tname = target.name
        cfg = WINNERS[tname]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        folds = _build_forward_folds(sorted(work["date"].dt.strftime("%Y%m%d").unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

        oof_geo = _run_geographic(work, target, cols, folds, cfg["w"])
        oof_reg = _run_regime(work, target, cols, folds, cfg["w"], cfg["k"])
        moe = cfg["alpha"] * oof_geo + (1 - cfg["alpha"]) * oof_reg

        out = work[["date", "lat", "lon", target.obs_col, target.model_col]].copy()
        out["date"] = out["date"].dt.strftime("%Y%m%d")
        out["pred_obs__moe_blend"] = moe

        # The locked predictions table holds the same rows but ordered
        # fold-by-fold, and (date, lat, lon) is not unique (duplicate casts),
        # so pair rows with a per-key occurrence counter: a bijection that a
        # plain key merge (cross-join on duplicates) or positional join
        # (different ordering) would both get wrong.
        pred = pd.read_parquet(PRED_PATHS[tname]).reset_index(drop=True)
        pred["date"] = pd.to_datetime(pred["date"]).dt.strftime("%Y%m%d")
        # pred holds only the OOF-evaluated subset (~60k of 90k rows); include
        # the observed value in the key so co-located duplicate casts pair
        # with the right prediction.
        keys = ["date", "lat", "lon", target.obs_col]
        out["_k"] = out.groupby(keys).cumcount()
        pred["_k"] = pred.groupby(keys).cumcount()
        out = out.merge(pred[keys + ["_k", "pred_obs__raw_rtofs", GLOBAL_COL[tname]]],
                        on=keys + ["_k"], how="left", validate="one_to_one").drop(columns="_k")
        out = out.rename(columns={GLOBAL_COL[tname]: "pred_obs__global_best"})
        out.to_parquet(OUT_DIR / f"moe_predictions_{tname}.parquet", index=False)

        ok = np.isfinite(out[["pred_obs__moe_blend", "pred_obs__global_best", "pred_obs__raw_rtofs"]]).all(axis=1).to_numpy()
        sub = out[ok].reset_index(drop=True)
        y = sub[target.obs_col].to_numpy(float)
        series = [sub["pred_obs__raw_rtofs"].to_numpy(float),
                  sub["pred_obs__global_best"].to_numpy(float),
                  sub["pred_obs__moe_blend"].to_numpy(float)]
        print(f"{tname}: rows={len(sub)}  " + "  ".join(
            f"{lab}: {_metrics(y, s)['mae']:.3f}" for lab, s in zip(("raw", "global", "moe"), series)))

        # 1 — density comparison
        xlim, ylim = _common_axis_limits(y, series[0], series[2])
        imgs = [_density_image(y, s, xlim=xlim, ylim=ylim)[0] for s in series]
        climits = _density_color_limits(*imgs)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), constrained_layout=True)
        mesh = None
        for ax, s, lab in zip(axes, series, LABELS):
            mesh, m, _ = _plot_density_panel(
                ax, y, s, title=f"{SHORT[tname]}: {lab}",
                xlabel=f"Observed {SHORT[tname]} ({UNITS[tname]})",
                ylabel=f"Model {SHORT[tname]} ({UNITS[tname]})",
                xlim=xlim, ylim=ylim, color_limits=climits)
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
        cbar.set_label("log10(PDF)")
        fig.suptitle(f"{SHORT[tname]} observed vs model — the recommended MoE against its predecessors "
                     f"(out-of-fold, n={len(sub):,})")
        fig.savefig(OUT_DIR / f"{tname}_moe_density_comparison.png", dpi=180)
        plt.close(fig)

        # 2 — named-box bars
        la, lo = _box_ids(sub["lat"].to_numpy(float), sub["lon"].to_numpy(float))
        fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
        x = np.arange(len(NAMED_BOXES))
        width = 0.26
        for i, (s, lab, colr) in enumerate(zip(series, LABELS, COLORS)):
            vals = []
            for b in NAMED_BOXES:
                sel = (la == b.lat0) & (lo == b.lon0)
                vals.append(_metrics(y[sel], s[sel])["mae"] if sel.sum() >= MIN_BOX_ROWS else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, color=colr, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([b.display for b in NAMED_BOXES], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: out-of-fold MAE per named 20° box")
        ax.grid(True, axis="y", alpha=0.15)
        ax.legend()
        fig.savefig(OUT_DIR / f"{tname}_moe_named_box_mae.png", dpi=180)
        plt.close(fig)

        # 3 — improvement map (MoE vs previous best)
        frame = pd.DataFrame({"la": la, "lo": lo,
                              "e_g": np.abs(series[1] - y), "e_m": np.abs(series[2] - y)})
        agg = frame.groupby(["la", "lo"]).agg(rows=("e_g", "size"), g=("e_g", "mean"), m=("e_m", "mean")).reset_index()
        agg = agg[agg["rows"] >= MIN_BOX_ROWS]
        agg["imp"] = agg["g"] - agg["m"]
        vlim = float(np.nanquantile(np.abs(agg["imp"]), 0.95))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        cmap = plt.get_cmap("RdBu_r")
        fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
        add_land_overlay(ax, zorder=2)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-70, 70)
        for _, rrow in agg.iterrows():
            ax.add_patch(Rectangle((rrow["lo"], rrow["la"]), BOX_DEG, BOX_DEG,
                                   facecolor=cmap(norm(rrow["imp"])), edgecolor="white", linewidth=0.4, zorder=1))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(f"MAE(single global) − MAE(MoE) ({UNITS[tname]})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{SHORT[tname]}: where the MoE improves on the single global model "
                     f"(red = MoE better; boxes with ≥{MIN_BOX_ROWS} rows)")
        fig.savefig(OUT_DIR / f"{tname}_moe_box_improvement_map.png", dpi=180)
        plt.close(fig)

        # 4 — conditional skill with the MoE included
        edges = np.unique(np.quantile(y, np.linspace(0, 1, N_BINS + 1)))
        idx = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for s, lab, colr in zip(series, LABELS, COLORS):
            centers, maes = [], []
            for b in range(len(edges) - 1):
                m = idx == b
                if m.sum() < 40:
                    continue
                centers.append(float(np.median(y[m])))
                maes.append(float(np.abs(s[m] - y[m]).mean()))
            ax.plot(centers, maes, marker="o", linewidth=2, color=colr, label=lab)
        ax.set_xlabel(f"observed {SHORT[tname]} ({UNITS[tname]})")
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: error vs observed value, all three models (out-of-fold)")
        ax.grid(True, alpha=0.15)
        ax.legend()
        fig.savefig(OUT_DIR / f"{tname}_moe_conditional_skill.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
