"""Follow-up diagnostics from the 2026-08 meeting notes (items 1-3).

1. Conditional skill by observed value: is raw RTOFS better than the corrected
   models in the upper tail? Bin rows by observed target value and compare
   MAE and bias of raw / corrected-global / corrected-local per bin, and
   report the crossover value where the correction stops helping.
   Run globally and for the Gulf of Mexico.
2. SHAP maps: each Gulf profile plotted at its position, colored by the SSH
   contribution to the prediction, global vs local model side by side.
3. Interaction-coloured dependence: the SSH dependence scatter recoloured by a
   third feature (longitude, then season) so the interaction branches that
   split the low-SSH cloud are visible.

Reuses the attribution machinery so folds, models and SHAP values match the
2026-07-22 analysis exactly.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import FEATURE_SETS_BY_TARGET, FOLD_PATH, _merge_feature_tables  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import (  # noqa: E402
    GOM,
    RECIPE,
    SHORT,
    UNITS,
    _fold_models_and_shap,
    _in_gom,
    _short_name,
)

RUN_DATE = "2026-08-11"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/gom_diagnostics_{RUN_DATE.replace('-', '')}")
PRED_PATHS = {
    "tchp": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet"),
    "d26": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet"),
}
BEST_PRED_COL = {
    "tchp": "pred_obs__global_pruned_plus_neighborhood",
    "d26": "pred_obs__drop_both_lat_interactions_plus_neighborhood",
}
OBS_COL = {"tchp": "argo_tchp_kj_per_cm2", "d26": "argo_d26_m"}
N_BINS = 12
MIN_BIN_ROWS = 40


# ---------------------------------------------------------------- item 1

def _conditional_skill(df: pd.DataFrame, obs_col: str, raw_col: str, corr_col: str) -> pd.DataFrame:
    obs = df[obs_col].to_numpy(float)
    raw = df[raw_col].to_numpy(float)
    corr = df[corr_col].to_numpy(float)
    ok = np.isfinite(obs) & np.isfinite(raw) & np.isfinite(corr)
    obs, raw, corr = obs[ok], raw[ok], corr[ok]
    edges = np.quantile(obs, np.linspace(0.0, 1.0, N_BINS + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(obs, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < MIN_BIN_ROWS:
            continue
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "bin_center": float(np.median(obs[m])), "rows": int(m.sum()),
            "mae_raw": float(np.abs(raw[m] - obs[m]).mean()),
            "mae_corrected": float(np.abs(corr[m] - obs[m]).mean()),
            "bias_raw": float((raw[m] - obs[m]).mean()),
            "bias_corrected": float((corr[m] - obs[m]).mean()),
        })
    out = pd.DataFrame(rows)
    out["mae_improvement"] = out["mae_raw"] - out["mae_corrected"]
    return out


def _plot_conditional_skill(tname: str, table_global: pd.DataFrame, table_gom: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    for col, (label, tab) in enumerate([("global ocean", table_global), ("Gulf of Mexico", table_gom)]):
        ax = axes[0, col]
        ax.plot(tab["bin_center"], tab["mae_raw"], marker="o", color="#dc2626", linewidth=2, label="raw RTOFS")
        ax.plot(tab["bin_center"], tab["mae_corrected"], marker="o", color="#2563eb", linewidth=2, label="corrected")
        ax.set_title(f"{SHORT[tname]} error vs observed value — {label}")
        ax.set_xlabel(f"observed {SHORT[tname]} ({UNITS[tname]})")
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.grid(True, alpha=0.15)
        ax.legend()

        ax2 = axes[1, col]
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in tab["mae_improvement"]]
        ax2.bar(tab["bin_center"], tab["mae_improvement"],
                width=0.8 * np.gradient(tab["bin_center"].to_numpy(float)), color=colors)
        ax2.axhline(0.0, color="black", linewidth=1.2)
        ax2.set_title("correction benefit (raw MAE − corrected MAE)")
        ax2.set_xlabel(f"observed {SHORT[tname]} ({UNITS[tname]})")
        ax2.set_ylabel(f"MAE reduction ({UNITS[tname]})")
        ax2.grid(True, axis="y", alpha=0.15)
        for _, r in tab.iterrows():
            ax2.annotate(f"{int(r['rows'])}", (r["bin_center"], 0), textcoords="offset points",
                         xytext=(0, 4 if r["mae_improvement"] < 0 else -12), ha="center", fontsize=8, alpha=0.7)
    fig.suptitle(
        f"{SHORT[tname]}: where the correction helps and where it does not\n"
        "Blue bars = correction better, red bars = raw RTOFS better; numbers are rows per bin"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _crossover(tab: pd.DataFrame) -> float | None:
    """Observed value above which the correction stops helping (last sign change)."""
    sign = np.sign(tab["mae_improvement"].to_numpy(float))
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] < 0 <= sign[i - 1]:
            return float(tab["bin_lo"].iloc[i])
    return None


# ---------------------------------------------------------------- items 2 & 3

def _plot_shap_maps(tname, gom, shap_g, shap_l, cols, feature, out_path):
    j = cols.index(feature)
    vlim = float(np.nanquantile(np.abs(np.concatenate([shap_g[:, j], shap_l[:, j]])), 0.98))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True)
    for ax, vals, label in [
        (axes[0], shap_g[:, j], "global model"),
        (axes[1], shap_l[:, j], "Gulf-local model"),
    ]:
        sc = ax.scatter(gom["lon"], gom["lat"], c=vals, s=13, cmap="RdBu_r", norm=norm, rasterized=True)
        add_land_overlay(ax, zorder=2)
        ax.set_xlim(GOM["lon_min"] - 1, GOM["lon_max"] + 1)
        ax.set_ylim(GOM["lat_min"] - 1, GOM["lat_max"] + 1)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{label}")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(f"{_short_name(feature)} contribution ({UNITS[tname]})")
    fig.suptitle(
        f"{SHORT[tname]}: where the {_short_name(feature)} signal drives the correction\n"
        "Red = pushes the value up at that location, blue = pushes it down"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_colored_dependence(tname, gom, shap_l, cols, feature, color_specs, out_path):
    j = cols.index(feature)
    x = pd.to_numeric(gom[feature], errors="coerce").to_numpy(float)
    phi = shap_l[:, j]
    fig, axes = plt.subplots(1, len(color_specs), figsize=(9 * len(color_specs), 7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (ccol, cmap, clabel) in zip(axes, color_specs):
        c = pd.to_numeric(gom[ccol], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(phi) & np.isfinite(c)
        sc = ax.scatter(x[ok], phi[ok], c=c[ok], s=14, cmap=cmap, alpha=0.85, rasterized=True)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel(_short_name(feature))
        ax.set_ylabel(f"{_short_name(feature)} contribution ({UNITS[tname]})")
        ax.set_title(f"coloured by {clabel}")
        ax.grid(True, alpha=0.15)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(clabel)
    fig.suptitle(
        f"{SHORT[tname]} (Gulf-local model): what splits the {_short_name(feature)} response into branches"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    summary: dict = {"run_date": RUN_DATE, "targets": {}}

    for target in TARGETS:
        tname = target.name
        # ---- item 1: conditional skill (global + Gulf), on the locked OOF table
        pred = pd.read_parquet(PRED_PATHS[tname])
        tab_global = _conditional_skill(pred, OBS_COL[tname], "pred_obs__raw_rtofs", BEST_PRED_COL[tname])
        tab_gom = _conditional_skill(pred[_in_gom(pred)], OBS_COL[tname], "pred_obs__raw_rtofs", BEST_PRED_COL[tname])
        tab_global.to_csv(OUT_DIR / f"{tname}_conditional_skill_global.csv", index=False)
        tab_gom.to_csv(OUT_DIR / f"{tname}_conditional_skill_gom.csv", index=False)
        p1 = OUT_DIR / f"{tname}_conditional_skill.png"
        _plot_conditional_skill(tname, tab_global, tab_gom, p1)

        # ---- items 2 & 3: rebuild the Gulf SHAP matrices (same folds/models)
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        gom = work[_in_gom(work)].copy().reset_index(drop=True)
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        folds = _build_forward_folds(
            sorted(gom["date"].dt.strftime("%Y%m%d").unique().tolist()),
            n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"],
        )
        shap_g, shap_l, _, _ = _fold_models_and_shap(work, gom, folds, cols, target)

        p2 = OUT_DIR / f"{tname}_ssh_shap_map.png"
        _plot_shap_maps(tname, gom, shap_g, shap_l, cols, "model_ssh_m", p2)
        p3 = OUT_DIR / f"{tname}_ssh_dependence_coloured.png"
        _plot_colored_dependence(
            tname, gom, shap_l, cols, "model_ssh_m",
            [("lon", "coolwarm", "longitude (°E)"), ("month_int", "twilight", "month")],
            p3,
        )

        summary["targets"][tname] = {
            "gom_rows": int(len(gom)),
            "crossover_global": _crossover(tab_global),
            "crossover_gom": _crossover(tab_gom),
            "worst_bin_global": tab_global.loc[tab_global["mae_improvement"].idxmin()].to_dict(),
            "figures": [str(p1), str(p2), str(p3)],
        }
        print(f"{tname}: crossover global={summary['targets'][tname]['crossover_global']}, "
              f"gom={summary['targets'][tname]['crossover_gom']}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: {"crossover_global": v["crossover_global"], "crossover_gom": v["crossover_gom"]}
                      for k, v in summary["targets"].items()}, indent=2))


if __name__ == "__main__":
    main()
