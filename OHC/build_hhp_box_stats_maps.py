"""Global maps of error statistics per 20-degree box (mentor request).

Instead of stacking dozens of per-box scatter panels, render one world map per
statistic where every 20-degree box is colored by its value, computed from the
locked out-of-fold predictions. Per target this produces a 4x2 panel figure:

    row 1: MAE, raw RTOFS          | MAE, corrected      (shared color scale)
    row 2: bias, raw RTOFS         | bias, corrected     (shared diverging scale)
    row 3: correlation, raw RTOFS  | correlation, corrected (shared scale)
    row 4: valid rows per box      | MAE improvement (raw - corrected)

plus a CSV of all per-box values.
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
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/box_stats_maps_2024_2025")
PRED_PATHS = {
    "tchp": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet"),
    "d26": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet"),
}
TARGETS = {
    "tchp": ("argo_tchp_kj_per_cm2", "pred_obs__raw_rtofs", "pred_obs__global_pruned", "kJ/cm²", "TCHP"),
    "d26": ("argo_d26_m", "pred_obs__raw_rtofs", "pred_obs__drop_both_lat_interactions", "m", "D26"),
}
BOX_DEG = 20
MIN_ROWS = 25


def _box_stats(df: pd.DataFrame, obs_col: str, raw_col: str, corr_col: str) -> pd.DataFrame:
    work = df.copy()
    work["lat0"] = np.floor(work["lat"].to_numpy(float) / BOX_DEG) * BOX_DEG
    work["lon0"] = np.floor(((work["lon"].to_numpy(float) + 180.0) % 360.0 - 180.0) / BOX_DEG) * BOX_DEG
    rows = []
    for (lat0, lon0), g in work.groupby(["lat0", "lon0"]):
        obs = g[obs_col].to_numpy(float)
        raw = g[raw_col].to_numpy(float)
        corr = g[corr_col].to_numpy(float)
        ok = np.isfinite(obs) & np.isfinite(raw) & np.isfinite(corr)
        if ok.sum() < MIN_ROWS:
            continue
        o, r, c = obs[ok], raw[ok], corr[ok]
        rows.append({
            "lat0": float(lat0), "lon0": float(lon0), "rows": int(ok.sum()),
            "mae_raw": float(np.abs(r - o).mean()),
            "mae_corrected": float(np.abs(c - o).mean()),
            "bias_raw": float((r - o).mean()),
            "bias_corrected": float((c - o).mean()),
            "corr_raw": float(pd.Series(o).corr(pd.Series(r))),
            "corr_corrected": float(pd.Series(o).corr(pd.Series(c))),
        })
    out = pd.DataFrame(rows)
    out["mae_improvement"] = out["mae_raw"] - out["mae_corrected"]
    return out


def _paint(ax, stats: pd.DataFrame, col: str, cmap, norm, label: str, units: str) -> None:
    add_land_overlay(ax, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 70)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    for _, row in stats.iterrows():
        ax.add_patch(Rectangle(
            (row["lon0"], row["lat0"]), BOX_DEG, BOX_DEG,
            facecolor=mapper.to_rgba(row[col]), edgecolor="white", linewidth=0.4, alpha=0.95, zorder=1,
        ))
    ax.set_title(label)
    cbar = plt.colorbar(mapper, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(units)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for tname, (obs_col, raw_col, corr_col, units, short) in TARGETS.items():
        df = pd.read_parquet(PRED_PATHS[tname])
        stats = _box_stats(df, obs_col, raw_col, corr_col)
        csv_path = OUT_DIR / f"{tname}_box_stats.csv"
        stats.to_csv(csv_path, index=False)

        fig, axes = plt.subplots(4, 2, figsize=(22, 22), constrained_layout=True)

        mae_max = float(np.nanquantile(stats[["mae_raw", "mae_corrected"]].to_numpy(), 0.98))
        mae_norm = Normalize(vmin=0.0, vmax=mae_max)
        _paint(axes[0, 0], stats, "mae_raw", "viridis", mae_norm, f"MAE, raw RTOFS ({short})", units)
        _paint(axes[0, 1], stats, "mae_corrected", "viridis", mae_norm, f"MAE, corrected ({short})", units)

        bias_lim = float(np.nanquantile(np.abs(stats[["bias_raw", "bias_corrected"]].to_numpy()), 0.98))
        bias_norm = TwoSlopeNorm(vcenter=0.0, vmin=-bias_lim, vmax=bias_lim)
        _paint(axes[1, 0], stats, "bias_raw", "RdBu_r", bias_norm, f"Bias (model − Argo), raw RTOFS ({short})", units)
        _paint(axes[1, 1], stats, "bias_corrected", "RdBu_r", bias_norm, f"Bias (model − Argo), corrected ({short})", units)

        corr_lo = float(np.nanquantile(stats[["corr_raw", "corr_corrected"]].to_numpy(), 0.02))
        corr_norm = Normalize(vmin=max(0.0, corr_lo), vmax=1.0)
        _paint(axes[2, 0], stats, "corr_raw", "magma", corr_norm, f"Correlation with Argo, raw RTOFS ({short})", "correlation")
        _paint(axes[2, 1], stats, "corr_corrected", "magma", corr_norm, f"Correlation with Argo, corrected ({short})", "correlation")

        n_norm = LogNorm(vmin=max(MIN_ROWS, 1), vmax=float(stats["rows"].max()))
        _paint(axes[3, 0], stats, "rows", "cividis", n_norm, "Valid rows per box", "rows")
        imp_lim = float(np.nanquantile(np.abs(stats["mae_improvement"]), 0.98))
        imp_norm = TwoSlopeNorm(vcenter=0.0, vmin=-imp_lim, vmax=imp_lim)
        _paint(axes[3, 1], stats, "mae_improvement", "RdBu_r", imp_norm,
               "MAE improvement (raw − corrected); red = correction helps", units)

        fig.suptitle(
            f"{short}: error statistics per {BOX_DEG}° box, locked out-of-fold evaluation "
            f"(boxes with ≥{MIN_ROWS} valid rows)"
        )
        fig_path = OUT_DIR / f"{tname}_box_stats_maps.png"
        fig.savefig(fig_path, dpi=170)
        plt.close(fig)
        manifest[tname] = {"figure": str(fig_path), "csv": str(csv_path), "boxes": int(len(stats))}
        print(f"{tname}: {len(stats)} boxes")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
