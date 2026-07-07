"""Build the mentor-facing spatial-ablation presentation figures.

Per target (TCHP, D26), on the full-calendar locked OOF prediction tables:

1. `<t>_observed_points_map.png` — where the truth lives: collocated points
   colored by the observed value, with a monthly support strip underneath.
2. `<t>_spatial_ablation_density.png` — 3-panel observed-vs-model density
   (raw RTOFS | corrected without any coordinate features | corrected best
   recipe with lat/lon/abs_lat + interactions), shared axes and color scale.
3. `<t>_correction_magnitude_map.png` — points colored by corrected - raw,
   showing the geographic structure of the learned correction.

Shared across targets:

4. `spatial_ablation_stats.png` — MAE / RMSE / |bias| bars for the three
   variants per target, with date-block bootstrap 95% intervals.

The 3-variant comparison uses the locked blocked-forward protocol; panels 2
and 3 of the density figure will look nearly identical by eye — the stats
figure carries the quantitative message.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.build_hhp_density_scatter_diagnostics import (  # noqa: E402
    _common_axis_limits,
    _density_color_limits,
    _density_image,
    _plot_density_panel,
)

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/presentation_spatial_ablation_2024_2025")
BASE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
PRED_PATHS = {
    "tchp": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet"),
    "d26": Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet"),
}
TARGETS = {
    "tchp": {
        "obs_col": "argo_tchp_kj_per_cm2",
        "raw_col": "pred_obs__raw_rtofs",
        "no_spatial_col": "pred_obs__global_pruned_no_spatial",
        "best_col": "pred_obs__global_pruned",
        "best_label": "global_pruned",
        "units": "kJ/cm²",
        "short": "TCHP",
    },
    "d26": {
        "obs_col": "argo_d26_m",
        "raw_col": "pred_obs__raw_rtofs",
        "no_spatial_col": "pred_obs__drop_both_lat_interactions_no_spatial",
        "best_col": "pred_obs__drop_both_lat_interactions",
        "best_label": "drop_both_lat_interactions",
        "units": "m",
        "short": "D26",
    },
}
VARIANT_LABELS = ["raw RTOFS", "corrected, no coordinate features", "corrected, best recipe (lat/lon/|lat|)"]
VARIANT_COLORS = ["#dc2626", "#f59e0b", "#2563eb"]
N_BOOT = 1000


def _load(target_key: str) -> pd.DataFrame:
    t = TARGETS[target_key]
    df = pd.read_parquet(PRED_PATHS[target_key]).copy()
    needed = [t["obs_col"], t["raw_col"], t["no_spatial_col"], t["best_col"]]
    df = df[np.isfinite(df[needed]).all(axis=1)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _plot_observed_points_map(target_key: str, df: pd.DataFrame, out_path: Path) -> None:
    """Map all finite observed rows; the strip contrasts full data support with
    the OOF-evaluated subset (the earliest blocked-forward block is train-only,
    so its months have data but no out-of-fold predictions)."""
    t = TARGETS[target_key]
    base = pd.read_parquet(BASE_PATH, columns=["date", "lat", "lon", t["obs_col"]])
    base = base[np.isfinite(base[t["obs_col"]])].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")

    fig, (ax_map, ax_strip) = plt.subplots(
        2, 1, figsize=(16, 9.6), constrained_layout=True, height_ratios=[4, 1]
    )
    obs = base[t["obs_col"]].to_numpy(float)
    vmax = float(np.nanquantile(obs, 0.99))
    sc = ax_map.scatter(base["lon"], base["lat"], c=obs, s=2.0, cmap="turbo", vmin=0.0, vmax=vmax, rasterized=True)
    add_land_overlay(ax_map, zorder=2)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-70, 70)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title(
        f"Observed Argo {t['short']} at collocated points, 2024-2025 full calendar\n"
        f"n={len(base):,} rows over {base['date'].nunique()} days"
    )
    cbar = fig.colorbar(sc, ax=ax_map, shrink=0.9, pad=0.01)
    cbar.set_label(f"Observed {t['short']} ({t['units']})")

    months = pd.period_range("2024-01", "2025-12", freq="M")
    x = np.arange(len(months))
    per_month_all = pd.to_datetime(base["date"]).dt.to_period("M").value_counts()
    per_month_oof = pd.to_datetime(df["date"]).dt.to_period("M").value_counts()
    counts_all = [int(per_month_all.get(m, 0)) for m in months]
    counts_oof = [int(per_month_oof.get(m, 0)) for m in months]
    ax_strip.bar(x, counts_all, color="#cbd5e1", width=0.85, label="all collocated rows")
    ax_strip.bar(x, counts_oof, color="#2563eb", width=0.85, label="evaluated OOF rows (earliest block is train-only)")
    ax_strip.set_xticks(x)
    ax_strip.set_xticklabels([str(m) for m in months], rotation=90, fontsize=7)
    ax_strip.set_ylabel("rows / month")
    ax_strip.set_title("Monthly support: data vs locked-protocol OOF evaluation", fontsize=10)
    ax_strip.legend(loc="upper left", fontsize=8)
    ax_strip.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_spatial_ablation_density(target_key: str, df: pd.DataFrame, out_path: Path) -> list[dict]:
    t = TARGETS[target_key]
    obs = df[t["obs_col"]].to_numpy(float)
    preds = [df[t["raw_col"]].to_numpy(float), df[t["no_spatial_col"]].to_numpy(float), df[t["best_col"]].to_numpy(float)]

    xlim, ylim = _common_axis_limits(obs, preds[0], preds[2])
    imgs = [_density_image(obs, p, xlim=xlim, ylim=ylim)[0] for p in preds]
    color_limits = _density_color_limits(*imgs)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), constrained_layout=True)
    mesh = None
    stats_rows = []
    for ax, pred, label in zip(axes, preds, VARIANT_LABELS):
        mesh, m, _ = _plot_density_panel(
            ax, obs, pred,
            title=f"{t['short']}: {label}",
            xlabel=f"Observed {t['short']} ({t['units']})",
            ylabel=f"Model {t['short']} ({t['units']})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        stats_rows.append({"target": target_key, "variant": label, **m})
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
    cbar.set_label("log10(PDF)")
    fig.suptitle(
        f"{t['short']} observed vs model — spatial-feature ablation "
        f"(locked blocked-forward OOF, n={len(df):,}, {df['date'].nunique()} days)",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return stats_rows


def _plot_correction_magnitude_map(target_key: str, df: pd.DataFrame, out_path: Path) -> None:
    t = TARGETS[target_key]
    delta = df[t["best_col"]].to_numpy(float) - df[t["raw_col"]].to_numpy(float)
    vlim = float(np.nanquantile(np.abs(delta), 0.98))
    fig, ax = plt.subplots(figsize=(16, 7.2), constrained_layout=True)
    sc = ax.scatter(df["lon"], df["lat"], c=delta, s=2.0, cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
    add_land_overlay(ax, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 70)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{t['short']}: learned correction (corrected best recipe − raw RTOFS) at collocated points\n"
        "Red = model raised the raw value, blue = lowered"
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.01)
    cbar.set_label(f"Correction ({t['units']})")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _bootstrap_metrics(df: pd.DataFrame, obs_col: str, pred_col: str, rng: np.random.Generator) -> dict:
    err = df[pred_col].to_numpy(float) - df[obs_col].to_numpy(float)
    dates = df["date"].to_numpy()
    unique_dates = np.unique(dates)
    idx_by_date = {d: np.where(dates == d)[0] for d in unique_dates}
    point = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())), "abs_bias": float(abs(err.mean()))}
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        sample_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        e = np.concatenate([err[idx_by_date[d]] for d in sample_dates])
        boot["mae"].append(np.abs(e).mean())
        boot["rmse"].append(np.sqrt((e ** 2).mean()))
        boot["abs_bias"].append(abs(e.mean()))
    out = {}
    for k in point:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = {"point": point[k], "lo": float(lo), "hi": float(hi)}
    return out


def _plot_stats(all_stats: dict, out_path: Path) -> None:
    metrics = [("mae", "MAE"), ("rmse", "RMSE"), ("abs_bias", "|Bias|")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    width = 0.25
    targets = list(TARGETS.keys())
    x = np.arange(len(targets))
    for ax, (mkey, mlabel) in zip(axes, metrics):
        for vi, vlabel in enumerate(VARIANT_LABELS):
            vals = [all_stats[t][vlabel][mkey]["point"] for t in targets]
            los = [all_stats[t][vlabel][mkey]["point"] - all_stats[t][vlabel][mkey]["lo"] for t in targets]
            his = [all_stats[t][vlabel][mkey]["hi"] - all_stats[t][vlabel][mkey]["point"] for t in targets]
            ax.bar(
                x + (vi - 1) * width, vals, width,
                yerr=[los, his], capsize=3,
                color=VARIANT_COLORS[vi], label=vlabel if mkey == "mae" else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([TARGETS[t]["short"] + f" ({TARGETS[t]['units']})" for t in targets])
        ax.set_title(mlabel)
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Spatial-feature ablation, locked blocked-forward OOF (2024-2025 full calendar)\n"
        "Error bars: 95% interval from date-block bootstrap "
        f"({N_BOOT} resamples of whole days)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_stats: dict = {}
    manifest: dict = {"figures": {}}
    for target_key, t in TARGETS.items():
        df = _load(target_key)
        points_map = OUT_DIR / f"{target_key}_observed_points_map.png"
        density = OUT_DIR / f"{target_key}_spatial_ablation_density.png"
        corr_map = OUT_DIR / f"{target_key}_correction_magnitude_map.png"
        _plot_observed_points_map(target_key, df, points_map)
        _plot_spatial_ablation_density(target_key, df, density)
        _plot_correction_magnitude_map(target_key, df, corr_map)

        all_stats[target_key] = {}
        for vlabel, col in zip(VARIANT_LABELS, [t["raw_col"], t["no_spatial_col"], t["best_col"]]):
            all_stats[target_key][vlabel] = _bootstrap_metrics(df, t["obs_col"], col, rng)
        manifest["figures"][target_key] = {
            "observed_points_map": str(points_map),
            "spatial_ablation_density": str(density),
            "correction_magnitude_map": str(corr_map),
            "rows": int(len(df)),
            "dates": int(df["date"].nunique()),
        }
        print(f"{target_key}: figures done ({len(df):,} rows)")

    stats_path = OUT_DIR / "spatial_ablation_stats.png"
    _plot_stats(all_stats, stats_path)
    manifest["figures"]["stats"] = str(stats_path)
    manifest["stats"] = all_stats
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {vk: vv["mae"] for vk, vv in v.items()} for k, v in all_stats.items()}, indent=2))


if __name__ == "__main__":
    main()
