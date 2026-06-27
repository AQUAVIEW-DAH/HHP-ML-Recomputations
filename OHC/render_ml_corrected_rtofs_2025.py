"""Render 2025 seasonal maps for ML-corrected RTOFS and Argo-minus-corrected differences.

Uses the best year-holdout XGBoost predictions:
- train on 2024 collocated rows
- test on 2025 collocated rows

Outputs:
- point maps for corrected RTOFS and Argo-minus-corrected fields
- gaussian interpolated winter/summer panels on a 0.25 degree grid
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (  # noqa: E402
    PARAMS,
    SEASONS,
    add_land_overlay,
    build_global_grid,
    gaussian_interpolate,
    linear_nd_interpolate,
    make_norm,
    rbf_gaussian_interpolate,
)

YEAR = 2025
MODEL_LABEL = "XGBoost correction (train 2024 -> test 2025)"
OUT_DIR = Path(__file__).resolve().parent / "output" / "ml_corrected_rtofs_2025"
POINTS_DIR = OUT_DIR / "points"
INTERP_DIR = OUT_DIR / "interpolated"
LINEAR_DIR = OUT_DIR / "linear_nd"
RBF_DIR = OUT_DIR / "rbf_gaussian"
DATA_DIR = OUT_DIR / "data"

TCHP_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "ml_benchmarks"
    / "year_holdout_sweep_best_predictions_tchp.parquet"
)
D26_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "ml_benchmarks"
    / "year_holdout_sweep_best_predictions_d26.parquet"
)

SEASON_KEYS = ["winter_jfm", "summer_jas"]
GRID_RES_DEG = 0.25
MASK_DISTANCE_KM = 100.0
METHOD_DIRS = {
    "gaussian": INTERP_DIR,
    "linear_nd": LINEAR_DIR,
    "rbf_gaussian": RBF_DIR,
}

POSITIVE_FIELDS = {
    "raw_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "raw_d26_m": "d26_m",
    "corrected_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "corrected_d26_m": "d26_m",
}


def _delta_norm(values: np.ndarray) -> TwoSlopeNorm:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = np.percentile(np.abs(vals), 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(np.abs(vals)) if len(vals) else 1.0
    vmax = float(max(vmax, 1e-6))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _load_collocated_predictions() -> pd.DataFrame:
    tchp = pd.read_parquet(TCHP_PATH).copy()
    d26 = pd.read_parquet(D26_PATH).copy()

    tchp = tchp.rename(
        columns={
            "pred_obs__xgb_sweep_best_2024_train_2025_test": "corrected_tchp_kj_per_cm2",
            "pred_obs__raw_rtofs": "raw_tchp_kj_per_cm2",
        }
    )
    d26 = d26.rename(
        columns={
            "pred_obs__xgb_sweep_best_2024_train_2025_test": "corrected_d26_m",
            "pred_obs__raw_rtofs": "raw_d26_m",
        }
    )

    join_cols = ["date", "year", "month", "lat", "lon", "region_group", "season_group"]
    merged = tchp[
        join_cols
        + [
            "argo_tchp_kj_per_cm2",
            "model_interp_tchp_kj_per_cm2",
            "raw_tchp_kj_per_cm2",
            "corrected_tchp_kj_per_cm2",
        ]
    ].merge(
        d26[
            join_cols
            + [
                "argo_d26_m",
                "model_interp_d26_m",
                "raw_d26_m",
                "corrected_d26_m",
            ]
        ],
        on=join_cols,
        how="inner",
    )

    merged = merged[merged["season_group"].isin(SEASON_KEYS)].copy()
    merged["season"] = merged["season_group"]
    merged["delta_raw_tchp_kj_per_cm2"] = merged["argo_tchp_kj_per_cm2"] - merged["raw_tchp_kj_per_cm2"]
    merged["delta_raw_d26_m"] = merged["argo_d26_m"] - merged["raw_d26_m"]
    merged["delta_corrected_tchp_kj_per_cm2"] = (
        merged["argo_tchp_kj_per_cm2"] - merged["corrected_tchp_kj_per_cm2"]
    )
    merged["delta_corrected_d26_m"] = merged["argo_d26_m"] - merged["corrected_d26_m"]
    return merged.reset_index(drop=True)


def _render_point_panels(df: pd.DataFrame, field: str, label: str, out_path: Path, *, delta: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    if delta:
        norm = _delta_norm(df[field].to_numpy(float))
        cmap = "RdBu_r"
    else:
        param_key = POSITIVE_FIELDS[field]
        norm = make_norm(param_key)
        cmap = PARAMS[param_key].cmap

    mappable = None
    for ax, season_key in zip(axes, SEASON_KEYS):
        sdf = df[df["season"] == season_key].copy()
        season_label = SEASONS[season_key][0]
        add_land_overlay(ax, zorder=0)
        sc = ax.scatter(
            sdf["lon"],
            sdf["lat"],
            c=sdf[field],
            s=5,
            alpha=0.8,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            zorder=2,
        )
        mappable = sc
        ax.set_title(f"{season_label}\n{len(sdf):,} collocated observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    fig.suptitle(f"{label} seasonal point maps ({YEAR})\n{MODEL_LABEL}", fontsize=14)
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_interpolated_panels(df: pd.DataFrame, field: str, label: str, out_path: Path, *, delta: bool = False) -> None:
    raise RuntimeError("Use _render_interpolated_panels_method instead")


def _interpolate(method: str, lat, lon, values, grid_lat, grid_lon, mask_distance_km):
    if method == "gaussian":
        return gaussian_interpolate(
            lat,
            lon,
            values,
            grid_lat,
            grid_lon,
            mask_distance_km=mask_distance_km,
        )
    if method == "linear_nd":
        return linear_nd_interpolate(
            lat,
            lon,
            values,
            grid_lat,
            grid_lon,
            mask_distance_km=mask_distance_km,
        )
    if method == "rbf_gaussian":
        return rbf_gaussian_interpolate(
            lat,
            lon,
            values,
            grid_lat,
            grid_lon,
            mask_distance_km=mask_distance_km,
        )
    raise ValueError(method)


def _render_interpolated_panels_method(
    df: pd.DataFrame,
    field: str,
    label: str,
    out_path: Path,
    *,
    method: str,
    mask_distance_km: float | None,
    delta: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    if delta:
        norm = _delta_norm(df[field].to_numpy(float))
        cmap = "RdBu_r"
    else:
        param_key = POSITIVE_FIELDS[field]
        norm = make_norm(param_key)
        cmap = PARAMS[param_key].cmap

    grid_lat, grid_lon = build_global_grid(GRID_RES_DEG)
    mappable = None
    for ax, season_key in zip(axes, SEASON_KEYS):
        sdf = df[df["season"] == season_key].copy()
        season_label = SEASONS[season_key][0]
        vals = sdf[field].to_numpy(float)
        valid = np.isfinite(sdf["lat"].to_numpy(float)) & np.isfinite(sdf["lon"].to_numpy(float)) & np.isfinite(vals)
        sdf = sdf.loc[valid]
        vals = vals[valid]
        if len(sdf) == 0:
            field_grid = np.full_like(grid_lat, np.nan, dtype=float)
        else:
            field_grid = _interpolate(
                method,
                sdf["lat"].to_numpy(float),
                sdf["lon"].to_numpy(float),
                vals,
                grid_lat,
                grid_lon,
                mask_distance_km,
            )
        artist = ax.pcolormesh(
            grid_lon,
            grid_lat,
            ma.masked_invalid(field_grid),
            shading="auto",
            cmap=cmap,
            norm=norm,
            zorder=1,
        )
        mappable = artist
        add_land_overlay(ax, zorder=5)
        ax.set_title(f"{season_label}\n{len(sdf):,} collocated observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    mask_label = f"masked_{int(mask_distance_km)}km" if mask_distance_km is not None else "nomask"
    fig.suptitle(
        f"{label} seasonal interpolated maps ({YEAR})\n{MODEL_LABEL}; method={method}, grid={GRID_RES_DEG}°, {mask_label}",
        fontsize=13,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    for d in [OUT_DIR, POINTS_DIR, INTERP_DIR, LINEAR_DIR, RBF_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = _load_collocated_predictions()
    colloc_path = DATA_DIR / f"ml_corrected_rtofs_{YEAR}_winter_summer_collocated.parquet"
    df.to_parquet(colloc_path, index=False)
    df.to_csv(colloc_path.with_suffix(".csv"), index=False)

    point_tasks = [
        ("raw_tchp_kj_per_cm2", "Raw RTOFS-at-Argo TCHP / OHC (kJ/cm²)", False),
        ("raw_d26_m", "Raw RTOFS-at-Argo D26 (m)", False),
        ("delta_raw_tchp_kj_per_cm2", "Argo - raw RTOFS TCHP difference (kJ/cm²)", True),
        ("delta_raw_d26_m", "Argo - raw RTOFS D26 difference (m)", True),
        ("corrected_tchp_kj_per_cm2", "ML-corrected RTOFS TCHP / OHC (kJ/cm²)", False),
        ("corrected_d26_m", "ML-corrected RTOFS D26 (m)", False),
        ("delta_corrected_tchp_kj_per_cm2", "Argo - ML-corrected RTOFS TCHP difference (kJ/cm²)", True),
        ("delta_corrected_d26_m", "Argo - ML-corrected RTOFS D26 difference (m)", True),
    ]
    for field, label, is_delta in point_tasks:
        _render_point_panels(
            df,
            field,
            label,
            POINTS_DIR / f"{field}_{YEAR}_xgb_train2024_test2025_points_winter_summer.png",
            delta=is_delta,
        )
        for method, out_dir in METHOD_DIRS.items():
            mask_options = [MASK_DISTANCE_KM] if method == "gaussian" else [MASK_DISTANCE_KM, None]
            for mask_distance_km in mask_options:
                mask_tag = f"masked_{int(mask_distance_km)}km" if mask_distance_km is not None else "nomask"
                _render_interpolated_panels_method(
                    df,
                    field,
                    label,
                    out_dir / f"{field}_{YEAR}_xgb_train2024_test2025_{method}_0p25deg_{mask_tag}_grid.png",
                    method=method,
                    mask_distance_km=mask_distance_km,
                    delta=is_delta,
                )

    summary = {
        "year": YEAR,
        "model_label": MODEL_LABEL,
        "collocated_rows": int(len(df)),
        "collocated_dates": int(df["date"].nunique()),
        "season_counts": {k: int((df["season"] == k).sum()) for k in SEASON_KEYS},
        "grid_resolution_deg": GRID_RES_DEG,
        "mask_distance_km": MASK_DISTANCE_KM,
        "outputs": {
            "collocated_parquet": str(colloc_path),
            "points_dir": str(POINTS_DIR),
            "method_dirs": {k: str(v) for k, v in METHOD_DIRS.items()},
        },
    }
    (DATA_DIR / f"summary_ml_corrected_rtofs_{YEAR}_winter_summer.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
