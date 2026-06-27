"""Render interpolated winter/summer panels from collocated Argo-RTOFS points.

Uses the saved 2024 collocated Argo/RTOFS point table and generates
comparison-ready interpolated maps for:
- RTOFS-at-Argo-point fields
- Argo-minus-RTOFS difference fields

Supported interpolation methods mirror the Argo-side seasonal products:
- gaussian
- linear_nd
- rbf_gaussian
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (
    PARAMS,
    SEASONS,
    add_land_overlay,
    build_global_grid,
    gaussian_interpolate,
    linear_nd_interpolate,
    make_norm,
    rbf_gaussian_interpolate,
)

YEAR = 2024
COLLOC_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "rtofs_at_argo_points_2024"
    / "data"
    / f"argo_rtofs_collocated_{YEAR}_winter_summer.parquet"
)
OUT_ROOT = Path(__file__).resolve().parent / "output" / "rtofs_at_argo_points_2024"

POSITIVE_FIELDS = {
    "model_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "model_d26_m": "d26_m",
}

DELTA_LABELS = {
    "delta_tchp_kj_per_cm2": "Argo - RTOFS TCHP difference (kJ/cm²)",
    "delta_d26_m": "Argo - RTOFS D26 difference (m)",
}

METHOD_DIRS = {
    "gaussian": "interpolated",
    "linear_nd": "linear_nd",
    "rbf_gaussian": "rbf_gaussian",
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
    raise ValueError(f"Unknown interpolation method: {method}")


def render_panels(
    df: pd.DataFrame,
    *,
    field: str,
    label: str,
    method: str,
    resolution_deg: float,
    mask_distance_km: float | None,
    smooth_display: bool,
    out_path: Path,
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

    grid_lat, grid_lon = build_global_grid(resolution_deg)
    mappable = None
    for ax, season_key in zip(axes, ("winter_jfm", "summer_jas")):
        season_label = SEASONS[season_key][0]
        sdf = df[df["season"] == season_key].copy()
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

        masked_field = ma.masked_invalid(field_grid)
        if smooth_display:
            artist = ax.imshow(
                masked_field,
                origin="lower",
                extent=[-180, 180, -90, 90],
                cmap=cmap,
                norm=norm,
                interpolation="bilinear",
                aspect="auto",
                zorder=1,
            )
        else:
            artist = ax.pcolormesh(
                grid_lon,
                grid_lat,
                masked_field,
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
    style_label = "smooth" if smooth_display else "grid"
    fig.suptitle(
        f"{label} seasonal interpolated maps ({YEAR})\nmethod={method}, grid={resolution_deg}°, {mask_label}, style={style_label}",
        fontsize=14,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gaussian", "linear_nd", "rbf_gaussian"], required=True)
    parser.add_argument("--resolution-deg", type=float, default=0.25)
    parser.add_argument("--mask-distance-km", type=float, default=100.0)
    parser.add_argument("--smooth-display", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(COLLOC_PATH)
    out_dir = OUT_ROOT / METHOD_DIRS[args.method]
    out_dir.mkdir(parents=True, exist_ok=True)
    res_tag = str(args.resolution_deg).replace(".", "p")
    mask_tag = f"masked_{int(args.mask_distance_km)}km"
    style_tag = "smooth" if args.smooth_display else "grid"

    tasks = [
        ("model_tchp_kj_per_cm2", "RTOFS TCHP / OHC (kJ/cm²) at Argo points", False),
        ("model_d26_m", "RTOFS D26 (m) at Argo points", False),
        ("delta_tchp_kj_per_cm2", "Argo - RTOFS TCHP difference (kJ/cm²)", True),
        ("delta_d26_m", "Argo - RTOFS D26 difference (m)", True),
    ]
    for field, label, is_delta in tasks:
        out_path = out_dir / f"{field}_{YEAR}_{args.method}_{res_tag}deg_{mask_tag}_{style_tag}.png"
        render_panels(
            df,
            field=field,
            label=label,
            method=args.method,
            resolution_deg=args.resolution_deg,
            mask_distance_km=args.mask_distance_km,
            smooth_display=args.smooth_display,
            out_path=out_path,
            delta=is_delta,
        )
        print(out_path)


if __name__ == "__main__":
    main()
