"""Render paper-style Argo steric-height mean and Argo-minus-WOA difference panels.

This is a validation-oriented plotting utility:
- load a processed Argo steric-height parquet table
- interpolate the Argo 0/2000 dbar steric height onto a 1 degree 0..360 grid
- derive a WOA23 proxy reference steric-height field from annual T/S climatology
- render paper-style mean and difference panels in dyn cm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gsw
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.build_argo_steric_height_2024 import _add_land_overlay_0_360, _build_grid
from OHC.seasonal_map_common import gaussian_interpolate


REF_TEMP = Path(__file__).resolve().parent / "output" / "steric_height" / "reference_cache" / "woa23_decav_t00_01.nc"
REF_SALT = Path(__file__).resolve().parent / "output" / "steric_height" / "reference_cache" / "woa23_decav_s00_01.nc"


def _argo_field_dyn_cm(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_lat, grid_lon = _build_grid(0.0, 360.0, 1.0)
    values_m = pd.to_numeric(df["steric_0_2000_m"], errors="coerce").to_numpy(float)
    mask = np.isfinite(values_m) & (values_m >= 0.3) & (values_m <= 3.2)
    sdf = df.loc[mask].copy()
    field_m = gaussian_interpolate(
        sdf["lat"].to_numpy(float),
        np.mod(sdf["lon"].to_numpy(float), 360.0),
        sdf["steric_0_2000_m"].to_numpy(float),
        grid_lat,
        grid_lon,
        length_scale_km=500.0,
        truncation_radius_km=1200.0,
        mask_distance_km=500.0,
    )
    return grid_lat, grid_lon, field_m * 100.0


def _woa_reference_dyn_cm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds_t = xr.open_dataset(REF_TEMP, decode_times=False)
    ds_s = xr.open_dataset(REF_SALT, decode_times=False)
    depth = np.asarray(ds_t["depth"].values, dtype=float)
    lat = np.asarray(ds_t["lat"].values, dtype=float)
    lon = np.asarray(ds_t["lon"].values, dtype=float)

    keep = depth <= 2000.0
    depth = depth[keep]
    temp = ds_t["t_an"].isel(time=0, depth=keep).to_numpy()
    salt = ds_s["s_an"].isel(time=0, depth=keep).to_numpy()

    out_dyn_cm = np.full((lat.size, lon.size), np.nan, dtype=float)
    p_ref = 2000.0
    p_profile = np.repeat(depth[:, None], lon.size, axis=1)

    for j, lat_j in enumerate(lat):
        temp_j = temp[:, j, :]
        salt_j = salt[:, j, :]
        finite = np.all(np.isfinite(temp_j), axis=0) & np.all(np.isfinite(salt_j), axis=0)
        if not np.any(finite):
            continue
        lon_j = lon[finite]
        p_j = p_profile[:, finite]
        salt_f = salt_j[:, finite]
        temp_f = temp_j[:, finite]
        lat_grid = np.full_like(p_j, lat_j, dtype=float)
        lon_grid = np.broadcast_to(lon_j[None, :], p_j.shape)
        sa = gsw.SA_from_SP(salt_f, p_j, lon_grid, lat_grid)
        ct = gsw.CT_from_t(sa, temp_f, p_j)
        dyn = gsw.geo_strf_dyn_height(sa, ct, p_j, p_ref=p_ref)
        out_dyn_cm[j, finite] = dyn[0, :] / 9.81 * 100.0
    lon_360 = np.mod(lon, 360.0)
    order = np.argsort(lon_360)
    return lat, lon_360[order], out_dyn_cm[:, order]


def _render(
    argo_label: str,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    argo_dyn_cm: np.ndarray,
    woa_lat: np.ndarray,
    woa_lon: np.ndarray,
    woa_dyn_cm: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 10.5), constrained_layout=True)

    mean_levels = np.arange(50.0, 311.0, 20.0)
    argo_masked = ma.masked_invalid(argo_dyn_cm)
    artist0 = axes[0].contourf(
        grid_lon,
        grid_lat,
        argo_masked,
        levels=mean_levels,
        cmap="turbo",
        extend="both",
        zorder=1,
    )
    contours0 = axes[0].contour(
        grid_lon,
        grid_lat,
        argo_masked,
        levels=mean_levels,
        colors="#4d2b00",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    axes[0].clabel(contours0, mean_levels[::2], fmt="%d", fontsize=8, inline=True)
    _add_land_overlay_0_360(axes[0], facecolor="black", edgecolor="black", linewidth=0.2, zorder=5)
    cbar0 = fig.colorbar(artist0, ax=axes[0], shrink=0.95, pad=0.02)
    cbar0.set_label("dyn cm")
    axes[0].set_title(f"a. Argo {argo_label} steric height, 0/2000 dbar")

    diff = argo_dyn_cm - woa_dyn_cm
    diff_levels = np.array([-35, -25, -15, -8, -4, 0, 4, 8, 15, 25, 35], dtype=float)
    diff_masked = ma.masked_invalid(diff)
    artist1 = axes[1].contourf(
        woa_lon,
        woa_lat,
        diff_masked,
        levels=diff_levels,
        cmap="RdYlGn_r",
        extend="both",
        zorder=1,
    )
    contours1 = axes[1].contour(
        woa_lon,
        woa_lat,
        diff_masked,
        levels=diff_levels,
        colors="black",
        linewidths=0.55,
        alpha=0.6,
        zorder=3,
    )
    axes[1].clabel(contours1, diff_levels[::2], fmt="%d", fontsize=8, inline=True)
    _add_land_overlay_0_360(axes[1], facecolor="black", edgecolor="black", linewidth=0.2, zorder=5)
    cbar1 = fig.colorbar(artist1, ax=axes[1], shrink=0.95, pad=0.02)
    cbar1.set_label("Argo minus WOA proxy (dyn cm)")
    axes[1].set_title("b. Argo minus WOA23 decav proxy reference")

    for ax in axes:
        ax.set_xlim(20.0, 380.0)
        ax.set_ylim(-60.0, 65.0)
        ax.set_xticks([60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
        ax.set_xticklabels(["60°E", "120°E", "180°", "120°W", "60°W", "0°"])
        ax.set_yticks([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])
        ax.set_yticklabels(["60°S", "40°S", "20°S", "0°", "20°N", "40°N", "60°N"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        "Paper-style steric-height validation panels\n"
        "Argo field from TEOS-10 dynamic height / 9.81, shown in dyn cm",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--argo-parquet", type=Path, required=True)
    parser.add_argument("--argo-label", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.argo_parquet)
    grid_lat, grid_lon, argo_dyn_cm = _argo_field_dyn_cm(df)
    woa_lat, woa_lon, woa_dyn_cm = _woa_reference_dyn_cm()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _render(args.argo_label, grid_lat, grid_lon, argo_dyn_cm, woa_lat, woa_lon, woa_dyn_cm, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
