"""Render side-by-side winter/summer interpolated maps for seasonal Argo parameters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (
    OUTPUT_DIR,
    PARAMS,
    SEASONS,
    add_land_overlay,
    build_global_grid,
    gaussian_interpolate,
    linear_nd_interpolate,
    load_clean_dataframe,
    make_norm,
    rbf_gaussian_interpolate,
)


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "nomask"}:
        return None
    return float(value)


def interpolate_field(method: str, df, resolution_deg: float, mask_distance_km: float | None):
    grid_lat, grid_lon = build_global_grid(resolution_deg)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    values = df[df.columns[-1]].to_numpy(dtype=float)  # overridden before call
    if method == "gaussian":
        field = gaussian_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=mask_distance_km)
    elif method == "linear_nd":
        field = linear_nd_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=mask_distance_km)
    elif method == "rbf_gaussian":
        field = rbf_gaussian_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=mask_distance_km)
    else:
        raise ValueError(f"Unknown interpolation method {method}")
    return grid_lat, grid_lon, field


def render_year_param(
    year: int,
    param_key: str,
    method: str,
    resolution_deg: float,
    mask_distance_km: float | None,
    out_dir: Path,
    *,
    smooth_display: bool = False,
) -> Path:
    cfg = PARAMS[param_key]
    norm = make_norm(param_key)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    mappable = None
    for ax, (season_key, (season_label, months)) in zip(axes, SEASONS.items()):
        df = load_clean_dataframe([year], months, param_key)
        values = df[param_key].to_numpy(dtype=float)
        grid_lat, grid_lon = build_global_grid(resolution_deg)
        if method == "gaussian":
            field = gaussian_interpolate(
                df["lat"].to_numpy(dtype=float),
                df["lon"].to_numpy(dtype=float),
                values,
                grid_lat,
                grid_lon,
                mask_distance_km=mask_distance_km,
            )
        elif method == "linear_nd":
            field = linear_nd_interpolate(
                df["lat"].to_numpy(dtype=float),
                df["lon"].to_numpy(dtype=float),
                values,
                grid_lat,
                grid_lon,
                mask_distance_km=mask_distance_km,
            )
        elif method == "rbf_gaussian":
            field = rbf_gaussian_interpolate(
                df["lat"].to_numpy(dtype=float),
                df["lon"].to_numpy(dtype=float),
                values,
                grid_lat,
                grid_lon,
                mask_distance_km=mask_distance_km,
            )
        else:
            raise ValueError(f"Unknown interpolation method {method}")

        masked_field = ma.masked_invalid(field)
        if smooth_display:
            artist = ax.imshow(
                masked_field,
                origin="lower",
                extent=[-180, 180, -90, 90],
                cmap=cfg.cmap,
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
                cmap=cfg.cmap,
                norm=norm,
                zorder=1,
            )
        mappable = artist
        add_land_overlay(ax, zorder=5)
        ax.set_title(f"{season_label}\n{len(df):,} observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    mask_label = f"masked_{int(mask_distance_km)}km" if mask_distance_km is not None else "nomask"
    style_label = "smooth" if smooth_display else "grid"
    fig.suptitle(
        f"{cfg.label} seasonal interpolated maps ({year})\nmethod={method}, grid={resolution_deg}°, {mask_label}, style={style_label}",
        fontsize=14,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(cfg.label)

    out_path = out_dir / f"{param_key}_{year}_{method}_{str(resolution_deg).replace('.', 'p')}deg_{mask_label}_{style_label}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    parser.add_argument("--params", nargs="+", choices=sorted(PARAMS), default=sorted(PARAMS))
    parser.add_argument("--method", choices=["gaussian", "linear_nd", "rbf_gaussian"], default="gaussian")
    parser.add_argument("--resolution-deg", type=float, default=0.25)
    parser.add_argument("--mask-distance-km", type=parse_optional_float, default=100.0)
    parser.add_argument("--smooth-display", action="store_true")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory under seasonal_maps to write images into. "
             "Defaults to 'interpolated'.",
    )
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / (args.output_subdir or "interpolated")
    out_dir.mkdir(parents=True, exist_ok=True)

    for param_key in args.params:
        for year in args.years:
            out = render_year_param(
                year=year,
                param_key=param_key,
                method=args.method,
                resolution_deg=args.resolution_deg,
                mask_distance_km=args.mask_distance_km,
                out_dir=out_dir,
                smooth_display=args.smooth_display,
            )
            print(out)


if __name__ == "__main__":
    main()
