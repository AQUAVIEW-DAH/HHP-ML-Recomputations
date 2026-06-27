"""Render side-by-side winter/summer RTOFS seasonal panels on a regular grid."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import OUTPUT_DIR, PARAMS, SEASONS, add_land_overlay, make_norm

INPUT_DIR = Path("/data/suramya/rtofs_ohc_fields_global/regridded_0p25deg")


def load_field(year: int, season_key: str, param_key: str, resolution_deg: float):
    path = INPUT_DIR / f"rtofs_{year}_{season_key}_{str(resolution_deg).replace('.', 'p')}deg.nc"
    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        field = ds[param_key].values
        n = int(ds.attrs.get("n_daily_fields", 0))
        note = ds.attrs.get("source_domain_note", "")
    return lat, lon, field, n, note


def render_year_param(
    year: int,
    param_key: str,
    out_dir: Path,
    *,
    resolution_deg: float,
    smooth_display: bool = False,
) -> Path:
    cfg = PARAMS[param_key]
    norm = make_norm(param_key)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    mappable = None
    domain_note = ""
    for ax, (season_key, (season_label, _months)) in zip(axes, SEASONS.items()):
        lat, lon, field, n, domain_note = load_field(year, season_key, param_key, resolution_deg)
        masked = ma.masked_invalid(field)
        if smooth_display:
            artist = ax.imshow(
                masked,
                origin="lower",
                extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
                cmap=cfg.cmap,
                norm=norm,
                interpolation="bilinear",
                aspect="auto",
                zorder=1,
            )
        else:
            artist = ax.pcolormesh(lon, lat, masked, shading="auto", cmap=cfg.cmap, norm=norm, zorder=1)
        mappable = artist
        add_land_overlay(ax, zorder=5)
        ax.set_title(f"{season_label}\n{n:,} daily fields")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    style_label = "smooth" if smooth_display else "grid"
    fig.suptitle(
        f"{cfg.label} RTOFS regridded seasonal maps ({year})\n"
        f"grid={resolution_deg}°, style={style_label}",
        fontsize=14,
    )
    if domain_note:
        fig.text(0.5, 0.01, domain_note, ha="center", va="bottom", fontsize=8)
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(cfg.label)

    out_path = out_dir / f"{param_key}_{year}_rtofs_regridded_{str(resolution_deg).replace('.', 'p')}deg_{style_label}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--params", nargs="+", choices=sorted(PARAMS), default=["tchp_kj_per_cm2", "d26_m"])
    parser.add_argument("--resolution-deg", type=float, default=0.25)
    parser.add_argument("--smooth-display", action="store_true")
    parser.add_argument("--output-subdir", default="rtofs_regridded")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for param_key in args.params:
        for year in args.years:
            out = render_year_param(
                year,
                param_key,
                out_dir,
                resolution_deg=args.resolution_deg,
                smooth_display=args.smooth_display,
            )
            print(out)


if __name__ == "__main__":
    main()
