"""Render side-by-side winter/summer point maps for seasonal Argo parameters."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import OUTPUT_DIR, PARAMS, SEASONS, add_land_overlay, load_clean_dataframe, make_norm


def render_year_param(year: int, param_key: str, out_dir: Path) -> Path:
    cfg = PARAMS[param_key]
    norm = make_norm(param_key)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    mappable = None
    for ax, (season_key, (season_label, months)) in zip(axes, SEASONS.items()):
        df = load_clean_dataframe([year], months, param_key)
        add_land_overlay(ax, zorder=0)
        sc = ax.scatter(
            df["lon"],
            df["lat"],
            c=df[param_key],
            s=5,
            alpha=0.8,
            cmap=cfg.cmap,
            norm=norm,
            linewidths=0,
            zorder=2,
        )
        mappable = sc
        ax.set_title(f"{season_label}\n{len(df):,} observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    fig.suptitle(f"{cfg.label} seasonal observation maps ({year})", fontsize=15)
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(cfg.label)

    out_path = out_dir / f"{param_key}_{year}_points_winter_summer.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023])
    parser.add_argument("--params", nargs="+", choices=sorted(PARAMS), default=sorted(PARAMS))
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "points"
    out_dir.mkdir(parents=True, exist_ok=True)

    for param_key in args.params:
        for year in args.years:
            out = render_year_param(year, param_key, out_dir)
            print(out)


if __name__ == "__main__":
    main()
