from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from OHC.seasonal_map_common import add_land_overlay

ARGO_PATH = Path('/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024')
OUT_DIR = Path(__file__).resolve().parent / 'output' / 'argo_raw_2024_distribution'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    df = pd.read_parquet(ARGO_PATH, columns=['date','year','month','lat','lon','error'])
    raw_2024 = df[df['year'] == 2024].copy()
    clean_2024 = raw_2024[raw_2024['error'].isna()].copy()
    winter = clean_2024[clean_2024['month'].isin([1,2,3])].copy()
    summer = clean_2024[clean_2024['month'].isin([7,8,9])].copy()

    # full-year raw cleaned distribution
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    add_land_overlay(ax, zorder=0)
    ax.scatter(clean_2024['lon'], clean_2024['lat'], s=3, alpha=0.45, linewidths=0, color='tab:blue', zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    ax.set_title(f'All cleaned 2024 Argo/HHP raw points on globe\ncount={len(clean_2024):,} (raw rows={len(raw_2024):,})')
    full_path = OUT_DIR / 'argo_2024_raw_global_distribution.png'
    fig.savefig(full_path, dpi=200)
    plt.close(fig)

    # seasonal split, raw point distribution only
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, sdf, title in zip(axes, [winter, summer], ['Winter (JFM)', 'Summer (JAS)']):
        add_land_overlay(ax, zorder=0)
        ax.scatter(sdf['lon'], sdf['lat'], s=4, alpha=0.55, linewidths=0, color='tab:blue', zorder=2)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
        ax.set_title(f'{title}\ncount={len(sdf):,}')
    fig.suptitle('2024 Argo/HHP raw point distribution used for seasonal comparison', fontsize=14)
    seasonal_path = OUT_DIR / 'argo_2024_raw_points_winter_summer.png'
    fig.savefig(seasonal_path, dpi=200)
    plt.close(fig)

    summary = {
        'raw_2024_rows': int(len(raw_2024)),
        'clean_2024_rows': int(len(clean_2024)),
        'winter_jfm_rows': int(len(winter)),
        'summer_jas_rows': int(len(summer)),
        'full_year_plot': str(full_path),
        'seasonal_plot': str(seasonal_path),
    }
    (OUT_DIR / 'summary_2024_raw_distribution.json').write_text(__import__('json').dumps(summary, indent=2))
    print(summary)

if __name__ == '__main__':
    main()
