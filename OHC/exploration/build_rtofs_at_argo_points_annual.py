"""Full-year RTOFS-at-Argo point maps, plus a coverage map explaining where D26 exists.

Companion to refresh_rtofs_at_argo_points.py (which does JFM/JAS only). For each
year: (1) all collocated profiles colored by whether a D26 target exists, and
(2) RTOFS D26 / TCHP and Argo-minus-RTOFS deltas at every collocated profile.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.build_rtofs_at_argo_points_2024 import _delta_norm  # noqa: E402
from OHC.seasonal_map_common import PARAMS, add_land_overlay, make_norm  # noqa: E402

SRC = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/rtofs_at_argo_points_20260826/annual")
FIELDS = [("model_interp_d26_m", "d26_m", "RTOFS D26 (m) at Argo points", False),
          ("model_interp_tchp_kj_per_cm2", "tchp_kj_per_cm2", "RTOFS TCHP (kJ/cm²) at Argo points", False),
          ("delta_d26_m", "d26_m", "Argo - RTOFS D26 (m)", True),
          ("delta_tchp_kj_per_cm2", "tchp_kj_per_cm2", "Argo - RTOFS TCHP (kJ/cm²)", True)]


def _axes(ax, title):
    add_land_overlay(ax, zorder=0)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    ax.set_title(title)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    df["year"] = pd.to_datetime(df["date"].astype(str)).dt.year
    for year in (2024, 2025):
        y = df[df["year"] == year]
        has = y["argo_d26_m"].notna()
        # 1 — coverage map
        fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
        _axes(ax, f"All collocated Argo profiles, {year}: {len(y):,} profiles on {y['date'].nunique()} RTOFS days\n"
                  f"D26 defined (26 °C crossing exists): {int(has.sum()):,}   |   no D26 (water never reaches 26 °C): {int((~has).sum()):,}")
        ax.scatter(y.loc[~has, "lon"], y.loc[~has, "lat"], s=2, c="#b0b0b0", linewidths=0, label="profile, D26 undefined", zorder=1)
        ax.scatter(y.loc[has, "lon"], y.loc[has, "lat"], s=2, c="#1f5fbf", linewidths=0, label="profile with D26", zorder=2)
        ax.legend(loc="lower left", markerscale=6)
        fig.savefig(OUT / f"coverage_d26_defined_{year}.png", dpi=180); plt.close(fig)
        # 2 — value maps
        for field, pkey, label, delta in FIELDS:
            s = y[y[field].notna()]
            fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
            _axes(ax, f"{label}, full year {year}: {len(s):,} profiles")
            if delta:
                norm, cmap = _delta_norm(s[field].to_numpy(float)), "RdBu_r"
            else:
                norm, cmap = make_norm(pkey), PARAMS[pkey].cmap
            sc = ax.scatter(s["lon"], s["lat"], c=s[field], s=3, alpha=0.8, cmap=cmap, norm=norm, linewidths=0, zorder=2)
            fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02).set_label(label)
            fig.savefig(OUT / f"{field}_{year}_annual_points.png", dpi=180); plt.close(fig)
        print(year, len(y), int(has.sum()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
