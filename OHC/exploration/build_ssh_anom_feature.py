"""SSH anomaly from the monthly climatology (Dr. Jacobs' suggestion, 2026-08).

Feature: ssh_anom_monthly = model_ssh_m - monthly_mean_ssh(lat, lon, month),
where the monthly mean comes from all 701 cached RTOFS days on the native
grid (build_ssh_monthly_climatology.py). Nearest-cell sampling; output is a
row-aligned parquet like the other feature tables.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402

IN_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
PHYS_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
CLIM_PATH = Path("/data/suramya/rtofs_ssh_climatology/monthly_mean_ssh.nc")
OUT_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_ssh_anom.parquet")
OUT_SUMMARY = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/summary_collocation_2024_2025_ssh_anom.json")


def main() -> None:
    base = pd.read_parquet(IN_PATH).reset_index(drop=True)
    base = base[np.isfinite(base["lat"]) & np.isfinite(base["lon"])].copy().reset_index(drop=True)
    phys = pd.read_parquet(PHYS_PATH).reset_index(drop=True)
    if len(phys) != len(base):
        raise RuntimeError("physics table not aligned with base")
    ssh = pd.to_numeric(phys["model_ssh_m"], errors="coerce").to_numpy(float)

    with xr.open_dataset(CLIM_PATH) as ds:
        clim = np.asarray(ds["mean_ssh"].values, dtype=np.float32)  # (12, Y, X)
        lat2d = np.asarray(ds["Latitude"].values, dtype=np.float64)
        lon2d = np.asarray(ds["Longitude"].values, dtype=np.float64)

    tree = cKDTree(latlon_to_xyz(lat2d.ravel(), lon2d.ravel()).astype(np.float32))
    _, flat_idx = tree.query(latlon_to_xyz(base["lat"].to_numpy(float), base["lon"].to_numpy(float)).astype(np.float32), k=1, workers=-1)
    y_idx, x_idx = np.unravel_index(flat_idx, lat2d.shape)
    months = base["month"].astype(int).to_numpy() - 1
    clim_at = clim[months, y_idx, x_idx].astype(np.float64)
    anom = (ssh - clim_at).astype(np.float32)

    base["ssh_clim_monthly_m"] = clim_at.astype(np.float32)
    base["ssh_anom_monthly_m"] = anom
    base.to_parquet(OUT_PATH, index=False)
    summary = {
        "output_path": str(OUT_PATH),
        "rows_total": int(len(base)),
        "valid_anom_rows": int(np.isfinite(anom).sum()),
        "anom_p5": float(np.nanquantile(anom, 0.05)),
        "anom_median": float(np.nanmedian(anom)),
        "anom_p95": float(np.nanquantile(anom, 0.95)),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
