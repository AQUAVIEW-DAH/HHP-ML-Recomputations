"""Build RTOFS-only neighborhood-context (stencil) features at collocated points.

Tier-0 spatial features from SPATIAL_FEATURES_DIRECTIONS.md: for each collocated
Argo point, summarize the surrounding reduced daily RTOFS field (TCHP, D26, SST)
inside ~0.5 deg, 1 deg, and 2 deg half-width windows:

- local mean and std per window
- gradient magnitude at native resolution (per 100 km)
- point anomaly relative to the 1 deg local mean

These encode mesoscale regime (eddy / front / quiescent) using nothing but the
RTOFS fields already cached on disk, so they stay inside the mentor constraint
of adding no dependencies beyond RTOFS and remain defined everywhere on the
globe, including the float-sparse ocean.

Window sizes are expressed in grid cells assuming the ~0.08 deg native spacing;
poleward of ~45 deg the geographic window shrinks somewhat, which is acceptable
for these regime descriptors (valid TCHP/D26 rows are overwhelmingly < 45 deg).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import latlon_to_xyz

logger = logging.getLogger(__name__)

IN_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data")
OUT_PATH = OUT_DIR / "argo_rtofs_collocated_2024_2025_neighborhood.parquet"
OUT_SUMMARY = OUT_DIR / "summary_collocation_2024_2025_neighborhood.json"

RTOFS_DAILY_DIR = {
    2024: Path("/data/suramya/rtofs_global_ohc_fields_2024"),
    2025: Path("/data/suramya/rtofs_global_ohc_fields_2025"),
}

FIELDS = {
    "tchp": "tchp_kj_per_cm2",
    "d26": "d26_m",
    "sst": "surface_temp_c",
}
# Window half-widths in native cells (~0.08 deg each).
SCALES = {"halfdeg": 6, "1deg": 12, "2deg": 25}
MIN_FINITE_FRAC = 0.25
EARTH_R_KM = 6371.0


def _daily_field_for_date(date_str: str, year: int) -> Path:
    return RTOFS_DAILY_DIR[year] / f"rtofs_tchp_{date_str}.nc"


def _first_available_daily_file() -> Path:
    for year_dir in RTOFS_DAILY_DIR.values():
        for path in sorted(year_dir.glob("rtofs_tchp_*.nc")):
            return path
    raise FileNotFoundError("No reduced RTOFS daily field files found.")


def _build_grid_lookup(sample_file: Path) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_file) as ds:
        lat = ds["Latitude"].values.astype(np.float64)
        lon = ds["Longitude"].values.astype(np.float64)
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel()).astype(np.float32)
    tree = cKDTree(xyz)
    y_idx, x_idx = np.unravel_index(np.arange(lat.size, dtype=np.int64), lat.shape)
    return tree, y_idx.astype(np.int32), x_idx.astype(np.int32), lat, lon


def _local_spacing_km(lat2d: np.ndarray, lon2d: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Approximate local grid spacing (dy_km, dx_km) around cell (y, x)."""
    ny, nx = lat2d.shape
    y0, y1 = max(y - 1, 0), min(y + 1, ny - 1)
    xm, xp = (x - 1) % nx, (x + 1) % nx
    dy_deg = (lat2d[y1, x] - lat2d[y0, x]) / max(y1 - y0, 1)
    dlon = (lon2d[y, xp] - lon2d[y, xm] + 540.0) % 360.0 - 180.0
    dx_deg = dlon / 2.0
    dy_km = abs(dy_deg) * 111.19
    dx_km = abs(dx_deg) * 111.19 * max(np.cos(np.deg2rad(lat2d[y, x])), 0.05)
    return max(dy_km, 1e-3), max(dx_km, 1e-3)


def _window(field: np.ndarray, y: int, x: int, hw: int) -> np.ndarray:
    ny, nx = field.shape
    y0, y1 = max(y - hw, 0), min(y + hw + 1, ny)
    xs = np.arange(x - hw, x + hw + 1) % nx
    return field[y0:y1][:, xs]


def _grad_mag_per_100km(field: np.ndarray, y: int, x: int, dy_km: float, dx_km: float) -> float:
    ny, nx = field.shape
    y0, y1 = max(y - 1, 0), min(y + 1, ny - 1)
    xm, xp = (x - 1) % nx, (x + 1) % nx
    fy0, fy1 = field[y0, x], field[y1, x]
    fx0, fx1 = field[y, xm], field[y, xp]
    if not (np.isfinite(fy0) and np.isfinite(fy1) and np.isfinite(fx0) and np.isfinite(fx1)):
        return np.nan
    gy = (fy1 - fy0) / (max(y1 - y0, 1) * dy_km)
    gx = (fx1 - fx0) / (2.0 * dx_km)
    return float(np.hypot(gy, gx) * 100.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH).copy()
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy().reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    df["year"] = df["year"].astype(int)

    sample = _first_available_daily_file()
    tree, all_y, all_x, lat2d, lon2d = _build_grid_lookup(sample)
    obs_xyz = latlon_to_xyz(df["lat"].to_numpy(float), df["lon"].to_numpy(float)).astype(np.float32)
    _, flat_idx = tree.query(obs_xyz, k=1, workers=-1)
    near_y = all_y[flat_idx]
    near_x = all_x[flat_idx]

    n = len(df)
    cols: dict[str, np.ndarray] = {}
    for fkey in FIELDS:
        for skey in SCALES:
            cols[f"model_{fkey}_local_mean_{skey}"] = np.full(n, np.nan, dtype=np.float32)
            cols[f"model_{fkey}_local_std_{skey}"] = np.full(n, np.nan, dtype=np.float32)
        cols[f"model_{fkey}_grad_mag_per_100km"] = np.full(n, np.nan, dtype=np.float32)
        cols[f"model_{fkey}_anom_from_1deg_mean"] = np.full(n, np.nan, dtype=np.float32)

    dates_done = 0
    for (year, date), idx in df.groupby(["year", "date"]).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        daily_path = _daily_field_for_date(date, int(year))
        if not daily_path.exists():
            continue
        with xr.open_dataset(daily_path) as ds:
            fields = {k: np.asarray(ds[v].values, dtype=np.float32) for k, v in FIELDS.items()}

        for i in idx:
            y, x = int(near_y[i]), int(near_x[i])
            dy_km, dx_km = _local_spacing_km(lat2d, lon2d, y, x)
            for fkey, field in fields.items():
                point_val = field[y, x]
                for skey, hw in SCALES.items():
                    w = _window(field, y, x, hw)
                    finite = np.isfinite(w)
                    if finite.mean() < MIN_FINITE_FRAC:
                        continue
                    vals = w[finite]
                    cols[f"model_{fkey}_local_mean_{skey}"][i] = vals.mean()
                    cols[f"model_{fkey}_local_std_{skey}"][i] = vals.std()
                cols[f"model_{fkey}_grad_mag_per_100km"][i] = _grad_mag_per_100km(field, y, x, dy_km, dx_km)
                mean_1deg = cols[f"model_{fkey}_local_mean_1deg"][i]
                if np.isfinite(point_val) and np.isfinite(mean_1deg):
                    cols[f"model_{fkey}_anom_from_1deg_mean"][i] = point_val - mean_1deg
        dates_done += 1
        if dates_done % 10 == 0:
            logger.info("Processed %d dates (%.1fs elapsed)", dates_done, time.perf_counter() - t_start)

    for name, arr in cols.items():
        df[name] = arr

    df.to_parquet(OUT_PATH, index=False)

    summary = {
        "input_path": str(IN_PATH),
        "output_path": str(OUT_PATH),
        "rows_total": int(len(df)),
        "dates_total": int(df["date"].nunique()),
        "dates_with_fields": dates_done,
        "scales_cells": SCALES,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "feature_availability": {
            c: int(np.isfinite(arr).sum()) for c, arr in cols.items()
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
