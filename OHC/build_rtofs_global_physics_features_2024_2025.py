"""Build global RTOFS-only physics features at collocated Argo points.

This enriches the existing multiyear collocation table with additional
RTOFS-derived upper-ocean structure features that are available globally from
the cached daily products and 2D diagnostics, without introducing any Argo
information into the feature set.

Current globally available additions:
- model_surface_temp_c
- model_ssh_m
- model_mixed_layer_thickness_m
- model_surface_boundary_layer_thickness_m
- thermocline / warm-layer interaction summaries derived from D26 and MLT

We intentionally keep this stage model-only so the correction model cannot
"cheat" by seeing observation-derived physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import latlon_to_xyz


IN_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data")
OUT_PATH = OUT_DIR / "argo_rtofs_collocated_2024_2025_physics.parquet"
OUT_CSV = OUT_DIR / "argo_rtofs_collocated_2024_2025_physics.csv"
OUT_SUMMARY = OUT_DIR / "summary_collocation_2024_2025_physics.json"

RTOFS_DAILY_DIR = {
    2024: Path("/data/suramya/rtofs_global_ohc_fields_2024"),
    2025: Path("/data/suramya/rtofs_global_ohc_fields_2025"),
}
RTOFS_GRID_CACHE_DIR = Path("/data/suramya/rtofs_global_cache")
K_NEIGHBORS = 8
EARTH_R_KM = 6371.0
REF_TEMP_C = 26.0


def _grid_file_for_date(date_str: str) -> Path:
    return RTOFS_GRID_CACHE_DIR / f"rtofs.{date_str}" / "rtofs_glo_2ds_f006_diag.nc"


def _daily_field_for_date(date_str: str, year: int) -> Path:
    return RTOFS_DAILY_DIR[year] / f"rtofs_tchp_{date_str}.nc"


def _first_available_grid_file() -> Path:
    for path in sorted(RTOFS_GRID_CACHE_DIR.glob("rtofs.*/rtofs_glo_2ds_f006_diag.nc")):
        return path
    raise FileNotFoundError("No cached RTOFS global grid/diagnostic file found in rtofs_global_cache.")


def _build_grid_lookup(sample_file: Path) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_file) as ds:
        lat = ds["Latitude"].values.astype(np.float32)
        lon = ds["Longitude"].values.astype(np.float32)
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel()).astype(np.float32)
    tree = cKDTree(xyz)
    y_idx, x_idx = np.unravel_index(np.arange(lat.size, dtype=np.int64), lat.shape)
    return tree, y_idx.astype(np.int32), x_idx.astype(np.int32)


def _interpolate_neighbor_values(
    values_2d: np.ndarray,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
    dist_km: np.ndarray,
) -> np.ndarray:
    vals = values_2d[y_idx, x_idx].astype(np.float64)
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(vals)

    exact = (dist_km <= 1e-6) & finite
    has_exact = exact.any(axis=1)
    if np.any(has_exact):
        first_exact = np.argmax(exact[has_exact], axis=1)
        rows = np.where(has_exact)[0]
        out[rows] = vals[rows, first_exact]

    remaining = ~has_exact
    if np.any(remaining):
        vals_r = vals[remaining]
        dist_r = dist_km[remaining]
        finite_r = finite[remaining]
        weights = np.zeros_like(vals_r, dtype=np.float64)
        weights[finite_r] = 1.0 / np.maximum(dist_r[finite_r], 1e-6) ** 2
        sw = weights.sum(axis=1)
        good = sw > 0
        if np.any(good):
            rows = np.where(remaining)[0][good]
            out[rows] = np.sum(weights[good] * vals_r[good], axis=1) / sw[good]

    return out.astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PATH).copy()
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy().reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    df["year"] = df["year"].astype(int)

    sample_grid = _first_available_grid_file()
    tree, all_y, all_x = _build_grid_lookup(sample_grid)
    obs_xyz = latlon_to_xyz(df["lat"].to_numpy(float), df["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=K_NEIGHBORS, workers=-1)
    if chord_dist.ndim == 1:
        chord_dist = chord_dist[:, None]
        flat_idx = flat_idx[:, None]
    y_idx = all_y[flat_idx]
    x_idx = all_x[flat_idx]
    dist_km = EARTH_R_KM * (2.0 * np.arcsin(np.clip(chord_dist.astype(np.float64) / 2.0, 0.0, 1.0)))

    n = len(df)
    surf_t = np.full(n, np.nan, dtype=np.float32)
    ssh = np.full(n, np.nan, dtype=np.float32)
    mlt = np.full(n, np.nan, dtype=np.float32)
    sblt = np.full(n, np.nan, dtype=np.float32)

    for (year, date), idx in df.groupby(["year", "date"]).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        daily_path = _daily_field_for_date(date, int(year))
        grid_path = _grid_file_for_date(date)
        if not daily_path.exists() or not grid_path.exists():
            continue

        with xr.open_dataset(daily_path) as ds:
            surf_t[idx] = _interpolate_neighbor_values(
                ds["surface_temp_c"].values,
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )

        with xr.open_dataset(grid_path) as ds:
            ssh[idx] = _interpolate_neighbor_values(
                np.asarray(ds["ssh"].isel(MT=0).values, dtype=np.float32),
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )
            mlt[idx] = _interpolate_neighbor_values(
                np.asarray(ds["mixed_layer_thickness"].isel(MT=0).values, dtype=np.float32),
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )
            sblt[idx] = _interpolate_neighbor_values(
                np.asarray(ds["surface_boundary_layer_thickness"].isel(MT=0).values, dtype=np.float32),
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )

    df["model_surface_temp_c"] = surf_t
    df["model_ssh_m"] = ssh
    df["model_mixed_layer_thickness_m"] = mlt
    df["model_surface_boundary_layer_thickness_m"] = sblt

    # Derived structure features kept model-only.
    temp_excess = np.where(np.isfinite(surf_t), surf_t - REF_TEMP_C, np.nan)
    df["model_temp_excess_26c"] = temp_excess.astype(np.float32)
    df["model_warm_surface_flag"] = np.where(np.isfinite(surf_t), (surf_t >= REF_TEMP_C).astype(np.int16), -1)

    d26 = df["model_interp_d26_m"].to_numpy(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["d26_minus_mlt_m"] = (d26 - mlt).astype(np.float32)
        df["d26_minus_sblt_m"] = (d26 - sblt).astype(np.float32)
        df["d26_to_mlt_ratio"] = (d26 / mlt).astype(np.float32)
        df["d26_to_sblt_ratio"] = (d26 / sblt).astype(np.float32)

    df["warm_layer_thickness_positive_m"] = np.where(
        np.isfinite(d26) & np.isfinite(mlt),
        np.maximum(d26 - mlt, 0.0),
        np.nan,
    ).astype(np.float32)
    df["model_ssh_x_abs_lat"] = (ssh.astype(np.float64) * np.abs(df["lat"].to_numpy(float))).astype(np.float32)
    df["model_mlt_x_abs_lat"] = (mlt.astype(np.float64) * np.abs(df["lat"].to_numpy(float))).astype(np.float32)
    df["model_temp_excess_x_abs_lat"] = (temp_excess.astype(np.float64) * np.abs(df["lat"].to_numpy(float))).astype(np.float32)

    df.to_parquet(OUT_PATH, index=False)
    df.to_csv(OUT_CSV, index=False)

    feature_cols = [
        "model_surface_temp_c",
        "model_ssh_m",
        "model_mixed_layer_thickness_m",
        "model_surface_boundary_layer_thickness_m",
        "model_temp_excess_26c",
        "d26_minus_mlt_m",
        "d26_minus_sblt_m",
        "d26_to_mlt_ratio",
        "d26_to_sblt_ratio",
        "warm_layer_thickness_positive_m",
        "model_ssh_x_abs_lat",
        "model_mlt_x_abs_lat",
        "model_temp_excess_x_abs_lat",
    ]
    summary = {
        "input_path": str(IN_PATH),
        "output_path": str(OUT_PATH),
        "rows_total": int(len(df)),
        "dates_total": int(df["date"].nunique()),
        "k_neighbors": K_NEIGHBORS,
        "feature_availability": {
            c: {
                "finite_rows": int(np.isfinite(pd.to_numeric(df[c], errors="coerce")).sum()),
            }
            for c in feature_cols
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
