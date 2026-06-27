"""Regrid reduced RTOFS seasonal mean fields to a regular global lat/lon grid.

This is a dense-grid remapping step, not sparse-data interpolation. We bin
native-grid values into a target regular grid and average within each cell.
Cells outside the underlying RTOFS domain remain NaN.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import PARAMS, build_global_grid

INPUT_DIR = Path("/data/suramya/rtofs_ohc_fields_global/seasonal_means")
DEFAULT_OUTPUT_DIR = Path("/data/suramya/rtofs_ohc_fields_global/regridded_0p25deg")


def regrid_mean_field(
    year: int,
    season_key: str,
    *,
    resolution_deg: float,
    out_dir: Path,
) -> Path:
    in_path = INPUT_DIR / f"rtofs_{year}_{season_key}_mean.nc"
    out_path = out_dir / f"rtofs_{year}_{season_key}_{str(resolution_deg).replace('.', 'p')}deg.nc"

    with xr.open_dataset(in_path) as ds:
        src_lat = ds["Latitude"].values
        src_lon = ds["Longitude"].values
        grid_lat, grid_lon = build_global_grid(resolution_deg)
        n_lat, n_lon = grid_lat.shape

        lat_idx = np.floor((src_lat + 90.0) / resolution_deg).astype(np.int64)
        lon_idx = np.floor((src_lon + 180.0) / resolution_deg).astype(np.int64)
        valid_geo = (
            np.isfinite(src_lat)
            & np.isfinite(src_lon)
            & (lat_idx >= 0)
            & (lat_idx < n_lat)
            & (lon_idx >= 0)
            & (lon_idx < n_lon)
        )

        flat_bin = (lat_idx * n_lon + lon_idx).ravel()
        valid_geo_flat = valid_geo.ravel()

        out_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}

        for param_key in PARAMS:
            flat_vals = ds[param_key].values.ravel()
            valid = valid_geo_flat & np.isfinite(flat_vals)
            counts = np.bincount(flat_bin[valid], minlength=n_lat * n_lon)
            sums = np.bincount(flat_bin[valid], weights=flat_vals[valid], minlength=n_lat * n_lon)
            out = np.full(n_lat * n_lon, np.nan, dtype=np.float32)
            mask = counts > 0
            out[mask] = (sums[mask] / counts[mask]).astype(np.float32)
            out_vars[param_key] = (("lat", "lon"), out.reshape(n_lat, n_lon))
            out_vars[f"{param_key}_count"] = (("lat", "lon"), counts.reshape(n_lat, n_lon).astype(np.int32))

        out_ds = xr.Dataset(
            data_vars=out_vars,
            coords={
                "lat": ("lat", grid_lat[:, 0].astype(np.float32)),
                "lon": ("lon", grid_lon[0, :].astype(np.float32)),
            },
            attrs={
                **ds.attrs,
                "regrid_resolution_deg": resolution_deg,
                "regrid_method": "bin_average",
                "source_domain_note": (
                    "Underlying cached RTOFS product is US-east / western Atlantic coverage; "
                    "cells outside source coverage remain NaN on the global grid."
                ),
            },
        )

    out_ds.to_netcdf(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--seasons", nargs="+", default=["winter_jfm", "summer_jas"])
    parser.add_argument("--resolution-deg", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        for season_key in args.seasons:
            out = regrid_mean_field(
                year,
                season_key,
                resolution_deg=args.resolution_deg,
                out_dir=args.out_dir,
            )
            print(out)


if __name__ == "__main__":
    main()
