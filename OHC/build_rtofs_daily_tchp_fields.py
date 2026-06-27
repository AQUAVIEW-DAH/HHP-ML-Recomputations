"""Build daily RTOFS TCHP/OHC fields from cached 3D model files.

This script converts each cached RTOFS 3D temperature/salinity file into a
compact 2D field product containing:

- ``tchp_kj_per_cm2``
- ``ohc_j_per_m2``
- ``d26_m``
- ``surface_temp_c``

The intended workflow is:

1. Materialize daily fields once, on the native RTOFS grid.
2. Build seasonal means from those reduced daily fields.
3. Regrid seasonal means to match the Argo comparison grid.

Parallelism is date-level via ``ProcessPoolExecutor`` so each worker handles
one NetCDF file independently.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import RTOFS_CACHE_DIR
from OHC.teos_ohc import compute_ohc_teos10

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/data/suramya/rtofs_ohc_fields")
DEFAULT_WINTER_MONTHS = {1, 2, 3}
DEFAULT_SUMMER_MONTHS = {7, 8, 9}


def cached_rtofs_dates(cache_dir: Path) -> list[str]:
    dates: list[str] = []
    for p in sorted(cache_dir.glob("rtofs.*/rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc")):
        name = p.parent.name
        if name.startswith("rtofs.") and len(name) == 14:
            dates.append(name.split(".", 1)[1])
    return dates


def select_dates(
    *,
    cache_dir: Path,
    years: list[int] | None,
    seasons: list[str] | None,
    explicit_dates: list[str] | None,
) -> list[str]:
    if explicit_dates:
        return sorted(explicit_dates)

    season_months: set[int] | None = None
    if seasons:
        season_months = set()
        for season in seasons:
            s = season.lower()
            if s in {"winter", "jfm", "winter_jfm"}:
                season_months |= DEFAULT_WINTER_MONTHS
            elif s in {"summer", "jas", "summer_jas"}:
                season_months |= DEFAULT_SUMMER_MONTHS
            else:
                raise ValueError(f"Unsupported season {season!r}")

    out: list[str] = []
    for d in cached_rtofs_dates(cache_dir):
        year = int(d[:4])
        month = int(d[4:6])
        if years and year not in years:
            continue
        if season_months and month not in season_months:
            continue
        out.append(d)
    return out


def build_output_path(out_dir: Path, yyyymmdd: str) -> Path:
    return out_dir / f"rtofs_tchp_{yyyymmdd}.nc"


def _subset_index_bounds(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[slice, slice]:
    if bbox is None:
        return slice(None), slice(None)

    west, south, east, north = bbox
    mask = (
        np.isfinite(lat2d)
        & np.isfinite(lon2d)
        & (lat2d >= south)
        & (lat2d <= north)
        & (lon2d >= west)
        & (lon2d <= east)
    )
    if not mask.any():
        raise ValueError(f"No RTOFS grid cells found inside bbox {bbox}")
    ys, xs = np.where(mask)
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def compute_daily_field_for_date(
    *,
    yyyymmdd: str,
    cache_dir: Path,
    out_dir: Path,
    bbox: tuple[float, float, float, float] | None,
    overwrite: bool,
) -> dict:
    t0 = time.perf_counter()
    source_path = cache_dir / f"rtofs.{yyyymmdd}" / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    out_path = build_output_path(out_dir, yyyymmdd)
    if out_path.exists() and not overwrite:
        return {
            "date": yyyymmdd,
            "status": "cached",
            "out_path": str(out_path),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    with xr.open_dataset(source_path) as ds:
        lat2d = np.asarray(ds["Latitude"].values, dtype=np.float32)
        lon2d = np.asarray(ds["Longitude"].values, dtype=np.float32)
        ys, xs = _subset_index_bounds(lat2d, lon2d, bbox)

        lat_sub = lat2d[ys, xs]
        lon_sub = lon2d[ys, xs]
        temp = np.asarray(ds["temperature"].isel(MT=0, Y=ys, X=xs).values, dtype=np.float32)
        sal = np.asarray(ds["salinity"].isel(MT=0, Y=ys, X=xs).values, dtype=np.float32)
        depth = np.asarray(ds["Depth"].values, dtype=np.float32)

    ny, nx = lat_sub.shape
    tchp = np.full((ny, nx), np.nan, dtype=np.float32)
    ohc = np.full((ny, nx), np.nan, dtype=np.float32)
    d26 = np.full((ny, nx), np.nan, dtype=np.float32)
    surf_t = np.full((ny, nx), np.nan, dtype=np.float32)

    surface = temp[0]
    valid_surface = np.isfinite(surface) & np.isfinite(lat_sub) & np.isfinite(lon_sub)
    wet_points = np.argwhere(valid_surface)

    for iy, ix in wet_points:
        t_col = temp[:, iy, ix]
        s_col = sal[:, iy, ix]
        mask = np.isfinite(depth) & np.isfinite(t_col) & np.isfinite(s_col)
        if mask.sum() < 2:
            continue
        result = compute_ohc_teos10(
            vertical=depth[mask],
            temp_c=t_col[mask],
            salinity_psu=s_col[mask],
            lat=float(lat_sub[iy, ix]),
            lon=float(lon_sub[iy, ix]),
            vertical_axis="depth",
        )
        if result.tchp_kj_per_cm2 is not None:
            tchp[iy, ix] = result.tchp_kj_per_cm2
        if result.ohc_j_per_m2 is not None:
            ohc[iy, ix] = result.ohc_j_per_m2
        if result.d26_m is not None:
            d26[iy, ix] = result.d26_m
        if result.surface_temp_c is not None:
            surf_t[iy, ix] = result.surface_temp_c

    out_ds = xr.Dataset(
        data_vars={
            "tchp_kj_per_cm2": (("Y", "X"), tchp),
            "ohc_j_per_m2": (("Y", "X"), ohc),
            "d26_m": (("Y", "X"), d26),
            "surface_temp_c": (("Y", "X"), surf_t),
            "Latitude": (("Y", "X"), lat_sub),
            "Longitude": (("Y", "X"), lon_sub),
        },
        coords={
            "Y": np.arange(ny, dtype=np.int32),
            "X": np.arange(nx, dtype=np.int32),
        },
        attrs={
            "source": str(source_path),
            "date": yyyymmdd,
            "bbox": json.dumps(bbox) if bbox is not None else "full_domain",
            "description": "Daily native-grid RTOFS TCHP/OHC field derived from cached 3D model output.",
        },
    )
    out_ds.to_netcdf(out_path)

    elapsed = time.perf_counter() - t0
    return {
        "date": yyyymmdd,
        "status": "computed",
        "out_path": str(out_path),
        "wet_cells": int(wet_points.shape[0]),
        "elapsed_s": round(elapsed, 2),
    }


def parse_bbox(values: list[float] | None) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("bbox must be west south east north")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def run_parallel(
    *,
    dates: list[str],
    cache_dir: Path,
    out_dir: Path,
    bbox: tuple[float, float, float, float] | None,
    overwrite: bool,
    workers: int,
) -> list[dict]:
    results: list[dict] = []
    pending_dates = list(dates)
    futures = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        while pending_dates and len(futures) < workers:
            d = pending_dates.pop(0)
            futures[ex.submit(
                compute_daily_field_for_date,
                yyyymmdd=d,
                cache_dir=cache_dir,
                out_dir=out_dir,
                bbox=bbox,
                overwrite=overwrite,
            )] = d

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                d = futures.pop(fut)
                result = fut.result()
                logger.info("%s", result)
                results.append(result)
                if pending_dates:
                    nxt = pending_dates.pop(0)
                    futures[ex.submit(
                        compute_daily_field_for_date,
                        yyyymmdd=nxt,
                        cache_dir=cache_dir,
                        out_dir=out_dir,
                        bbox=bbox,
                        overwrite=overwrite,
                    )] = nxt
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--seasons", nargs="+", default=["winter", "summer"])
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument("--bbox", nargs=4, type=float, default=None)
    parser.add_argument("--cache-dir", type=Path, default=RTOFS_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    dates = select_dates(
        cache_dir=args.cache_dir,
        years=args.years,
        seasons=args.seasons,
        explicit_dates=args.dates,
    )
    if not dates:
        raise SystemExit("No matching cached RTOFS dates found.")

    bbox = parse_bbox(args.bbox)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building RTOFS daily TCHP fields for %d dates with workers=%d, bbox=%s, out_dir=%s",
        len(dates), args.workers, bbox, args.out_dir,
    )
    results = run_parallel(
        dates=dates,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        bbox=bbox,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    summary = {
        "dates_requested": len(dates),
        "computed": sum(1 for r in results if r["status"] == "computed"),
        "cached": sum(1 for r in results if r["status"] == "cached"),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
