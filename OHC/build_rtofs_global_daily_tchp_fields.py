"""Build daily global RTOFS TCHP/OHC fields from HYCOM archive products.

This is the real global path, using the NOAA RTOFS/HYCOM archive:

- global grid coordinates from ``rtofs_glo_2ds_f006_diag.nc``
- 3D temperature/salinity/thickness from ``rtofs_glo.t00z.f06.archv.[ab]``

The workflow mirrors the earlier regional prototype, but uses chunked row
processing so we can handle the 4500 x 3298 global grid without loading the
full 3D volume into memory at once.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import RTOFS_CACHE_DIR, RTOFS_GLOBAL_CACHE_DIR
from ml.sources.rtofs_global_source import (
    cached_global_archv_a_path,
    download_global_archv_a,
    download_global_archv_b,
    download_global_grid,
    extracted_global_archv_a_path,
    extract_global_archv_a,
    normalize_longitude,
    parse_archv_b,
    records_for_field,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/data/suramya/rtofs_global_ohc_fields")
REF_TEMP_C = 26.0
PA_PER_M = 9806.0
DEFAULT_WINTER_MONTHS = {1, 2, 3}
DEFAULT_SUMMER_MONTHS = {7, 8, 9}


def cached_regional_dates(cache_dir: Path) -> list[str]:
    dates: list[str] = []
    for p in sorted(cache_dir.glob("rtofs.*/rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc")):
        name = p.parent.name
        if name.startswith("rtofs.") and len(name) == 14:
            dates.append(name.split(".", 1)[1])
    return dates


def select_dates(*, years: list[int] | None, seasons: list[str] | None, explicit_dates: list[str] | None) -> list[str]:
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
    for d in cached_regional_dates(RTOFS_CACHE_DIR):
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


def _read_record_rows(mm: np.memmap, *, record_index: int, record_words: int, idm: int, y0: int, y1: int) -> np.ndarray:
    start = record_index * record_words + y0 * idm
    count = (y1 - y0) * idm
    arr = np.asarray(mm[start:start + count], dtype=">f4").astype(np.float32, copy=False).reshape(y1 - y0, idm)
    arr = np.where(arr > 1.0e30, np.nan, arr)
    return arr


def _compute_chunk_fields(
    *,
    lat_chunk: np.ndarray,
    lon_chunk: np.ndarray,
    temp: np.ndarray,
    sal: np.ndarray,
    thk_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TCHP/OHC/D26/surface temperature for one row chunk.

    Arrays are shaped:
      - lat/lon: (ny, nx)
      - temp/sal/thk: (nz, ny, nx)
    """
    valid_layer = np.isfinite(temp) & np.isfinite(sal) & np.isfinite(thk_pa) & (thk_pa > 0.0)
    thickness_m = np.where(valid_layer, thk_pa / PA_PER_M, 0.0).astype(np.float32)

    depth_top = np.cumsum(thickness_m, axis=0, dtype=np.float32) - thickness_m
    depth_bottom = depth_top + thickness_m
    depth_center = depth_top + 0.5 * thickness_m

    surf_t = np.where(valid_layer[0], temp[0], np.nan).astype(np.float32)
    warm_surface = valid_layer[0] & (temp[0] >= REF_TEMP_C)

    below = valid_layer & (temp <= REF_TEMP_C)
    any_below = below.any(axis=0) & warm_surface
    cross_idx = np.argmax(below, axis=0)
    prev_idx = np.clip(cross_idx - 1, 0, temp.shape[0] - 1)

    def gather(arr3d: np.ndarray, idx2d: np.ndarray) -> np.ndarray:
        return np.take_along_axis(arr3d, idx2d[None, :, :], axis=0)[0]

    z1 = gather(depth_center, prev_idx)
    z2 = gather(depth_center, cross_idx)
    t1 = gather(temp, prev_idx)
    t2 = gather(temp, cross_idx)

    d26 = np.full_like(surf_t, np.nan, dtype=np.float32)
    good = any_below & np.isfinite(t1) & np.isfinite(t2) & np.isfinite(z1) & np.isfinite(z2) & (t1 != t2)
    frac = np.zeros_like(surf_t, dtype=np.float32)
    frac[good] = ((t1[good] - REF_TEMP_C) / (t1[good] - t2[good])).astype(np.float32)
    d26[good] = z1[good] + frac[good] * (z2[good] - z1[good])

    lat3 = lat_chunk[None, :, :]
    lon3 = lon_chunk[None, :, :]
    pressure = gsw.p_from_z(-depth_center.astype(np.float64), lat3.astype(np.float64))
    abs_sal = gsw.SA_from_SP(sal.astype(np.float64), pressure, lon3.astype(np.float64), lat3.astype(np.float64))
    rho = gsw.rho_t_exact(abs_sal, temp.astype(np.float64), pressure)
    cp = gsw.cp_t_exact(abs_sal, temp.astype(np.float64), pressure)

    heat = np.clip(temp.astype(np.float64) - REF_TEMP_C, 0.0, None)
    d26_3 = d26[None, :, :]
    full = valid_layer & np.isfinite(d26_3) & (depth_bottom <= d26_3)
    partial = valid_layer & np.isfinite(d26_3) & (depth_top < d26_3) & (depth_bottom > d26_3)

    eff_dz = np.zeros_like(thickness_m, dtype=np.float32)
    eff_dz[full] = thickness_m[full]
    d26_b = np.broadcast_to(d26_3, depth_top.shape)
    eff_dz[partial] = (d26_b[partial] - depth_top[partial]).astype(np.float32)

    ohc = np.sum(heat * rho * cp * eff_dz.astype(np.float64), axis=0)
    ohc = np.where(np.isfinite(d26), ohc, np.nan)
    tchp = ohc / 1.0e7

    return (
        tchp.astype(np.float32),
        ohc.astype(np.float32),
        d26.astype(np.float32),
        surf_t.astype(np.float32),
    )


def compute_daily_field_for_date(
    *,
    yyyymmdd: str,
    grid_date: str,
    cache_dir: Path,
    out_dir: Path,
    row_chunk: int,
    overwrite: bool,
    keep_tgz: bool,
    cleanup_archv_a: bool,
) -> dict:
    t0 = time.perf_counter()
    out_path = build_output_path(out_dir, yyyymmdd)
    if out_path.exists() and not overwrite:
        return {"date": yyyymmdd, "status": "cached", "out_path": str(out_path), "elapsed_s": round(time.perf_counter() - t0, 2)}

    grid_path = download_global_grid(grid_date, cache_dir)
    download_global_archv_b(yyyymmdd, cache_dir)
    extracted_path = extracted_global_archv_a_path(yyyymmdd, cache_dir)
    if extracted_path.exists() and extracted_path.stat().st_size > 0:
        data_path = extracted_path
    else:
        download_global_archv_a(yyyymmdd, cache_dir)
        data_path = extract_global_archv_a(yyyymmdd, cache_dir)
    if not keep_tgz:
        tgz_path = cached_global_archv_a_path(yyyymmdd, cache_dir)
        if tgz_path.exists():
            tgz_path.unlink()

    header = parse_archv_b(cache_dir / f"rtofs.{yyyymmdd}" / "rtofs_glo.t00z.f06.archv.b")
    temp_recs = records_for_field(header, "temp")
    sal_recs = records_for_field(header, "salin")
    thk_recs = records_for_field(header, "thknss")

    with xr.open_dataset(grid_path) as grid_ds:
        lat2d = np.asarray(grid_ds["Latitude"].values, dtype=np.float32)
        lon2d = normalize_longitude(np.asarray(grid_ds["Longitude"].values, dtype=np.float32))

    ny, nx = lat2d.shape
    tchp = np.full((ny, nx), np.nan, dtype=np.float32)
    ohc = np.full((ny, nx), np.nan, dtype=np.float32)
    d26 = np.full((ny, nx), np.nan, dtype=np.float32)
    surf_t = np.full((ny, nx), np.nan, dtype=np.float32)

    mm = np.memmap(data_path, dtype=">f4", mode="r")
    for y0 in range(0, ny, row_chunk):
        y1 = min(ny, y0 + row_chunk)
        lat_chunk = lat2d[y0:y1]
        lon_chunk = lon2d[y0:y1]

        temp_chunk = np.stack([
            _read_record_rows(mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y0=y0, y1=y1)
            for rec in temp_recs
        ], axis=0)
        sal_chunk = np.stack([
            _read_record_rows(mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y0=y0, y1=y1)
            for rec in sal_recs
        ], axis=0)
        thk_chunk = np.stack([
            _read_record_rows(mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y0=y0, y1=y1)
            for rec in thk_recs
        ], axis=0)

        tchp_chunk, ohc_chunk, d26_chunk, surf_chunk = _compute_chunk_fields(
            lat_chunk=lat_chunk,
            lon_chunk=lon_chunk,
            temp=temp_chunk,
            sal=sal_chunk,
            thk_pa=thk_chunk,
        )
        tchp[y0:y1] = tchp_chunk
        ohc[y0:y1] = ohc_chunk
        d26[y0:y1] = d26_chunk
        surf_t[y0:y1] = surf_chunk
        logger.info("Processed %s rows %d:%d", yyyymmdd, y0, y1)

    out_ds = xr.Dataset(
        data_vars={
            "tchp_kj_per_cm2": (("Y", "X"), tchp),
            "ohc_j_per_m2": (("Y", "X"), ohc),
            "d26_m": (("Y", "X"), d26),
            "surface_temp_c": (("Y", "X"), surf_t),
            "Latitude": (("Y", "X"), lat2d),
            "Longitude": (("Y", "X"), lon2d),
        },
        coords={"Y": np.arange(ny, dtype=np.int32), "X": np.arange(nx, dtype=np.int32)},
        attrs={
            "source_date": yyyymmdd,
            "grid_date": grid_date,
            "source_type": "global_hycom_archv",
            "row_chunk": row_chunk,
            "description": "Daily global RTOFS/HYCOM TCHP/OHC field derived from global archv archive.",
        },
    )
    out_ds.to_netcdf(out_path)
    if cleanup_archv_a and data_path.exists():
        data_path.unlink()

    elapsed = time.perf_counter() - t0
    finite = int(np.isfinite(tchp).sum())
    return {
        "date": yyyymmdd,
        "status": "computed",
        "out_path": str(out_path),
        "finite_tchp_cells": finite,
        "elapsed_s": round(elapsed, 2),
    }


def run_parallel(
    *,
    dates: list[str],
    grid_date: str,
    cache_dir: Path,
    out_dir: Path,
    row_chunk: int,
    overwrite: bool,
    keep_tgz: bool,
    cleanup_archv_a: bool,
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
                grid_date=grid_date,
                cache_dir=cache_dir,
                out_dir=out_dir,
                row_chunk=row_chunk,
                overwrite=overwrite,
                keep_tgz=keep_tgz,
                cleanup_archv_a=cleanup_archv_a,
            )] = d

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                d = futures.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "date": d,
                        "status": "failed",
                        "error": repr(exc),
                    }
                    logger.exception("Global RTOFS date %s failed", d)
                logger.info("%s", result)
                results.append(result)
                if pending_dates:
                    nxt = pending_dates.pop(0)
                    futures[ex.submit(
                        compute_daily_field_for_date,
                        yyyymmdd=nxt,
                        grid_date=grid_date,
                        cache_dir=cache_dir,
                        out_dir=out_dir,
                        row_chunk=row_chunk,
                        overwrite=overwrite,
                        keep_tgz=keep_tgz,
                        cleanup_archv_a=cleanup_archv_a,
                    )] = nxt
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--seasons", nargs="+", default=["winter", "summer"])
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--grid-date", default="20240131")
    parser.add_argument("--cache-dir", type=Path, default=RTOFS_GLOBAL_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-tgz", action="store_true")
    parser.add_argument("--cleanup-archv-a", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    dates = select_dates(years=args.years, seasons=args.seasons, explicit_dates=args.dates)
    if not dates:
        raise SystemExit("No matching dates found.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building global RTOFS daily TCHP fields for %d dates with workers=%d, row_chunk=%d, out_dir=%s",
        len(dates), args.workers, args.row_chunk, args.out_dir,
    )
    results = run_parallel(
        dates=dates,
        grid_date=args.grid_date,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        row_chunk=args.row_chunk,
        overwrite=args.overwrite,
        keep_tgz=args.keep_tgz,
        cleanup_archv_a=args.cleanup_archv_a,
        workers=max(1, args.workers),
    )
    computed = sum(1 for r in results if r["status"] == "computed")
    cached = sum(1 for r in results if r["status"] == "cached")
    failed = [r for r in results if r["status"] == "failed"]
    print(json.dumps({
        "dates_requested": len(dates),
        "computed": computed,
        "cached": cached,
        "failed": len(failed),
        "out_dir": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
