"""SIDE EXPLORATION: reduce HYCOM GOFS 3.1 reanalysis snapshots to daily OHC fields.

Downloads one 12Z t000 snapshot per requested date from the AWS Open Data bucket
``hycom-gofs-3pt1-reanalysis`` (NetCDF-3, ~4.8 GB each), computes the same 2D
diagnostics as the RTOFS reduced fields (TCHP kJ/cm2, OHC J/m2, D26 m, SST C,
plus SSH m), writes a compact NetCDF per date, and deletes the raw snapshot.

Unlike the RTOFS archv path, GOFS files are already on 40 standard z-levels, so
the vertical integration works directly on depth coordinates.

See exploration/README.md for why this is a side track and its caveats
(different model than RTOFS; NCODA assimilates Argo, so residuals vs Argo are
not independent).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

BUCKET = "hycom-gofs-3pt1-reanalysis"
DEFAULT_CACHE_DIR = Path("/data/suramya/gofs31_cache")
DEFAULT_OUT_DIR = Path("/data/suramya/gofs31_ohc_fields_2015")
REF_TEMP_C = 26.0
MAX_DEPTH_M = 1000.0
ROW_CHUNK = 128


def _s3_key_for_date(yyyymmdd: str) -> str:
    """Return the earliest-tau snapshot of the 12Z run for this date (t000 is absent for some dates)."""
    year = yyyymmdd[:4]
    listing = subprocess.run(
        ["aws", "s3", "ls", f"s3://{BUCKET}/{year}/hycom_GLBv0.08_", "--no-sign-request"],
        capture_output=True, text=True, timeout=300,
    )
    candidates = []
    for line in listing.stdout.splitlines():
        name = line.split()[-1]
        if f"_{yyyymmdd}12_t" in name and name.endswith(".nc"):
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError(f"No 12Z snapshot for {yyyymmdd} in s3://{BUCKET}/{year}/")
    return f"{year}/{sorted(candidates)[0]}"


def _download(key: str, cache_dir: Path) -> Path:
    dest = cache_dir / Path(key).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s", BUCKET, key)
    subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dest), "--no-sign-request", "--only-show-errors"],
        check=True, timeout=3600,
    )
    return dest


def _compute_fields(snapshot: Path) -> xr.Dataset:
    with xr.open_dataset(snapshot, decode_times=False) as ds:
        depth_full = np.asarray(ds["depth"].values, dtype=np.float64)
        n_lev = int(np.searchsorted(depth_full, MAX_DEPTH_M, side="right"))
        depth = depth_full[:n_lev]
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        ny, nx = lat.size, lon.size

        tchp = np.full((ny, nx), np.nan, dtype=np.float32)
        ohc = np.full((ny, nx), np.nan, dtype=np.float32)
        d26 = np.full((ny, nx), np.nan, dtype=np.float32)
        sst = np.full((ny, nx), np.nan, dtype=np.float32)
        ssh = np.asarray(ds["surf_el"].isel(time=0).values, dtype=np.float32)

        # Layer bounds: midpoints between z-levels, top at 0.
        edges = np.concatenate([[0.0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1]]])
        dz_top = edges[:-1]
        dz_bottom = edges[1:]

        for y0 in range(0, ny, ROW_CHUNK):
            y1 = min(ny, y0 + ROW_CHUNK)
            t = np.asarray(ds["water_temp"].isel(time=0, depth=slice(0, n_lev), lat=slice(y0, y1)).values, dtype=np.float64)
            s = np.asarray(ds["salinity"].isel(time=0, depth=slice(0, n_lev), lat=slice(y0, y1)).values, dtype=np.float64)

            valid = np.isfinite(t) & np.isfinite(s)
            sst_c = np.where(valid[0], t[0], np.nan)
            warm = valid[0] & (t[0] >= REF_TEMP_C)

            below = valid & (t <= REF_TEMP_C)
            any_below = below.any(axis=0) & warm
            cross = np.argmax(below, axis=0)
            prev = np.clip(cross - 1, 0, n_lev - 1)

            def take(a3, idx2):
                return np.take_along_axis(a3, idx2[None], axis=0)[0]

            z1 = depth[prev]
            z2 = depth[cross]
            t1 = take(t, prev)
            t2 = take(t, cross)
            d26_c = np.full_like(sst_c, np.nan)
            good = any_below & np.isfinite(t1) & np.isfinite(t2) & (t1 != t2)
            frac = np.zeros_like(sst_c)
            frac[good] = (t1[good] - REF_TEMP_C) / (t1[good] - t2[good])
            d26_c[good] = z1[good] + frac[good] * (z2[good] - z1[good])

            lat_chunk = np.broadcast_to(lat[y0:y1, None], sst_c.shape)
            lon_chunk = np.broadcast_to(lon[None, :], sst_c.shape)
            p = gsw.p_from_z(-depth[:, None, None], lat_chunk[None])
            sa = gsw.SA_from_SP(s, p, lon_chunk[None], lat_chunk[None])
            rho = gsw.rho_t_exact(sa, t, p)
            cp = gsw.cp_t_exact(sa, t, p)

            heat = np.clip(t - REF_TEMP_C, 0.0, None)
            d26_3 = d26_c[None]
            top = dz_top[:, None, None]
            bottom = dz_bottom[:, None, None]
            full = valid & np.isfinite(d26_3) & (bottom <= d26_3)
            partial = valid & np.isfinite(d26_3) & (top < d26_3) & (bottom > d26_3)
            eff_dz = np.zeros_like(t)
            eff_dz[full] = np.broadcast_to(bottom - top, t.shape)[full]
            eff_dz[partial] = (np.broadcast_to(d26_3, t.shape) - np.broadcast_to(top, t.shape))[partial]

            ohc_c = np.sum(heat * rho * cp * eff_dz, axis=0)
            ohc_c = np.where(np.isfinite(d26_c), ohc_c, np.nan)

            sst[y0:y1] = sst_c.astype(np.float32)
            d26[y0:y1] = d26_c.astype(np.float32)
            ohc[y0:y1] = ohc_c.astype(np.float32)
            tchp[y0:y1] = (ohc_c / 1.0e7).astype(np.float32)

    lat2d = np.broadcast_to(lat[:, None], (ny, nx)).astype(np.float32)
    lon2d = np.broadcast_to(((lon + 180.0) % 360.0 - 180.0)[None, :], (ny, nx)).astype(np.float32)
    return xr.Dataset(
        data_vars={
            "tchp_kj_per_cm2": (("Y", "X"), tchp),
            "ohc_j_per_m2": (("Y", "X"), ohc),
            "d26_m": (("Y", "X"), d26),
            "surface_temp_c": (("Y", "X"), sst),
            "ssh_m": (("Y", "X"), ssh),
            "Latitude": (("Y", "X"), lat2d),
            "Longitude": (("Y", "X"), lon2d),
        },
        coords={"Y": np.arange(ny, dtype=np.int32), "X": np.arange(nx, dtype=np.int32)},
        attrs={
            "source": f"s3://{BUCKET}",
            "source_type": "gofs31_reanalysis_z_levels",
            "max_depth_m": MAX_DEPTH_M,
            "description": "SIDE EXPLORATION: GOFS 3.1 reanalysis daily TCHP/OHC/D26 field (see OHC/exploration/README.md).",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--keep-source", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for date in args.dates:
        t0 = time.perf_counter()
        out_path = args.out_dir / f"gofs31_tchp_{date}.nc"
        if out_path.exists():
            results.append({"date": date, "status": "cached"})
            continue
        try:
            key = _s3_key_for_date(date)
            snapshot = _download(key, args.cache_dir)
            out = _compute_fields(snapshot)
            out.attrs["source_date"] = date
            out.to_netcdf(out_path)
            if not args.keep_source:
                snapshot.unlink()
            finite = int(np.isfinite(out["tchp_kj_per_cm2"].values).sum())
            results.append({"date": date, "status": "computed", "finite_tchp_cells": finite, "elapsed_s": round(time.perf_counter() - t0, 1)})
            logger.info("%s", results[-1])
        except Exception:
            logger.exception("GOFS date %s failed", date)
            results.append({"date": date, "status": "failed"})
    print(json.dumps({
        "requested": len(args.dates),
        "computed": sum(1 for r in results if r["status"] == "computed"),
        "cached": sum(1 for r in results if r["status"] == "cached"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    }, indent=2))


if __name__ == "__main__":
    main()
