"""SIDE EXPLORATION: WOA23 climatological TCHP/D26 priors at collocated points.

Tier-1 features from SPATIAL_FEATURES_DIRECTIONS.md. Downloads the WOA23
decadal-average monthly temperature and salinity climatology (1 deg pilot
resolution, 0-1500 m), computes climatological TCHP/D26 per month with the
same z-level TEOS-10 integration used elsewhere, and samples them at each
collocated Argo point by calendar month:

- woa_tchp_clim_kj_per_cm2, woa_d26_clim_m, woa_sst_clim_c
- model_minus_woa_tchp, model_minus_woa_d26 (RTOFS anomaly vs climatology)

This is the domain-standard "where am I, physically" encoding (the operational
AOML/NESDIS TCHP products are climatology + altimetry based). It adds a
non-RTOFS dependency, hence side exploration. Upgrade path: 0.25 deg files
(same URL pattern with 0.25/_04 suffix).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

IN_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_woa_clim.parquet")
OUT_SUMMARY = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/summary_collocation_2024_2025_woa_clim.json")
WOA_CACHE = Path("/data/suramya/woa23_cache")
CLIM_CACHE = Path("/data/suramya/woa23_clim_fields")
BASE_URL = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"
REF_TEMP_C = 26.0


def _download(var: str, month: int) -> Path:
    letter = var[0]
    name = f"woa23_decav_{letter}{month:02d}_01.nc"
    dest = WOA_CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    WOA_CACHE.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{var}/netcdf/decav/1.00/{name}"
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest


def _monthly_clim_fields(month: int) -> xr.Dataset:
    CLIM_CACHE.mkdir(parents=True, exist_ok=True)
    cached = CLIM_CACHE / f"woa23_clim_ohc_m{month:02d}.nc"
    if cached.exists():
        return xr.open_dataset(cached)

    t_path = _download("temperature", month)
    s_path = _download("salinity", month)
    with xr.open_dataset(t_path, decode_times=False) as tds, xr.open_dataset(s_path, decode_times=False) as sds:
        depth = np.asarray(tds["depth"].values, dtype=np.float64)
        lat = np.asarray(tds["lat"].values, dtype=np.float64)
        lon = np.asarray(tds["lon"].values, dtype=np.float64)
        t = np.asarray(tds["t_an"].isel(time=0).values, dtype=np.float64)
        s = np.asarray(sds["s_an"].isel(time=0).values, dtype=np.float64)

    n_lev, ny, nx = t.shape
    valid = np.isfinite(t) & np.isfinite(s)
    sst = np.where(valid[0], t[0], np.nan)
    warm = valid[0] & (t[0] >= REF_TEMP_C)

    below = valid & (t <= REF_TEMP_C)
    any_below = below.any(axis=0) & warm
    cross = np.argmax(below, axis=0)
    prev = np.clip(cross - 1, 0, n_lev - 1)

    def take(a3, idx2):
        return np.take_along_axis(a3, idx2[None], axis=0)[0]

    z1, z2 = depth[prev], depth[cross]
    t1, t2 = take(t, prev), take(t, cross)
    d26 = np.full_like(sst, np.nan)
    good = any_below & np.isfinite(t1) & np.isfinite(t2) & (t1 != t2)
    frac = np.zeros_like(sst)
    frac[good] = (t1[good] - REF_TEMP_C) / (t1[good] - t2[good])
    d26[good] = z1[good] + frac[good] * (z2[good] - z1[good])

    lat3 = np.broadcast_to(lat[None, :, None], t.shape)
    lon3 = np.broadcast_to(lon[None, None, :], t.shape)
    p = gsw.p_from_z(-depth[:, None, None], lat3)
    sa = gsw.SA_from_SP(s, p, lon3, lat3)
    rho = gsw.rho_t_exact(sa, t, p)
    cp = gsw.cp_t_exact(sa, t, p)

    edges = np.concatenate([[0.0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1]]])
    top = edges[:-1][:, None, None]
    bottom = edges[1:][:, None, None]
    heat = np.clip(t - REF_TEMP_C, 0.0, None)
    d26_3 = d26[None]
    full = valid & np.isfinite(d26_3) & (bottom <= d26_3)
    partial = valid & np.isfinite(d26_3) & (top < d26_3) & (bottom > d26_3)
    eff_dz = np.zeros_like(t)
    eff_dz[full] = np.broadcast_to(bottom - top, t.shape)[full]
    eff_dz[partial] = (np.broadcast_to(d26_3, t.shape) - np.broadcast_to(top, t.shape))[partial]
    ohc = np.sum(heat * rho * cp * eff_dz, axis=0)
    ohc = np.where(np.isfinite(d26), ohc, np.nan)

    out = xr.Dataset(
        data_vars={
            "woa_tchp_clim_kj_per_cm2": (("lat", "lon"), (ohc / 1.0e7).astype(np.float32)),
            "woa_d26_clim_m": (("lat", "lon"), d26.astype(np.float32)),
            "woa_sst_clim_c": (("lat", "lon"), sst.astype(np.float32)),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"source": "WOA23 decav 1.00deg monthly", "month": month},
    )
    out.to_netcdf(cached)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    t0 = time.perf_counter()
    df = pd.read_parquet(IN_PATH).copy()
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy().reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    df["year"] = df["year"].astype(int)

    n = len(df)
    tchp_clim = np.full(n, np.nan, dtype=np.float32)
    d26_clim = np.full(n, np.nan, dtype=np.float32)
    sst_clim = np.full(n, np.nan, dtype=np.float32)

    months = df["month"].astype(int).to_numpy()
    lats = df["lat"].to_numpy(float)
    lons = ((df["lon"].to_numpy(float) + 180.0) % 360.0) - 180.0

    for month in sorted(np.unique(months)):
        clim = _monthly_clim_fields(int(month))
        idx = np.where(months == month)[0]
        lat_idx = np.clip(np.searchsorted(clim["lat"].values, lats[idx]), 0, clim.sizes["lat"] - 1)
        lon_idx = np.clip(np.searchsorted(clim["lon"].values, lons[idx]), 0, clim.sizes["lon"] - 1)
        tchp_clim[idx] = clim["woa_tchp_clim_kj_per_cm2"].values[lat_idx, lon_idx]
        d26_clim[idx] = clim["woa_d26_clim_m"].values[lat_idx, lon_idx]
        sst_clim[idx] = clim["woa_sst_clim_c"].values[lat_idx, lon_idx]
        clim.close()
        logger.info("month %02d: %d rows sampled", month, len(idx))

    df["woa_tchp_clim_kj_per_cm2"] = tchp_clim
    df["woa_d26_clim_m"] = d26_clim
    df["woa_sst_clim_c"] = sst_clim
    df["model_minus_woa_tchp"] = (df["model_interp_tchp_kj_per_cm2"].to_numpy(np.float64) - tchp_clim).astype(np.float32)
    df["model_minus_woa_d26"] = (df["model_interp_d26_m"].to_numpy(np.float64) - d26_clim).astype(np.float32)

    df.to_parquet(OUT_PATH, index=False)
    new_cols = ["woa_tchp_clim_kj_per_cm2", "woa_d26_clim_m", "woa_sst_clim_c", "model_minus_woa_tchp", "model_minus_woa_d26"]
    summary = {
        "output_path": str(OUT_PATH),
        "rows_total": n,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "feature_availability": {c: int(np.isfinite(pd.to_numeric(df[c], errors="coerce")).sum()) for c in new_cols},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
