"""Build the co-location pairs table for Hurricane Milton 2024.

Given the manifest of Argo profiles (milton_argo_profile_manifest.csv), match
each profile to its same-day RTOFS 3dz file, extract the model T/S column at
the nearest valid ocean grid point, compute TCHP on both the observation and
the model, and emit:

  artifacts/datasets/milton_pairs.csv
      one row per profile with lat, lon, obs_time, D26 (obs/model), TCHP
      (obs/model), delta, distance to matched grid point, and metadata

  artifacts/datasets/milton_pair_profiles.parquet
      per-profile arrays (P/Z, T, S) for both Argo and the matched RTOFS
      column, keyed by cast_id. Used later for Leipper-style profile plots.
"""
from __future__ import annotations

import logging
import sys
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hhp_core import compute_d26, compute_tchp
from ml.paths import ARGO_CACHE_DIR, DATASETS_DIR, RTOFS_CACHE_DIR
from ml.sources.rtofs_source import cached_rtofs_path

logger = logging.getLogger(__name__)

MANIFEST_CSV = DATASETS_DIR / "milton_argo_profile_manifest.csv"
PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
PROFILES_PARQUET = DATASETS_DIR / "milton_pair_profiles.parquet"


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371.0


def _obs_date_yyyymmdd(obs_time: str) -> str:
    return obs_time[:10].replace("-", "")


def _load_argo_profile(cast_path: str) -> dict:
    """Re-open the original Argo NetCDF and return its first QC-clean profile's
    P, T, S arrays. We don't persist these in the manifest, so reload on demand.
    """
    local = ARGO_CACHE_DIR / cast_path
    with xr.open_dataset(local, decode_timedelta=False) as ds:
        # Match the logic used in argo_gdac_source.py — prefer *_ADJUSTED when
        # any finite values exist, and accept QC flags 1 or 2.
        def pref(name_raw, name_adj):
            if name_adj in ds and np.isfinite(ds[name_adj].values).any():
                qc = f"{name_adj}_QC"
                return name_adj, qc if qc in ds else None
            qc = f"{name_raw}_QC"
            return name_raw, qc if qc in ds else None

        p_var, p_qc = pref("PRES", "PRES_ADJUSTED")
        t_var, t_qc = pref("TEMP", "TEMP_ADJUSTED")
        s_var, s_qc = pref("PSAL", "PSAL_ADJUSTED")

        p = np.asarray(ds[p_var].values, dtype=float)
        t = np.asarray(ds[t_var].values, dtype=float)
        s = np.asarray(ds[s_var].values, dtype=float)
        if p.ndim == 1:
            p, t, s = p.reshape(1, -1), t.reshape(1, -1), s.reshape(1, -1)

        def _qc_mask(name):
            if name is None:
                return np.ones_like(p, dtype=bool)
            vals = ds[name].values
            flat = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals.ravel()])
            flat = flat.reshape(vals.shape)
            return np.isin(flat, ["1", "2"])

        p_qc_m = _qc_mask(p_qc)
        t_qc_m = _qc_mask(t_qc)
        s_qc_m = _qc_mask(s_qc)

        # Use the first profile whose surviving depth reaches 200 dbar (matches
        # the manifest filter so we get exactly one).
        for idx in range(p.shape[0]):
            pi, ti, si = p[idx], t[idx], s[idx]
            valid = (np.isfinite(pi) & np.isfinite(ti) & np.isfinite(si)
                     & p_qc_m[idx] & t_qc_m[idx] & s_qc_m[idx])
            pi, ti, si = pi[valid], ti[valid], si[valid]
            if len(pi) < 5 or pi.max() < 200:
                continue
            order = np.argsort(pi)
            return {
                "pressure_dbar": pi[order].astype(float),
                "temperature_c": ti[order].astype(float),
                "salinity_psu": si[order].astype(float),
            }
        return {}


def _nearest_rtofs_column(ds: xr.Dataset, lat: float, lon: float) -> dict | None:
    """Return T(z), S(z), grid lat/lon, distance for nearest valid ocean column."""
    lats = ds["Latitude"].values
    lons = ds["Longitude"].values
    surf_t = ds["temperature"].isel(MT=0, Depth=0).values
    valid = np.isfinite(surf_t) & np.isfinite(lats) & np.isfinite(lons)
    if not valid.any():
        return None

    dist2 = (lats - lat) ** 2 + (lons - lon) ** 2
    dist2[~valid] = np.inf
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    grid_lat = float(lats[iy, ix])
    grid_lon = float(lons[iy, ix])
    dist_km = _haversine_km(lon, lat, grid_lon, grid_lat)

    depth = np.asarray(ds["Depth"].values, dtype=float)
    t_col = np.asarray(ds["temperature"].isel(MT=0, Y=iy, X=ix).values, dtype=float)
    s_col = np.asarray(ds["salinity"].isel(MT=0, Y=iy, X=ix).values, dtype=float)
    mask = np.isfinite(depth) & np.isfinite(t_col) & np.isfinite(s_col)
    if not mask.any():
        return None
    return {
        "depth_m": depth[mask],
        "temperature_c": t_col[mask],
        "salinity_psu": s_col[mask],
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "y_index": int(iy),
        "x_index": int(ix),
        "distance_km": round(dist_km, 3),
    }


def build_pairs() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_CSV)
    df["obs_date"] = df["obs_time"].apply(_obs_date_yyyymmdd)

    rows = []
    profile_records = []
    for obs_date, group in df.groupby("obs_date"):
        rtofs_path = cached_rtofs_path(obs_date, RTOFS_CACHE_DIR)
        if not rtofs_path.exists():
            logger.warning("Skipping %s — no RTOFS file at %s", obs_date, rtofs_path)
            continue
        logger.info("Pairing %s: %d Argo profiles", obs_date, len(group))
        ds = xr.open_dataset(rtofs_path)
        try:
            for _, row in group.iterrows():
                cast_path = row["source_file"]
                argo = _load_argo_profile(cast_path)
                if not argo:
                    logger.warning("No usable Argo profile in %s", cast_path)
                    continue

                obs_res = compute_tchp(
                    argo["pressure_dbar"],
                    argo["temperature_c"],
                    salinity_psu=argo["salinity_psu"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    vertical_axis="pressure",
                )
                col = _nearest_rtofs_column(ds, float(row["lat"]), float(row["lon"]))
                if col is None:
                    logger.warning("No valid RTOFS column near %s", cast_path)
                    continue
                mod_res = compute_tchp(
                    col["depth_m"],
                    col["temperature_c"],
                    salinity_psu=col["salinity_psu"],
                    lat=float(col["grid_lat"]),
                    lon=float(col["grid_lon"]),
                    vertical_axis="depth",
                )

                obs_tchp = obs_res.tchp_kj_per_cm2
                mod_tchp = mod_res.tchp_kj_per_cm2
                delta = (obs_tchp - mod_tchp) if (obs_tchp is not None and mod_tchp is not None) else None

                rows.append({
                    "cast_id": cast_path,
                    "platform": row["platform"],
                    "obs_time": row["obs_time"],
                    "obs_date": obs_date,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "obs_surface_t_c": obs_res.surface_temp_c,
                    "obs_d26_m": obs_res.d26_m,
                    "obs_tchp_kj_cm2": obs_tchp,
                    "obs_max_depth_m": obs_res.max_depth_m,
                    "model_grid_lat": col["grid_lat"],
                    "model_grid_lon": col["grid_lon"],
                    "model_grid_distance_km": col["distance_km"],
                    "model_surface_t_c": mod_res.surface_temp_c,
                    "model_d26_m": mod_res.d26_m,
                    "model_tchp_kj_cm2": mod_tchp,
                    "model_max_depth_m": mod_res.max_depth_m,
                    "tchp_delta_kj_cm2": delta,
                })
                profile_records.append({
                    "cast_id": cast_path,
                    "obs_date": obs_date,
                    "obs_pressure_dbar": argo["pressure_dbar"].tolist(),
                    "obs_temperature_c": argo["temperature_c"].tolist(),
                    "obs_salinity_psu": argo["salinity_psu"].tolist(),
                    "model_depth_m": col["depth_m"].tolist(),
                    "model_temperature_c": col["temperature_c"].tolist(),
                    "model_salinity_psu": col["salinity_psu"].tolist(),
                })
        finally:
            ds.close()

    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(PAIRS_CSV, index=False)
    logger.info("Wrote %d pairs to %s", len(pairs_df), PAIRS_CSV)

    profiles_df = pd.DataFrame(profile_records)
    try:
        profiles_df.to_parquet(PROFILES_PARQUET, index=False)
        logger.info("Wrote %d profile records to %s", len(profiles_df), PROFILES_PARQUET)
    except Exception as exc:
        # Parquet may fail without pyarrow; fall back to pickle
        fallback = PROFILES_PARQUET.with_suffix(".pkl")
        profiles_df.to_pickle(fallback)
        logger.warning("Parquet write failed (%s); wrote pickle instead: %s", exc, fallback)

    return pairs_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_pairs()
