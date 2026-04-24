"""HHP query orchestration and provenance assembly."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hhp_core import compute_tchp
from ml.features import build_inference_feature_row, safe_float

logger = logging.getLogger(__name__)

DEFAULT_SUPPORT_RADIUS_KM = 200.0
DEFAULT_SUPPORT_WINDOW_HR = 72


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371.0


@dataclass
class HHPEstimateResult:
    query_lat: float
    query_lon: float
    query_time: str
    best_tchp_kj_cm2: float | None
    best_d26_m: float | None
    confidence: str
    model_tchp_kj_cm2: float | None
    model_d26_m: float | None
    model_surface_t_c: float | None
    model_grid_lat: float
    model_grid_lon: float
    correction_delta_kj_cm2: float | None = None
    correction_source: str = "raw_rtofs"
    nearby_observations: list[dict[str, Any]] = field(default_factory=list)
    support_radius_km: float = DEFAULT_SUPPORT_RADIUS_KM
    support_window_hr: int = DEFAULT_SUPPORT_WINDOW_HR


def _nearest_rtofs_column(ds: xr.Dataset, lat: float, lon: float) -> dict | None:
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
    depth = np.asarray(ds["Depth"].values, dtype=float)
    t_col = np.asarray(ds["temperature"].isel(MT=0, Y=iy, X=ix).values, dtype=float)
    s_col = np.asarray(ds["salinity"].isel(MT=0, Y=iy, X=ix).values, dtype=float)
    mask = np.isfinite(depth) & np.isfinite(t_col) & np.isfinite(s_col)
    return {
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "depth_m": depth[mask],
        "temperature_c": t_col[mask],
        "salinity_psu": s_col[mask],
        "distance_km": _haversine_km(lon, lat, grid_lon, grid_lat),
    }


def _confidence(model_loaded: bool, nearby_count: int) -> str:
    if not model_loaded:
        return "Low"
    if nearby_count >= 3:
        return "High"
    if nearby_count >= 1:
        return "Medium"
    return "Low"


def find_nearby_observations(
    pairs_df: pd.DataFrame,
    lat: float,
    lon: float,
    query_time: str,
    radius_km: float = DEFAULT_SUPPORT_RADIUS_KM,
    time_window_hr: int = DEFAULT_SUPPORT_WINDOW_HR,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if pairs_df.empty:
        return []
    qt = pd.to_datetime(query_time, utc=True)
    obs_times = pd.to_datetime(pairs_df["obs_time"], utc=True, errors="coerce")
    mask = (obs_times >= qt - pd.Timedelta(hours=time_window_hr)) & (
        obs_times <= qt + pd.Timedelta(hours=time_window_hr)
    )
    window = pairs_df.loc[mask].copy()
    if window.empty:
        return []
    window["distance_km"] = window.apply(
        lambda r: _haversine_km(lon, lat, float(r["lon"]), float(r["lat"])), axis=1
    )
    window = window[window["distance_km"] <= radius_km]
    if window.empty:
        return []
    window["time_delta_hr"] = (
        pd.to_datetime(window["obs_time"], utc=True) - qt
    ).abs().dt.total_seconds() / 3600.0
    window = window.sort_values(["distance_km", "time_delta_hr"]).head(limit)

    out: list[dict[str, Any]] = []
    for _, row in window.iterrows():
        out.append({
            "cast_id": row["cast_id"],
            "platform": str(row["platform"]),
            "obs_time": row["obs_time"],
            "lat": round(float(row["lat"]), 4),
            "lon": round(float(row["lon"]), 4),
            "distance_km": round(float(row["distance_km"]), 2),
            "time_delta_hr": round(float(row["time_delta_hr"]), 2),
            "obs_tchp_kj_cm2": round(float(row["obs_tchp_kj_cm2"]), 2) if pd.notna(row["obs_tchp_kj_cm2"]) else None,
            "obs_d26_m": round(float(row["obs_d26_m"]), 1) if pd.notna(row["obs_d26_m"]) else None,
            "obs_surface_t_c": round(float(row["obs_surface_t_c"]), 2) if pd.notna(row["obs_surface_t_c"]) else None,
            "model_tchp_kj_cm2": round(float(row["model_tchp_kj_cm2"]), 2) if pd.notna(row["model_tchp_kj_cm2"]) else None,
            "tchp_delta_kj_cm2": round(float(row["tchp_delta_kj_cm2"]), 2) if pd.notna(row["tchp_delta_kj_cm2"]) else None,
        })
    return out


def get_hhp_estimate(
    lat: float,
    lon: float,
    query_time: str,
    rtofs_ds: xr.Dataset,
    pairs_df: pd.DataFrame | None = None,
    ml_bundle: dict[str, Any] | None = None,
    support_radius_km: float = DEFAULT_SUPPORT_RADIUS_KM,
    support_window_hr: int = DEFAULT_SUPPORT_WINDOW_HR,
) -> HHPEstimateResult:
    col = _nearest_rtofs_column(rtofs_ds, lat, lon)
    if col is None:
        raise ValueError("No valid RTOFS ocean column near this point.")

    model_res = compute_tchp(
        col["depth_m"],
        col["temperature_c"],
        salinity_psu=col["salinity_psu"],
        lat=col["grid_lat"],
        lon=col["grid_lon"],
        vertical_axis="depth",
    )
    nearby = []
    if pairs_df is not None:
        nearby = find_nearby_observations(
            pairs_df, lat, lon, query_time,
            radius_km=support_radius_km, time_window_hr=support_window_hr,
        )

    best_tchp = model_res.tchp_kj_per_cm2
    best_d26 = model_res.d26_m
    raw_model_tchp = model_res.tchp_kj_per_cm2
    model_loaded = True
    correction_delta = None
    correction_source = "raw_rtofs"

    if ml_bundle is not None and model_res.tchp_kj_per_cm2 is not None:
        model = ml_bundle.get("model")
        if model is not None:
            feat = build_inference_feature_row(
                query_time=query_time,
                lat=lat,
                lon=lon,
                model_tchp_kj_cm2=model_res.tchp_kj_per_cm2,
                model_d26_m=model_res.d26_m,
                model_surface_t_c=model_res.surface_temp_c,
                model_grid_distance_km=col["distance_km"],
            )
            predicted_delta = safe_float(model.predict(feat)[0])
            if predicted_delta is not None:
                correction_delta = predicted_delta
                best_tchp = model_res.tchp_kj_per_cm2 + predicted_delta
                correction_source = str(ml_bundle.get("model_name", "ml_correction"))

    return HHPEstimateResult(
        query_lat=lat,
        query_lon=lon,
        query_time=query_time,
        best_tchp_kj_cm2=round(best_tchp, 2) if best_tchp is not None else None,
        best_d26_m=round(best_d26, 2) if best_d26 is not None else None,
        confidence=_confidence(model_loaded, len(nearby)),
        model_tchp_kj_cm2=round(raw_model_tchp, 2) if raw_model_tchp is not None else None,
        model_d26_m=round(best_d26, 2) if best_d26 is not None else None,
        model_surface_t_c=round(model_res.surface_temp_c, 2) if model_res.surface_temp_c is not None else None,
        model_grid_lat=round(col["grid_lat"], 4),
        model_grid_lon=round(col["grid_lon"], 4),
        correction_delta_kj_cm2=round(correction_delta, 2) if correction_delta is not None else None,
        correction_source=correction_source,
        nearby_observations=nearby,
        support_radius_km=support_radius_km,
        support_window_hr=support_window_hr,
    )
