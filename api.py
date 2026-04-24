"""HHP Prediction FastAPI service.

Milton 2024 replay sandbox served from a single FastAPI process together with
the built dashboard. The default ML artifact is trained on Argo-GDAC-only,
TEOS-10-backed HHP matchups from the 2024 storm set.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hhp_core import REF_TEMP_C, compute_tchp
from hhp_pipeline import get_hhp_estimate
from ml.features import build_inference_feature_row, safe_float
from ml.modeling import DEFAULT_MODEL_PATH, load_model_bundle
from ml.paths import DATASETS_DIR, IBTRACS_CACHE_DIR, LAYERS_DIR, REPO_ROOT, RTOFS_CACHE_DIR
from ml.sources.rtofs_source import cached_rtofs_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

APP_MODE = os.getenv("APP_MODE", "milton_replay").strip().lower()
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", REPO_ROOT / "dashboard" / "dist"))

PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
PROFILES_PARQUET = DATASETS_DIR / "milton_pair_profiles.parquet"
PROFILES_PKL = DATASETS_DIR / "milton_pair_profiles.pkl"
TRACK_CSV = IBTRACS_CACHE_DIR / "ibtracs_MILTON_2024.csv"

LAYER_BBOX = {"lat_min": 18.0, "lat_max": 31.0, "lon_min": -98.0, "lon_max": -80.0}
DEFAULT_LAYER_STRIDE = 10
MAX_LAYER_POINTS = 700


# in-memory state
pairs_df: pd.DataFrame | None = None
profiles_df: pd.DataFrame | None = None
track_df: pd.DataFrame | None = None
available_dates: list[str] = []
rtofs_cache: dict[str, xr.Dataset] = {}
layer_cache: dict[tuple[str, str, int], dict] = {}
ml_bundle: dict[str, Any] | None = None


def _disk_layer_path(date_yyyymmdd: str, layer: str, stride: int) -> Path:
    return LAYERS_DIR / f"{date_yyyymmdd}_{layer}_s{stride}.json"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pairs_df, profiles_df, track_df, available_dates, ml_bundle
    logger.info("Loading HHP replay state…")
    pairs_df = pd.read_csv(PAIRS_CSV)
    pairs_df["obs_date"] = pairs_df["obs_date"].astype(str)
    if PROFILES_PARQUET.exists():
        profiles_df = pd.read_parquet(PROFILES_PARQUET)
    else:
        profiles_df = pd.read_pickle(PROFILES_PKL)
    track_df = pd.read_csv(TRACK_CSV)
    ml_bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    # Available replay dates = dates we have both RTOFS files AND pairs for
    candidate_dates = sorted(pairs_df["obs_date"].unique())
    available_dates = [d for d in candidate_dates if cached_rtofs_path(d, RTOFS_CACHE_DIR).exists()]
    logger.info("Replay dates: %s", available_dates)
    logger.info("Loaded %d pairs, %d profile records, %d track points",
                len(pairs_df), len(profiles_df), len(track_df))
    logger.info("ML correction model loaded: %s", bool(ml_bundle))

    # Warm-up: prime xarray coordinate caches (per-dataset) and sklearn
    # first-predict JIT so the first user click on any replay date doesn't pay
    # the ~300 ms cold-start tax. Each RTOFS NetCDF has its own coord cache,
    # so we warm one dummy query per available date.
    if available_dates:
        warmed = 0
        for warm_date in available_dates:
            try:
                warm_ds = _get_rtofs_ds(warm_date)
                warm_iso = f"{warm_date[:4]}-{warm_date[4:6]}-{warm_date[6:]}T12:00:00Z"
                get_hhp_estimate(
                    25.0, -88.0, warm_iso, warm_ds,
                    pairs_df=pairs_df, ml_bundle=ml_bundle,
                )
                warmed += 1
            except Exception as exc:  # pragma: no cover — best-effort warm-up
                logger.warning("Warm-up failed for %s: %s", warm_date, exc)
        logger.info("API warm-up complete (%d/%d dates)", warmed, len(available_dates))

    yield
    for ds in rtofs_cache.values():
        ds.close()


app = FastAPI(
    title="HHP-Prediction API",
    description="ML-corrected Hurricane Heat Potential estimates for the Milton 2024 replay using Argo GDAC matchups and TEOS-10-backed HHP values.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_rtofs_ds(yyyymmdd: str) -> xr.Dataset:
    if yyyymmdd in rtofs_cache:
        return rtofs_cache[yyyymmdd]
    path = cached_rtofs_path(yyyymmdd, RTOFS_CACHE_DIR)
    if not path.exists():
        raise HTTPException(404, f"No RTOFS file cached for {yyyymmdd}")
    rtofs_cache[yyyymmdd] = xr.open_dataset(path)
    return rtofs_cache[yyyymmdd]


def _parse_iso_date(t: str) -> str:
    try:
        return pd.to_datetime(t, utc=True).strftime("%Y%m%d")
    except Exception as exc:
        raise HTTPException(400, f"Invalid time: {t} ({exc})") from exc


class HHPQuery(BaseModel):
    lat: float
    lon: float
    time: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": APP_MODE,
        "available_dates": available_dates,
        "pairs_loaded": int(len(pairs_df)) if pairs_df is not None else 0,
        "cached_rtofs_datasets": len(rtofs_cache),
        "cached_layers": len(layer_cache),
        "ml_model_loaded": bool(ml_bundle),
        "ml_model_name": ml_bundle.get("model_name") if ml_bundle else None,
        "ml_model_path": str(DEFAULT_MODEL_PATH),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if pairs_df is None or track_df is None:
        raise HTTPException(500, "Replay state not loaded")
    default = available_dates[-1] if available_dates else None
    return {
        "mode": APP_MODE,
        "storm": {
            "name": "MILTON",
            "season": 2024,
            "basin": "NA",
            "peak_wind_kt": int(pd.to_numeric(track_df["usa_wind"], errors="coerce").max()),
            "peak_category": int(pd.to_numeric(track_df["usa_sshs"], errors="coerce").max()),
        },
        "available_dates": available_dates,
        "available_date_count": len(available_dates),
        "default_query_time": f"{default[:4]}-{default[4:6]}-{default[6:]}T12:00:00Z" if default else None,
        "replay_pair_count": int(len(pairs_df)),
        "replay_platforms": int(pairs_df["platform"].nunique()),
        "support_radius_km": 200.0,
        "support_window_hr": 72,
        "ref_temp_c": REF_TEMP_C,
        "ml_model_loaded": bool(ml_bundle),
        "ml_model_name": ml_bundle.get("model_name") if ml_bundle else None,
        "ml_model_train_events": ml_bundle.get("train_events") if ml_bundle else None,
        "ml_model_train_rows": ml_bundle.get("train_rows") if ml_bundle else None,
        "ml_model_path": str(DEFAULT_MODEL_PATH),
        "demo_note": "Milton remains the replay event. The default ML artifact is trained on Argo GDAC TEOS-10 matchups from the 2024 storm set.",
        "bbox": LAYER_BBOX,
    }


@app.get("/track/milton")
def track_milton() -> dict[str, Any]:
    if track_df is None:
        raise HTTPException(500, "Track not loaded")
    rows = []
    for _, r in track_df.iterrows():
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
        except (ValueError, TypeError):
            continue
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        rows.append({
            "iso_time": r["iso_time"],
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "wind_kt": int(r["usa_wind"]) if pd.notna(r["usa_wind"]) else None,
            "pres_mb": int(r["usa_pres"]) if pd.notna(r["usa_pres"]) else None,
            "sshs": int(r["usa_sshs"]) if pd.notna(r["usa_sshs"]) else None,
            "nature": r["nature"] if pd.notna(r["nature"]) else None,
        })
    return {"storm": "MILTON_2024", "points": rows, "count": len(rows)}


@app.post("/tchp")
def tchp_query(req: HHPQuery) -> dict[str, Any]:
    yyyymmdd = _parse_iso_date(req.time)
    if yyyymmdd not in available_dates:
        raise HTTPException(400, f"No replay data for {yyyymmdd}. Available: {available_dates}")
    ds = _get_rtofs_ds(yyyymmdd)
    try:
        res = get_hhp_estimate(req.lat, req.lon, req.time, ds, pairs_df=pairs_df, ml_bundle=ml_bundle)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return asdict(res)


@app.get("/profile")
def profile(cast_id: str = Query(...)) -> dict[str, Any]:
    if profiles_df is None:
        raise HTTPException(500, "Profile store not loaded")
    rows = profiles_df[profiles_df["cast_id"] == cast_id]
    if rows.empty:
        raise HTTPException(404, f"No profile arrays for cast {cast_id}")
    r = rows.iloc[0]

    def _compute(vertical, temp, salinity, *, lat, lon, vertical_axis):
        res = compute_tchp(
            np.asarray(vertical),
            np.asarray(temp),
            salinity_psu=np.asarray(salinity),
            lat=float(lat),
            lon=float(lon),
            vertical_axis=vertical_axis,
        )
        return {"tchp_kj_cm2": res.tchp_kj_per_cm2, "d26_m": res.d26_m, "surface_t_c": res.surface_temp_c}

    argo_depth = np.asarray(r["obs_pressure_dbar"], dtype=float)
    argo_temp = np.asarray(r["obs_temperature_c"], dtype=float)
    argo_sal = np.asarray(r["obs_salinity_psu"], dtype=float)
    model_depth = np.asarray(r["model_depth_m"], dtype=float)
    model_temp = np.asarray(r["model_temperature_c"], dtype=float)
    model_sal = np.asarray(r["model_salinity_psu"], dtype=float)
    meta_row = pairs_df[pairs_df["cast_id"] == cast_id].iloc[0]

    argo_stats = _compute(
        argo_depth,
        argo_temp,
        argo_sal,
        lat=float(meta_row["lat"]),
        lon=float(meta_row["lon"]),
        vertical_axis="pressure",
    )
    model_stats = _compute(
        model_depth,
        model_temp,
        model_sal,
        lat=float(meta_row["model_grid_lat"]),
        lon=float(meta_row["model_grid_lon"]),
        vertical_axis="depth",
    )

    correction_delta = None
    correction_source = "raw_rtofs"
    corrected_tchp = model_stats["tchp_kj_cm2"]
    corrected_d26 = model_stats["d26_m"]
    corrected_depth = model_depth
    corrected_temp = model_temp.copy()

    if ml_bundle is not None and model_stats["tchp_kj_cm2"] is not None:
        model = ml_bundle.get("model")
        if model is not None:
            feat = build_inference_feature_row(
                query_time=str(meta_row["obs_time"]),
                lat=float(meta_row["lat"]),
                lon=float(meta_row["lon"]),
                model_tchp_kj_cm2=model_stats["tchp_kj_cm2"],
                model_d26_m=model_stats["d26_m"],
                model_surface_t_c=model_stats["surface_t_c"],
                model_grid_distance_km=float(meta_row["model_grid_distance_km"]),
            )
            predicted_delta = safe_float(model.predict(feat)[0])
            if predicted_delta is not None:
                correction_delta = float(predicted_delta)
                correction_source = str(ml_bundle.get("model_name", "ml_correction"))
                if corrected_tchp is not None:
                    corrected_tchp = max(0.0, corrected_tchp + predicted_delta)
                raw_tchp = model_stats["tchp_kj_cm2"]
                if raw_tchp and raw_tchp > 0 and corrected_tchp is not None:
                    excess = np.clip(model_temp - REF_TEMP_C, 0.0, None)
                    scale = max(0.0, corrected_tchp / raw_tchp)
                    corrected_temp = np.where(model_temp >= REF_TEMP_C, REF_TEMP_C + excess * scale, model_temp)

    return {
        "cast_id": cast_id,
        "obs_time": meta_row["obs_time"],
        "lat": float(meta_row["lat"]),
        "lon": float(meta_row["lon"]),
        "platform": str(meta_row["platform"]),
        "argo": {
            "depth_m": list(map(float, argo_depth)),  # dbar ≈ m in upper ocean
            "temperature_c": list(map(float, argo_temp)),
            "salinity_psu": list(map(float, argo_sal)),
            **argo_stats,
        },
        "model": {
            "depth_m": list(map(float, model_depth)),
            "temperature_c": list(map(float, model_temp)),
            "salinity_psu": list(map(float, model_sal)),
            **model_stats,
        },
        "corrected": {
            "correction_delta_kj_cm2": round(float(correction_delta), 2) if correction_delta is not None else None,
            "correction_source": correction_source,
            "tchp_kj_cm2": corrected_tchp,
            "d26_m": corrected_d26,
            "depth_m": list(map(float, corrected_depth)),
            "temperature_c": list(map(float, corrected_temp)),
            "curve_note": "Proxy corrected curve: the model profile shape is preserved while heat content above 26C is rescaled to the ML-corrected TCHP.",
        },
        "delta_tchp_kj_cm2": (
            argo_stats["tchp_kj_cm2"] - model_stats["tchp_kj_cm2"]
            if (argo_stats["tchp_kj_cm2"] is not None and model_stats["tchp_kj_cm2"] is not None)
            else None
        ),
    }


def _predict_delta_for_point(
    *,
    lat: float,
    lon: float,
    query_time: str,
    model_tchp_kj_cm2: float | None,
    model_d26_m: float | None,
    model_surface_t_c: float | None,
    model_grid_distance_km: float,
) -> float | None:
    if ml_bundle is None or model_tchp_kj_cm2 is None:
        return None
    model = ml_bundle.get("model")
    if model is None:
        return None
    feat = build_inference_feature_row(
        query_time=query_time,
        lat=lat,
        lon=lon,
        model_tchp_kj_cm2=model_tchp_kj_cm2,
        model_d26_m=model_d26_m,
        model_surface_t_c=model_surface_t_c,
        model_grid_distance_km=model_grid_distance_km,
    )
    return safe_float(model.predict(feat)[0])


def _build_layer(ds: xr.Dataset, layer_kind: str, stride: int, query_time: str) -> dict:
    stride = max(1, int(stride))
    lats = ds["Latitude"].values
    lons = ds["Longitude"].values
    depth = np.asarray(ds["Depth"].values, dtype=float)
    temp = np.asarray(ds["temperature"].isel(MT=0).values, dtype=float)
    sal = np.asarray(ds["salinity"].isel(MT=0).values, dtype=float)
    layer_time = str(ds["MT"].isel(MT=0).values)

    points = []
    for y in range(0, lats.shape[0], stride):
        for x in range(0, lats.shape[1], stride):
            lat = float(lats[y, x]); lon = float(lons[y, x])
            if not (LAYER_BBOX["lat_min"] <= lat <= LAYER_BBOX["lat_max"]):
                continue
            if not (LAYER_BBOX["lon_min"] <= lon <= LAYER_BBOX["lon_max"]):
                continue
            col_t = temp[:, y, x]
            if not np.isfinite(col_t[0]):
                continue
            col_s = sal[:, y, x]
            res = compute_tchp(
                depth,
                col_t,
                salinity_psu=col_s,
                lat=lat,
                lon=lon,
                vertical_axis="depth",
            )
            if layer_kind == "tchp":
                val = res.tchp_kj_per_cm2
            elif layer_kind == "corrected_tchp":
                delta = _predict_delta_for_point(
                    lat=lat,
                    lon=lon,
                    query_time=query_time,
                    model_tchp_kj_cm2=res.tchp_kj_per_cm2,
                    model_d26_m=res.d26_m,
                    model_surface_t_c=res.surface_temp_c,
                    model_grid_distance_km=0.0,
                )
                val = (res.tchp_kj_per_cm2 + delta) if (res.tchp_kj_per_cm2 is not None and delta is not None) else res.tchp_kj_per_cm2
            elif layer_kind == "correction":
                val = _predict_delta_for_point(
                    lat=lat,
                    lon=lon,
                    query_time=query_time,
                    model_tchp_kj_cm2=res.tchp_kj_per_cm2,
                    model_d26_m=res.d26_m,
                    model_surface_t_c=res.surface_temp_c,
                    model_grid_distance_km=0.0,
                )
            elif layer_kind == "d26":
                val = res.d26_m
            else:
                raise HTTPException(400, f"Unsupported layer {layer_kind}")
            if val is None or not np.isfinite(val):
                continue
            points.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "value": round(float(val), 2),
            })
            if len(points) >= MAX_LAYER_POINTS:
                break
        if len(points) >= MAX_LAYER_POINTS:
            break
    values = [p["value"] for p in points]
    return {
        "layer": layer_kind,
        "time": layer_time,
        "point_count": len(points),
        "stride": stride,
        "bbox": LAYER_BBOX,
        "value_min": min(values) if values else None,
        "value_max": max(values) if values else None,
        "points": points,
    }


@app.get("/map_layer")
def map_layer(
    time: str = Query(...),
    layer: str = Query("tchp"),
    stride: int = Query(DEFAULT_LAYER_STRIDE, ge=2, le=40),
) -> dict[str, Any]:
    yyyymmdd = _parse_iso_date(time)
    if yyyymmdd not in available_dates:
        raise HTTPException(400, f"No replay data for {yyyymmdd}")
    if layer not in {"tchp", "corrected_tchp", "correction", "d26", "observations"}:
        raise HTTPException(400, f"Unsupported layer '{layer}'")

    disk_path = _disk_layer_path(yyyymmdd, layer, stride)
    if disk_path.exists():
        return json.loads(disk_path.read_text(encoding="utf-8"))

    if layer == "observations":
        key = (f"obs:{yyyymmdd}", layer, stride)
        if key not in layer_cache:
            day = pairs_df[pairs_df["obs_date"] == yyyymmdd]
            points = [{
                "lat": round(float(r["lat"]), 4),
                "lon": round(float(r["lon"]), 4),
                "value": round(float(r["obs_tchp_kj_cm2"]), 2) if pd.notna(r["obs_tchp_kj_cm2"]) else None,
                "model_value": round(float(r["model_tchp_kj_cm2"]), 2) if pd.notna(r["model_tchp_kj_cm2"]) else None,
                "cast_id": r["cast_id"],
                "platform": str(r["platform"]),
                "obs_time": r["obs_time"],
            } for _, r in day.iterrows()]
            layer_cache[key] = {"layer": "observations", "time": yyyymmdd, "points": points,
                                 "point_count": len(points)}
        return layer_cache[key]

    key = (yyyymmdd, layer, stride)
    if key not in layer_cache:
        layer_cache[key] = _build_layer(_get_rtofs_ds(yyyymmdd), layer, stride, time)
    return layer_cache[key]


# Catch-all to serve the built dashboard (must be LAST route)
@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    dist_dir = FRONTEND_DIST_DIR.resolve()
    index_path = dist_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend build not found. Run dashboard build first.")
    requested = (dist_dir / full_path).resolve()
    if requested.is_file() and requested.is_relative_to(dist_dir):
        return FileResponse(requested)
    return FileResponse(index_path)
