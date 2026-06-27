"""Common helpers for seasonal OHC/TCHP map rendering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import json

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import PolyCollection
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

DATA_DIR = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "seasonal_maps"
LAND_GEOJSON = Path(__file__).resolve().parent / "assets" / "ne_110m_land.geojson"

EARTH_R_KM = 6371.0
SEASONS = {
    "winter_jfm": ("Winter (Jan-Feb-Mar)", [1, 2, 3]),
    "summer_jas": ("Summer (Jul-Aug-Sep)", [7, 8, 9]),
}


@dataclass(frozen=True)
class ParamConfig:
    key: str
    label: str
    bins: tuple[float, ...]
    cmap: str
    valid_min: float | None = None
    valid_max: float | None = None


PARAMS: dict[str, ParamConfig] = {
    "tchp_kj_per_cm2": ParamConfig(
        key="tchp_kj_per_cm2",
        label="TCHP / OHC (kJ/cm²)",
        bins=(0, 5, 15, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300),
        cmap="plasma",
        valid_min=0.0,
    ),
    "d26_m": ParamConfig(
        key="d26_m",
        label="D26 (m)",
        bins=(0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 220),
        cmap="viridis",
        valid_min=0.0,
    ),
}

_LAND_POLYGONS: list[np.ndarray] | None = None


def load_clean_dataframe(years: list[int], months: list[int], param_key: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR, columns=["year", "month", "lat", "lon", param_key, "error"])
    df = df[df["error"].isna()].copy()
    df = df[df["year"].isin(years) & df["month"].isin(months)].copy()
    cfg = PARAMS[param_key]
    df = df[np.isfinite(df[param_key])].copy()
    if cfg.valid_min is not None:
        df = df[df[param_key] >= cfg.valid_min].copy()
    if cfg.valid_max is not None:
        df = df[df[param_key] <= cfg.valid_max].copy()
    return df


def make_norm(param_key: str) -> BoundaryNorm:
    cfg = PARAMS[param_key]
    return BoundaryNorm(cfg.bins, ncolors=256, clip=True)


def _load_land_polygons() -> list[np.ndarray]:
    global _LAND_POLYGONS
    if _LAND_POLYGONS is not None:
        return _LAND_POLYGONS
    obj = json.loads(LAND_GEOJSON.read_text())
    polys: list[np.ndarray] = []
    for feature in obj.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            if coords:
                polys.append(np.asarray(coords[0], dtype=float))
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly:
                    polys.append(np.asarray(poly[0], dtype=float))
    _LAND_POLYGONS = polys
    return polys


def add_land_overlay(ax, *, facecolor="#d8d8d8", edgecolor="#444444", linewidth=0.35, zorder=5):
    polys = _load_land_polygons()
    coll = PolyCollection(
        polys,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        closed=True,
        zorder=zorder,
    )
    ax.add_collection(coll)
    return coll


def build_global_grid(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat_centers = np.arange(-90.0 + resolution_deg / 2.0, 90.0, resolution_deg)
    lon_centers = np.arange(-180.0 + resolution_deg / 2.0, 180.0, resolution_deg)
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)
    return grid_lat, grid_lon


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


def chord_from_km(distance_km: float) -> float:
    return 2.0 * math.sin((distance_km / EARTH_R_KM) / 2.0)


def gaussian_interpolate(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    *,
    length_scale_km: float = 100.0,
    truncation_radius_km: float = 300.0,
    mask_distance_km: float | None = None,
) -> np.ndarray:
    obs_xyz = latlon_to_xyz(lat, lon)
    tree = cKDTree(obs_xyz)
    query_xyz = latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel())
    radius_chord = chord_from_km(truncation_radius_km)
    idx_lists = tree.query_ball_point(query_xyz, r=radius_chord, workers=-1)
    out = np.full(query_xyz.shape[0], np.nan, dtype=float)
    mask_chord = chord_from_km(mask_distance_km) if mask_distance_km is not None else None
    for i, idxs in enumerate(idx_lists):
        if not idxs:
            continue
        pts = obs_xyz[idxs]
        chord = np.linalg.norm(pts - query_xyz[i], axis=1)
        if mask_chord is not None and chord.min(initial=np.inf) > mask_chord:
            continue
        angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        dist_km = EARTH_R_KM * angle
        weights = np.exp(-(dist_km ** 2) / (length_scale_km ** 2))
        sw = weights.sum()
        if sw <= 0:
            continue
        out[i] = float(np.dot(weights, values[idxs]) / sw)
    return out.reshape(grid_lat.shape)


def linear_nd_interpolate(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    *,
    mask_distance_km: float | None = None,
) -> np.ndarray:
    interpolator = LinearNDInterpolator(np.column_stack((lon, lat)), values, fill_value=np.nan)
    grid = interpolator(grid_lon, grid_lat)
    if mask_distance_km is not None:
        obs_xyz = latlon_to_xyz(lat, lon)
        tree = cKDTree(obs_xyz)
        q_xyz = latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel())
        dists, _ = tree.query(q_xyz, k=1, workers=-1)
        grid = grid.ravel()
        grid[dists > chord_from_km(mask_distance_km)] = np.nan
        grid = grid.reshape(grid_lat.shape)
    return grid


def rbf_gaussian_interpolate(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    *,
    epsilon_km: float = 100.0,
    neighbors: int = 64,
    smoothing: float = 1e-6,
    mask_distance_km: float | None = None,
) -> np.ndarray:
    obs_xyz = latlon_to_xyz(lat, lon)
    interpolator = RBFInterpolator(
        obs_xyz,
        values,
        kernel="gaussian",
        epsilon=chord_from_km(epsilon_km),
        neighbors=neighbors,
        smoothing=smoothing,
        degree=-1,
    )
    q_xyz = latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel())
    grid = interpolator(q_xyz)
    if mask_distance_km is not None:
        tree = cKDTree(obs_xyz)
        dists, _ = tree.query(q_xyz, k=1, workers=-1)
        grid[dists > chord_from_km(mask_distance_km)] = np.nan
    return grid.reshape(grid_lat.shape)
