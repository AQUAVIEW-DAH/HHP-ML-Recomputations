"""Benchmark seasonal interpolation strategies on pooled Argo OHC/TCHP data.

This script measures build/evaluation time for three interpolation strategies
using the pooled seasonal Argo dataset:

- Gaussian-weighted local averaging (L = 100 km, truncated at 3L)
- scipy.interpolate.LinearNDInterpolator
- scipy.interpolate.RBFInterpolator with Gaussian kernel

It reports timing estimates for the exact grid variants discussed:

- 1/8 degree global grid, no support mask
- 1/4 degree global grid, no support mask
- 1/4 degree global grid, with a nearest-observation support mask at 100 km

Notes:
- "No mask" means evaluate on the full global grid.
- The Gaussian method is implemented with a practical truncation radius of 3L
  for speed; weights beyond that are negligible.
- RBF uses ``neighbors=64`` so the benchmark remains feasible on the
  current ~180k-season dataset.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

DATA_DIR = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_CSV = OUT_DIR / "seasonal_interpolation_benchmark.csv"
OUT_TXT = OUT_DIR / "seasonal_interpolation_benchmark.txt"

EARTH_R_KM = 6371.0
GAUSSIAN_L_KM = 100.0
GAUSSIAN_RADIUS_KM = 300.0  # 3L truncation for practical local evaluation
RBF_NEIGHBORS = 64
RBF_SMOOTHING = 1e-6
SAMPLE_POINTS = 100_000
RNG_SEED = 42

SEASONS = {
    "winter_jfm": [1, 2, 3],
    "summer_jas": [7, 8, 9],
}

GRID_CASES = {
    "global_1_8_nomask": {"resolution_deg": 0.125, "mask_100km": False},
    "global_1_4_nomask": {"resolution_deg": 0.25, "mask_100km": False},
    "global_1_4_mask100km": {"resolution_deg": 0.25, "mask_100km": True},
}


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


def chord_from_km(distance_km: float) -> float:
    return 2.0 * math.sin((distance_km / EARTH_R_KM) / 2.0)


def build_global_grid(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat_centers = np.arange(-90.0 + resolution_deg / 2.0, 90.0, resolution_deg)
    lon_centers = np.arange(-180.0 + resolution_deg / 2.0, 180.0, resolution_deg)
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)
    return grid_lat.ravel(), grid_lon.ravel()


def seasonal_dataframe(season_name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR)
    df = df[df["error"].isna()].copy()
    months = SEASONS[season_name]
    return df[df["month"].isin(months)].copy()


def gaussian_weighted_chunk(
    tree: cKDTree,
    obs_xyz: np.ndarray,
    obs_values: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    radius = chord_from_km(GAUSSIAN_RADIUS_KM)
    idx_lists = tree.query_ball_point(query_xyz, r=radius, workers=-1)
    out = np.full(query_xyz.shape[0], np.nan, dtype=float)
    for i, idxs in enumerate(idx_lists):
        if not idxs:
            continue
        pts = obs_xyz[idxs]
        chord = np.linalg.norm(pts - query_xyz[i], axis=1)
        angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        dist_km = EARTH_R_KM * angle
        weights = np.exp(-(dist_km ** 2) / (GAUSSIAN_L_KM ** 2))
        sw = weights.sum()
        if sw <= 0:
            continue
        out[i] = float(np.dot(weights, obs_values[idxs]) / sw)
    return out


@dataclass
class BenchmarkResult:
    season: str
    grid_case: str
    resolution_deg: float
    mask_100km: bool
    method: str
    n_obs: int
    grid_points_total: int
    grid_points_effective: int
    build_seconds: float
    mask_seconds: float
    sample_points: int
    sample_eval_seconds: float
    estimated_full_eval_seconds: float
    estimated_total_seconds: float


def benchmark_method(
    method: str,
    season_name: str,
    grid_case_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    support_tree: cKDTree,
    obs_xyz: np.ndarray,
) -> BenchmarkResult:
    cfg = GRID_CASES[grid_case_name]
    mask_seconds = 0.0

    # Mask calculation first, shared logic across methods.
    if cfg["mask_100km"]:
        t0 = time.perf_counter()
        query_xyz_all = latlon_to_xyz(query_lat, query_lon)
        dists, _ = support_tree.query(query_xyz_all, k=1, workers=-1)
        mask = dists <= chord_from_km(GAUSSIAN_L_KM)
        mask_seconds = time.perf_counter() - t0
        effective_lat = query_lat[mask]
        effective_lon = query_lon[mask]
    else:
        effective_lat = query_lat
        effective_lon = query_lon

    rng = np.random.default_rng(RNG_SEED)
    sample_n = min(SAMPLE_POINTS, effective_lat.shape[0])
    sample_idx = rng.choice(effective_lat.shape[0], size=sample_n, replace=False)
    sample_lat = effective_lat[sample_idx]
    sample_lon = effective_lon[sample_idx]

    build_seconds = 0.0
    sample_eval_seconds = float("nan")

    if method == "gaussian_weighted":
        t0 = time.perf_counter()
        tree = cKDTree(obs_xyz)
        build_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = gaussian_weighted_chunk(
            tree=tree,
            obs_xyz=obs_xyz,
            obs_values=values,
            query_xyz=latlon_to_xyz(sample_lat, sample_lon),
        )
        sample_eval_seconds = time.perf_counter() - t1

    elif method == "linear_nd":
        points = np.column_stack((lon, lat))
        t0 = time.perf_counter()
        interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
        build_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = interpolator(sample_lon, sample_lat)
        sample_eval_seconds = time.perf_counter() - t1

    elif method == "rbf_gaussian":
        points = obs_xyz
        epsilon = chord_from_km(GAUSSIAN_L_KM)
        t0 = time.perf_counter()
        interpolator = RBFInterpolator(
            points,
            values,
            kernel="gaussian",
            epsilon=epsilon,
            neighbors=RBF_NEIGHBORS,
            smoothing=RBF_SMOOTHING,
            degree=-1,
        )
        build_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = interpolator(latlon_to_xyz(sample_lat, sample_lon))
        sample_eval_seconds = time.perf_counter() - t1

    else:
        raise ValueError(f"Unknown method {method}")

    estimated_full_eval_seconds = sample_eval_seconds * (effective_lat.shape[0] / sample_n)
    estimated_total_seconds = build_seconds + mask_seconds + estimated_full_eval_seconds

    return BenchmarkResult(
        season=season_name,
        grid_case=grid_case_name,
        resolution_deg=cfg["resolution_deg"],
        mask_100km=cfg["mask_100km"],
        method=method,
        n_obs=int(lat.shape[0]),
        grid_points_total=int(query_lat.shape[0]),
        grid_points_effective=int(effective_lat.shape[0]),
        build_seconds=build_seconds,
        mask_seconds=mask_seconds,
        sample_points=sample_n,
        sample_eval_seconds=sample_eval_seconds,
        estimated_full_eval_seconds=estimated_full_eval_seconds,
        estimated_total_seconds=estimated_total_seconds,
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for season_name in SEASONS:
        sdf = seasonal_dataframe(season_name)
        lat = sdf["lat"].to_numpy(dtype=float)
        lon = sdf["lon"].to_numpy(dtype=float)
        values = sdf["tchp_kj_per_cm2"].to_numpy(dtype=float)
        obs_xyz = latlon_to_xyz(lat, lon)
        support_tree = cKDTree(obs_xyz)

        for method_grid_case, cfg in GRID_CASES.items():
            query_lat, query_lon = build_global_grid(cfg["resolution_deg"])
            for method_name in ("gaussian_weighted", "linear_nd", "rbf_gaussian"):
                print(f"Benchmarking {season_name} | {method_grid_case} | {method_name} ...", flush=True)
                result = benchmark_method(
                    method=method_name,
                    season_name=season_name,
                    grid_case_name=method_grid_case,
                    lat=lat,
                    lon=lon,
                    values=values,
                    query_lat=query_lat,
                    query_lon=query_lon,
                    support_tree=support_tree,
                    obs_xyz=obs_xyz,
                )
                results.append(result)

    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_df.to_csv(OUT_CSV, index=False)
    with OUT_TXT.open("w") as f:
        f.write(out_df.to_string(index=False))
        f.write("\n")

    print(out_df.to_string(index=False))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TXT}")
