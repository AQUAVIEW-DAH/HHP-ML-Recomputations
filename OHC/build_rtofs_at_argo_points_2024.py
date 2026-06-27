"""Collocate 2024 seasonal Argo observations with global daily RTOFS fields.

For winter (JFM) and summer (JAS) 2024, sample the precomputed global daily
RTOFS TCHP/D26 fields at the nearest model grid point for every Argo
observation date where a matching RTOFS daily field exists.

Outputs:
- parquet/csv table of collocated point pairs
- JSON/CSV stats summary for Argo-RTOFS differences
- winter/summer point maps for RTOFS-at-Argo values and Argo-minus-RTOFS deltas
- winter/summer gaussian-interpolated panels on 0.25° grid for the same fields
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (
    PARAMS,
    SEASONS,
    add_land_overlay,
    build_global_grid,
    gaussian_interpolate,
    latlon_to_xyz,
    make_norm,
)

ARGO_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
RTOFS_DIR = Path("/data/suramya/rtofs_global_ohc_fields_2024")
OUT_DIR = Path(__file__).resolve().parent / "output" / "rtofs_at_argo_points_2024"
POINTS_DIR = OUT_DIR / "points"
INTERP_DIR = OUT_DIR / "interpolated"
DATA_DIR = OUT_DIR / "data"

YEAR = 2024
SEASON_KEYS = ["winter_jfm", "summer_jas"]
MASK_DISTANCE_KM = 100.0
GRID_RES_DEG = 0.25

POSITIVE_FIELDS = {
    "obs_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "model_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "obs_d26_m": "d26_m",
    "model_d26_m": "d26_m",
}

DELTA_FIELDS = {
    "delta_tchp_kj_per_cm2": "TCHP difference (Argo - RTOFS, kJ/cm²)",
    "delta_d26_m": "D26 difference (Argo - RTOFS, m)",
}


def _load_argo_subset() -> pd.DataFrame:
    df = pd.read_parquet(
        ARGO_PATH,
        columns=["date", "year", "month", "lat", "lon", "tchp_kj_per_cm2", "d26_m", "error"],
    )
    df = df[df["error"].isna()].copy()
    df = df[(df["year"] == YEAR) & (df["month"].isin([1, 2, 3, 7, 8, 9]))].copy()
    df["date"] = df["date"].astype(str)
    return df.reset_index(drop=True)


def _available_rtofs_dates() -> set[str]:
    return {p.stem.split("_")[-1] for p in RTOFS_DIR.glob("rtofs_tchp_*.nc")}


def _build_grid_lookup(sample_file: Path) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_file) as ds:
        lat = ds["Latitude"].values.astype(np.float32)
        lon = ds["Longitude"].values.astype(np.float32)
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel()).astype(np.float32)
    tree = cKDTree(xyz)
    y_idx, x_idx = np.unravel_index(np.arange(lat.size, dtype=np.int64), lat.shape)
    return tree, y_idx.astype(np.int32), x_idx.astype(np.int32)


def _collocate_points(df: pd.DataFrame) -> pd.DataFrame:
    rtofs_dates = _available_rtofs_dates()
    df = df[df["date"].isin(rtofs_dates)].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No overlapping Argo/RTOFS dates found for 2024 seasonal subset.")

    sample_file = RTOFS_DIR / f"rtofs_tchp_{sorted(rtofs_dates)[0]}.nc"
    tree, all_y, all_x = _build_grid_lookup(sample_file)

    obs_xyz = latlon_to_xyz(df["lat"].to_numpy(float), df["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=1, workers=-1)
    df["rtofs_y"] = all_y[flat_idx]
    df["rtofs_x"] = all_x[flat_idx]
    df["rtofs_chord_dist"] = chord_dist

    # Haversine-like geodesic from xyz chord distance
    chord = np.clip(chord_dist.astype(float), 0.0, 2.0)
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    df["rtofs_grid_distance_km"] = 6371.0 * angle

    model_tchp = np.full(len(df), np.nan, dtype=np.float32)
    model_d26 = np.full(len(df), np.nan, dtype=np.float32)
    model_lat = np.full(len(df), np.nan, dtype=np.float32)
    model_lon = np.full(len(df), np.nan, dtype=np.float32)

    for date, idx in df.groupby("date").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        path = RTOFS_DIR / f"rtofs_tchp_{date}.nc"
        with xr.open_dataset(path) as ds:
            y = df.loc[idx, "rtofs_y"].to_numpy(int)
            x = df.loc[idx, "rtofs_x"].to_numpy(int)
            model_tchp[idx] = ds["tchp_kj_per_cm2"].values[y, x]
            model_d26[idx] = ds["d26_m"].values[y, x]
            model_lat[idx] = ds["Latitude"].values[y, x]
            model_lon[idx] = ds["Longitude"].values[y, x]

    df["model_tchp_kj_per_cm2"] = model_tchp
    df["model_d26_m"] = model_d26
    df["model_grid_lat"] = model_lat
    df["model_grid_lon"] = model_lon
    df["delta_tchp_kj_per_cm2"] = df["tchp_kj_per_cm2"] - df["model_tchp_kj_per_cm2"]
    df["delta_d26_m"] = df["d26_m"] - df["model_d26_m"]

    def season_from_month(m: float) -> str:
        m = int(m)
        return "winter_jfm" if m in (1, 2, 3) else "summer_jas"

    df["season"] = df["month"].apply(season_from_month)
    return df


def _season_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season_key in SEASON_KEYS:
        sdf = df[df["season"] == season_key].copy()
        date_count = sdf["date"].nunique()
        for field in ["delta_tchp_kj_per_cm2", "delta_d26_m"]:
            vals = sdf[field].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "season": season_key,
                "field": field,
                "count": int(len(vals)),
                "date_count": int(date_count),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "mae": float(np.mean(np.abs(vals))),
                "rmse": float(np.sqrt(np.mean(vals ** 2))),
                "p05": float(np.percentile(vals, 5)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })
    return pd.DataFrame(rows)


def _delta_norm(values: np.ndarray) -> TwoSlopeNorm:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    vmax = np.percentile(np.abs(vals), 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(np.abs(vals)) if len(vals) else 1.0
    vmax = float(max(vmax, 1e-6))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _render_point_panels(df: pd.DataFrame, field: str, label: str, out_path: Path, *, delta: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    if delta:
        norm = _delta_norm(df[field].to_numpy(float))
        cmap = "RdBu_r"
    else:
        param_key = POSITIVE_FIELDS[field]
        norm = make_norm(param_key)
        cmap = PARAMS[param_key].cmap

    mappable = None
    for ax, season_key in zip(axes, SEASON_KEYS):
        season_label = SEASONS[season_key][0]
        sdf = df[df["season"] == season_key]
        add_land_overlay(ax, zorder=0)
        sc = ax.scatter(
            sdf["lon"],
            sdf["lat"],
            c=sdf[field],
            s=5,
            alpha=0.8,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            zorder=2,
        )
        mappable = sc
        ax.set_title(f"{season_label}\n{len(sdf):,} collocated observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    fig.suptitle(f"{label} seasonal point maps ({YEAR})", fontsize=15)
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_interpolated_panels(df: pd.DataFrame, field: str, label: str, out_path: Path, *, delta: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    if delta:
        norm = _delta_norm(df[field].to_numpy(float))
        cmap = "RdBu_r"
    else:
        param_key = POSITIVE_FIELDS[field]
        norm = make_norm(param_key)
        cmap = PARAMS[param_key].cmap

    mappable = None
    for ax, season_key in zip(axes, SEASON_KEYS):
        season_label = SEASONS[season_key][0]
        sdf = df[df["season"] == season_key]
        grid_lat, grid_lon = build_global_grid(GRID_RES_DEG)
        field_grid = gaussian_interpolate(
            sdf["lat"].to_numpy(float),
            sdf["lon"].to_numpy(float),
            sdf[field].to_numpy(float),
            grid_lat,
            grid_lon,
            mask_distance_km=MASK_DISTANCE_KM,
        )
        artist = ax.pcolormesh(
            grid_lon,
            grid_lat,
            ma.masked_invalid(field_grid),
            shading="auto",
            cmap=cmap,
            norm=norm,
            zorder=1,
        )
        mappable = artist
        add_land_overlay(ax, zorder=5)
        ax.set_title(f"{season_label}\n{len(sdf):,} collocated observations")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    fig.suptitle(
        f"{label} seasonal interpolated maps ({YEAR})\nmethod=gaussian, grid={GRID_RES_DEG}°, masked_{int(MASK_DISTANCE_KM)}km",
        fontsize=14,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    for d in [POINTS_DIR, INTERP_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    argo = _load_argo_subset()
    colloc = _collocate_points(argo)

    colloc_path = DATA_DIR / f"argo_rtofs_collocated_{YEAR}_winter_summer.parquet"
    colloc.to_parquet(colloc_path, index=False)
    colloc.to_csv(colloc_path.with_suffix('.csv'), index=False)

    stats = _season_stats(colloc)
    stats_path = DATA_DIR / f"argo_rtofs_diff_stats_{YEAR}_winter_summer.csv"
    stats.to_csv(stats_path, index=False)
    stats_json = stats.to_dict(orient="records")
    (DATA_DIR / f"argo_rtofs_diff_stats_{YEAR}_winter_summer.json").write_text(json.dumps(stats_json, indent=2))

    _render_point_panels(
        colloc,
        "model_tchp_kj_per_cm2",
        "RTOFS TCHP / OHC (kJ/cm²) at Argo points",
        POINTS_DIR / f"model_tchp_kj_per_cm2_{YEAR}_points_winter_summer.png",
    )
    _render_point_panels(
        colloc,
        "model_d26_m",
        "RTOFS D26 (m) at Argo points",
        POINTS_DIR / f"model_d26_m_{YEAR}_points_winter_summer.png",
    )
    _render_point_panels(
        colloc,
        "delta_tchp_kj_per_cm2",
        "Argo - RTOFS TCHP difference (kJ/cm²)",
        POINTS_DIR / f"delta_tchp_kj_per_cm2_{YEAR}_points_winter_summer.png",
        delta=True,
    )
    _render_point_panels(
        colloc,
        "delta_d26_m",
        "Argo - RTOFS D26 difference (m)",
        POINTS_DIR / f"delta_d26_m_{YEAR}_points_winter_summer.png",
        delta=True,
    )

    _render_interpolated_panels(
        colloc,
        "model_tchp_kj_per_cm2",
        "RTOFS TCHP / OHC (kJ/cm²) at Argo points",
        INTERP_DIR / f"model_tchp_kj_per_cm2_{YEAR}_gaussian_0p25deg_masked_100km_grid.png",
    )
    _render_interpolated_panels(
        colloc,
        "model_d26_m",
        "RTOFS D26 (m) at Argo points",
        INTERP_DIR / f"model_d26_m_{YEAR}_gaussian_0p25deg_masked_100km_grid.png",
    )
    _render_interpolated_panels(
        colloc,
        "delta_tchp_kj_per_cm2",
        "Argo - RTOFS TCHP difference (kJ/cm²)",
        INTERP_DIR / f"delta_tchp_kj_per_cm2_{YEAR}_gaussian_0p25deg_masked_100km_grid.png",
        delta=True,
    )
    _render_interpolated_panels(
        colloc,
        "delta_d26_m",
        "Argo - RTOFS D26 difference (m)",
        INTERP_DIR / f"delta_d26_m_{YEAR}_gaussian_0p25deg_masked_100km_grid.png",
        delta=True,
    )

    summary = {
        "year": YEAR,
        "argo_rows_input": int(len(argo)),
        "collocated_rows": int(len(colloc)),
        "collocated_dates": int(colloc['date'].nunique()),
        "collocated_by_month": {str(int(k)): int(v) for k, v in colloc.groupby('month').size().to_dict().items()},
        "outputs": {
            "collocated_parquet": str(colloc_path),
            "stats_csv": str(stats_path),
            "points_dir": str(POINTS_DIR),
            "interpolated_dir": str(INTERP_DIR),
        },
    }
    (DATA_DIR / f"summary_{YEAR}_winter_summer.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
