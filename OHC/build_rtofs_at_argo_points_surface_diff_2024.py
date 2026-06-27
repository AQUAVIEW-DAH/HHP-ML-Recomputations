"""Compare 2024 seasonal Argo fields with RTOFS interpolated to exact Argo points.

Workflow:
1. For each Argo observation in JFM/JAS 2024 that has a matching daily RTOFS field,
   interpolate RTOFS from nearby native model cells to the exact Argo location.
2. Compute pointwise Argo-minus-RTOFS differences and summary statistics.
3. Build seasonal interpolated surfaces separately for:
   - Argo
   - RTOFS-at-Argo-locations
4. Compute seasonal surface differences as:
   Argo_surface - RTOFS_surface

Outputs:
- collocated parquet/csv with exact-location RTOFS estimates
- pointwise difference stats
- gridded difference stats by method/season/field
- point maps
- seasonal interpolated maps for Argo, RTOFS, and surface-difference
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
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (
    PARAMS,
    SEASONS,
    add_land_overlay,
    build_global_grid,
    gaussian_interpolate,
    latlon_to_xyz,
    linear_nd_interpolate,
    make_norm,
    rbf_gaussian_interpolate,
)

ARGO_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
RTOFS_DIR = Path("/data/suramya/rtofs_global_ohc_fields_2024")
OUT_DIR = Path(__file__).resolve().parent / "output" / "rtofs_at_argo_points_2024_surface_diff"
POINTS_DIR = OUT_DIR / "points"
DATA_DIR = OUT_DIR / "data"

YEAR = 2024
SEASON_KEYS = ["winter_jfm", "summer_jas"]
GRID_RES_DEG = 0.25
MASK_DISTANCE_KM = 100.0
K_NEIGHBORS = 8
METHOD_DIRS = {
    "gaussian": OUT_DIR / "interpolated",
    "linear_nd": OUT_DIR / "linear_nd",
    "rbf_gaussian": OUT_DIR / "rbf_gaussian",
}

POSITIVE_FIELDS = {
    "argo_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "argo_d26_m": "d26_m",
    "model_interp_tchp_kj_per_cm2": "tchp_kj_per_cm2",
    "model_interp_d26_m": "d26_m",
}


def _load_argo_subset() -> pd.DataFrame:
    df = pd.read_parquet(
        ARGO_PATH,
        columns=["date", "year", "month", "lat", "lon", "tchp_kj_per_cm2", "d26_m", "error"],
    )
    df = df[df["error"].isna()].copy()
    df = df[(df["year"] == YEAR) & (df["month"].isin([1, 2, 3, 7, 8, 9]))].copy()
    df["date"] = df["date"].astype(str)
    df = df.rename(columns={"tchp_kj_per_cm2": "argo_tchp_kj_per_cm2", "d26_m": "argo_d26_m"})
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


def _interpolate_neighbor_values(values_2d: np.ndarray, y_idx: np.ndarray, x_idx: np.ndarray, dist_km: np.ndarray) -> np.ndarray:
    vals = values_2d[y_idx, x_idx].astype(np.float64)
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    exact = (dist_km <= 1e-6) & finite
    has_exact = exact.any(axis=1)
    if np.any(has_exact):
        first_exact = np.argmax(exact[has_exact], axis=1)
        rows = np.where(has_exact)[0]
        out[rows] = vals[rows, first_exact]

    remaining = ~has_exact
    if np.any(remaining):
        vals_r = vals[remaining]
        dist_r = dist_km[remaining]
        finite_r = finite[remaining]
        weights = np.zeros_like(vals_r, dtype=np.float64)
        weights[finite_r] = 1.0 / np.maximum(dist_r[finite_r], 1e-6) ** 2
        sw = weights.sum(axis=1)
        good = sw > 0
        if np.any(good):
            out_rows = np.where(remaining)[0][good]
            out[out_rows] = np.sum(weights[good] * vals_r[good], axis=1) / sw[good]
    return out.astype(np.float32)


def _collocate_points(df: pd.DataFrame) -> pd.DataFrame:
    rtofs_dates = _available_rtofs_dates()
    df = df[df["date"].isin(rtofs_dates)].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No overlapping Argo/RTOFS dates found for 2024 seasonal subset.")

    sample_file = RTOFS_DIR / f"rtofs_tchp_{sorted(rtofs_dates)[0]}.nc"
    tree, all_y, all_x = _build_grid_lookup(sample_file)

    obs_xyz = latlon_to_xyz(df["lat"].to_numpy(float), df["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=K_NEIGHBORS, workers=-1)
    if chord_dist.ndim == 1:
        chord_dist = chord_dist[:, None]
        flat_idx = flat_idx[:, None]

    df["nearest_rtofs_grid_distance_km"] = 6371.0 * (2.0 * np.arcsin(np.clip(chord_dist[:, 0] / 2.0, 0.0, 1.0)))
    y_idx = all_y[flat_idx]
    x_idx = all_x[flat_idx]
    dist_km = 6371.0 * (2.0 * np.arcsin(np.clip(chord_dist.astype(np.float64) / 2.0, 0.0, 1.0)))

    model_tchp = np.full(len(df), np.nan, dtype=np.float32)
    model_d26 = np.full(len(df), np.nan, dtype=np.float32)

    for date, idx in df.groupby("date").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        path = RTOFS_DIR / f"rtofs_tchp_{date}.nc"
        with xr.open_dataset(path) as ds:
            model_tchp[idx] = _interpolate_neighbor_values(
                ds["tchp_kj_per_cm2"].values,
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )
            model_d26[idx] = _interpolate_neighbor_values(
                ds["d26_m"].values,
                y_idx[idx],
                x_idx[idx],
                dist_km[idx],
            )

    df["model_interp_tchp_kj_per_cm2"] = model_tchp
    df["model_interp_d26_m"] = model_d26
    df["delta_tchp_kj_per_cm2"] = df["argo_tchp_kj_per_cm2"] - df["model_interp_tchp_kj_per_cm2"]
    df["delta_d26_m"] = df["argo_d26_m"] - df["model_interp_d26_m"]
    df["season"] = np.where(df["month"].isin([1, 2, 3]), "winter_jfm", "summer_jas")
    return df


def _point_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    pairs = [
        ("argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "delta_tchp_kj_per_cm2"),
        ("argo_d26_m", "model_interp_d26_m", "delta_d26_m"),
    ]
    for season_key in SEASON_KEYS:
        sdf = df[df["season"] == season_key].copy()
        date_count = sdf["date"].nunique()
        for argo_field, model_field, delta_field in pairs:
            sub = sdf[[argo_field, model_field, delta_field]].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty:
                continue
            vals = sub[delta_field].to_numpy(float)
            corr = float(np.corrcoef(sub[argo_field], sub[model_field])[0, 1]) if len(sub) > 1 else np.nan
            rows.append({
                "season": season_key,
                "field": delta_field,
                "count": int(len(sub)),
                "date_count": int(date_count),
                "argo_mean": float(sub[argo_field].mean()),
                "rtofs_mean": float(sub[model_field].mean()),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "mae": float(np.mean(np.abs(vals))),
                "rmse": float(np.sqrt(np.mean(vals ** 2))),
                "corr": corr,
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


def _interp(method: str, lat, lon, values, grid_lat, grid_lon) -> np.ndarray:
    if method == "gaussian":
        return gaussian_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=MASK_DISTANCE_KM)
    if method == "linear_nd":
        return linear_nd_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=MASK_DISTANCE_KM)
    if method == "rbf_gaussian":
        return rbf_gaussian_interpolate(lat, lon, values, grid_lat, grid_lon, mask_distance_km=MASK_DISTANCE_KM)
    raise ValueError(method)


def _surface_stats(values: np.ndarray) -> dict:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return {"count": 0}
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _render_surface_panels(
    *,
    method: str,
    df: pd.DataFrame,
    argo_field: str,
    model_field: str,
    delta_field: str,
    label_base: str,
) -> list[dict]:
    out_dir = METHOD_DIRS[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_lat, grid_lon = build_global_grid(GRID_RES_DEG)
    surface_stats: list[dict] = []

    # Build season surfaces first so we can use matched norms.
    season_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for season_key in SEASON_KEYS:
        sdf = df[df["season"] == season_key].copy()
        argo_vals = sdf[argo_field].to_numpy(float)
        model_vals = sdf[model_field].to_numpy(float)
        lat = sdf["lat"].to_numpy(float)
        lon = sdf["lon"].to_numpy(float)
        valid_argo = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(argo_vals)
        valid_model = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(model_vals)
        argo_grid = _interp(method, lat[valid_argo], lon[valid_argo], argo_vals[valid_argo], grid_lat, grid_lon)
        model_grid = _interp(method, lat[valid_model], lon[valid_model], model_vals[valid_model], grid_lat, grid_lon)
        diff_grid = argo_grid - model_grid
        season_surfaces[season_key] = (argo_grid, model_grid, diff_grid, len(sdf))

    positive_norm = make_norm(POSITIVE_FIELDS[argo_field])
    diff_point_vals = df[delta_field].to_numpy(float)
    diff_norm = _delta_norm(diff_point_vals)

    for kind, cmap, norm, field_prefix, label in [
        ("argo", PARAMS[POSITIVE_FIELDS[argo_field]].cmap, positive_norm, argo_field, f"Argo {label_base}"),
        ("model", PARAMS[POSITIVE_FIELDS[model_field]].cmap, positive_norm, model_field, f"RTOFS-at-Argo {label_base}"),
        ("delta_surface", "RdBu_r", diff_norm, delta_field, f"Argo - RTOFS {label_base}"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        mappable = None
        for ax, season_key in zip(axes, SEASON_KEYS):
            season_label = SEASONS[season_key][0]
            argo_grid, model_grid, diff_grid, obs_count = season_surfaces[season_key]
            field_grid = argo_grid if kind == "argo" else model_grid if kind == "model" else diff_grid
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
            ax.set_title(f"{season_label}\n{obs_count:,} observations")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

            if kind == "delta_surface":
                stat_row = {"method": method, "season": season_key, "field": delta_field, **_surface_stats(diff_grid)}
                surface_stats.append(stat_row)

        fig.suptitle(
            f"{label} seasonal interpolated maps ({YEAR})\nmethod={method}, grid={GRID_RES_DEG}°, masked_{int(MASK_DISTANCE_KM)}km",
            fontsize=14,
        )
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
        cbar.set_label(label)
        out_path = out_dir / f"{field_prefix}_{YEAR}_{method}_0p25deg_masked_100km_grid.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    return surface_stats


def main() -> None:
    for d in [POINTS_DIR, DATA_DIR, *METHOD_DIRS.values()]:
        d.mkdir(parents=True, exist_ok=True)

    argo = _load_argo_subset()
    colloc = _collocate_points(argo)

    colloc_path = DATA_DIR / f"argo_rtofs_surface_diff_collocated_{YEAR}_winter_summer.parquet"
    colloc.to_parquet(colloc_path, index=False)
    colloc.to_csv(colloc_path.with_suffix(".csv"), index=False)

    stats = _point_stats(colloc)
    stats_path = DATA_DIR / f"argo_rtofs_surface_diff_point_stats_{YEAR}_winter_summer.csv"
    stats.to_csv(stats_path, index=False)
    (DATA_DIR / f"argo_rtofs_surface_diff_point_stats_{YEAR}_winter_summer.json").write_text(
        json.dumps(stats.to_dict(orient="records"), indent=2)
    )

    _render_point_panels(
        colloc,
        "model_interp_tchp_kj_per_cm2",
        "RTOFS interpolated-to-Argo TCHP / OHC (kJ/cm²)",
        POINTS_DIR / f"model_interp_tchp_kj_per_cm2_{YEAR}_points_winter_summer.png",
    )
    _render_point_panels(
        colloc,
        "model_interp_d26_m",
        "RTOFS interpolated-to-Argo D26 (m)",
        POINTS_DIR / f"model_interp_d26_m_{YEAR}_points_winter_summer.png",
    )
    _render_point_panels(
        colloc,
        "delta_tchp_kj_per_cm2",
        "Pointwise Argo - RTOFS TCHP difference (kJ/cm²)",
        POINTS_DIR / f"delta_tchp_kj_per_cm2_{YEAR}_points_winter_summer.png",
        delta=True,
    )
    _render_point_panels(
        colloc,
        "delta_d26_m",
        "Pointwise Argo - RTOFS D26 difference (m)",
        POINTS_DIR / f"delta_d26_m_{YEAR}_points_winter_summer.png",
        delta=True,
    )

    surface_stats_all: list[dict] = []
    for method in ["gaussian", "linear_nd", "rbf_gaussian"]:
        surface_stats_all.extend(
            _render_surface_panels(
                method=method,
                df=colloc,
                argo_field="argo_tchp_kj_per_cm2",
                model_field="model_interp_tchp_kj_per_cm2",
                delta_field="delta_tchp_kj_per_cm2",
                label_base="TCHP / OHC (kJ/cm²)",
            )
        )
        surface_stats_all.extend(
            _render_surface_panels(
                method=method,
                df=colloc,
                argo_field="argo_d26_m",
                model_field="model_interp_d26_m",
                delta_field="delta_d26_m",
                label_base="D26 (m)",
            )
        )

    surface_stats_df = pd.DataFrame(surface_stats_all)
    surface_stats_csv = DATA_DIR / f"argo_rtofs_surface_diff_grid_stats_{YEAR}_winter_summer.csv"
    surface_stats_df.to_csv(surface_stats_csv, index=False)
    (DATA_DIR / f"argo_rtofs_surface_diff_grid_stats_{YEAR}_winter_summer.json").write_text(
        json.dumps(surface_stats_df.to_dict(orient="records"), indent=2)
    )

    summary = {
        "year": YEAR,
        "argo_rows_input": int(len(argo)),
        "collocated_rows": int(len(colloc)),
        "collocated_dates": int(colloc["date"].nunique()),
        "rtofs_to_argo_point_method": f"inverse_distance_squared_k{K_NEIGHBORS}",
        "render_methods": ["gaussian", "linear_nd", "rbf_gaussian"],
        "mask_distance_km": MASK_DISTANCE_KM,
        "grid_resolution_deg": GRID_RES_DEG,
        "outputs": {
            "collocated_parquet": str(colloc_path),
            "point_stats_csv": str(stats_path),
            "grid_stats_csv": str(surface_stats_csv),
            "points_dir": str(POINTS_DIR),
            "method_dirs": {k: str(v) for k, v in METHOD_DIRS.items()},
        },
    }
    (DATA_DIR / f"summary_{YEAR}_winter_summer.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
