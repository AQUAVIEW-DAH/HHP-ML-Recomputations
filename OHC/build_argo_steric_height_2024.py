"""Compute Argo steric-height diagnostics and render annual maps.

This first pass follows the mentor note:
- derive steric height from TEOS-10 dynamic height
- use 2000 dbar as the primary comparison reference
- also compute the 1000 dbar reference and their difference

Outputs:
- per-profile parquet/csv with steric-height diagnostics
- summary json
- annual interpolated map panels for 2024
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import gsw
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import (
    _load_land_polygons,
    add_land_overlay,
    gaussian_interpolate,
)
from ml.sources.argo_gdac_source import _extract_profiles_from_file


ARGO_ROOT = Path("/data/suramya/argo_cache_hhp")
ARGO_DATASETS = {
    "2020_2024": {
        "path": Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024"),
        "year_filter": None,
    },
    "2024": {
        "path": Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024"),
        "year_filter": 2024,
    },
    "2025": {
        "path": Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2025"),
        "year_filter": None,
    },
}
OUT_DIR = Path(__file__).resolve().parent / "output" / "steric_height"
G_METERS_PER_S2 = 9.81

LOGGER = logging.getLogger("argo_steric_height")


def _compute_surface_steric_m(profile, p_ref: float) -> float | None:
    p = np.asarray(profile.pressure_dbar, dtype=float)
    t = np.asarray(profile.temperature_c, dtype=float)
    sp = np.asarray(profile.salinity_psu, dtype=float)
    if p.size < 5 or not np.isfinite(p).all() or float(np.nanmax(p)) < p_ref:
        return None

    order = np.argsort(p)
    p = p[order]
    t = t[order]
    sp = sp[order]
    keep = np.isfinite(p) & np.isfinite(t) & np.isfinite(sp)
    p = p[keep]
    t = t[keep]
    sp = sp[keep]
    if p.size < 5 or float(p[-1]) < p_ref:
        return None

    sa = gsw.SA_from_SP(sp, p, profile.lon, profile.lat)
    ct = gsw.CT_from_t(sa, t, p)
    dyn = gsw.geo_strf_dyn_height(sa, ct, p, p_ref=p_ref)
    if dyn.size == 0 or not np.isfinite(dyn[0]):
        return None
    return float(dyn[0] / G_METERS_PER_S2)


def _process_row(row: dict) -> dict:
    cast_id = row["cast_id"]
    path = ARGO_ROOT / cast_id
    out = {
        "cast_id": cast_id,
        "platform": row["platform"],
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "date": str(row["date"]),
        "year": int(row["year"]),
        "month": int(row["month"]),
        "max_depth_m": float(row["max_depth_m"]),
        "surface_t_c": float(row["surface_t_c"]) if pd.notna(row["surface_t_c"]) else np.nan,
        "steric_0_1000_m": np.nan,
        "steric_0_2000_m": np.nan,
        "steric_1000_ref2000_m": np.nan,
        "error": None,
    }
    try:
        profiles = _extract_profiles_from_file(path, cast_id)
        if not profiles:
            out["error"] = "empty_profile"
            return out
        prof = profiles[0]
        s1000 = _compute_surface_steric_m(prof, 1000.0)
        s2000 = _compute_surface_steric_m(prof, 2000.0)
        if s1000 is not None:
            out["steric_0_1000_m"] = s1000
        if s2000 is not None:
            out["steric_0_2000_m"] = s2000
        if s1000 is not None and s2000 is not None:
            out["steric_1000_ref2000_m"] = s2000 - s1000
        return out
    except Exception as exc:  # pragma: no cover - defensive batch guard
        out["error"] = f"compute:{exc}"
        return out


def _load_source_rows(dataset_key: str) -> pd.DataFrame:
    spec = ARGO_DATASETS[dataset_key]
    df = pd.read_parquet(
        spec["path"],
        columns=[
            "cast_id",
            "platform",
            "lat",
            "lon",
            "date",
            "year",
            "month",
            "max_depth_m",
            "surface_t_c",
            "error",
        ],
    )
    if spec["year_filter"] is not None:
        df = df[df["year"] == spec["year_filter"]].copy()
    df = df[df["error"].isna()].copy()
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy()
    df = df[df["max_depth_m"] >= 1000.0].copy()
    return df.reset_index(drop=True)


def _summarize(df: pd.DataFrame, dataset_key: str) -> dict:
    def _stats(col: str) -> dict:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"finite_rows": 0}
        return {
            "finite_rows": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }

    return {
        "dataset": dataset_key,
        "rows_total": int(len(df)),
        "rows_with_errors": int(df["error"].notna().sum()),
        "steric_0_1000_m": _stats("steric_0_1000_m"),
        "steric_0_2000_m": _stats("steric_0_2000_m"),
        "steric_1000_ref2000_m": _stats("steric_1000_ref2000_m"),
    }


def _render_annual_maps(df: pd.DataFrame, label: str, out_dir: Path, *, filtered: bool) -> Path:
    grid_lat, grid_lon = _build_grid(-180.0, 180.0, 1.0)
    plot_cfg = [
        ("steric_0_2000_m", "Surface Steric Height (ref 2000 dbar)", "viridis", (-1.0, 3.5), (0.0, 1.6)),
        ("steric_0_1000_m", "Surface Steric Height (ref 1000 dbar)", "plasma", (-1.0, 3.5), (0.0, 1.2)),
        ("steric_1000_ref2000_m", "Steric Height at 1000 dbar (ref 2000 dbar)", "magma", (-1.0, 1.5), (0.0, 0.6)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), constrained_layout=True)
    lat = df["lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)

    for ax, (col, title, cmap, keep_range, show_range) in zip(axes, plot_cfg):
        series = pd.to_numeric(df[col], errors="coerce")
        mask = np.isfinite(series)
        if filtered:
            mask &= (series >= keep_range[0]) & (series <= keep_range[1])
        sdf = df[mask].copy()
        values = sdf[col].to_numpy(float)
        field = gaussian_interpolate(
            sdf["lat"].to_numpy(float),
            sdf["lon"].to_numpy(float),
            values,
            grid_lat,
            grid_lon,
            length_scale_km=350.0,
            truncation_radius_km=900.0,
            mask_distance_km=300.0,
        )
        masked = ma.masked_invalid(field)
        artist = ax.pcolormesh(
            grid_lon,
            grid_lat,
            masked,
            shading="auto",
            cmap=cmap,
            vmin=show_range[0],
            vmax=show_range[1],
            zorder=1,
        )
        add_land_overlay(ax, zorder=5)
        ax.scatter(lon[::250], lat[::250], s=0.7, c="white", alpha=0.08, linewidths=0, zorder=4)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 80)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        suffix = " after plausibility filter" if filtered else ""
        ax.set_title(f"{title}\n{len(sdf):,} profiles{suffix}")
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.25)
        cbar = fig.colorbar(artist, ax=ax, shrink=0.86, pad=0.02)
        cbar.set_label("meters")

    fig.suptitle(
        f"Argo Steric Height Diagnostics ({label}{', filtered display' if filtered else ''})\n"
        "TEOS-10 dynamic height converted to steric height using g = 9.81 m s^-2",
        fontsize=14,
    )
    tag = "filtered" if filtered else "raw"
    safe = label.replace("-", "_")
    out_path = out_dir / f"argo_steric_height_{safe}_annual_maps_{tag}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _build_grid(lon_start: float, lon_stop: float, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat_centers = np.arange(-90.0 + resolution_deg / 2.0, 90.0, resolution_deg)
    lon_centers = np.arange(lon_start + resolution_deg / 2.0, lon_stop, resolution_deg)
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)
    return grid_lat, grid_lon


def _add_land_overlay_0_360(
    ax,
    *,
    facecolor: str = "black",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    zorder: int = 6,
):
    # Draw both native and +360-shifted polygons so features west of Greenwich
    # appear correctly on a 0..360 axis without depending on cartopy/basemap.
    polys = _load_land_polygons()
    shifted: list[np.ndarray] = []
    for poly in polys:
        base = np.asarray(poly, dtype=float).copy()
        shifted.append(base)
        plus_360 = base.copy()
        plus_360[:, 0] += 360.0
        shifted.append(plus_360)
    coll = PolyCollection(
        shifted,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        closed=True,
        zorder=zorder,
    )
    ax.add_collection(coll)
    return coll


def _render_paper_style_ref2000(df: pd.DataFrame, label: str, out_dir: Path) -> Path:
    grid_lat, grid_lon = _build_grid(0.0, 360.0, 1.0)
    lon_360 = np.mod(df["lon"].to_numpy(float), 360.0)
    values_m = pd.to_numeric(df["steric_0_2000_m"], errors="coerce").to_numpy(float)
    mask = np.isfinite(values_m) & (values_m >= 0.3) & (values_m <= 3.2)
    sdf = df.loc[mask].copy()
    field_m = gaussian_interpolate(
        sdf["lat"].to_numpy(float),
        np.mod(sdf["lon"].to_numpy(float), 360.0),
        sdf["steric_0_2000_m"].to_numpy(float),
        grid_lat,
        grid_lon,
        length_scale_km=500.0,
        truncation_radius_km=1200.0,
        mask_distance_km=500.0,
    )
    field_dyn_cm = field_m * 100.0
    masked = ma.masked_invalid(field_dyn_cm)

    fig, ax = plt.subplots(figsize=(14.5, 6.8), constrained_layout=True)
    levels = np.arange(50.0, 311.0, 20.0)
    artist = ax.contourf(
        grid_lon,
        grid_lat,
        masked,
        levels=levels,
        cmap="turbo",
        extend="both",
        zorder=1,
    )
    contours = ax.contour(
        grid_lon,
        grid_lat,
        masked,
        levels=levels,
        colors="#4d2b00",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    ax.clabel(contours, levels[::2], fmt="%d", fontsize=8, inline=True)
    _add_land_overlay_0_360(ax, facecolor="black", edgecolor="black", linewidth=0.2, zorder=5)
    ax.set_xlim(20.0, 380.0)
    ax.set_ylim(-60.0, 65.0)
    ax.set_xticks([60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    ax.set_xticklabels(["60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticks([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])
    ax.set_yticklabels(["60°S", "40°S", "20°S", "0°", "20°N", "40°N", "60°N"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Argo {label} steric height, 0/2000 dbar\n"
        f"Paper-style validation render ({len(sdf):,} plausible profiles, shown in dyn cm)"
    )
    cbar = fig.colorbar(artist, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label("dyn cm")
    safe = label.replace("-", "_")
    out_path = out_dir / f"argo_steric_height_{safe}_ref2000_paperstyle.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _render_paper_style_ref2000_meters(df: pd.DataFrame, label: str, out_dir: Path) -> Path:
    grid_lat, grid_lon = _build_grid(0.0, 360.0, 1.0)
    values_m = pd.to_numeric(df["steric_0_2000_m"], errors="coerce").to_numpy(float)
    mask = np.isfinite(values_m) & (values_m >= 0.3) & (values_m <= 3.2)
    sdf = df.loc[mask].copy()
    field_m = gaussian_interpolate(
        sdf["lat"].to_numpy(float),
        np.mod(sdf["lon"].to_numpy(float), 360.0),
        sdf["steric_0_2000_m"].to_numpy(float),
        grid_lat,
        grid_lon,
        length_scale_km=500.0,
        truncation_radius_km=1200.0,
        mask_distance_km=500.0,
    )
    masked = ma.masked_invalid(field_m)

    fig, ax = plt.subplots(figsize=(14.5, 6.8), constrained_layout=True)
    levels = np.arange(0.5, 3.11, 0.2)
    artist = ax.contourf(
        grid_lon,
        grid_lat,
        masked,
        levels=levels,
        cmap="turbo",
        extend="both",
        zorder=1,
    )
    contours = ax.contour(
        grid_lon,
        grid_lat,
        masked,
        levels=levels,
        colors="#4d2b00",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    ax.clabel(contours, levels[::2], fmt="%.1f", fontsize=8, inline=True)
    _add_land_overlay_0_360(ax, facecolor="black", edgecolor="black", linewidth=0.2, zorder=5)
    ax.set_xlim(20.0, 380.0)
    ax.set_ylim(-60.0, 65.0)
    ax.set_xticks([60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    ax.set_xticklabels(["60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticks([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])
    ax.set_yticklabels(["60°S", "40°S", "20°S", "0°", "20°N", "40°N", "60°N"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Argo {label} steric height, 0/2000 dbar\n"
        f"Paper-style validation render ({len(sdf):,} plausible profiles, shown in meters)"
    )
    cbar = fig.colorbar(artist, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label("meters")
    safe = label.replace("-", "_")
    out_path = out_dir / f"argo_steric_height_{safe}_ref2000_paperstyle_meters.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="2024", choices=sorted(ARGO_DATASETS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out_dir = OUT_DIR / str(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = _load_source_rows(args.dataset)
    LOGGER.info("Loaded %d Argo rows for steric computation (%d reach >=1000 m).", len(src), len(src))

    rows = src.to_dict("records")
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(_process_row, rows, chunksize=100), start=1):
            results.append(rec)
            if i % 5000 == 0 or i == len(rows):
                LOGGER.info("Steric progress %d/%d", i, len(rows))

    df = pd.DataFrame(results).sort_values(["date", "cast_id"]).reset_index(drop=True)
    parquet_path = out_dir / f"argo_steric_height_{args.dataset}.parquet"
    csv_path = out_dir / f"argo_steric_height_{args.dataset}.csv"
    summary_path = out_dir / f"argo_steric_height_{args.dataset}_summary.json"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    summary = _summarize(df, args.dataset)
    summary_path.write_text(json.dumps(summary, indent=2))
    fig_raw = _render_annual_maps(df, args.dataset, out_dir, filtered=False)
    fig_filtered = _render_annual_maps(df, args.dataset, out_dir, filtered=True)
    fig_paper = _render_paper_style_ref2000(df, args.dataset, out_dir)
    fig_paper_m = _render_paper_style_ref2000_meters(df, args.dataset, out_dir)

    payload = {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "figures": {
            "raw": str(fig_raw),
            "filtered": str(fig_filtered),
            "paperstyle_ref2000": str(fig_paper),
            "paperstyle_ref2000_meters": str(fig_paper_m),
        },
        "stats": summary,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
