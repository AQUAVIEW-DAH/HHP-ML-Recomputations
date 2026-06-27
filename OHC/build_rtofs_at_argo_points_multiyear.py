"""Build exact-location Argo/RTOFS collocation tables for multiple years.

This is the training-table builder for the ML correction pipeline. It uses:

- year-specific Argo-derived TCHP/D26 tables
- year-specific global RTOFS daily TCHP/D26 fields

For each Argo observation whose date has a matching RTOFS daily field, the
script interpolates RTOFS from the 8 nearest native grid cells to the exact
Argo location using inverse-distance-squared weighting.

Outputs:
- one collocated parquet/csv per year
- one combined parquet/csv across all requested years
- summary JSON/CSV with overlap counts and finite-target counts
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import latlon_to_xyz


YEAR_CONFIG = {
    2024: {
        "argo_path": Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024"),
        "rtofs_dir": Path("/data/suramya/rtofs_global_ohc_fields_2024"),
    },
    2025: {
        "argo_path": Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2025"),
        "rtofs_dir": Path("/data/suramya/rtofs_global_ohc_fields_2025"),
    },
}

DEFAULT_YEARS = [2024, 2025]
K_NEIGHBORS = 8

OUT_DIR = Path(__file__).resolve().parent / "output" / "ml_collocation"
DATA_DIR = OUT_DIR / "data"


def _season_from_month(month: int) -> str:
    if month in (1, 2, 3):
        return "winter_jfm"
    if month in (7, 8, 9):
        return "summer_jas"
    return "other"


def _available_rtofs_dates(rtofs_dir: Path) -> list[str]:
    return sorted(p.stem.split("_")[-1] for p in rtofs_dir.glob("rtofs_tchp_*.nc"))


def _build_grid_lookup(sample_file: Path) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_file) as ds:
        lat = ds["Latitude"].values.astype(np.float32)
        lon = ds["Longitude"].values.astype(np.float32)
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel()).astype(np.float32)
    tree = cKDTree(xyz)
    y_idx, x_idx = np.unravel_index(np.arange(lat.size, dtype=np.int64), lat.shape)
    return tree, y_idx.astype(np.int32), x_idx.astype(np.int32)


def _interpolate_neighbor_values(
    values_2d: np.ndarray,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
    dist_km: np.ndarray,
) -> np.ndarray:
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
            rows = np.where(remaining)[0][good]
            out[rows] = np.sum(weights[good] * vals_r[good], axis=1) / sw[good]

    return out.astype(np.float32)


def _load_argo_table(argo_path: Path, year: int) -> pd.DataFrame:
    df = pd.read_parquet(
        argo_path,
        columns=["date", "year", "month", "lat", "lon", "tchp_kj_per_cm2", "d26_m", "error"],
    )
    df = df[df["error"].isna()].copy()
    if "year" in df.columns:
        df = df[df["year"] == year].copy()
    df["date"] = df["date"].astype(str)
    df["year"] = year
    df = df.rename(columns={"tchp_kj_per_cm2": "argo_tchp_kj_per_cm2", "d26_m": "argo_d26_m"})
    return df.reset_index(drop=True)


def collocate_year(*, year: int, argo_path: Path, rtofs_dir: Path) -> pd.DataFrame:
    argo = _load_argo_table(argo_path, year)
    argo = argo[np.isfinite(argo["lat"]) & np.isfinite(argo["lon"])].copy().reset_index(drop=True)
    rtofs_dates = _available_rtofs_dates(rtofs_dir)
    argo = argo[argo["date"].isin(rtofs_dates)].copy().reset_index(drop=True)
    if argo.empty:
        raise RuntimeError(f"No overlapping Argo/RTOFS dates found for {year}.")

    sample_file = rtofs_dir / f"rtofs_tchp_{rtofs_dates[0]}.nc"
    tree, all_y, all_x = _build_grid_lookup(sample_file)

    obs_xyz = latlon_to_xyz(argo["lat"].to_numpy(float), argo["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=K_NEIGHBORS, workers=-1)
    if chord_dist.ndim == 1:
        chord_dist = chord_dist[:, None]
        flat_idx = flat_idx[:, None]
    y_idx = all_y[flat_idx]
    x_idx = all_x[flat_idx]
    dist_km = 6371.0 * (2.0 * np.arcsin(np.clip(chord_dist.astype(np.float64) / 2.0, 0.0, 1.0)))

    argo["nearest_rtofs_grid_distance_km"] = dist_km[:, 0]

    model_tchp = np.full(len(argo), np.nan, dtype=np.float32)
    model_d26 = np.full(len(argo), np.nan, dtype=np.float32)

    for date, idx in argo.groupby("date").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        path = rtofs_dir / f"rtofs_tchp_{date}.nc"
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

    argo["model_interp_tchp_kj_per_cm2"] = model_tchp
    argo["model_interp_d26_m"] = model_d26
    argo["delta_tchp_kj_per_cm2"] = argo["argo_tchp_kj_per_cm2"] - argo["model_interp_tchp_kj_per_cm2"]
    argo["delta_d26_m"] = argo["argo_d26_m"] - argo["model_interp_d26_m"]
    argo["season"] = argo["month"].astype(int).map(_season_from_month)
    argo["target_tchp_available"] = np.isfinite(argo["argo_tchp_kj_per_cm2"])
    argo["target_d26_available"] = np.isfinite(argo["argo_d26_m"])
    return argo


def _summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "rows_total": int(len(df)),
        "dates_total": int(df["date"].nunique()),
        "rows_by_year": {str(int(k)): int(v) for k, v in df.groupby("year").size().to_dict().items()},
        "rows_by_month": {str(int(k)): int(v) for k, v in df.groupby("month").size().to_dict().items()},
        "rows_by_season": {str(k): int(v) for k, v in df.groupby("season").size().to_dict().items()},
        "finite_argo_tchp_rows": int(np.isfinite(df["argo_tchp_kj_per_cm2"]).sum()),
        "finite_argo_d26_rows": int(np.isfinite(df["argo_d26_m"]).sum()),
        "finite_model_tchp_rows": int(np.isfinite(df["model_interp_tchp_kj_per_cm2"]).sum()),
        "finite_model_d26_rows": int(np.isfinite(df["model_interp_d26_m"]).sum()),
        "finite_delta_tchp_rows": int(np.isfinite(df["delta_tchp_kj_per_cm2"]).sum()),
        "finite_delta_d26_rows": int(np.isfinite(df["delta_d26_m"]).sum()),
    }


def main(years: list[int] | None = None) -> None:
    years = years or DEFAULT_YEARS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for year in years:
        cfg = YEAR_CONFIG[year]
        df = collocate_year(year=year, argo_path=cfg["argo_path"], rtofs_dir=cfg["rtofs_dir"])
        frames.append(df)

        year_tag = str(year)
        parquet_path = DATA_DIR / f"argo_rtofs_collocated_{year_tag}.parquet"
        csv_path = DATA_DIR / f"argo_rtofs_collocated_{year_tag}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        summaries.append(_summarize(df, year_tag))

    combined = pd.concat(frames, ignore_index=True)
    years_tag = "_".join(str(y) for y in years)
    combined_parquet = DATA_DIR / f"argo_rtofs_collocated_{years_tag}.parquet"
    combined_csv = DATA_DIR / f"argo_rtofs_collocated_{years_tag}.csv"
    combined.to_parquet(combined_parquet, index=False)
    combined.to_csv(combined_csv, index=False)
    summaries.append(_summarize(combined, years_tag))

    summary = {
        "years": years,
        "method": f"inverse_distance_squared_k{K_NEIGHBORS}",
        "outputs": {
            "data_dir": str(DATA_DIR),
            "combined_parquet": str(combined_parquet),
            "combined_csv": str(combined_csv),
        },
        "summaries": summaries,
    }
    summary_json = DATA_DIR / f"summary_collocation_{years_tag}.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
