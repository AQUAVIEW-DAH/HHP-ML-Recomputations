"""Build model-only profile physics features at collocated Argo points.

This enriches the multiyear collocation table with features that require the
raw global 3D RTOFS/HYCOM archive columns:

- steric height summaries derived from TEOS-10 dynamic height
- Brunt-Vaisala frequency (N^2) summaries

The enrichment remains model-only: all new columns are computed from RTOFS
temperature/salinity/thickness profiles blended to the exact collocated point
using the same 8-neighbor inverse-distance-squared weighting as the existing
surface collocation path.

Because the public 3D archives are large, this script is resumable and can be
run over a small date subset first for smoke testing before a full pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import latlon_to_xyz
from ml.sources.rtofs_global_source import (
    download_global_archv_a,
    download_global_archv_b,
    download_global_grid,
    extract_global_archv_a,
    extracted_global_archv_a_path,
    parse_archv_b,
    records_for_field,
)


IN_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data")
OUT_PATH = OUT_DIR / "argo_rtofs_collocated_2024_2025_profile_physics.parquet"
OUT_CSV = OUT_DIR / "argo_rtofs_collocated_2024_2025_profile_physics.csv"
OUT_SUMMARY = OUT_DIR / "summary_collocation_2024_2025_profile_physics.json"
STATUS_PATH = OUT_DIR / "summary_collocation_2024_2025_profile_physics_dates.json"

RTOFS_GLOBAL_CACHE_DIR = Path("/data/suramya/rtofs_global_cache")
K_NEIGHBORS = 8
EARTH_R_KM = 6371.0
PA_PER_M = 9806.0
G_METERS_PER_S2 = 9.81
VOID_THRESHOLD = 1.0e30

FEATURE_COLUMNS = [
    "model_steric_0_1000_m",
    "model_steric_0_2000_m",
    "model_steric_1000_ref2000_m",
    "model_n2_mean_upper200_s2",
    "model_n2_max_upper200_s2",
    "model_n2_mean_to_d26_s2",
    "model_n2_max_to_d26_s2",
]


def _build_grid_lookup(grid_file: Path) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    with xr.open_dataset(grid_file) as ds:
        lat = ds["Latitude"].values.astype(np.float32)
        lon = ds["Longitude"].values.astype(np.float32)
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel()).astype(np.float32)
    tree = cKDTree(xyz)
    y_idx, x_idx = np.unravel_index(np.arange(lat.size, dtype=np.int64), lat.shape)
    return tree, y_idx.astype(np.int32), x_idx.astype(np.int32)


def _prepare_neighbor_lookup(df: pd.DataFrame, grid_file: Path):
    tree, all_y, all_x = _build_grid_lookup(grid_file)
    obs_xyz = latlon_to_xyz(df["lat"].to_numpy(float), df["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=K_NEIGHBORS, workers=-1)
    if chord_dist.ndim == 1:
        chord_dist = chord_dist[:, None]
        flat_idx = flat_idx[:, None]
    y_idx = all_y[flat_idx]
    x_idx = all_x[flat_idx]
    dist_km = EARTH_R_KM * (2.0 * np.arcsin(np.clip(chord_dist.astype(np.float64) / 2.0, 0.0, 1.0)))
    return y_idx, x_idx, dist_km


def _sample_record_points(
    mm: np.memmap,
    *,
    record_index: int,
    record_words: int,
    idm: int,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
) -> np.ndarray:
    offsets = record_index * record_words + (y_idx.astype(np.int64) * idm) + x_idx.astype(np.int64)
    vals = np.asarray(mm[offsets], dtype=">f4").astype(np.float32)
    vals = np.where(vals > VOID_THRESHOLD, np.nan, vals)
    return vals


def _interpolate_neighbor_profile(values_nzk: np.ndarray, dist_km: np.ndarray) -> np.ndarray:
    # values_nzk shape: (n_rows, k, nz)
    n_rows, _, nz = values_nzk.shape
    out = np.full((n_rows, nz), np.nan, dtype=np.float32)
    finite = np.isfinite(values_nzk)
    exact = dist_km <= 1e-6
    has_exact = exact.any(axis=1)
    if np.any(has_exact):
        exact_idx = np.argmax(exact[has_exact], axis=1)
        rows = np.where(has_exact)[0]
        out[rows] = values_nzk[rows, exact_idx, :]

    remaining = ~has_exact
    if np.any(remaining):
        vals_r = values_nzk[remaining]
        finite_r = finite[remaining]
        dist_r = dist_km[remaining]
        weights = np.zeros_like(dist_r, dtype=np.float64)
        valid_dist = np.isfinite(dist_r)
        weights[valid_dist] = 1.0 / np.maximum(dist_r[valid_dist], 1e-6) ** 2
        num = np.nansum(vals_r.astype(np.float64) * weights[:, :, None], axis=1)
        den = np.sum(weights[:, :, None] * finite_r, axis=1)
        good = den > 0
        interp = np.full((vals_r.shape[0], nz), np.nan, dtype=np.float64)
        interp[good] = num[good] / den[good]
        out[remaining] = interp.astype(np.float32)
    return out


def _compute_row_profile_features(
    *,
    lat: float,
    lon: float,
    temp_c: np.ndarray,
    sal_psu: np.ndarray,
    thickness_pa: np.ndarray,
    model_d26_m: float,
) -> dict[str, float]:
    out = {col: np.nan for col in FEATURE_COLUMNS}
    valid = np.isfinite(temp_c) & np.isfinite(sal_psu) & np.isfinite(thickness_pa) & (thickness_pa > 0.0)
    if valid.sum() < 5:
        return out

    thk_m = np.where(valid, thickness_pa / PA_PER_M, 0.0)
    depth_top = np.cumsum(thk_m) - thk_m
    depth_bottom = depth_top + thk_m
    depth_center = depth_top + 0.5 * thk_m

    p = gsw.p_from_z(-depth_center.astype(np.float64), float(lat))
    keep = valid & np.isfinite(p)
    if keep.sum() < 5:
        return out

    p = p[keep]
    t = temp_c[keep].astype(np.float64)
    sp = sal_psu[keep].astype(np.float64)
    order = np.argsort(p)
    p = p[order]
    t = t[order]
    sp = sp[order]
    keep_unique = np.concatenate(([True], np.diff(p) > 1e-9))
    p = p[keep_unique]
    t = t[keep_unique]
    sp = sp[keep_unique]
    if p.size < 5:
        return out

    if np.nanmax(p) >= 1000.0:
        sa = gsw.SA_from_SP(sp, p, float(lon), float(lat))
        ct = gsw.CT_from_t(sa, t, p)
        dyn_1000 = gsw.geo_strf_dyn_height(sa, ct, p, p_ref=1000.0)
        if dyn_1000.size and np.isfinite(dyn_1000[0]):
            out["model_steric_0_1000_m"] = float(dyn_1000[0] / G_METERS_PER_S2)
    else:
        sa = None
        ct = None

    if np.nanmax(p) >= 2000.0:
        if sa is None or ct is None:
            sa = gsw.SA_from_SP(sp, p, float(lon), float(lat))
            ct = gsw.CT_from_t(sa, t, p)
        dyn_2000 = gsw.geo_strf_dyn_height(sa, ct, p, p_ref=2000.0)
        if dyn_2000.size and np.isfinite(dyn_2000[0]):
            out["model_steric_0_2000_m"] = float(dyn_2000[0] / G_METERS_PER_S2)

    if np.isfinite(out["model_steric_0_1000_m"]) and np.isfinite(out["model_steric_0_2000_m"]):
        out["model_steric_1000_ref2000_m"] = float(
            out["model_steric_0_2000_m"] - out["model_steric_0_1000_m"]
        )

    if sa is None or ct is None:
        sa = gsw.SA_from_SP(sp, p, float(lon), float(lat))
        ct = gsw.CT_from_t(sa, t, p)

    if p.size >= 6:
        n2, p_mid = gsw.Nsquared(sa, ct, p, float(lat))
        depth_mid = -gsw.z_from_p(p_mid, float(lat))
        finite_n2 = np.isfinite(n2) & np.isfinite(depth_mid)
        upper200 = finite_n2 & (depth_mid <= 200.0)
        if np.any(upper200):
            out["model_n2_mean_upper200_s2"] = float(np.nanmean(n2[upper200]))
            out["model_n2_max_upper200_s2"] = float(np.nanmax(n2[upper200]))
        if np.isfinite(model_d26_m):
            to_d26 = finite_n2 & (depth_mid <= float(model_d26_m))
            if np.any(to_d26):
                out["model_n2_mean_to_d26_s2"] = float(np.nanmean(n2[to_d26]))
                out["model_n2_max_to_d26_s2"] = float(np.nanmax(n2[to_d26]))

    return out


def _load_or_init_df(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_parquet(output_path).copy()
    df = pd.read_parquet(input_path).copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _cleanup_archive_files(date: str, *, remove_tgz: bool, remove_archv_a: bool) -> None:
    date_dir = RTOFS_GLOBAL_CACHE_DIR / f"rtofs.{date}"
    if remove_archv_a:
        archv_a = extracted_global_archv_a_path(date, RTOFS_GLOBAL_CACHE_DIR)
        if archv_a.exists():
            archv_a.unlink()
    if remove_tgz:
        tgz = date_dir / "rtofs_glo.t00z.f06.archv.a.tgz"
        if tgz.exists():
            tgz.unlink()


def enrich_date(
    date: str,
    work: pd.DataFrame,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
    dist_km: np.ndarray,
    *,
    cleanup_tgz: bool,
    cleanup_archv_a: bool,
) -> tuple[pd.DataFrame, dict]:
    sub = work[work["date"] == date].copy()
    if sub.empty:
        return work, {"date": date, "status": "empty"}

    date_dir = RTOFS_GLOBAL_CACHE_DIR / f"rtofs.{date}"
    download_global_archv_b(date, RTOFS_GLOBAL_CACHE_DIR)
    download_global_grid(date, RTOFS_GLOBAL_CACHE_DIR)
    download_global_archv_a(date, RTOFS_GLOBAL_CACHE_DIR)
    archv_a = extract_global_archv_a(date, RTOFS_GLOBAL_CACHE_DIR)

    header = parse_archv_b(date_dir / "rtofs_glo.t00z.f06.archv.b")
    temp_recs = records_for_field(header, "temp")
    sal_recs = records_for_field(header, "salin")
    thk_recs = records_for_field(header, "thknss")
    if not temp_recs or not sal_recs or not thk_recs:
        raise RuntimeError(f"Missing required temp/salin/thknss records for {date}")

    idx = sub.index.to_numpy(np.int64)
    y = y_idx[idx]
    x = x_idx[idx]
    d = dist_km[idx]

    mm = np.memmap(archv_a, dtype=">f4", mode="r")
    nz = len(temp_recs)
    n_rows = len(sub)
    temp_points = np.empty((n_rows, K_NEIGHBORS, nz), dtype=np.float32)
    sal_points = np.empty((n_rows, K_NEIGHBORS, nz), dtype=np.float32)
    thk_points = np.empty((n_rows, K_NEIGHBORS, nz), dtype=np.float32)

    for z, rec in enumerate(temp_recs):
        temp_points[:, :, z] = _sample_record_points(
            mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y_idx=y, x_idx=x
        )
    for z, rec in enumerate(sal_recs):
        sal_points[:, :, z] = _sample_record_points(
            mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y_idx=y, x_idx=x
        )
    for z, rec in enumerate(thk_recs):
        thk_points[:, :, z] = _sample_record_points(
            mm, record_index=rec.record_index, record_words=header.record_words, idm=header.idm, y_idx=y, x_idx=x
        )

    temp_prof = _interpolate_neighbor_profile(temp_points, d)
    sal_prof = _interpolate_neighbor_profile(sal_points, d)
    thk_prof = _interpolate_neighbor_profile(thk_points, d)

    feature_rows = []
    for row_i, (_, row) in enumerate(sub.iterrows()):
        feature_rows.append(
            _compute_row_profile_features(
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                temp_c=temp_prof[row_i],
                sal_psu=sal_prof[row_i],
                thickness_pa=thk_prof[row_i],
                model_d26_m=float(row["model_interp_d26_m"]) if pd.notna(row["model_interp_d26_m"]) else np.nan,
            )
        )
    feat_df = pd.DataFrame(feature_rows, index=sub.index)
    for col in FEATURE_COLUMNS:
        work.loc[sub.index, col] = feat_df[col].astype(np.float32)

    _cleanup_archive_files(date, remove_tgz=cleanup_tgz, remove_archv_a=cleanup_archv_a)
    finite_summary = {col: int(np.isfinite(feat_df[col]).sum()) for col in FEATURE_COLUMNS}
    return work, {"date": date, "status": "processed", "rows": int(len(sub)), "finite": finite_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=IN_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--summary-path", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--cleanup-tgz", action="store_true")
    parser.add_argument("--cleanup-archv-a", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = _load_or_init_df(args.input_path, args.output_path)
    work["date"] = work["date"].astype(str)
    work = work[np.isfinite(work["lat"]) & np.isfinite(work["lon"])].copy().reset_index(drop=True)

    grid_files = sorted(RTOFS_GLOBAL_CACHE_DIR.glob("rtofs.*/rtofs_glo_2ds_f006_diag.nc"))
    if not grid_files:
        raise FileNotFoundError("No cached RTOFS global grid/diagnostic file found in rtofs_global_cache.")
    grid_file = grid_files[0]
    y_idx, x_idx, dist_km = _prepare_neighbor_lookup(work, grid_file)

    all_dates = sorted(work["date"].unique().tolist())
    target_dates = args.dates if args.dates else all_dates
    if args.max_dates is not None:
        target_dates = target_dates[: args.max_dates]

    statuses: list[dict] = []
    for date in target_dates:
        already = work.loc[work["date"] == date, FEATURE_COLUMNS]
        if not already.empty and np.isfinite(already.to_numpy(float)).any():
            statuses.append({"date": date, "status": "skipped_existing", "rows": int(len(already))})
            continue
        work, status = enrich_date(
            date,
            work,
            y_idx,
            x_idx,
            dist_km,
            cleanup_tgz=args.cleanup_tgz,
            cleanup_archv_a=args.cleanup_archv_a,
        )
        statuses.append(status)
        work.to_parquet(args.output_path, index=False)
        args.status_path.write_text(json.dumps(statuses, indent=2))

    work.to_parquet(args.output_path, index=False)
    work.to_csv(args.output_csv, index=False)

    summary = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "rows_total": int(len(work)),
        "dates_requested": int(len(target_dates)),
        "dates_processed": int(sum(1 for s in statuses if s["status"] == "processed")),
        "feature_availability": {
            col: {"finite_rows": int(np.isfinite(pd.to_numeric(work[col], errors="coerce")).sum())}
            for col in FEATURE_COLUMNS
        },
        "status_path": str(args.status_path),
        "statuses": statuses,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
