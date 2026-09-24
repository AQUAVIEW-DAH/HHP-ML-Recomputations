"""Rebuild the collocation tables after the 2026-09-23 audit fixes.

Why: `_interpolate_neighbor_values` multiplied zero weights by NaN values, so
one missing neighbour out of eight voided the interpolation. TCHP/D26 grids
are NaN below 26 C as well as on land, so the failure concentrated on the
isotherm edge. The same function was copied into the global-physics builder.

What this does, in stages, never overwriting before verifying:
  1. Re-collocate Argo onto RTOFS with the fixed interpolation, carrying the
     Argo metadata the old loader discarded (cast, platform, profile index,
     level count, max depth). Writes to a staging file first.
  2. Verify row-for-row identity with the pre-fix table on every column that
     must not change. Abort without touching anything if it differs.
  3. Enrich: exact observation time from the GDAC profile index, and data
     mode, cycle and direction parsed from the file name. Flag one primary
     profile per cast (several profiles of one cast share a file: position 0
     is always "Primary sampling"; older batches never recorded the index, and
     "most levels" picks the primary in 99.8% of casts where it is known).
     Rows are FLAGGED, not deleted, so every row-aligned feature table stays
     aligned; deduplicate at analysis time with `is_primary_profile`.
  4. Promote the new base table, rebuild the global-physics table (same bug),
     refresh the key-column copies in the row-aligned feature tables whose own
     features are unaffected (neighbourhood and SSH-anomaly use a nearest-cell
     lookup; profile physics already handled NaN correctly), then run the
     existing alignment guard.

Backup of every pre-fix table: /data/suramya/table_backups/collocation_pre_fix_20260924/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import OHC.build_rtofs_at_argo_points_multiyear as colloc  # noqa: E402

DATA = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data")
BASE = DATA / "argo_rtofs_collocated_2024_2025.parquet"
STAGE = DATA / "_staging_argo_rtofs_collocated_2024_2025.parquet"
BACKUP = Path("/data/suramya/table_backups/collocation_pre_fix_20260924")
GDAC_INDEX = Path("/data/suramya/argo_cache_hhp/ar_index_global_prof.txt")
REPORT = DATA / "rebuild_report_20260924.json"

# must be identical before and after: the fix only changes model-side values
IDENTITY_COLS = ["date", "year", "month", "lat", "lon", "nearest_rtofs_grid_distance_km",
                 "argo_tchp_kj_per_cm2", "argo_d26_m"]
MODEL_COLS = ["model_interp_tchp_kj_per_cm2", "model_interp_d26_m"]
KEY_COLS = IDENTITY_COLS + MODEL_COLS + ["delta_tchp_kj_per_cm2", "delta_d26_m"]
ROW_ALIGNED = {
    "neighborhood": DATA / "argo_rtofs_collocated_2024_2025_neighborhood.parquet",
    "profile_physics": DATA / "argo_rtofs_collocated_2024_2025_profile_physics.parquet",
    "ssh_anom": DATA / "argo_rtofs_collocated_2024_2025_ssh_anom.parquet",
    "woa_clim": DATA / "argo_rtofs_collocated_2024_2025_woa_clim.parquet",
}


def same(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
    return a.astype(str).equals(b.astype(str))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.read_csv(GDAC_INDEX, comment="#", usecols=["file", "date"], dtype={"date": str})
    idx = idx.drop_duplicates("file").rename(columns={"file": "cast_id", "date": "gdac_datetime"})
    out = df.merge(idx, on="cast_id", how="left")
    if len(out) != len(df):
        raise RuntimeError("GDAC join changed the row count")
    t = pd.to_datetime(out["gdac_datetime"], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    out["obs_time_utc"] = t
    out["obs_hour_utc"] = t.dt.hour + t.dt.minute / 60.0
    base = out["cast_id"].fillna("").str.split("/").str[-1]
    out["argo_data_mode"] = base.str.extract(r"^([A-Z]+)\d", expand=False)
    out["argo_cycle"] = pd.to_numeric(base.str.extract(r"_(\d+)D?\.nc$", expand=False), errors="coerce")
    out["argo_descending"] = base.str.endswith("D.nc")
    # one primary profile per cast: prefer position 0, fall back to most levels
    rank = pd.DataFrame({
        "cast_id": out["cast_id"],
        "is0": (out["profile_index"] == 0).astype(int),
        "lev": out["n_levels"].fillna(-1),
        "pos": np.arange(len(out)),
    })
    rank = rank.sort_values(["cast_id", "is0", "lev", "pos"], ascending=[True, False, False, True])
    first = rank.drop_duplicates("cast_id", keep="first")["pos"].to_numpy()
    prim = np.zeros(len(out), dtype=bool)
    prim[first] = True
    prim[out["cast_id"].isna().to_numpy()] = True  # no cast id: cannot be a duplicate of anything
    out["is_primary_profile"] = prim
    return out.drop(columns=["gdac_datetime"])


def main() -> None:
    report: dict = {}
    old = pd.read_parquet(BACKUP / BASE.name)

    # ---- stage 1: re-collocate with the fix, to staging
    frames = []
    for year in (2024, 2025):
        cfg = colloc.YEAR_CONFIG[year]
        frames.append(colloc.collocate_year(year=year, argo_path=cfg["argo_path"], rtofs_dir=cfg["rtofs_dir"]))
        print("collocated", year, len(frames[-1]), flush=True)
    new = pd.concat(frames, ignore_index=True)
    new.to_parquet(STAGE, index=False)

    # ---- stage 2: verify identity before touching anything live
    if len(new) != len(old):
        raise RuntimeError(f"row count changed: {len(old)} -> {len(new)}; nothing promoted")
    bad = [c for c in IDENTITY_COLS if not same(old[c], new[c])]
    if bad:
        raise RuntimeError(f"rows are not identical on {bad}; nothing promoted (staging kept at {STAGE})")
    for c in MODEL_COLS:
        o, n = old[c].to_numpy(float), new[c].to_numpy(float)
        both = np.isfinite(o) & np.isfinite(n)
        report[c] = {
            "recovered_nan_to_value": int((np.isnan(o) & np.isfinite(n)).sum()),
            "lost_value_to_nan": int((np.isfinite(o) & np.isnan(n)).sum()),
            "max_abs_change_where_both_valid": float(np.max(np.abs(o[both] - n[both]))) if both.any() else 0.0,
        }
        if report[c]["lost_value_to_nan"] or report[c]["max_abs_change_where_both_valid"] > 1e-4:
            raise RuntimeError(f"{c}: fix changed previously valid values {report[c]}; nothing promoted")
    print("identity verified", json.dumps({c: report[c] for c in MODEL_COLS}), flush=True)

    # ---- stage 3: enrich, then promote
    ordered = list(old.columns) + [c for c in new.columns if c not in old.columns]
    new = enrich(new[ordered])
    report["gdac_time_matched_pct"] = float(100 * new["obs_time_utc"].notna().mean())
    report["is_primary_profile_false"] = int((~new["is_primary_profile"]).sum())
    warm = new["argo_tchp_kj_per_cm2"].notna() & new["model_interp_tchp_kj_per_cm2"].notna()
    report["warm_rows_before_fix"] = int((old["argo_tchp_kj_per_cm2"].notna() & old["model_interp_tchp_kj_per_cm2"].notna()).sum())
    report["warm_rows_after_fix"] = int(warm.sum())
    report["warm_rows_after_fix_primary_only"] = int((warm & new["is_primary_profile"]).sum())
    report["data_mode_counts"] = new["argo_data_mode"].value_counts(dropna=False).astype(int).to_dict()
    new.to_parquet(BASE, index=False)
    for year in (2024, 2025):
        y = new[new["year"] == year]
        y.to_parquet(DATA / f"argo_rtofs_collocated_{year}.parquet", index=False)
    STAGE.unlink()
    print("base promoted", flush=True)

    # ---- stage 4a: rebuild global physics (same interpolation bug)
    import OHC.build_rtofs_global_physics_features_2024_2025 as phys
    phys.main()
    print("physics rebuilt", flush=True)

    # ---- stage 4b: refresh key-column copies in row-aligned tables
    report["refreshed"] = {}
    for name, path in ROW_ALIGNED.items():
        if not path.exists():
            continue
        t = pd.read_parquet(path).reset_index(drop=True)
        if len(t) != len(new):
            raise RuntimeError(f"{name}: length {len(t)} != base {len(new)}")
        chk = [c for c in ("date", "lat", "lon") if c in t.columns]
        if not all(same(t[c], new[c]) for c in chk):
            raise RuntimeError(f"{name}: not row-aligned with base on {chk}")
        upd = [c for c in KEY_COLS if c in t.columns]
        for c in upd:
            t[c] = new[c].to_numpy()
        t.to_parquet(path, index=False)
        report["refreshed"][name] = upd
        print("refreshed", name, flush=True)

    # ---- stage 4c: the existing alignment guard must pass
    import OHC.run_locked_xgb_physics_semi_ablation as abl
    merged = abl._merge_feature_tables()
    report["merge_guard"] = f"passed, {len(merged)} rows"
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
