"""Validate TEOS-backed OHC/TCHP outputs and flag suspicious rows.

Usage:
    ./hhp-env/bin/python OHC/validate_teos_ohc.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")


def load_outputs() -> pd.DataFrame:
    frames = []
    for path in sorted(OUT_DIR.glob("batch_*.parquet")):
        df = pd.read_parquet(path)
        if "ohc_j_per_m2" not in df.columns and "integral_j_per_m2" in df.columns:
            df = df.rename(columns={"integral_j_per_m2": "ohc_j_per_m2"})
        frames.append(df)
    if not frames:
        raise SystemExit(f"No parquet batches found in {OUT_DIR}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    all_df = load_outputs()
    ok = all_df[all_df["error"].isna()].copy() if "error" in all_df.columns else all_df.copy()

    counts = {
        "rows_total": len(all_df),
        "rows_ok": len(ok),
        "negative_d26": int(ok["d26_m"].lt(0).sum()),
        "d26_gt_max_depth": int(((ok["d26_m"] > ok["max_depth_m"]) & ok["d26_m"].notna()).sum()),
        "negative_tchp": int(ok["tchp_kj_per_cm2"].lt(0).sum()),
        "negative_ohc": int(ok["ohc_j_per_m2"].lt(0).sum()),
        "cold_surface_positive_tchp": int(((ok["surface_t_c"] < 26) & (ok["tchp_kj_per_cm2"] > 0)).sum()),
        "warm_surface_missing_d26": int(((ok["surface_t_c"] >= 26) & ok["d26_m"].isna()).sum()),
        "duplicate_cast_ids": int((ok["cast_id"].value_counts() > 1).sum()),
    }

    print("Validation counts")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    print("\nTCHP quantiles (kJ/cm^2)")
    print(ok["tchp_kj_per_cm2"].quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]).to_string())

    print("\nD26 quantiles (m)")
    print(ok["d26_m"].dropna().quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]).to_string())

    suspicious = ok.loc[
        ok["d26_m"].lt(0)
        | ((ok["surface_t_c"] < 26) & (ok["tchp_kj_per_cm2"] > 0))
        | ((ok["surface_t_c"] >= 26) & ok["d26_m"].isna()),
        [c for c in [
            "cast_id",
            "profile_key",
            "profile_index",
            "date",
            "lat",
            "lon",
            "surface_t_c",
            "d26_m",
            "tchp_kj_per_cm2",
            "ohc_j_per_m2",
            "max_depth_m",
            "levels_above_d26",
        ] if c in ok.columns],
    ]
    print("\nSuspicious rows")
    print(suspicious.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
