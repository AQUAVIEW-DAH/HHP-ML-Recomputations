"""Build the 2026 forward-holdout tables. PROCESSING ONLY: nothing is scored here.

2026 is the one period no model choice was tuned on, so it is kept physically
separate from the development tables in OHC/output/ml_collocation/data/ and
must be evaluated exactly once, after the final configuration is frozen.

Same pipeline as the 2026-09-24 rebuild of 2024-2025, so the two are directly
comparable:
  1. collocate 2026 Argo profiles onto 2026 RTOFS with the fixed
     inverse-distance interpolation, carrying the Argo metadata;
  2. add exact observation time, data mode, cycle, and the one-primary-
     profile-per-cast flag (same rule as the rebuild);
  3. build the global-physics and neighbourhood features by pointing the
     existing builders at the 2026 inputs and the holdout folder.
The deep-profile features (steric height, N^2) are not built: they need the
3D archives, which are deleted after each day is processed.

Output: OHC/output/ml_collocation/holdout_2026/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import OHC.build_rtofs_at_argo_points_multiyear as colloc  # noqa: E402
import OHC.build_rtofs_global_physics_features_2024_2025 as phys  # noqa: E402
import OHC.build_rtofs_neighborhood_features_2024_2025 as nbhd  # noqa: E402
from OHC.rebuild_tables_20260924 import enrich  # noqa: E402

HOLD = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/holdout_2026")
ARGO = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2026")
FIELDS = Path("/data/suramya/rtofs_global_ohc_fields_2026")
BASE = HOLD / "argo_rtofs_collocated_2026.parquet"


def main() -> None:
    HOLD.mkdir(parents=True, exist_ok=True)
    (HOLD / "README.txt").write_text(
        "2026 FORWARD HOLDOUT. Do not tune, select, or evaluate on these tables until the final\n"
        "configuration is frozen; then evaluate exactly once. Built by OHC/build_holdout_2026_tables.py.\n")
    report = {}

    df = colloc.collocate_year(year=2026, argo_path=ARGO, rtofs_dir=FIELDS)
    df = enrich(df)
    df.to_parquet(BASE, index=False)
    warm = df["argo_tchp_kj_per_cm2"].notna() & df["model_interp_tchp_kj_per_cm2"].notna()
    report.update({"rows": int(len(df)), "dates": int(df["date"].nunique()),
                   "date_range": [str(df["date"].min()), str(df["date"].max())],
                   "warm_rows": int(warm.sum()),
                   "warm_primary_rows": int((warm & df["is_primary_profile"]).sum()),
                   "obs_time_matched_pct": float(100 * df["obs_time_utc"].notna().mean())})
    print("collocated", json.dumps(report), flush=True)

    phys.IN_PATH = BASE
    phys.OUT_DIR = HOLD
    phys.OUT_PATH = HOLD / "argo_rtofs_collocated_2026_physics.parquet"
    phys.OUT_CSV = HOLD / "argo_rtofs_collocated_2026_physics.csv"
    phys.OUT_SUMMARY = HOLD / "summary_2026_physics.json"
    phys.RTOFS_DAILY_DIR[2026] = FIELDS
    phys.main()
    print("physics done", flush=True)

    nbhd.IN_PATH = BASE
    nbhd.OUT_DIR = HOLD
    nbhd.OUT_PATH = HOLD / "argo_rtofs_collocated_2026_neighborhood.parquet"
    nbhd.OUT_SUMMARY = HOLD / "summary_2026_neighborhood.json"
    nbhd.RTOFS_DAILY_DIR[2026] = FIELDS
    nbhd.main()
    print("neighbourhood done", flush=True)

    for name in ("physics", "neighborhood"):
        t = pd.read_parquet(HOLD / f"argo_rtofs_collocated_2026_{name}.parquet")
        if len(t) != len(df) or not (t["lat"].to_numpy() == df["lat"].to_numpy()).all():
            raise RuntimeError(f"{name} table is not row-aligned with the 2026 base")
    report["aligned"] = True
    (HOLD / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
