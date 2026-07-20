"""SIDE EXPLORATION: collocate 2015 Argo TCHP/D26 against GOFS 3.1 reanalysis fields.

Mirrors the RTOFS multiyear collocation exactly (same 8-neighbor
inverse-distance-squared sampling, same table schema) so downstream analyses
can treat the two model datasets interchangeably. See exploration/README.md
for the cross-model caveats (different model bias; NCODA assimilates Argo).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402
from OHC.build_rtofs_at_argo_points_multiyear import (  # noqa: E402
    K_NEIGHBORS,
    _build_grid_lookup,
    _interpolate_neighbor_values,
    _load_argo_table,
    _season_from_month,
)

GOFS_DIR = Path("/data/suramya/gofs31_ohc_fields_2015")
ARGO_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2015")
OUT_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_gofs31_collocated_2015.parquet")
OUT_SUMMARY = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/summary_argo_gofs31_collocated_2015.json")


def main() -> None:
    argo = _load_argo_table(ARGO_PATH, 2015)
    argo = argo[np.isfinite(argo["lat"]) & np.isfinite(argo["lon"])].copy().reset_index(drop=True)
    gofs_dates = sorted(p.stem.split("_")[-1] for p in GOFS_DIR.glob("gofs31_tchp_*.nc"))
    argo = argo[argo["date"].isin(gofs_dates)].copy().reset_index(drop=True)
    if argo.empty:
        raise RuntimeError("No overlapping Argo/GOFS dates found for 2015.")

    sample_file = GOFS_DIR / f"gofs31_tchp_{gofs_dates[0]}.nc"
    tree, all_y, all_x = _build_grid_lookup(sample_file)

    obs_xyz = latlon_to_xyz(argo["lat"].to_numpy(float), argo["lon"].to_numpy(float)).astype(np.float32)
    chord_dist, flat_idx = tree.query(obs_xyz, k=K_NEIGHBORS, workers=-1)
    y_idx = all_y[flat_idx]
    x_idx = all_x[flat_idx]
    dist_km = 6371.0 * (2.0 * np.arcsin(np.clip(chord_dist.astype(np.float64) / 2.0, 0.0, 1.0)))
    argo["nearest_rtofs_grid_distance_km"] = dist_km[:, 0]

    model_tchp = np.full(len(argo), np.nan, dtype=np.float32)
    model_d26 = np.full(len(argo), np.nan, dtype=np.float32)
    for date, idx in argo.groupby("date").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        with xr.open_dataset(GOFS_DIR / f"gofs31_tchp_{date}.nc") as ds:
            model_tchp[idx] = _interpolate_neighbor_values(ds["tchp_kj_per_cm2"].values, y_idx[idx], x_idx[idx], dist_km[idx])
            model_d26[idx] = _interpolate_neighbor_values(ds["d26_m"].values, y_idx[idx], x_idx[idx], dist_km[idx])

    argo["model_interp_tchp_kj_per_cm2"] = model_tchp
    argo["model_interp_d26_m"] = model_d26
    argo["delta_tchp_kj_per_cm2"] = argo["argo_tchp_kj_per_cm2"] - argo["model_interp_tchp_kj_per_cm2"]
    argo["delta_d26_m"] = argo["argo_d26_m"] - argo["model_interp_d26_m"]
    argo["season"] = argo["month"].astype(int).map(_season_from_month)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    argo.to_parquet(OUT_PATH, index=False)
    summary = {
        "rows_total": int(len(argo)),
        "dates_total": int(argo["date"].nunique()),
        "finite_tchp_residuals": int(np.isfinite(argo["delta_tchp_kj_per_cm2"]).sum()),
        "finite_d26_residuals": int(np.isfinite(argo["delta_d26_m"]).sum()),
        "output_path": str(OUT_PATH),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
