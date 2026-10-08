"""SIDE EXPLORATION: collocate Argo with the GOFS 3.1 analysis daily fields (2021-09-05 .. 2024-09-04).

Same sampling as the RTOFS tables (8 nearest grid cells, inverse-distance-squared weights, missing
neighbours excluded) and the same one-profile-per-cast rule (prefer position 0, else most levels), so
GOFS and RTOFS errors can be compared on identical floats where the records overlap (2024-01-31 ..
2024-09-04). Only profiles inside the downloaded latitude band (42 S - 48 N) are kept.

Fields come from build_gofs31_analysis_daily_fields.py: TCHP and D26 from GOFS temperature with a
constant salinity of 35 (TCHP changes <= 0.19%), SST, SSH, and a temperature-criterion mixed layer.
Output: OHC/output/gofs31_analysis_20261008/argo_gofs31a_collocated.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402
from OHC.build_rtofs_at_argo_points_multiyear import (  # noqa: E402
    K_NEIGHBORS, _build_grid_lookup, _interpolate_neighbor_values, _load_argo_table)

GOFS_DIR = Path("/data/suramya/gofs31_analysis_fields")
ARGO_PATH = Path("/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/gofs31_analysis_20261008")
FIELDS = {"gofs_tchp_kj_per_cm2": "tchp_kj_per_cm2", "gofs_d26_m": "d26_m", "gofs_sst_c": "surface_temp_c",
          "gofs_ssh_m": "ssh_m", "gofs_mld_t02_m": "mld_t02_m"}
LAT_BAND = (-41.5, 47.5)   # half a degree inside the downloaded band, so all 8 neighbours exist


def primary_only(df: pd.DataFrame) -> pd.DataFrame:
    rank = pd.DataFrame({"cast_id": df["cast_id"], "is0": (df.get("profile_index") == 0).astype(int)
                         if "profile_index" in df else 0, "lev": df["n_levels"].fillna(-1), "pos": np.arange(len(df))})
    rank = rank.sort_values(["cast_id", "is0", "lev", "pos"], ascending=[True, False, False, True])
    keep = np.zeros(len(df), bool)
    keep[rank.drop_duplicates("cast_id", keep="first")["pos"].to_numpy()] = True
    keep[df["cast_id"].isna().to_numpy()] = True
    return df[keep].reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dates = sorted(p.stem.split("_")[-1] for p in GOFS_DIR.glob("gofs31a_fields_*.nc"))
    argo = pd.concat([_load_argo_table(ARGO_PATH, y) for y in (2021, 2022, 2023, 2024)], ignore_index=True)
    argo = argo[argo["date"].isin(dates) & argo["lat"].between(*LAT_BAND) & np.isfinite(argo["lon"])]
    argo = primary_only(argo.reset_index(drop=True))
    tree, all_y, all_x = _build_grid_lookup(GOFS_DIR / f"gofs31a_fields_{dates[0]}.nc")
    chord, flat = tree.query(latlon_to_xyz(argo["lat"].to_numpy(float), argo["lon"].to_numpy(float)).astype(np.float32),
                             k=K_NEIGHBORS, workers=-1)
    yi, xi = all_y[flat], all_x[flat]
    dist_km = 6371.0 * 2.0 * np.arcsin(np.clip(chord.astype(np.float64) / 2.0, 0.0, 1.0))
    out = {k: np.full(len(argo), np.nan, np.float32) for k in FIELDS}
    for i, (date, idx) in enumerate(argo.groupby("date").groups.items()):
        idx = np.asarray(list(idx), dtype=np.int64)
        with xr.open_dataset(GOFS_DIR / f"gofs31a_fields_{date}.nc") as ds:
            for k, v in FIELDS.items():
                out[k][idx] = _interpolate_neighbor_values(ds[v].values, yi[idx], xi[idx], dist_km[idx])
        if i % 100 == 0:
            print(f"{i} dates", flush=True)
    for k, v in out.items():
        argo[k] = v
    argo["gofs_delta_tchp"] = argo["argo_tchp_kj_per_cm2"] - argo["gofs_tchp_kj_per_cm2"]
    argo["gofs_delta_d26"] = argo["argo_d26_m"] - argo["gofs_d26_m"]
    argo.to_parquet(OUT / "argo_gofs31a_collocated.parquet", index=False)
    warm = argo["argo_tchp_kj_per_cm2"].notna() & argo["gofs_tchp_kj_per_cm2"].notna()
    summary = {"field_dates": len(dates), "profiles": int(len(argo)), "warm_profiles": int(warm.sum()),
               "date_range": [dates[0], dates[-1]],
               "tchp_mae": float(argo.loc[warm, "gofs_delta_tchp"].abs().mean()),
               "tchp_mean_error_argo_minus_gofs": float(argo.loc[warm, "gofs_delta_tchp"].mean()),
               "d26_mae": float(argo.loc[warm, "gofs_delta_d26"].abs().mean()),
               "d26_mean_error_argo_minus_gofs": float(argo.loc[warm, "gofs_delta_d26"].mean())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
