"""Re-render the RTOFS-at-Argo-points seasonal maps from the current collocation.

The original `rtofs_at_argo_points_2024` figures (2026-05-26) were built from a
pre-backfill collocation with only 61 RTOFS dates. The backfill completed the
2024-2025 archive (701 days) but those figures were never regenerated. This
script subsets the current ML collocation table to winter (JFM) / summer (JAS)
per year and reuses the original point and interpolated renderers.

Output: OHC/output/rtofs_at_argo_points_<RUN_DATE>/<year>/{data,points,interpolated}/
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.build_rtofs_at_argo_points_2024 as pts  # noqa: E402
import OHC.render_rtofs_at_argo_points_interpolated as interp  # noqa: E402

RUN_DATE = "20260826"
SRC = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_ROOT = Path(f"/home/suramya/HHP-Prediction/OHC/output/rtofs_at_argo_points_{RUN_DATE}")
FIELDS = [
    ("model_tchp_kj_per_cm2", "RTOFS TCHP / OHC (kJ/cm²) at Argo points", False),
    ("model_d26_m", "RTOFS D26 (m) at Argo points", False),
    ("delta_tchp_kj_per_cm2", "Argo - RTOFS TCHP difference (kJ/cm²)", True),
    ("delta_d26_m", "Argo - RTOFS D26 difference (m)", True),
]


def main() -> None:
    df = pd.read_parquet(SRC)
    d = pd.to_datetime(df["date"].astype(str))
    df["year"], df["month"] = d.dt.year, d.dt.month
    df = df.rename(columns={"model_interp_tchp_kj_per_cm2": "model_tchp_kj_per_cm2",
                            "model_interp_d26_m": "model_d26_m",
                            "argo_tchp_kj_per_cm2": "tchp_kj_per_cm2", "argo_d26_m": "d26_m"})
    df["season"] = df["month"].map(lambda m: "winter_jfm" if m in (1, 2, 3) else "summer_jas" if m in (7, 8, 9) else None)
    df = df[df["season"].notna()].copy()
    for year in (2024, 2025):
        ydf = df[df["year"] == year].copy()
        pts.YEAR = interp.YEAR = year
        out = OUT_ROOT / str(year)
        for sub in ("data", "points", "interpolated"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        ydf.to_parquet(out / "data" / f"argo_rtofs_collocated_{year}_winter_summer.parquet", index=False)
        for field, label, delta in FIELDS:
            sub = ydf[ydf[field].notna()]
            print(year, field, {k: int(v) for k, v in sub["season"].value_counts().items()}, "dates", sub["date"].nunique())
            pts._render_point_panels(sub, field, label, out / "points" / f"{field}_{year}_points_winter_summer.png", delta=delta)
            interp.render_panels(sub, field=field, label=label, method="gaussian", resolution_deg=0.25,
                                 mask_distance_km=100.0, smooth_display=False, delta=delta,
                                 out_path=out / "interpolated" / f"{field}_{year}_gaussian_0p25deg_masked_100km_grid.png")
    print("wrote", OUT_ROOT)


if __name__ == "__main__":
    main()
