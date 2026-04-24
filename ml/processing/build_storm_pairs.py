"""Generic storm-event Argo/RTOFS pairing pipeline for HHP.

This script generalizes the Milton-specific prototype so we can build paired
event tables for any storm with an IBTrACS best track and same-day RTOFS 3dz
files. It writes three artifacts per event:

  artifacts/datasets/<storm>_<season>_argo_profile_manifest.csv
  artifacts/datasets/<storm>_<season>_pairs.csv
  artifacts/datasets/<storm>_<season>_pair_profiles.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.paths import DATASETS_DIR, RTOFS_CACHE_DIR
from ml.processing.build_milton_pairs import _load_argo_profile, _nearest_rtofs_column
from ml.sources.argo_gdac_source import download_and_extract
from ml.sources.ibtracs_source import fetch_storm_track, track_bbox, track_dates_yyyymmdd
from ml.sources.rtofs_source import cached_rtofs_path, download_rtofs_date_range
from hhp_core import compute_tchp
import xarray as xr

logger = logging.getLogger(__name__)


def _slug(name: str, season: int) -> str:
    return f"{name.lower()}_{season}"


def build_event(
    *,
    name: str,
    season: int,
    padding_deg: float = 5.0,
    max_profiles: int | None = None,
    max_per_platform: int | None = None,
    force_track: bool = False,
) -> dict[str, Path]:
    slug = _slug(name, season)
    manifest_csv = DATASETS_DIR / f"{slug}_argo_profile_manifest.csv"
    pairs_csv = DATASETS_DIR / f"{slug}_pairs.csv"
    profiles_pkl = DATASETS_DIR / f"{slug}_pair_profiles.pkl"

    track_df = fetch_storm_track(name, season, force=force_track)
    dates = track_dates_yyyymmdd(track_df)
    if not dates:
        raise RuntimeError(f"No IBTrACS dates found for {name} {season}")
    bbox = list(track_bbox(track_df, padding_deg=padding_deg))

    logger.info(
        "Building event %s %d with bbox=%s date=%s/%s",
        name.upper(), season, bbox, dates[0], dates[-1],
    )

    download_rtofs_date_range(dates, RTOFS_CACHE_DIR)

    profiles = download_and_extract(
        bbox=bbox,
        start_yyyymmdd=dates[0],
        end_yyyymmdd=dates[-1],
        max_profiles=max_profiles,
        max_per_platform=max_per_platform,
    )
    manifest_df = pd.DataFrame([{
        "cast_id": p.cast_id,
        "source_file": p.cast_id,
        "platform": p.platform,
        "cycle": p.cycle,
        "obs_time": p.obs_time,
        "lat": p.lat,
        "lon": p.lon,
        "max_depth_m": p.max_depth_m,
        "storm_name": name.upper(),
        "storm_season": season,
    } for p in profiles])
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info("Wrote manifest: %s (%d rows)", manifest_csv, len(manifest_df))

    manifest_df["obs_date"] = manifest_df["obs_time"].str[:10].str.replace("-", "", regex=False)
    rows = []
    profile_records = []
    for obs_date, group in manifest_df.groupby("obs_date"):
        rtofs_path = cached_rtofs_path(str(obs_date), RTOFS_CACHE_DIR)
        if not rtofs_path.exists():
            logger.warning("Skipping %s — no RTOFS file at %s", obs_date, rtofs_path)
            continue
        logger.info("Pairing %s: %d profiles", obs_date, len(group))
        ds = xr.open_dataset(rtofs_path)
        try:
            for _, row in group.iterrows():
                argo = _load_argo_profile(row["source_file"])
                if not argo:
                    continue
                obs_res = compute_tchp(
                    argo["pressure_dbar"],
                    argo["temperature_c"],
                    salinity_psu=argo["salinity_psu"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    vertical_axis="pressure",
                )
                col = _nearest_rtofs_column(ds, float(row["lat"]), float(row["lon"]))
                if col is None:
                    continue
                mod_res = compute_tchp(
                    col["depth_m"],
                    col["temperature_c"],
                    salinity_psu=col["salinity_psu"],
                    lat=float(col["grid_lat"]),
                    lon=float(col["grid_lon"]),
                    vertical_axis="depth",
                )
                obs_tchp = obs_res.tchp_kj_per_cm2
                mod_tchp = mod_res.tchp_kj_per_cm2
                delta = (obs_tchp - mod_tchp) if (obs_tchp is not None and mod_tchp is not None) else None

                rows.append({
                    "cast_id": row["source_file"],
                    "platform": row["platform"],
                    "obs_time": row["obs_time"],
                    "obs_date": obs_date,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "storm_name": name.upper(),
                    "storm_season": season,
                    "obs_surface_t_c": obs_res.surface_temp_c,
                    "obs_d26_m": obs_res.d26_m,
                    "obs_tchp_kj_cm2": obs_tchp,
                    "obs_max_depth_m": obs_res.max_depth_m,
                    "model_grid_lat": col["grid_lat"],
                    "model_grid_lon": col["grid_lon"],
                    "model_grid_distance_km": col["distance_km"],
                    "model_surface_t_c": mod_res.surface_temp_c,
                    "model_d26_m": mod_res.d26_m,
                    "model_tchp_kj_cm2": mod_tchp,
                    "model_max_depth_m": mod_res.max_depth_m,
                    "tchp_delta_kj_cm2": delta,
                })
                profile_records.append({
                    "cast_id": row["source_file"],
                    "obs_date": obs_date,
                    "storm_name": name.upper(),
                    "storm_season": season,
                    "obs_pressure_dbar": argo["pressure_dbar"],
                    "obs_temperature_c": argo["temperature_c"],
                    "obs_salinity_psu": argo["salinity_psu"],
                    "model_depth_m": col["depth_m"].tolist(),
                    "model_temperature_c": col["temperature_c"].tolist(),
                    "model_salinity_psu": col["salinity_psu"].tolist(),
                })
        finally:
            ds.close()

    pd.DataFrame(rows).to_csv(pairs_csv, index=False)
    pd.DataFrame(profile_records).to_pickle(profiles_pkl)
    logger.info("Wrote pairs: %s (%d rows)", pairs_csv, len(rows))
    logger.info("Wrote profile arrays: %s (%d rows)", profiles_pkl, len(profile_records))
    return {"manifest_csv": manifest_csv, "pairs_csv": pairs_csv, "profiles_pkl": profiles_pkl}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Argo/RTOFS event pairs for one storm.")
    parser.add_argument("--name", required=True, help="Storm name, e.g. HELENE")
    parser.add_argument("--season", required=True, type=int, help="Storm season/year, e.g. 2024")
    parser.add_argument("--padding-deg", type=float, default=5.0)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--max-per-platform", type=int, default=None)
    parser.add_argument("--force-track", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    paths = build_event(
        name=args.name,
        season=args.season,
        padding_deg=args.padding_deg,
        max_profiles=args.max_profiles,
        max_per_platform=args.max_per_platform,
        force_track=args.force_track,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
