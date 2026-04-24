"""IBTrACS hurricane track fetcher via NOAA AOML ERDDAP.

Backed by the `IBTrACS_since1980_1` dataset indexed in AQUAVIEW (collection
`NOAA_AOML_HDB`). The upstream source is AOML's ERDDAP tabledap endpoint;
AQUAVIEW provides discovery/metadata, but the actual track rows are pulled
directly as CSV via ERDDAP query parameters.

Per-storm access: filter by `name` and `season`. Columns returned are the
subset needed for map overlay and track-aware co-location queries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from ml.paths import IBTRACS_CACHE_DIR

logger = logging.getLogger(__name__)

IBTRACS_ERDDAP_CSV = (
    "https://erddap.aoml.noaa.gov/hdb/erddap/tabledap/IBTrACS_since1980_1.csv"
)

# Minimal column set for track visualisation + co-location work
DEFAULT_COLUMNS = [
    "sid", "season", "basin", "name",
    "iso_time", "latitude", "longitude",
    "wmo_wind", "wmo_pres",
    "usa_wind", "usa_pres", "usa_sshs",
    "nature", "dist2land", "landfall",
    "storm_speed", "storm_dir",
]


def fetch_storm_track(
    name: str,
    season: int,
    columns: list[str] | None = None,
    cache_dir: Path = IBTRACS_CACHE_DIR,
    timeout: float = 60.0,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch best-track rows for one storm from AOML ERDDAP.

    Returns a DataFrame with at least (iso_time, latitude, longitude) and
    intensity/nature columns. Cached to disk as CSV per (name, season).
    """
    columns = columns or DEFAULT_COLUMNS
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ibtracs_{name.upper()}_{season}.csv"

    if cache_path.exists() and not force:
        logger.info("Cached IBTrACS %s %d -> %s", name.upper(), season, cache_path)
        return pd.read_csv(cache_path)

    # ERDDAP tabledap: col list + & filters. name is a quoted string.
    col_csv = ",".join(columns)
    name_expr = 'name="{}"'.format(name.upper())
    season_expr = f"season={season}"
    quote_name = quote(name_expr, safe='="')
    quote_season = quote(season_expr, safe="=")
    url = f'{IBTRACS_ERDDAP_CSV}?{quote(col_csv, safe=",")}&{quote_name}&{quote_season}'

    logger.info("Fetching IBTrACS %s %d from %s", name.upper(), season, url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_bytes(resp.content)
    tmp_path.replace(cache_path)

    # First line = column names, second line = units. Skip units line.
    df = pd.read_csv(cache_path, skiprows=[1])
    logger.info("IBTrACS %s %d: %d track points", name.upper(), season, len(df))
    return df


def track_dates_yyyymmdd(track_df: pd.DataFrame) -> list[str]:
    """Return unique YYYYMMDD strings covered by a storm track."""
    if "iso_time" not in track_df.columns:
        return []
    times = pd.to_datetime(track_df["iso_time"], utc=True, errors="coerce").dropna()
    return sorted({t.strftime("%Y%m%d") for t in times})


def track_bbox(track_df: pd.DataFrame, padding_deg: float = 5.0) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) bbox covering the track, with padding."""
    lat = pd.to_numeric(track_df["latitude"], errors="coerce").dropna()
    lon = pd.to_numeric(track_df["longitude"], errors="coerce").dropna()
    return (
        float(lon.min() - padding_deg),
        float(lat.min() - padding_deg),
        float(lon.max() + padding_deg),
        float(lat.max() + padding_deg),
    )
