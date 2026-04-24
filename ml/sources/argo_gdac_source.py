"""Argo GDAC profile ingestion for HHP.

Forked from the MLD project's argo_gdac_source.py and adapted:
- Keeps salinity (needed for TEOS-10 density and specific heat)
- Drops MLD-specific 10 m reference-depth bracketing QC
- Requires max_depth >= 200 m so profiles reach below a plausible D26
- Widens the default basin to the Atlantic hurricane region

Index-based discovery: use the Argo global profile index (one row per profile)
to locate candidate NetCDF files, then download only the matched ones and parse
T, S, P with QC flags 1-2 accepted.
"""
from __future__ import annotations

import logging
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from ml.paths import ARGO_CACHE_DIR

logger = logging.getLogger(__name__)

ARGO_GDAC_BASE = "https://data-argo.ifremer.fr/dac"
ARGO_INDEX_URL = "https://data-argo.ifremer.fr/ar_index_global_prof.txt"
INDEX_CACHE_NAME = "ar_index_global_prof.txt"

MIN_DEPTH_LEVELS = 5
MIN_MAX_DEPTH_M_FOR_HHP = 200.0  # Must reach below likely D26
GOOD_QC_FLAGS = {"1", "2"}

ssl_ctx = ssl._create_unverified_context()


@dataclass(frozen=True)
class ArgoHHPProfile:
    source: str
    instrument: str
    cast_id: str          # path-form cast id (e.g. "aoml/4903556/profiles/R4903556_157.nc")
    platform: str         # platform number
    cycle: str
    lat: float
    lon: float
    obs_time: str         # ISO8601 with Z
    pressure_dbar: list[float]
    temperature_c: list[float]
    salinity_psu: list[float]

    @property
    def max_depth_m(self) -> float:
        # In the upper ocean, 1 dbar ≈ 1 m to ~1% accuracy. Good enough for a
        # coarse depth-coverage filter; TEOS-10 computations below use true p.
        return float(max(self.pressure_dbar)) if self.pressure_dbar else 0.0


def _fetch_bytes(url: str, timeout_seconds: int = 300) -> bytes:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_seconds, context=ssl_ctx) as resp:
        return resp.read()


def _download_to_cache(url: str, local_path: Path, timeout_seconds: int = 300) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    tmp_path.write_bytes(_fetch_bytes(url, timeout_seconds=timeout_seconds))
    tmp_path.replace(local_path)
    return local_path


def download_argo_index(index_url: str = ARGO_INDEX_URL) -> Path:
    return _download_to_cache(index_url, ARGO_CACHE_DIR / INDEX_CACHE_NAME, timeout_seconds=600)


def _parse_index_date(value: str) -> str:
    return value.strip()[:8]


def matching_index_rows(
    bbox: list[float],
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    max_profiles: int | None = None,
    max_per_platform: int | None = None,
    index_url: str = ARGO_INDEX_URL,
) -> list[dict]:
    """Return per-profile index rows (each is one cast) inside bbox/date window."""
    west, south, east, north = bbox
    index_path = download_argo_index(index_url)

    header: list[str] | None = None
    matches: list[dict] = []
    platform_counts: dict[str, int] = {}
    total_rows = 0

    for raw_line in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if header is None:
            header = [v.strip() for v in line.split(",")]
            continue

        values = [v.strip() for v in line.split(",")]
        if len(values) < len(header):
            continue
        item = dict(zip(header, values))
        total_rows += 1

        try:
            lat = float(item.get("latitude", "nan"))
            lon = float(item.get("longitude", "nan"))
        except ValueError:
            continue

        date = _parse_index_date(item.get("date", ""))
        if not (start_yyyymmdd <= date <= end_yyyymmdd):
            continue
        if not (south <= lat <= north and west <= lon <= east):
            continue

        file_name = item.get("file", "")
        if not file_name:
            continue
        platform_id = file_name.split("/")[1] if "/" in file_name else file_name
        if max_per_platform and platform_counts.get(platform_id, 0) >= max_per_platform:
            continue

        matches.append({
            "file": file_name,
            "platform": platform_id,
            "lat": lat,
            "lon": lon,
            "date": date,
        })
        platform_counts[platform_id] = platform_counts.get(platform_id, 0) + 1
        if max_profiles and len(matches) >= max_profiles:
            break

    logger.info(
        "Argo index scan: %d rows scanned, %d matches across %d platforms "
        "in bbox=%s date=%s/%s",
        total_rows, len(matches), len(platform_counts),
        bbox, start_yyyymmdd, end_yyyymmdd,
    )
    return matches


def _decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()
    return str(value).strip()


def _qc_mask(qc_values: np.ndarray) -> np.ndarray:
    qc_str = np.array([_decode_scalar(v) for v in qc_values.ravel()]).reshape(qc_values.shape)
    return np.isin(qc_str, list(GOOD_QC_FLAGS))


def _preferred_var(ds: xr.Dataset, raw_name: str, adjusted_name: str) -> tuple[str, str | None]:
    if adjusted_name in ds and np.isfinite(ds[adjusted_name].values).any():
        qc_name = f"{adjusted_name}_QC"
        return adjusted_name, qc_name if qc_name in ds else None
    qc_name = f"{raw_name}_QC"
    return raw_name, qc_name if qc_name in ds else None


def _extract_profiles_from_file(local_path: Path, source_file: str) -> list[ArgoHHPProfile]:
    profiles: list[ArgoHHPProfile] = []
    with xr.open_dataset(local_path, decode_timedelta=False) as ds:
        required = ("PRES", "TEMP", "PSAL", "LATITUDE", "LONGITUDE")
        if any(v not in ds for v in required):
            return profiles

        pres_var, pres_qc_var = _preferred_var(ds, "PRES", "PRES_ADJUSTED")
        temp_var, temp_qc_var = _preferred_var(ds, "TEMP", "TEMP_ADJUSTED")
        psal_var, psal_qc_var = _preferred_var(ds, "PSAL", "PSAL_ADJUSTED")

        pres = ds[pres_var].values
        temp = ds[temp_var].values
        psal = ds[psal_var].values
        n_prof = int(ds.sizes.get("N_PROF", pres.shape[0] if pres.ndim > 1 else 1))

        if pres.ndim == 1:
            pres = pres.reshape(1, -1)
            temp = temp.reshape(1, -1)
            psal = psal.reshape(1, -1)

        pres_qc = _qc_mask(ds[pres_qc_var].values) if pres_qc_var else np.ones_like(pres, dtype=bool)
        temp_qc = _qc_mask(ds[temp_qc_var].values) if temp_qc_var else np.ones_like(temp, dtype=bool)
        psal_qc = _qc_mask(ds[psal_qc_var].values) if psal_qc_var else np.ones_like(psal, dtype=bool)

        for idx in range(n_prof):
            p = np.asarray(pres[idx], dtype=float)
            t = np.asarray(temp[idx], dtype=float)
            s = np.asarray(psal[idx], dtype=float)
            valid = (
                np.isfinite(p) & np.isfinite(t) & np.isfinite(s)
                & pres_qc[idx] & temp_qc[idx] & psal_qc[idx]
            )
            p = p[valid]
            t = t[valid]
            s = s[valid]

            if len(p) < MIN_DEPTH_LEVELS:
                continue
            if float(p.max()) < MIN_MAX_DEPTH_M_FOR_HHP:
                continue

            order = np.argsort(p)
            p, t, s = p[order], t[order], s[order]

            platform = _decode_scalar(ds["PLATFORM_NUMBER"].values[idx]) if "PLATFORM_NUMBER" in ds else "unknown"
            cycle = _decode_scalar(ds["CYCLE_NUMBER"].values[idx]) if "CYCLE_NUMBER" in ds else str(idx)
            try:
                obs_time = np.datetime_as_string(ds["JULD"].values[idx], unit="s") + "Z"
            except Exception:
                obs_time = _decode_scalar(ds["JULD"].values[idx]) if "JULD" in ds else ""

            profiles.append(
                ArgoHHPProfile(
                    source="ARGO_GDAC",
                    instrument="pfl",
                    cast_id=source_file,
                    platform=platform,
                    cycle=cycle,
                    lat=float(ds["LATITUDE"].values[idx]),
                    lon=float(ds["LONGITUDE"].values[idx]),
                    obs_time=obs_time,
                    pressure_dbar=p.tolist(),
                    temperature_c=t.tolist(),
                    salinity_psu=s.tolist(),
                )
            )
    return profiles


def download_and_extract(
    bbox: list[float],
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    max_profiles: int | None = None,
    max_per_platform: int | None = None,
    index_url: str = ARGO_INDEX_URL,
) -> list[ArgoHHPProfile]:
    """End-to-end: scan the Argo index, download matched profile NetCDFs, parse."""
    rows = matching_index_rows(
        bbox=bbox,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
        max_profiles=max_profiles,
        max_per_platform=max_per_platform,
        index_url=index_url,
    )

    out: list[ArgoHHPProfile] = []
    skipped_fetch = skipped_parse = 0
    for i, row in enumerate(rows, start=1):
        source_file = row["file"]
        local = ARGO_CACHE_DIR / source_file
        url = f"{ARGO_GDAC_BASE}/{source_file}"
        if i == 1 or i % 25 == 0 or i == len(rows):
            logger.info("Argo fetch/parse %d/%d", i, len(rows))
        try:
            path = _download_to_cache(url, local, timeout_seconds=120)
            profiles = _extract_profiles_from_file(path, source_file)
        except Exception as exc:
            skipped_fetch += 1
            logger.warning("Argo fetch/parse failed for %s: %s", source_file, exc)
            continue
        if not profiles:
            skipped_parse += 1
        out.extend(profiles)

    logger.info(
        "Argo HHP profiles: %d usable from %d files (%d empty, %d fetch-failed)",
        len(out), len(rows), skipped_parse, skipped_fetch,
    )
    return out
