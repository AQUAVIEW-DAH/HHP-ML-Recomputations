"""Helpers for the actual global RTOFS/HYCOM archive products.

The public NOAA RTOFS bucket exposes:

- a global 2D diagnostics NetCDF (`rtofs_glo_2ds_f006_diag.nc`) that carries
  the full global curvilinear Latitude/Longitude grid and some surface fields
  on the 4500 x 3298 domain;
- a global HYCOM archive pair for 3D state at the same forecast lead:
  `rtofs_glo.t00z.f06.archv.a.tgz` and `rtofs_glo.t00z.f06.archv.b`.

The `.b` file is a small plain-text header that describes the record sequence
stored in the raw IEEE float archive `.a` file.
"""
from __future__ import annotations

import logging
import math
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

RTOFS_S3_BASE = "https://noaa-nws-rtofs-pds.s3.amazonaws.com"
GLOBAL_GRID_FILE = "rtofs_glo_2ds_f006_diag.nc"
GLOBAL_ARCHV_A_FILE = "rtofs_glo.t00z.f06.archv.a.tgz"
GLOBAL_ARCHV_B_FILE = "rtofs_glo.t00z.f06.archv.b"


def global_grid_url_for_date(yyyymmdd: str) -> str:
    return f"{RTOFS_S3_BASE}/rtofs.{yyyymmdd}/{GLOBAL_GRID_FILE}"


def global_archv_a_url_for_date(yyyymmdd: str) -> str:
    return f"{RTOFS_S3_BASE}/rtofs.{yyyymmdd}/{GLOBAL_ARCHV_A_FILE}"


def global_archv_b_url_for_date(yyyymmdd: str) -> str:
    return f"{RTOFS_S3_BASE}/rtofs.{yyyymmdd}/{GLOBAL_ARCHV_B_FILE}"


def cached_global_grid_path(cache_dir: Path, grid_date: str) -> Path:
    return cache_dir / f"rtofs.{grid_date}" / GLOBAL_GRID_FILE


def cached_global_archv_a_path(yyyymmdd: str, cache_dir: Path) -> Path:
    return cache_dir / f"rtofs.{yyyymmdd}" / GLOBAL_ARCHV_A_FILE


def cached_global_archv_b_path(yyyymmdd: str, cache_dir: Path) -> Path:
    return cache_dir / f"rtofs.{yyyymmdd}" / GLOBAL_ARCHV_B_FILE


def extracted_global_archv_a_path(yyyymmdd: str, cache_dir: Path) -> Path:
    return cache_dir / f"rtofs.{yyyymmdd}" / GLOBAL_ARCHV_A_FILE.removesuffix(".tgz")


def _download_file(
    url: str,
    local_path: Path,
    *,
    timeout: float = 600.0,
    force: bool = False,
    max_retries: int = 4,
) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        logger.info("Cached %s -> %s", url, local_path)
        return local_path
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    expected_size = None
    if not force:
        try:
            head = requests.head(url, timeout=min(timeout, 60.0))
            head.raise_for_status()
            raw = head.headers.get("Content-Length")
            expected_size = int(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("HEAD failed for %s: %s", url, exc)

    if tmp_path.exists() and expected_size is not None and tmp_path.stat().st_size >= expected_size:
        tmp_path.replace(local_path)
        logger.info("Recovered completed tmp download for %s -> %s", url, local_path)
        return local_path

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        resume_from = 0 if force else (tmp_path.stat().st_size if tmp_path.exists() else 0)
        mode = "ab" if resume_from > 0 else "wb"
        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else None
        logger.info(
            "Downloading %s -> %s (attempt %d/%d, resume_from=%d)",
            url, local_path, attempt, max_retries, resume_from,
        )
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                if resume_from > 0 and response.status_code == 200:
                    logger.warning("Server ignored Range for %s; restarting tmp download from zero", url)
                    resume_from = 0
                    mode = "wb"
                response.raise_for_status()
                total_bytes = resume_from
                with tmp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            total_bytes += len(chunk)
            final_size = tmp_path.stat().st_size
            if expected_size is not None and final_size < expected_size:
                raise IOError(
                    f"Incomplete download for {url}: got {final_size} bytes, expected {expected_size}"
                )
            tmp_path.replace(local_path)
            logger.info("Wrote %s (%.1f MB)", local_path, final_size / (1024 * 1024))
            return local_path
        except Exception as exc:
            last_exc = exc
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
            if attempt == max_retries:
                break
            time.sleep(min(30, 2 ** attempt))
    assert last_exc is not None
    raise last_exc
    logger.info("Wrote %s (%.1f MB)", local_path, total_bytes / (1024 * 1024))
    return local_path


def head_global_product_sizes(yyyymmdd: str, *, timeout: float = 60.0) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for key, url in {
        "grid_nc_bytes": global_grid_url_for_date(yyyymmdd),
        "archv_a_tgz_bytes": global_archv_a_url_for_date(yyyymmdd),
        "archv_b_bytes": global_archv_b_url_for_date(yyyymmdd),
    }.items():
        response = requests.head(url, timeout=timeout)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        out[key] = int(value) if value is not None else None
    return out


def download_global_grid(grid_date: str, cache_dir: Path, *, timeout: float = 600.0, force: bool = False) -> Path:
    return _download_file(
        global_grid_url_for_date(grid_date),
        cached_global_grid_path(cache_dir, grid_date),
        timeout=timeout,
        force=force,
    )


def download_global_archv_b(yyyymmdd: str, cache_dir: Path, *, timeout: float = 600.0, force: bool = False) -> Path:
    return _download_file(
        global_archv_b_url_for_date(yyyymmdd),
        cached_global_archv_b_path(yyyymmdd, cache_dir),
        timeout=timeout,
        force=force,
    )


def download_global_archv_a(yyyymmdd: str, cache_dir: Path, *, timeout: float = 600.0, force: bool = False) -> Path:
    return _download_file(
        global_archv_a_url_for_date(yyyymmdd),
        cached_global_archv_a_path(yyyymmdd, cache_dir),
        timeout=timeout,
        force=force,
    )


def extract_global_archv_a(yyyymmdd: str, cache_dir: Path, *, force: bool = False) -> Path:
    tgz_path = cached_global_archv_a_path(yyyymmdd, cache_dir)
    out_path = extracted_global_archv_a_path(yyyymmdd, cache_dir)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        logger.info("Cached extracted archive %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise ValueError(f"No file members found in {tgz_path}")
        if len(members) > 1:
            logger.warning("Multiple file members in %s; using first member %s", tgz_path, members[0].name)
        member = members[0]
        with tf.extractfile(member) as src, out_path.open("wb") as dst:
            if src is None:
                raise ValueError(f"Could not extract {member.name} from {tgz_path}")
            while True:
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    logger.info("Extracted %s -> %s", tgz_path, out_path)
    return out_path


@dataclass(frozen=True)
class ArchvFieldRecord:
    field_name: str
    timestep: int
    model_day: float
    k: int
    dens: float
    min_value: float
    max_value: float
    record_index: int


@dataclass(frozen=True)
class ArchvHeader:
    title_lines: tuple[str, ...]
    idm: int
    jdm: int
    record_words: int
    fields: tuple[ArchvFieldRecord, ...]


_DIM_RE = re.compile(r"\s*(\d+)\s+'(idm|jdm)\s+'")
_FIELD_RE = re.compile(
    r"^\s*(?P<field>[^=]+?)\s*=\s*"
    r"(?P<timestep>-?\d+)\s+"
    r"(?P<model_day>-?\d+(?:\.\d+)?)\s+"
    r"(?P<k>-?\d+)\s+"
    r"(?P<dens>-?\d+(?:\.\d+)?)\s+"
    r"(?P<min>[+-]?\d+\.\d+E[+-]\d+)\s+"
    r"(?P<max>[+-]?\d+\.\d+E[+-]\d+)\s*$"
)


def parse_archv_b(header_path: Path) -> ArchvHeader:
    lines = header_path.read_text().splitlines()
    idm = None
    jdm = None
    title_lines: list[str] = []
    fields: list[ArchvFieldRecord] = []

    for line in lines:
        m = _DIM_RE.match(line)
        if m:
            value = int(m.group(1))
            if m.group(2) == "idm":
                idm = value
            else:
                jdm = value
            continue
        if "'idm" in line or "'jdm" in line:
            continue
        fm = _FIELD_RE.match(line)
        if fm:
            fields.append(
                ArchvFieldRecord(
                    field_name=fm.group("field").strip(),
                    timestep=int(fm.group("timestep")),
                    model_day=float(fm.group("model_day")),
                    k=int(fm.group("k")),
                    dens=float(fm.group("dens")),
                    min_value=float(fm.group("min")),
                    max_value=float(fm.group("max")),
                    record_index=len(fields),
                )
            )
        elif not fields:
            title_lines.append(line.rstrip())

    if idm is None or jdm is None:
        raise ValueError(f"Could not parse idm/jdm from {header_path}")
    record_words = math.ceil((idm * jdm) / 4096) * 4096
    return ArchvHeader(
        title_lines=tuple(title_lines),
        idm=idm,
        jdm=jdm,
        record_words=record_words,
        fields=tuple(fields),
    )


def summarize_archv_fields(header: ArchvHeader) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in header.fields:
        counts[field.field_name] = counts.get(field.field_name, 0) + 1
    return counts


def normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """Convert HYCOM-style wrapped longitudes to the conventional [-180, 180) range."""
    return ((lon + 180.0) % 360.0) - 180.0


def records_for_field(header: ArchvHeader, field_name: str) -> list[ArchvFieldRecord]:
    return [field for field in header.fields if field.field_name == field_name]


def read_archv_record(
    data_path: Path,
    header: ArchvHeader,
    record_index: int,
    *,
    mask_void: bool = True,
) -> np.ndarray:
    """Read one 2-D HYCOM archive record as big-endian float32.

    HYCOM documents describe archive files as big-endian IEEE REAL*4 arrays.
    Each 2-D array is padded to a multiple of 4096 32-bit words.
    """
    words = header.record_words
    offset_bytes = record_index * words * 4
    n_words = words
    with data_path.open("rb") as handle:
        handle.seek(offset_bytes)
        arr = np.fromfile(handle, dtype=">f4", count=n_words)
    if arr.size != n_words:
        raise ValueError(
            f"Short read for record {record_index}: expected {n_words} words, got {arr.size}"
        )
    arr = arr[: header.idm * header.jdm].astype(np.float32, copy=False).reshape(header.jdm, header.idm)
    if mask_void:
        arr = np.where(arr > 1.0e30, np.nan, arr)
    return arr
