"""Build the global Argo TCHP/D26 dataset for 2020-2024.

Pulls every Argo cast in the global GDAC index for the date window, parses
T/S/P with QC flags 1-2, and computes TCHP and D26 via TEOS-10
(``hhp_core.compute_tchp`` → ``gsw.rho_t_exact`` / ``gsw.cp_t_exact``).

Outputs are written incrementally as numbered parquet batch files into a
single directory, so:
- a kill mid-run loses at most one batch (≤ ``CHECKPOINT_EVERY`` rows);
- resuming on the next launch automatically skips cast IDs already on disk;
- ``pd.read_parquet(out_dir)`` reads all batches as one frame.

Run:
    ./hhp-env/bin/python OHC/build_global_argo_2020_2024.py [--workers 16]

Output dir: ``ARGO_CACHE_DIR / "global_argo_tchp_d26_2020_2024/"``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhp_core import compute_tchp
from ml.paths import ARGO_CACHE_DIR
from ml.sources.argo_gdac_source import (
    ARGO_GDAC_BASE,
    _download_to_cache,
    _extract_profiles_from_file,
    matching_index_rows,
)

OUT_DIR = ARGO_CACHE_DIR / "global_argo_tchp_d26_2020_2024"
CHECKPOINT_EVERY = 2000   # rows to buffer before writing one batch parquet
DEFAULT_WORKERS = 8
MAX_IN_FLIGHT_PER_WORKER = 4
DEFAULT_START = "20200101"
DEFAULT_END = "20241231"

logger = logging.getLogger("build_global_argo")


def _process_row(row: dict) -> list[dict]:
    """Download one Argo file, parse profiles, compute TCHP/D26.

    Returns a list of result dicts (1-2 entries per file). On fetch/parse
    failure, returns a single dict with an ``error`` field instead of fields.
    """
    source_file = row["file"]
    local = ARGO_CACHE_DIR / source_file
    url = f"{ARGO_GDAC_BASE}/{source_file}"
    try:
        path = _download_to_cache(url, local, timeout_seconds=180)
    except Exception as exc:
        return [{"cast_id": source_file, "platform": row.get("platform", ""),
                 "date": row.get("date", ""), "error": f"fetch:{exc}"}]
    try:
        profiles = _extract_profiles_from_file(path, source_file)
    except Exception as exc:
        return [{"cast_id": source_file, "platform": row.get("platform", ""),
                 "date": row.get("date", ""), "error": f"parse:{exc}"}]
    if not profiles:
        return [{"cast_id": source_file, "platform": row.get("platform", ""),
                 "date": row.get("date", ""), "error": "empty"}]

    out: list[dict] = []
    for prof in profiles:
        try:
            res = compute_tchp(
                np.asarray(prof.pressure_dbar, dtype=float),
                np.asarray(prof.temperature_c, dtype=float),
                salinity_psu=np.asarray(prof.salinity_psu, dtype=float),
                lat=prof.lat,
                lon=prof.lon,
                vertical_axis="pressure",
            )
            yyyymmdd = prof.obs_time[:10].replace("-", "")
            out.append({
                "cast_id": prof.cast_id,
                "profile_key": prof.profile_key,
                "profile_index": prof.profile_index,
                "platform": prof.platform,
                "lat": prof.lat,
                "lon": prof.lon,
                "date": yyyymmdd,
                "year": int(yyyymmdd[:4]) if yyyymmdd[:4].isdigit() else None,
                "month": int(yyyymmdd[4:6]) if yyyymmdd[4:6].isdigit() else None,
                "n_levels": len(prof.pressure_dbar),
                "max_depth_m": prof.max_depth_m,
                "surface_t_c": float(prof.temperature_c[0]) if prof.temperature_c else None,
                "d26_m": res.d26_m,
                "tchp_kj_per_cm2": res.tchp_kj_per_cm2,
                "integral_j_per_m2": res.integral_j_per_m2,
                "levels_above_d26": res.levels_above_d26,
                "error": None,
            })
        except Exception as exc:
            out.append({
                "cast_id": prof.cast_id, "platform": prof.platform,
                "date": prof.obs_time[:10].replace("-", ""),
                "error": f"compute:{exc}",
            })
    return out


def _existing_cast_ids(out_dir: Path) -> set[str]:
    if not out_dir.exists():
        return set()
    cast_ids: set[str] = set()
    for p in sorted(out_dir.glob("batch_*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["cast_id"])
        except Exception as exc:
            logger.warning("Could not read %s (%s); ignoring for resume.", p, exc)
            continue
        cast_ids.update(df["cast_id"].astype(str).tolist())
    return cast_ids


def _next_batch_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob("batch_*.parquet")))
    return out_dir / f"batch_{n:05d}.parquet"


def _flush(buffer: list[dict], out_dir: Path) -> None:
    if not buffer:
        return
    path = _next_batch_path(out_dir)
    pd.DataFrame(buffer).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N pending index rows (for smoke tests).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("Scanning global Argo index for %s..%s", args.start, args.end)
    rows = matching_index_rows(
        bbox=[-180.0, -90.0, 180.0, 90.0],
        start_yyyymmdd=args.start,
        end_yyyymmdd=args.end,
    )
    logger.info("Index match: %d candidate rows", len(rows))

    completed = _existing_cast_ids(args.out_dir)
    pending = [r for r in rows if r["file"] not in completed]
    logger.info(
        "Resume state: %d cast IDs already in %s; %d pending.",
        len(completed), args.out_dir, len(pending),
    )

    if args.limit is not None:
        pending = pending[: args.limit]
        logger.info("--limit applied: processing %d rows.", len(pending))

    buffer: list[dict] = []
    n_done = n_ok = n_err = 0
    t_start = time.time()

    max_in_flight = max(args.workers * MAX_IN_FLIGHT_PER_WORKER, args.workers)

    def _handle_future_result(fut) -> None:
        nonlocal n_done, n_ok, n_err, buffer
        n_done += 1
        try:
            results = fut.result()
        except Exception as exc:
            n_err += 1
            buffer.append({"cast_id": "?", "error": f"worker:{exc}"})
            return
        for rec in results:
            buffer.append(rec)
            if rec.get("error"):
                n_err += 1
            else:
                n_ok += 1

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending_iter = iter(pending)
        in_flight = set()

        for _ in range(min(max_in_flight, len(pending))):
            try:
                row = next(pending_iter)
            except StopIteration:
                break
            in_flight.add(ex.submit(_process_row, row))

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                _handle_future_result(fut)
                try:
                    row = next(pending_iter)
                except StopIteration:
                    row = None
                if row is not None:
                    in_flight.add(ex.submit(_process_row, row))

            if len(buffer) >= CHECKPOINT_EVERY or (n_done and n_done % 250 == 0):
                _flush(buffer, args.out_dir)
                buffer.clear()
                elapsed = max(time.time() - t_start, 1e-3)
                rate = n_done / elapsed
                remaining = len(pending) - n_done
                eta_min = (remaining / rate) / 60 if rate > 0 else float("inf")
                logger.info(
                    "Progress: %d/%d  (ok=%d err=%d  rate=%.1f/s  elapsed=%.1f min  ETA=%.1f min  in_flight=%d)",
                    n_done, len(pending), n_ok, n_err, rate, elapsed / 60, eta_min, len(in_flight),
                )

    _flush(buffer, args.out_dir)
    elapsed = time.time() - t_start
    logger.info(
        "DONE in %.1f min: %d processed, %d ok, %d err.",
        elapsed / 60, n_done, n_ok, n_err,
    )
    logger.info("Output dir: %s", args.out_dir)
    logger.info("Read with: pd.read_parquet('%s')", args.out_dir)


if __name__ == "__main__":
    main()
