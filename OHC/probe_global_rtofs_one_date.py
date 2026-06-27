"""Probe one global RTOFS date and inspect archive structure.

This is a lightweight sanity tool for the real global pipeline. By default it
downloads only the global grid NetCDF and the small `archv.b` header. If
`--download-archv-a` is passed, it also downloads/extracts the large binary
archive and reads the first layer of `temp`, `salin`, and `thknss`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.paths import RTOFS_GLOBAL_CACHE_DIR
from ml.sources.rtofs_global_source import (
    download_global_archv_a,
    download_global_archv_b,
    download_global_grid,
    extract_global_archv_a,
    normalize_longitude,
    parse_archv_b,
    read_archv_record,
    records_for_field,
    summarize_archv_fields,
)


def summarize_array(arr: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(arr)
    vals = arr[finite]
    return {
        "finite_cells": int(finite.sum()),
        "min": float(np.nanmin(vals)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p95": float(np.nanpercentile(vals, 95)),
        "max": float(np.nanmax(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cache-dir", type=Path, default=RTOFS_GLOBAL_CACHE_DIR)
    parser.add_argument("--grid-date", default="20240131")
    parser.add_argument("--download-archv-a", action="store_true")
    args = parser.parse_args()

    grid_path = download_global_grid(args.grid_date, args.cache_dir)
    header_path = download_global_archv_b(args.date, args.cache_dir)
    header = parse_archv_b(header_path)

    with xr.open_dataset(grid_path) as ds:
        lat = ds["Latitude"].values
        lon = normalize_longitude(ds["Longitude"].values)
        grid_summary = {
            "shape": list(lat.shape),
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
        }

    out: dict[str, object] = {
        "date": args.date,
        "grid_date": args.grid_date,
        "grid_summary": grid_summary,
        "header_summary": {
            "idm": header.idm,
            "jdm": header.jdm,
            "record_words": header.record_words,
            "n_records": len(header.fields),
            "field_counts": summarize_archv_fields(header),
        },
    }

    if args.download_archv_a:
        download_global_archv_a(args.date, args.cache_dir)
        data_path = extract_global_archv_a(args.date, args.cache_dir)
        field_samples = {}
        for field_name in ("temp", "salin", "thknss"):
            rec = records_for_field(header, field_name)[0]
            arr = read_archv_record(data_path, header, rec.record_index)
            field_samples[field_name] = summarize_array(arr)
        out["field_samples_layer1"] = field_samples

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
