"""Build seasonal mean RTOFS TCHP/OHC fields from daily reduced outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

INPUT_DIR = Path('/data/suramya/rtofs_ohc_fields_global')
DEFAULT_OUT_DIR = INPUT_DIR / 'seasonal_means'
SEASONS = {
    'winter_jfm': ('Winter (Jan-Feb-Mar)', {1, 2, 3}),
    'summer_jas': ('Summer (Jul-Aug-Sep)', {7, 8, 9}),
}
PARAMS = ('tchp_kj_per_cm2', 'ohc_j_per_m2', 'd26_m', 'surface_temp_c')


def list_daily_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob('rtofs_tchp_*.nc'))


def parse_date(path: Path) -> str:
    stem = path.stem
    return stem.split('_')[-1]


def select_files(input_dir: Path, year: int, months: set[int]) -> list[Path]:
    out = []
    for p in list_daily_files(input_dir):
        d = parse_date(p)
        if int(d[:4]) == year and int(d[4:6]) in months:
            out.append(p)
    return out


def build_mean(paths: list[Path]) -> xr.Dataset:
    if not paths:
        raise ValueError('No daily RTOFS fields found for selection')

    with xr.open_dataset(paths[0]) as first:
        lat = first['Latitude'].load()
        lon = first['Longitude'].load()
        template_shape = first['Latitude'].shape

    sums = {k: np.zeros(template_shape, dtype=np.float64) for k in PARAMS}
    counts = {k: np.zeros(template_shape, dtype=np.int32) for k in PARAMS}

    for p in paths:
        with xr.open_dataset(p) as ds:
            for key in PARAMS:
                arr = ds[key].values.astype(np.float64)
                mask = np.isfinite(arr)
                sums[key][mask] += arr[mask]
                counts[key][mask] += 1

    data_vars = {}
    for key in PARAMS:
        mean = np.full(template_shape, np.nan, dtype=np.float32)
        valid = counts[key] > 0
        mean[valid] = (sums[key][valid] / counts[key][valid]).astype(np.float32)
        data_vars[key] = (('Y', 'X'), mean)
        data_vars[f'{key}_count'] = (('Y', 'X'), counts[key].astype(np.int32))

    out = xr.Dataset(
        data_vars={
            **data_vars,
            'Latitude': (('Y', 'X'), lat.values.astype(np.float32)),
            'Longitude': (('Y', 'X'), lon.values.astype(np.float32)),
        },
        coords={
            'Y': np.arange(template_shape[0], dtype=np.int32),
            'X': np.arange(template_shape[1], dtype=np.int32),
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--years', nargs='+', type=int, default=[2024])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for year in args.years:
        for season_key, (_, months) in SEASONS.items():
            paths = select_files(args.input_dir, year, months)
            if not paths:
                print(json.dumps({'year': year, 'season': season_key, 'status': 'skipped', 'reason': 'no files'}))
                continue
            ds = build_mean(paths)
            ds.attrs.update({
                'year': year,
                'season': season_key,
                'n_daily_fields': len(paths),
                'daily_dates': json.dumps([parse_date(p) for p in paths]),
                'description': 'Seasonal mean RTOFS TCHP/OHC field built from reduced daily native-grid fields.',
            })
            out_path = args.out_dir / f'rtofs_{year}_{season_key}_mean.nc'
            ds.to_netcdf(out_path)
            ds.close()
            row = {'year': year, 'season': season_key, 'status': 'written', 'n_files': len(paths), 'out_path': str(out_path)}
            summary.append(row)
            print(json.dumps(row))

    print(json.dumps({'done': True, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
