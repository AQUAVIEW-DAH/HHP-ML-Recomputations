"""SIDE EXPLORATION: daily TCHP/D26 fields from the GOFS 3.1 *analysis* (2021-09-05 .. 2024-09-04).

Three years of a sibling HYCOM system for training a TCHP/D26 error model
against Argo, chosen to (a) overlap our RTOFS record for 2024-01-27..09-04, so
the GOFS and RTOFS errors can be compared on the same floats, and (b) span
the 2021-23 La Nina and the 2023-24 El Nino. Argo is already cached for all
of it (argo_cache_hhp/global_argo_tchp_d26_2020_2024).

Source: HYCOM file server, GLBy0.08/expt_93.0 hindcasts (1/12.5 deg, 40 z-levels,
3-hourly, NetCDF-3, 3.1 GB per 3D file). The server caps us at ~2.4 MB/s in
total, so whole files are never downloaded. The NetCDF-3 header gives each
variable's byte offset, and with one time step a variable is stored
contiguously as (depth, lat, lon), so each depth level's latitude band is one
HTTP range request. Only what TCHP/D26 need is fetched: temperature on the
levels down to 250 m over 42 S - 48 N (warm water plus a margin for the
2-degree neighbourhood features), ~0.5 GB per day, plus the 2D SSH.
(netCDF-C's own `#mode=bytes` remote reads were tried first: >30 min per day.)

Choices that differ from the RTOFS fields, all deliberate:
  * Valid time 06Z, the same as RTOFS (00Z cycle + 6 h): the 12Z run of the
    previous day at tau 18. Fallbacks are recorded in the file attributes.
  * Salinity is not downloaded: rho and cp use SA from a constant S = 35.
    Measured on 1,577 warm RTOFS columns, this changes TCHP by 0.04% on
    average and at most 0.19% (D26 does not depend on salinity).
  * `water_temp` is labelled `sea_water_temperature` (in-situ), so unlike the
    RTOFS archives it is used as TEOS-10 expects.
  * Mixed-layer depth is computed here (HYCOM's own diagnostic is not in these
    files): depth where T falls 0.2 C below T(10 m) (de Boyer Montegut et al.
    2004). It is NOT the same definition as RTOFS `mixed_layer_thickness`.
  * Layers are the z-level cells (bounds at mid-points between levels), as in
    the 2015 reanalysis pilot (build_gofs31_daily_ohc_fields.py).

Output: /data/suramya/gofs31_analysis_fields/gofs31a_fields_YYYYMMDD.nc
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import struct
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import gsw
import numpy as np
import requests
import xarray as xr

logger = logging.getLogger(__name__)

BASE = "https://data.hycom.org/datasets/GLBy0.08/expt_93.0/data/hindcasts"
OUT_DIR = Path("/data/suramya/gofs31_analysis_fields")
START, END = "20210905", "20240904"
RTOFS_OVERLAP = ("20240127", "20240904")
REF_TEMP_C = 26.0
MAX_DEPTH_M = 250.0
LAT_BAND = (-42.0, 48.0)
S_CONST = 35.0
MLD_DT, MLD_REF_M = 0.2, 10.0
ROW_CHUNK = 250
RETRIES = 5
THREADS = 4
HEADER_BYTES = 131072
NC_DTYPE = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}


def _candidates(date: str) -> list[tuple[str, int]]:
    """(run, tau) pairs for this date, preferred first: 06Z valid, then the nearest other times."""
    d = dt.datetime.strptime(date, "%Y%m%d")
    prev = (d - dt.timedelta(days=1)).strftime("%Y%m%d")
    return [(f"{prev}12", 18), (f"{prev}12", 15), (f"{prev}12", 21), (f"{date}12", 0)]


def _url(run: str, tau: int, kind: str) -> str:
    return f"{BASE}/{run[:4]}/hycom_glby_930_{run}_t{tau:03d}_{kind}.nc"


def _get_range(url: str, start: int, length: int) -> bytes:
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=900)
            r.raise_for_status()
            if len(r.content) == length:
                return r.content
            raise OSError(f"short read {len(r.content)} of {length}")
        except (requests.RequestException, OSError) as e:
            if attempt == RETRIES:
                raise
            logger.warning("range retry %d for %s @%d: %s", attempt, url.rsplit("/", 1)[-1], start, e)
            time.sleep(30 * attempt)
    raise AssertionError


class NC3:
    """Minimal NetCDF-3 (classic / 64-bit offset) header reader for remote byte-range access."""

    def __init__(self, url: str):
        self.url = url
        h = _get_range(url, 0, HEADER_BYTES)
        if h[:3] != b"CDF" or h[3] not in (1, 2):
            raise ValueError(f"not a NetCDF-3 file: {url}")
        off = 8 if h[3] == 2 else 4
        pos = 4

        def u32():
            nonlocal pos
            v = struct.unpack(">I", h[pos:pos + 4])[0]
            pos += 4
            return v

        def name():
            nonlocal pos
            n = u32()
            s = h[pos:pos + n].decode()
            pos += n + (-n % 4)
            return s

        def atts():
            nonlocal pos
            u32()
            out = {}
            for _ in range(u32()):
                k = name()
                t, m = u32(), u32()
                nb = m * np.dtype(NC_DTYPE[t]).itemsize
                out[k] = np.frombuffer(h[pos:pos + nb], NC_DTYPE[t]) if t != 2 else h[pos:pos + nb].decode()
                pos += nb + (-nb % 4)
            return out

        self.numrecs = u32()
        u32()
        dims = [(name(), u32()) for _ in range(u32())]
        atts()
        u32()
        self.vars = {}
        for _ in range(u32()):
            n = name()
            ids = [u32() for _ in range(u32())]
            a = atts()
            t, _vsize = u32(), u32()
            begin = struct.unpack(">Q" if off == 8 else ">I", h[pos:pos + off])[0]
            pos += off
            shape = [d[1] if d[1] else self.numrecs for d in (dims[i] for i in ids)]
            self.vars[n] = {"shape": shape, "dtype": NC_DTYPE[t], "begin": begin, "atts": a}
        self._head = h

    def coord(self, name: str) -> np.ndarray:
        v = self.vars[name]
        n = int(np.prod(v["shape"]))
        nb = n * np.dtype(v["dtype"]).itemsize
        raw = self._head[v["begin"]:v["begin"] + nb] if v["begin"] + nb <= len(self._head) else \
            _get_range(self.url, v["begin"], nb)
        return np.frombuffer(raw, v["dtype"]).astype(np.float64)

    def rows(self, name: str, level: int | None, y0: int, y1: int) -> np.ndarray:
        """Rows y0:y1 of one (time=0[, level]) slab, decoded to float32 with NaN for fill."""
        v = self.vars[name]
        nlat, nlon = v["shape"][-2], v["shape"][-1]
        isz = np.dtype(v["dtype"]).itemsize
        start = v["begin"] + (((level or 0) * nlat + y0) * nlon) * isz
        raw = np.frombuffer(_get_range(self.url, start, (y1 - y0) * nlon * isz), v["dtype"]).reshape(y1 - y0, nlon)
        a = v["atts"]
        out = raw.astype(np.float32)
        if "_FillValue" in a:
            out[raw == a["_FillValue"][0]] = np.nan
        return out * float(a.get("scale_factor", [1.0])[0]) + float(a.get("add_offset", [0.0])[0])


def _column_fields(t: np.ndarray, depth: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """TCHP, OHC, D26, SST and MLD for one row chunk; t is (level, row, col) in-situ temperature."""
    valid = np.isfinite(t)
    sst = np.where(valid[0], t[0], np.nan)
    warm = valid[0] & (t[0] >= REF_TEMP_C)
    below = valid & (t <= REF_TEMP_C)
    cross = np.argmax(below, axis=0)
    prev = np.clip(cross - 1, 0, len(depth) - 1)
    take = lambda a, i: np.take_along_axis(a, i[None], axis=0)[0]
    t1, t2 = take(t, prev), take(t, cross)
    good = below.any(axis=0) & warm & (cross > 0) & np.isfinite(t1) & np.isfinite(t2) & (t1 != t2)
    d26 = np.full(sst.shape, np.nan, dtype=np.float64)
    frac = np.where(good, (t1 - REF_TEMP_C) / np.where(good, t1 - t2, 1.0), 0.0)
    d26[good] = (depth[prev] + frac * (depth[cross] - depth[prev]))[good]

    lat2 = np.broadcast_to(lat[:, None], sst.shape)
    lon2 = np.broadcast_to(lon[None, :], sst.shape)
    p = gsw.p_from_z(-depth[:, None, None], lat2[None])
    sa = gsw.SA_from_SP(np.full(t.shape, S_CONST), p, lon2[None], lat2[None])
    tt = np.where(valid, t, REF_TEMP_C).astype(np.float64)
    heat = np.clip(tt - REF_TEMP_C, 0.0, None) * gsw.rho_t_exact(sa, tt, p) * gsw.cp_t_exact(sa, tt, p)
    edges = np.concatenate([[0.0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1]]])
    top, bot = edges[:-1][:, None, None], edges[1:][:, None, None]
    d3 = d26[None]
    eff = np.where(valid & np.isfinite(d3), np.clip(np.minimum(bot, d3) - top, 0.0, None), 0.0)
    ohc = np.where(np.isfinite(d26), np.sum(heat * eff, axis=0), np.nan)

    # temperature-criterion mixed layer: first depth below 10 m where T <= T(10 m) - 0.2
    k10 = int(np.searchsorted(depth, MLD_REF_M))
    tref = t[k10] - MLD_DT
    deeper = valid & (t <= tref[None]) & (depth[:, None, None] > MLD_REF_M)
    kk = np.argmax(deeper, axis=0)
    kp = np.clip(kk - 1, 0, len(depth) - 1)
    ta, tb = take(t, kp), take(t, kk)
    ok = deeper.any(axis=0) & np.isfinite(tref) & (ta != tb)
    deepest = depth[np.clip(valid.sum(axis=0) - 1, 0, len(depth) - 1)]
    mld = np.where(valid[k10], deepest, np.nan)   # never reached: capped at the column bottom (or 250 m)
    f = np.where(ok, (ta - tref) / np.where(ok, ta - tb, 1.0), 0.0)
    mld = np.where(ok, depth[kp] + f * (depth[kk] - depth[kp]), mld)
    return {"tchp_kj_per_cm2": ohc / 1e7, "ohc_j_per_m2": ohc, "d26_m": d26, "surface_temp_c": sst, "mld_t02_m": mld}


def _find_source(date: str) -> tuple[str, int]:
    for run, tau in _candidates(date):
        if requests.head(_url(run, tau, "ts3z"), timeout=120).status_code == 200 and \
                requests.head(_url(run, tau, "ssh"), timeout=120).status_code == 200:
            return run, tau
    raise FileNotFoundError(f"no GOFS analysis ts3z+ssh pair for {date}")


def build_date(date: str, out_dir: Path) -> dict:
    out_path = out_dir / f"gofs31a_fields_{date}.nc"
    if out_path.exists():
        return {"date": date, "status": "cached"}
    t0 = time.perf_counter()
    run, tau = _find_source(date)
    ts = NC3(_url(run, tau, "ts3z"))
    depth_all, lat_all, lon = ts.coord("depth"), ts.coord("lat"), ts.coord("lon")
    n_lev = int(np.searchsorted(depth_all, MAX_DEPTH_M, side="right"))
    depth = depth_all[:n_lev]
    y0, y1 = int(np.searchsorted(lat_all, LAT_BAND[0])), int(np.searchsorted(lat_all, LAT_BAND[1], side="right"))
    lat = lat_all[y0:y1]
    with ThreadPoolExecutor(THREADS) as ex:   # one range request per depth level
        temp = np.stack(list(ex.map(lambda k: ts.rows("water_temp", k, y0, y1), range(n_lev))))
    sh = NC3(_url(run, tau, "ssh"))
    ssh = sh.rows("surf_el", None, y0, y1)
    fields = {k: np.full((y1 - y0, lon.size), np.nan, np.float32)
              for k in ("tchp_kj_per_cm2", "ohc_j_per_m2", "d26_m", "surface_temp_c", "mld_t02_m")}
    for r0 in range(0, y1 - y0, ROW_CHUNK):
        r1 = min(y1 - y0, r0 + ROW_CHUNK)
        for k, v in _column_fields(temp[:, r0:r1], depth, lat[r0:r1], lon).items():
            fields[k][r0:r1] = v.astype(np.float32)

    lon180 = ((lon + 180.0) % 360.0) - 180.0
    valid_time = dt.datetime(2000, 1, 1) + dt.timedelta(hours=float(ts.coord("time")[0]))
    ds = xr.Dataset(
        {k: (("Y", "X"), v) for k, v in {**fields, "ssh_m": ssh}.items()}
        | {"Latitude": (("Y", "X"), np.broadcast_to(lat[:, None], ssh.shape).astype(np.float32)),
           "Longitude": (("Y", "X"), np.broadcast_to(lon180[None, :], ssh.shape).astype(np.float32))},
        coords={"Y": np.arange(lat.size, dtype=np.int32), "X": np.arange(lon.size, dtype=np.int32)},
        attrs={"source": f"{BASE} (GLBy0.08 expt_93.0, GOFS 3.1 analysis)", "run": run, "tau": tau,
               "valid_time_utc": valid_time.isoformat() + "Z", "source_date": date,
               "salinity": f"constant S = {S_CONST} (not downloaded)", "max_depth_m": MAX_DEPTH_M,
               "lat_band": str(LAT_BAND), "mld_definition": "T(10 m) - 0.2 C, de Boyer Montegut et al. 2004",
               "description": "SIDE EXPLORATION: GOFS 3.1 analysis daily TCHP/D26 fields "
                              "(OHC/exploration/build_gofs31_analysis_daily_fields.py)."})
    tmp = out_path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    tmp.rename(out_path)
    return {"date": date, "status": "computed", "valid": ds.attrs["valid_time_utc"], "run": run, "tau": tau,
            "finite_tchp": int(np.isfinite(fields["tchp_kj_per_cm2"]).sum()),
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def date_list(start: str, end: str, overlap_first: bool) -> list[str]:
    d0, d1 = (dt.datetime.strptime(x, "%Y%m%d") for x in (start, end))
    dates = [(d0 + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]
    if overlap_first:   # the RTOFS-overlap months first, so the transfer test can start early
        dates.sort(key=lambda x: (not (RTOFS_OVERLAP[0] <= x <= RTOFS_OVERLAP[1]), x))
    return dates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--workers", type=int, default=2, help="dates in flight (each uses THREADS connections)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dates = args.dates or date_list(args.start, args.end, overlap_first=True)
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_date, d, args.out_dir): d for d in dates}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                done += 1
                logger.info("%s", json.dumps(r))
            except Exception:
                failed += 1
                logger.exception("GOFS date %s failed", futs[fut])
            if (done + failed) % 20 == 0:
                logger.info("progress: %d done, %d failed, %d remaining", done, failed, len(dates) - done - failed)
    print(json.dumps({"requested": len(dates), "done": done, "failed": failed}))


if __name__ == "__main__":
    main()
