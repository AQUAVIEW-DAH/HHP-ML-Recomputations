"""How much does treating RTOFS potential temperature as in-situ temperature matter?

HYCOM (and so RTOFS) stores potential temperature: the RTOFS 3D z-level files
label it `sea_water_potential_temperature`. The daily TCHP/D26 builder and the
profile-physics builder pass it to TEOS-10 functions that expect in-situ
temperature (gsw.rho_t_exact, gsw.cp_t_exact, gsw.CT_from_t), and compare it
with 26 C as if it were in-situ. Argo TEMP is in-situ, so the Argo targets are
computed correctly.

This measures the size of the mix-up on real RTOFS columns: every quantity is
computed twice, once as the pipeline does (potential temperature used as
in-situ) and once correctly (theta -> CT -> in-situ t via gsw.CT_from_pt and
gsw.t_from_CT). The 3D archives used for the global fields are deleted after
processing, so the regional US-East z-level file (same model run) is used.

Output: OHC/output/potential_temperature_check_20260924/result.json
"""
from __future__ import annotations

import json
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np

SRC = Path("/data/suramya/rtofs_time_matched/rtofs.20241007/rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/potential_temperature_check_20260924")
REF = 26.0
N_COLUMNS = 6000


def d26_tchp(temp: np.ndarray, sa: np.ndarray, z: np.ndarray, lat: float) -> tuple[float, float]:
    """D26 by linear interpolation; TCHP = integral of rho cp (T - 26) from the surface to D26, kJ/cm^2."""
    if not np.isfinite(temp[0]) or temp[0] < REF:
        return np.nan, np.nan
    below = np.where(np.isfinite(temp) & (temp <= REF))[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan, np.nan
    k = below[0]
    f = (temp[k - 1] - REF) / (temp[k - 1] - temp[k])
    d26 = z[k - 1] + f * (z[k] - z[k - 1])
    zz, tt = np.r_[z[:k], d26], np.r_[temp[:k], REF]
    ss = np.r_[sa[:k], sa[k - 1] + f * (sa[k] - sa[k - 1])]
    pp = gsw.p_from_z(-zz, lat)
    heat = gsw.rho_t_exact(ss, tt, pp) * gsw.cp_t_exact(ss, tt, pp) * (tt - REF)
    return float(d26), float(np.trapezoid(heat, zz) / 1e7)


def steric_1000_ref2000(sa: np.ndarray, ct: np.ndarray, p: np.ndarray) -> float:
    dyn = lambda p_ref: gsw.geo_strf_dyn_height(sa, ct, p, p_ref=p_ref)[0]
    return float((dyn(2000.0) - dyn(1000.0)) / 9.81)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = nc.Dataset(SRC)
    z_all = np.asarray(d["Depth"][:], float)
    lat, lon = np.asarray(d["Latitude"][:], float), np.asarray(d["Longitude"][:], float)
    rng = np.random.default_rng(0)
    iy, ix = rng.integers(0, lat.shape[0], N_COLUMNS), rng.integers(0, lat.shape[1], N_COLUMNS)
    theta = np.ma.filled(d["temperature"][0][:, iy, ix].astype(float), np.nan)
    salt = np.ma.filled(d["salinity"][0][:, iy, ix].astype(float), np.nan)
    la = lat[iy, ix]
    lo = np.where(lon[iy, ix] > 180, lon[iy, ix] - 360, lon[iy, ix])

    warm, gap, steric = [], {60: [], 100: [], 200: []}, []
    for j in range(N_COLUMNS):
        ok = np.isfinite(theta[:, j]) & np.isfinite(salt[:, j])
        if ok.sum() < 5:
            continue
        z, th, sp = z_all[ok], theta[ok, j], salt[ok, j]
        p = gsw.p_from_z(-z, la[j])
        sa = gsw.SA_from_SP(sp, p, lo[j], la[j])
        ct = gsw.CT_from_pt(sa, th)
        t = gsw.t_from_CT(sa, ct, p)
        for depth in gap:
            if z[-1] >= depth:
                gap[depth].append(float(np.interp(depth, z, t - th)))
        as_built, correct = d26_tchp(th, sa, z, la[j]), d26_tchp(t, sa, z, la[j])
        if np.all(np.isfinite(as_built + correct)):
            warm.append(as_built + correct)
        if p[-1] >= 2000:
            steric.append((steric_1000_ref2000(sa, gsw.CT_from_t(sa, th, p), p), steric_1000_ref2000(sa, ct, p)))

    w, s = np.array(warm), np.array(steric)
    dd, dh, ds = w[:, 0] - w[:, 2], w[:, 1] - w[:, 3], s[:, 0] - s[:, 1]
    res = {
        "source": str(SRC), "columns_sampled": N_COLUMNS,
        "insitu_minus_potential_temperature_c": {f"{k}m": float(np.mean(v)) for k, v in gap.items()},
        "warm_columns": int(len(w)),
        "d26_as_built_minus_correct_m": {"mean": float(dd.mean()), "p99_abs": float(np.percentile(np.abs(dd), 99)),
                                         "mean_d26_m": float(w[:, 2].mean())},
        "tchp_as_built_minus_correct_kj_cm2": {"mean": float(dh.mean()), "p99_abs": float(np.percentile(np.abs(dh), 99)),
                                               "mean_tchp": float(w[:, 3].mean()),
                                               "mean_pct": float(100 * dh.mean() / w[:, 3].mean())},
        "deep_columns": int(len(s)),
        "steric_1000_ref2000_as_built_minus_correct_m": {"mean": float(ds.mean()), "std": float(ds.std()),
                                                         "p99_abs": float(np.percentile(np.abs(ds), 99))},
    }
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
