"""TEOS-10-backed OHC/TCHP calculations for Argo and RTOFS profiles.

This module is intentionally self-contained so we have a stable, documented
home for the ocean-heat-content calculation outside the app glue code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gsw
import numpy as np

REF_TEMP_C = 26.0
J_PER_M2_TO_KJ_PER_CM2 = 1.0 / 1.0e7


@dataclass(frozen=True)
class OHCResult:
    d26_m: float | None
    ohc_j_per_m2: float | None
    tchp_kj_per_cm2: float | None
    surface_temp_c: float | None
    max_depth_m: float
    levels_above_d26: int
    method: str = "teos10"


def _clean_profile(
    vertical: np.ndarray,
    temp_c: np.ndarray,
    salinity_psu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical = np.asarray(vertical, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    salinity_psu = np.asarray(salinity_psu, dtype=float)
    mask = np.isfinite(vertical) & np.isfinite(temp_c) & np.isfinite(salinity_psu)
    vertical = vertical[mask]
    temp_c = temp_c[mask]
    salinity_psu = salinity_psu[mask]
    order = np.argsort(vertical)
    return vertical[order], temp_c[order], salinity_psu[order]


def compute_d26(depth_m: np.ndarray, temp_c: np.ndarray, ref_temp_c: float = REF_TEMP_C) -> float | None:
    """Return the depth where the profile crosses downward through 26 C."""
    depth_m = np.asarray(depth_m, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    mask = np.isfinite(depth_m) & np.isfinite(temp_c)
    depth_m = depth_m[mask]
    temp_c = temp_c[mask]
    if depth_m.size < 2:
        return None
    order = np.argsort(depth_m)
    depth_m = depth_m[order]
    temp_c = temp_c[order]

    if temp_c[0] < ref_temp_c:
        return None

    for idx in range(1, depth_m.size):
        if temp_c[idx] <= ref_temp_c:
            z1, z2 = depth_m[idx - 1], depth_m[idx]
            t1, t2 = temp_c[idx - 1], temp_c[idx]
            if t1 == t2:
                return float(z2)
            frac = (t1 - ref_temp_c) / (t1 - t2)
            return float(z1 + frac * (z2 - z1))
    return None


def compute_ohc_teos10(
    *,
    vertical: np.ndarray,
    temp_c: np.ndarray,
    salinity_psu: np.ndarray,
    lat: float,
    lon: float,
    vertical_axis: Literal["pressure", "depth"] = "pressure",
    ref_temp_c: float = REF_TEMP_C,
) -> OHCResult:
    """Compute OHC/TCHP above the 26 C isotherm using TEOS-10/GSW.

    Parameters
    ----------
    vertical
        Pressure in dbar for Argo-style profiles, or depth in meters for
        model-style profiles.
    temp_c
        In-situ temperature profile in degrees Celsius.
    salinity_psu
        Practical salinity profile.
    lat, lon
        Geographic coordinates used by TEOS-10 conversions.
    vertical_axis
        `"pressure"` for Argo-style input or `"depth"` for RTOFS-style input.
    ref_temp_c
        The reference isotherm, usually 26 C for TCHP.
    """
    vertical, temp_c, salinity_psu = _clean_profile(vertical, temp_c, salinity_psu)
    if vertical.size < 2:
        return OHCResult(None, None, None, None, float(vertical.max()) if vertical.size else 0.0, 0)

    if vertical_axis == "pressure":
        pressure_dbar = vertical.astype(float)
        depth_m = -gsw.z_from_p(pressure_dbar, lat)
    elif vertical_axis == "depth":
        depth_m = vertical.astype(float)
        pressure_dbar = gsw.p_from_z(-depth_m, lat)
    else:
        raise ValueError(f"Unsupported vertical_axis '{vertical_axis}'")

    d26_m = compute_d26(depth_m, temp_c, ref_temp_c=ref_temp_c)
    surface_temp_c = float(temp_c[0])
    max_depth_m = float(depth_m[-1])
    if d26_m is None:
        return OHCResult(None, None, None, surface_temp_c, max_depth_m, 0)

    mask = depth_m < d26_m
    z_nodes = np.concatenate([depth_m[mask], [d26_m]])
    p_nodes = np.concatenate([pressure_dbar[mask], [np.interp(d26_m, depth_m, pressure_dbar)]])
    sp_nodes = np.concatenate([salinity_psu[mask], [np.interp(d26_m, depth_m, salinity_psu)]])
    t_nodes = np.concatenate([temp_c[mask], [ref_temp_c]])

    absolute_salinity = gsw.SA_from_SP(sp_nodes, p_nodes, lon, lat)
    rho = gsw.rho_t_exact(absolute_salinity, t_nodes, p_nodes)
    cp = gsw.cp_t_exact(absolute_salinity, t_nodes, p_nodes)

    heat_excess_j_per_m3 = np.clip(t_nodes - ref_temp_c, 0.0, None) * rho * cp
    ohc_j_per_m2 = float(np.trapezoid(heat_excess_j_per_m3, z_nodes))
    tchp_kj_per_cm2 = ohc_j_per_m2 * J_PER_M2_TO_KJ_PER_CM2

    return OHCResult(
        d26_m=float(d26_m),
        ohc_j_per_m2=ohc_j_per_m2,
        tchp_kj_per_cm2=float(tchp_kj_per_cm2),
        surface_temp_c=surface_temp_c,
        max_depth_m=max_depth_m,
        levels_above_d26=int(mask.sum()),
    )

