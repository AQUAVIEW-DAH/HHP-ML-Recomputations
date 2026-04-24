"""Core physics for Tropical Cyclone Heat Potential (TCHP) computation.

TCHP = ∫[0 → D26] ρ(z) · Cp(z) · (T(z) − 26°C) dz

The preferred implementation uses TEOS-10 via Gibbs SeaWater (GSW):

- Practical Salinity -> Absolute Salinity
- in-situ temperature with pressure/depth context
- depth-varying density and heat capacity

Output convention: kJ/cm² (divide J/m² by 1e7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gsw
import numpy as np

REF_TEMP_C = 26.0
SEAWATER_CP_J_PER_KG_K = 4180.0
SEAWATER_DENSITY_KG_PER_M3 = 1026.0
J_PER_M2_TO_KJ_PER_CM2 = 1.0 / 1.0e7


@dataclass(frozen=True)
class TCHPResult:
    d26_m: Optional[float]
    tchp_kj_per_cm2: Optional[float]
    integral_j_per_m2: Optional[float]
    surface_temp_c: Optional[float]
    max_depth_m: float
    levels_above_d26: int
    method: str = "constant_density"


def _clean_profile(depth: np.ndarray, temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=float)
    temp = np.asarray(temp, dtype=float)
    mask = np.isfinite(depth) & np.isfinite(temp)
    depth = depth[mask]
    temp = temp[mask]
    order = np.argsort(depth)
    return depth[order], temp[order]


def _clean_teos_profile(
    vertical: np.ndarray,
    temp: np.ndarray,
    salinity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical = np.asarray(vertical, dtype=float)
    temp = np.asarray(temp, dtype=float)
    salinity = np.asarray(salinity, dtype=float)
    mask = np.isfinite(vertical) & np.isfinite(temp) & np.isfinite(salinity)
    vertical = vertical[mask]
    temp = temp[mask]
    salinity = salinity[mask]
    order = np.argsort(vertical)
    return vertical[order], temp[order], salinity[order]


def compute_d26(depth: np.ndarray, temp: np.ndarray, ref_temp_c: float = REF_TEMP_C) -> Optional[float]:
    """Depth at which the profile crosses 26 °C, linearly interpolated.

    Returns None if the profile never reaches 26 °C (too cold throughout) or
    never crosses down past 26 °C (surface already below 26 °C).
    """
    depth, temp = _clean_profile(depth, temp)
    if depth.size < 2:
        return None

    # Surface must be warmer than ref; otherwise no 26 °C crossing going down.
    if temp[0] < ref_temp_c:
        return None

    for idx in range(1, depth.size):
        if temp[idx] <= ref_temp_c:
            z1, z2 = depth[idx - 1], depth[idx]
            t1, t2 = temp[idx - 1], temp[idx]
            if t1 == t2:
                return float(z2)
            frac = (t1 - ref_temp_c) / (t1 - t2)
            return float(z1 + frac * (z2 - z1))

    # Profile stays above 26 °C throughout — TCHP integrates to max_depth which
    # is an underestimate; caller should flag this.
    return None


def compute_tchp(
    depth: np.ndarray,
    temp: np.ndarray,
    ref_temp_c: float = REF_TEMP_C,
    density_kg_m3: float = SEAWATER_DENSITY_KG_PER_M3,
    cp_j_per_kg_k: float = SEAWATER_CP_J_PER_KG_K,
    salinity_psu: np.ndarray | None = None,
    lat: float | None = None,
    lon: float | None = None,
    vertical_axis: str = "depth",
) -> TCHPResult:
    """Integrate heat content above the 26 °C isotherm.

    Uses the trapezoidal rule with the profile's native depth levels plus the
    interpolated D26 endpoint. Returns TCHPResult with provenance fields;
    tchp_kj_per_cm2 is None when D26 cannot be resolved.
    """
    if salinity_psu is not None and lat is not None and lon is not None:
        return compute_tchp_teos(
            vertical=depth,
            temp=temp,
            salinity_psu=salinity_psu,
            lat=lat,
            lon=lon,
            ref_temp_c=ref_temp_c,
            vertical_axis=vertical_axis,
        )

    depth, temp = _clean_profile(depth, temp)
    if depth.size < 2:
        return TCHPResult(
            d26_m=None,
            tchp_kj_per_cm2=None,
            integral_j_per_m2=None,
            surface_temp_c=None,
            max_depth_m=float(depth.max()) if depth.size else 0.0,
            levels_above_d26=0,
            method="constant_density",
        )

    d26 = compute_d26(depth, temp, ref_temp_c)
    surface_temp = float(temp[0])
    max_depth = float(depth[-1])

    if d26 is None:
        return TCHPResult(
            d26_m=None,
            tchp_kj_per_cm2=None,
            integral_j_per_m2=None,
            surface_temp_c=surface_temp,
            max_depth_m=max_depth,
            levels_above_d26=0,
            method="constant_density",
        )

    # Build the integration grid: native levels shallower than d26, then d26.
    mask = depth < d26
    z_nodes = np.concatenate([depth[mask], [d26]])
    t_nodes = np.concatenate([temp[mask], [ref_temp_c]])
    excess = np.clip(t_nodes - ref_temp_c, 0.0, None)

    integral_k_m = np.trapezoid(excess, z_nodes)  # units: °C · m (= K · m)
    integral_j_per_m2 = density_kg_m3 * cp_j_per_kg_k * float(integral_k_m)
    tchp_kj_per_cm2 = integral_j_per_m2 * J_PER_M2_TO_KJ_PER_CM2

    return TCHPResult(
        d26_m=float(d26),
        tchp_kj_per_cm2=float(tchp_kj_per_cm2),
        integral_j_per_m2=float(integral_j_per_m2),
        surface_temp_c=surface_temp,
        max_depth_m=max_depth,
        levels_above_d26=int(mask.sum()),
        method="constant_density",
    )


def compute_tchp_teos(
    *,
    vertical: np.ndarray,
    temp: np.ndarray,
    salinity_psu: np.ndarray,
    lat: float,
    lon: float,
    ref_temp_c: float = REF_TEMP_C,
    vertical_axis: str = "depth",
) -> TCHPResult:
    """Compute TCHP using TEOS-10/GSW with depth-varying rho and Cp."""
    vertical, temp, salinity_psu = _clean_teos_profile(vertical, temp, salinity_psu)
    if vertical.size < 2:
        return TCHPResult(
            d26_m=None,
            tchp_kj_per_cm2=None,
            integral_j_per_m2=None,
            surface_temp_c=None,
            max_depth_m=float(vertical.max()) if vertical.size else 0.0,
            levels_above_d26=0,
            method="teos10",
        )

    if vertical_axis == "pressure":
        pressure = vertical.astype(float)
        depth_m = -gsw.z_from_p(pressure, lat)
    elif vertical_axis == "depth":
        depth_m = vertical.astype(float)
        pressure = gsw.p_from_z(-depth_m, lat)
    else:
        raise ValueError(f"Unsupported vertical axis '{vertical_axis}'")

    d26 = compute_d26(depth_m, temp, ref_temp_c)
    surface_temp = float(temp[0])
    max_depth = float(depth_m[-1])
    if d26 is None:
        return TCHPResult(
            d26_m=None,
            tchp_kj_per_cm2=None,
            integral_j_per_m2=None,
            surface_temp_c=surface_temp,
            max_depth_m=max_depth,
            levels_above_d26=0,
            method="teos10",
        )

    mask = depth_m < d26
    z_nodes = np.concatenate([depth_m[mask], [d26]])
    p_nodes = np.concatenate([pressure[mask], [np.interp(d26, depth_m, pressure)]])
    s_nodes = np.concatenate([salinity_psu[mask], [np.interp(d26, depth_m, salinity_psu)]])
    t_nodes = np.concatenate([temp[mask], [ref_temp_c]])

    SA = gsw.SA_from_SP(s_nodes, p_nodes, lon, lat)
    rho = gsw.rho_t_exact(SA, t_nodes, p_nodes)
    cp = gsw.cp_t_exact(SA, t_nodes, p_nodes)
    heat_density = np.clip(t_nodes - ref_temp_c, 0.0, None) * rho * cp
    integral_j_per_m2 = float(np.trapezoid(heat_density, z_nodes))
    tchp_kj_per_cm2 = integral_j_per_m2 * J_PER_M2_TO_KJ_PER_CM2

    return TCHPResult(
        d26_m=float(d26),
        tchp_kj_per_cm2=float(tchp_kj_per_cm2),
        integral_j_per_m2=integral_j_per_m2,
        surface_temp_c=surface_temp,
        max_depth_m=max_depth,
        levels_above_d26=int(mask.sum()),
        method="teos10",
    )
