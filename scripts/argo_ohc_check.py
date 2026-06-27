"""Standalone Argo OHC sanity check.

Picks one Argo profile from the Milton-window matchup set, computes Ocean
Heat Content (TCHP) via TEOS-10, and renders a temperature-vs-depth plot
with the >26 °C integration region shaded. The bottom of the figure carries
the full TEOS-10 equation that produced the printed OHC.

Default target is the 2024-10-04 warm Loop Current cast at (25.47 °N,
86.24 °W) — a known high-OHC pre-Milton point. Override via CLI flags.

References used by the implementation:
- IOC, SCOR, IAPSO (2010). The international thermodynamic equation of
  seawater - 2010 (TEOS-10). UNESCO IOC manual 56.
- McDougall, T. J. & Barker, P. M. (2011). Getting started with TEOS-10
  and the Gibbs Seawater (GSW) Oceanographic Toolbox.
- Leipper, D. F. & Volgenau, D. (1972). Hurricane Heat Potential of the
  Gulf of Mexico. J. Phys. Oceanogr., 2(3), 218-224.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhp_core import REF_TEMP_C, compute_tchp
from ml.paths import DATASETS_DIR, REPO_ROOT

PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
PROFILES_PARQUET = DATASETS_DIR / "milton_pair_profiles.parquet"
PROFILES_PKL = DATASETS_DIR / "milton_pair_profiles.pkl"
DEFAULT_OUT = REPO_ROOT / "docs" / "argo_ohc_check.png"

ARGO_COLOR = "#0ea5e9"
REF_COLOR = "#dc2626"
SHADE_COLOR = "#0ea5e9"
MAX_DEPTH_PLOT_DEFAULT = 350.0


def _load_profile_arrays(cast_id: str) -> dict:
    if PROFILES_PARQUET.exists():
        df = pd.read_parquet(PROFILES_PARQUET)
    else:
        df = pd.read_pickle(PROFILES_PKL)
    rows = df[df["cast_id"] == cast_id]
    if rows.empty:
        raise SystemExit(f"No stored profile arrays for cast {cast_id}")
    r = rows.iloc[0]
    return {
        "pressure_dbar": np.asarray(r["obs_pressure_dbar"], dtype=float),
        "temperature_c": np.asarray(r["obs_temperature_c"], dtype=float),
        "salinity_psu": np.asarray(r["obs_salinity_psu"], dtype=float),
    }


def _pick_cast(target_lat: float, target_lon: float, target_date: str) -> dict:
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["obs_date"] = pairs["obs_date"].astype(str)
    candidates = pairs[pairs["obs_date"] == target_date].copy()
    if candidates.empty:
        raise SystemExit(f"No pairs on {target_date}; available: {sorted(pairs['obs_date'].unique())}")
    candidates["d2"] = (candidates["lat"] - target_lat) ** 2 + (candidates["lon"] - target_lon) ** 2
    return candidates.sort_values("d2").iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=25.47)
    parser.add_argument("--lon", type=float, default=-86.24)
    parser.add_argument("--date", default="20241004")
    parser.add_argument("--max-depth-plot", type=float, default=MAX_DEPTH_PLOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cast = _pick_cast(args.lat, args.lon, args.date)
    cast_id = cast["cast_id"]
    arrays = _load_profile_arrays(cast_id)
    lat = float(cast["lat"])
    lon = float(cast["lon"])

    res = compute_tchp(
        arrays["pressure_dbar"],
        arrays["temperature_c"],
        salinity_psu=arrays["salinity_psu"],
        lat=lat,
        lon=lon,
        vertical_axis="pressure",
    )

    print("=== Argo OHC check ===")
    print(f"Cast:        {cast_id}")
    print(f"Platform:    {cast['platform']}")
    obs_date = str(cast["obs_date"])
    print(f"Date:        {obs_date[:4]}-{obs_date[4:6]}-{obs_date[6:]}")
    print(f"Location:    ({lat:.3f} °N, {abs(lon):.3f} °W)")
    print(f"Method:      {res.method} (TEOS-10 via Gibbs SeaWater)")
    print()
    print(f"Surface T:   {res.surface_temp_c:.2f} °C")
    print(f"D26:         {res.d26_m:.1f} m")
    print(f"OHC (TCHP):  {res.tchp_kj_per_cm2:.1f} kJ/cm²")
    print(f"             {res.integral_j_per_m2:.3e} J/m²")
    print(f"Levels above D26: {res.levels_above_d26}  (max profile depth: {res.max_depth_m:.0f} m)")
    print()
    ri_threshold = 50.0
    status = "ABOVE" if res.tchp_kj_per_cm2 > ri_threshold else "BELOW"
    print(f"Shay et al. 2000 RI threshold: {ri_threshold:.0f} kJ/cm²  →  this profile is {status} threshold")

    pressure = arrays["pressure_dbar"]
    depth_m = -gsw.z_from_p(pressure, lat)
    temp = arrays["temperature_c"]
    plot_mask = depth_m <= args.max_depth_plot
    z_plot = depth_m[plot_mask]
    t_plot = temp[plot_mask]

    d26 = res.d26_m
    shade_mask = z_plot < d26
    z_shade = np.concatenate([z_plot[shade_mask], [d26]])
    t_shade = np.concatenate([t_plot[shade_mask], [REF_TEMP_C]])

    # ---------------- figure layout ----------------
    # Two rows: tall main plot on top, equation/provenance block below.
    fig = plt.figure(figsize=(8.6, 11.0), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[3.6, 1.0],
        left=0.10, right=0.96, top=0.93, bottom=0.04,
        hspace=0.20,
    )
    ax = fig.add_subplot(gs[0])
    ax_eq = fig.add_subplot(gs[1])
    ax_eq.axis("off")

    # ---------------- title + subtitle (clean, no overlap) ----------------
    fig.suptitle(
        f"Argo float {cast['platform']}  —  OHC sanity check",
        fontsize=15, fontweight="bold", y=0.985,
    )
    subtitle = (
        f"{obs_date[:4]}-{obs_date[4:6]}-{obs_date[6:]}   ·   "
        f"({lat:.2f} °N, {abs(lon):.2f} °W)   ·   "
        f"TEOS-10  (variable ρ, Cp from GSW)"
    )
    fig.text(0.5, 0.955, subtitle, ha="center", color="#475569", fontsize=10.5)

    # ---------------- main panel ----------------
    ax.fill_betweenx(
        z_shade, REF_TEMP_C, t_shade,
        where=(t_shade >= REF_TEMP_C),
        color=SHADE_COLOR, alpha=0.30,
        label="Heat excess above 26 °C  (×ρ·Cp = OHC)",
    )
    ax.plot(t_plot, z_plot, color=ARGO_COLOR, lw=2.6, label="Argo temperature profile")
    ax.axvline(REF_TEMP_C, color=REF_COLOR, lw=1.4, ls="--", label="26 °C reference")
    ax.axhline(d26, color=ARGO_COLOR, lw=0.8, ls=":", alpha=0.65)
    ax.annotate(
        f"D26 = {d26:.0f} m",
        xy=(REF_TEMP_C, d26),
        xytext=(REF_TEMP_C + 0.4, d26 + 6),
        fontsize=11, color=ARGO_COLOR, fontweight="bold",
    )

    ax.invert_yaxis()
    ax.set_xlim(20, 32)
    ax.set_ylim(args.max_depth_plot, 0)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)

    ax.text(
        0.04, 0.04,
        f"OHC = {res.tchp_kj_per_cm2:.0f} kJ/cm²",
        transform=ax.transAxes, fontsize=22, fontweight="bold",
        color=ARGO_COLOR,
        bbox=dict(facecolor="white", edgecolor=ARGO_COLOR, alpha=0.95, pad=8),
    )
    provenance_inline = (
        f"D26       = {d26:.0f} m\n"
        f"Surface T = {res.surface_temp_c:.2f} °C\n"
        f"Levels    = {res.levels_above_d26} above D26\n"
        f"Max depth = {res.max_depth_m:.0f} m"
    )
    ax.text(
        0.96, 0.04, provenance_inline,
        transform=ax.transAxes, fontsize=9.5, color="#0f172a", ha="right",
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.92, pad=6),
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # ---------------- equation panel ----------------
    # Compute the intermediate quantities the panel displays so the arithmetic
    # actually matches what the integral evaluated to.
    p_full = arrays["pressure_dbar"]
    s_full = arrays["salinity_psu"]
    t_full = arrays["temperature_c"]
    z_full = -gsw.z_from_p(p_full, lat)
    above = z_full < d26
    z_nodes = np.concatenate([z_full[above], [d26]])
    p_nodes = np.concatenate([p_full[above], [np.interp(d26, z_full, p_full)]])
    s_nodes = np.concatenate([s_full[above], [np.interp(d26, z_full, s_full)]])
    t_nodes = np.concatenate([t_full[above], [REF_TEMP_C]])
    SA_nodes = gsw.SA_from_SP(s_nodes, p_nodes, lon, lat)
    rho_nodes = gsw.rho_t_exact(SA_nodes, t_nodes, p_nodes)
    cp_nodes = gsw.cp_t_exact(SA_nodes, t_nodes, p_nodes)
    excess_nodes = np.clip(t_nodes - REF_TEMP_C, 0.0, None)
    integral_km = float(np.trapezoid(excess_nodes, z_nodes))            # K·m
    rho_avg = float(np.mean(rho_nodes))                                  # kg/m³
    cp_avg = float(np.mean(cp_nodes))                                    # J/(kg·K)
    j_per_m2 = res.integral_j_per_m2
    kj_per_cm2 = res.tchp_kj_per_cm2

    eq_main = (
        r"$\mathrm{OHC} \,=\, \int_{0}^{D_{26}}\, "
        r"\rho(z)\; C_p(z)\; [\, T(z) - 26\,^{\circ}\mathrm{C}\, ]\; dz$"
    )
    ax_eq.text(
        0.5, 0.94, eq_main,
        transform=ax_eq.transAxes, ha="center", va="top",
        fontsize=14.5, color="#0f172a",
    )

    rho_min = float(np.min(rho_nodes)); rho_max = float(np.max(rho_nodes))
    cp_min = float(np.min(cp_nodes)); cp_max = float(np.max(cp_nodes))
    where_line = (
        rf"$\rho(z)$ and $C_p(z)$ are evaluated at each depth from TEOS-10  "
        rf"(in this profile $\rho{{=}}{rho_min:,.0f}{{-}}{rho_max:,.0f}$, "
        rf"$C_p{{=}}{cp_min:,.0f}{{-}}{cp_max:,.0f}$).  Trapezoidal integration gives:"
    )
    ax_eq.text(
        0.5, 0.74, where_line,
        transform=ax_eq.transAxes, ha="center", va="top",
        fontsize=10.5, color="#0f172a",
    )

    arith_lines = [
        rf"$\int_{{0}}^{{D_{{26}}}} [T(z)-26\,^{{\circ}}\mathrm{{C}}]\, dz \;\approx\; {integral_km:.0f}\,\mathrm{{K{{\cdot}}m}}$",
        rf"$\int_{{0}}^{{D_{{26}}}} \rho(z)\,C_p(z)\,[T(z)-26\,^{{\circ}}\mathrm{{C}}]\, dz "
        rf"\;\approx\; {j_per_m2:.2e}\,\mathrm{{J/m^{{2}}}}$",
        rf"        $\approx\; \mathbf{{{kj_per_cm2:.0f}\,\mathrm{{kJ/cm^{{2}}}}}}$"
        rf"   (depth-mean $\rho \approx {rho_avg:,.0f}$, $C_p \approx {cp_avg:,.0f}$ — for reference only)",
    ]
    y0 = 0.56
    dy = 0.135
    for i, line in enumerate(arith_lines):
        ax_eq.text(
            0.5, y0 - i * dy, line,
            transform=ax_eq.transAxes, ha="center", va="top",
            fontsize=12.0, color="#0f172a",
        )

    citation = (
        "TEOS-10 thermodynamics: $\\rho =$ gsw.rho_t_exact$(S_A, t, p)$,  "
        "$C_p =$ gsw.cp_t_exact$(S_A, t, p)$.   "
        "OHC definition: Leipper & Volgenau (1972)."
    )
    ax_eq.text(
        0.5, 0.06, citation,
        transform=ax_eq.transAxes, ha="center", va="bottom",
        fontsize=8.8, color="#475569", style="italic",
    )

    # subtle border around equation panel for visual separation
    ax_eq.add_patch(
        plt.Rectangle(
            (0.005, 0.005), 0.99, 0.99,
            transform=ax_eq.transAxes,
            fill=False, edgecolor="#cbd5e1", linewidth=0.8,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved to: {args.output}")


if __name__ == "__main__":
    main()
