"""Standalone Argo salinity / density companion check.

Companion to ``scripts/argo_ohc_check.py``. For the same Argo cast, plots
the temperature profile alongside salinity, in-situ density, and potential
density (referenced to 0 dbar), all as a function of depth.

Motivation
----------
Argo casts in the Loop Current and adjacent Gulf of Mexico waters often show
warm-water-below-cold-water inversions in the temperature profile. Salinity
inversions of the opposite sign (saltier water deeper) keep density
monotonically increasing with depth, so the column remains statically stable.
This script makes that story explicit on a per-cast basis and prints the
ΔT / ΔS / Δσ_θ across each detected inversion band.

CLI defaults match ``argo_ohc_check.py`` — same ``--lat --lon --date`` pulls
the same cast and renders the salinity/density companion to that figure.
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

from ml.paths import DATASETS_DIR, REPO_ROOT

PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
PROFILES_PARQUET = DATASETS_DIR / "milton_pair_profiles.parquet"
PROFILES_PKL = DATASETS_DIR / "milton_pair_profiles.pkl"
DEFAULT_OUT = REPO_ROOT / "docs" / "argo_ts_density_check.png"

TEMP_COLOR = "#0ea5e9"
SAL_COLOR = "#0891b2"
RHO_COLOR = "#7c3aed"
SIGMA_COLOR = "#16a34a"
INVERSION_COLOR = "#fb7185"

MAX_DEPTH_PLOT_DEFAULT = 350.0
INVERSION_MIN_DT = 0.05  # °C; minimum (deeper - shallower) to flag as a T-inversion band
NSQ_NOISE_FLOOR = -1.0e-7  # s^-2; N² below this is treated as a real static instability
                           # (above the floor = within CTD noise, column is effectively stable)


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
        raise SystemExit(
            f"No pairs on {target_date}; available: {sorted(pairs['obs_date'].unique())}"
        )
    candidates["d2"] = (candidates["lat"] - target_lat) ** 2 + (candidates["lon"] - target_lon) ** 2
    return candidates.sort_values("d2").iloc[0].to_dict()


def _find_inversion_bands(
    z: np.ndarray, t: np.ndarray, min_dt: float
) -> list[tuple[float, float, float]]:
    """Return [(z_top, z_bot, ΔT)] for contiguous bands where T rises with depth.

    A band starts at the first sample of a warming run and ends where warming
    stops; bands shallower than ``min_dt`` in cumulative ΔT are dropped.
    """
    bands: list[tuple[float, float, float]] = []
    in_band = False
    z_start = 0.0
    t_start = 0.0
    for i in range(1, len(z)):
        warming = t[i] - t[i - 1] > 0.0
        if warming and not in_band:
            in_band = True
            z_start = z[i - 1]
            t_start = t[i - 1]
        elif not warming and in_band:
            in_band = False
            dt = t[i - 1] - t_start
            if dt >= min_dt:
                bands.append((z_start, z[i - 1], dt))
    if in_band:
        dt = t[-1] - t_start
        if dt >= min_dt:
            bands.append((z_start, z[-1], dt))
    return bands


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

    p = arrays["pressure_dbar"]
    t = arrays["temperature_c"]
    sp = arrays["salinity_psu"]
    finite = np.isfinite(p) & np.isfinite(t) & np.isfinite(sp)
    p, t, sp = p[finite], t[finite], sp[finite]
    order = np.argsort(p)
    p, t, sp = p[order], t[order], sp[order]
    z = -gsw.z_from_p(p, lat)

    SA = gsw.SA_from_SP(sp, p, lon, lat)
    CT = gsw.CT_from_t(SA, t, p)
    rho_in_situ = gsw.rho(SA, CT, p)
    rho_pot = gsw.rho(SA, CT, np.zeros_like(p))  # potential density referenced to 0 dbar

    plot_mask = z <= args.max_depth_plot
    z_plot = z[plot_mask]
    t_plot = t[plot_mask]
    sp_plot = sp[plot_mask]
    rho_in_situ_plot = rho_in_situ[plot_mask]
    rho_pot_plot = rho_pot[plot_mask]
    sigma_pot_plot = rho_pot_plot - 1000.0

    bands = _find_inversion_bands(z_plot, t_plot, INVERSION_MIN_DT)

    # Per-band stability check: Δσ_θ across each T-inversion band must be > 0
    # for the column to be statically stable across that band despite warm-on-cold.
    band_drho: list[float] = []
    for (z_top, z_bot, _dt) in bands:
        sig_top = float(np.interp(z_top, z_plot, sigma_pot_plot))
        sig_bot = float(np.interp(z_bot, z_plot, sigma_pot_plot))
        band_drho.append(sig_bot - sig_top)
    bands_unstable = sum(1 for d in band_drho if d < 0.0)

    # Fine-scale N² is reported as a secondary diagnostic only.
    n2, _p_mid = gsw.Nsquared(SA, CT, p, lat)
    n2_min = float(np.nanmin(n2)) if n2.size else np.nan
    n2_unstable_fine = int((n2 < NSQ_NOISE_FLOOR).sum())

    obs_date = str(cast["obs_date"])
    print("=== Argo S / ρ / σ_θ companion check ===")
    print(f"Cast:          {cast_id}")
    print(f"Platform:      {cast['platform']}")
    print(f"Date:          {obs_date[:4]}-{obs_date[4:6]}-{obs_date[6:]}")
    print(f"Location:      ({lat:.3f} °N, {abs(lon):.3f} °W)")
    print(f"S range:       {np.nanmin(sp_plot):.3f} – {np.nanmax(sp_plot):.3f} PSU")
    print(f"ρ in-situ:     {np.nanmin(rho_in_situ_plot):.2f} – {np.nanmax(rho_in_situ_plot):.2f} kg/m³")
    print(f"σ_θ:           {np.nanmin(sigma_pot_plot):.3f} – {np.nanmax(sigma_pot_plot):.3f} kg/m³")
    print(f"T inversions ≥ {INVERSION_MIN_DT:.2f} °C: {len(bands)}")
    for (z_top, z_bot, dt) in bands:
        s_top = float(np.interp(z_top, z_plot, sp_plot))
        s_bot = float(np.interp(z_bot, z_plot, sp_plot))
        sig_top = float(np.interp(z_top, z_plot, sigma_pot_plot))
        sig_bot = float(np.interp(z_bot, z_plot, sigma_pot_plot))
        print(
            f"  band {z_top:6.1f}–{z_bot:6.1f} m: "
            f"ΔT = +{dt:.3f} °C, ΔS = {s_bot - s_top:+.3f} PSU, "
            f"Δσ_θ = {sig_bot - sig_top:+.3f} kg/m³"
        )
    print(f"Bands with Δσ_θ < 0 (statically unstable across band): {bands_unstable} of {len(bands)}")
    print(f"Fine-scale diagnostic: min N² = {n2_min:.2e} s⁻²,  "
          f"N² below {NSQ_NOISE_FLOOR:.0e} at {n2_unstable_fine} of {n2.size} levels "
          f"(typically reflects CTD jitter / double-diffusion, not bulk overturning).")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(
        1, 4, figsize=(15.0, 9.0), sharey=True, facecolor="white"
    )
    fig.suptitle(
        f"Argo float {cast['platform']}  —  salinity, in-situ ρ, potential density σ_θ",
        fontsize=15, fontweight="bold", y=0.985,
    )
    subtitle = (
        f"{obs_date[:4]}-{obs_date[4:6]}-{obs_date[6:]}   ·   "
        f"({lat:.2f} °N, {abs(lon):.2f} °W)   ·   "
        f"TEOS-10  (S_A, C_T, ρ from GSW)"
    )
    fig.text(0.5, 0.953, subtitle, ha="center", color="#475569", fontsize=10.5)

    ax_t, ax_s, ax_rho, ax_sigma = axes

    ax_t.plot(t_plot, z_plot, color=TEMP_COLOR, lw=2.4)
    ax_t.set_xlabel("Temperature (°C)", fontsize=12)
    ax_t.set_ylabel("Depth (m)", fontsize=12)

    ax_s.plot(sp_plot, z_plot, color=SAL_COLOR, lw=2.4)
    ax_s.set_xlabel("Salinity (PSU)", fontsize=12)

    ax_rho.plot(rho_in_situ_plot, z_plot, color=RHO_COLOR, lw=2.4)
    ax_rho.set_xlabel(r"In-situ density $\rho(S_A, C_T, p)$  (kg/m³)", fontsize=12)

    ax_sigma.plot(rho_pot_plot, z_plot, color=SIGMA_COLOR, lw=2.4)
    ax_sigma.set_xlabel(r"Potential density $\rho(S_A, C_T, p{=}0)$  (kg/m³)", fontsize=12)

    for ax in axes:
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)

    for (z_top, z_bot, dt) in bands:
        for ax in axes:
            ax.axhspan(z_top, z_bot, color=INVERSION_COLOR, alpha=0.18, zorder=0)
        ax_t.text(
            0.97, (z_top + z_bot) / 2.0,
            f"T-inv  ΔT = +{dt:.2f} °C",
            transform=ax_t.get_yaxis_transform(),
            ha="right", va="center", fontsize=8.8,
            color="#9f1239",
            bbox=dict(facecolor="white", edgecolor="#fecaca", alpha=0.92, pad=2),
        )

    for ax in axes:
        ax.invert_yaxis()
        ax.set_ylim(args.max_depth_plot, 0)

    if bands and bands_unstable == 0:
        max_drho = max(band_drho)
        verdict_line1 = (
            f"{len(bands)} T-inversion band(s) detected (ΔT ≥ {INVERSION_MIN_DT:.2f} °C); "
            rf"each has $\Delta\sigma_\theta > 0$ across the band "
            rf"(largest $+{max_drho:.3f}\,\mathrm{{kg/m^3}}$)."
        )
        verdict_line2 = (
            r"$\Rightarrow$ salinity-driven density gain compensates the warm-on-cold structure; "
            r"column is statically stable across every inversion."
        )
        verdict_color = "#0f172a"
    elif not bands:
        verdict_line1 = "No T-inversions above threshold detected in this cast."
        verdict_line2 = ""
        verdict_color = "#0f172a"
    else:
        verdict_line1 = (
            f"{len(bands)} T-inversion band(s); "
            rf"{bands_unstable} band(s) have $\Delta\sigma_\theta < 0$."
        )
        verdict_line2 = "Flag this cast for QC review (real or apparent overturning)."
        verdict_color = "#9f1239"

    fig.text(
        0.5, 0.045, verdict_line1,
        ha="center", va="bottom", fontsize=10.5, color=verdict_color,
    )
    if verdict_line2:
        fig.text(
            0.5, 0.018, verdict_line2,
            ha="center", va="bottom", fontsize=10.5, color=verdict_color,
        )

    fig.tight_layout(rect=(0.02, 0.085, 0.98, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved to: {args.output}")


if __name__ == "__main__":
    main()
