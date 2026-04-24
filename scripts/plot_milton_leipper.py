"""Leipper-&-Volgenau-style figure for a single Milton pair.

Reproduces the 1972 Fig 1 aesthetic (temperature vs depth, 26 °C reference
line, shaded integration region) but augments it with three things L&V could
not do:
  * Argo observation and RTOFS model profile on the same pair of axes
  * Explicit TCHP and D26 values labeled on each panel
  * A context inset showing where the profile sits relative to Hurricane
    Milton's 2024 track

Default target: Oct 4 2024, (25.47 °N, 86.24 °W) — the central GoM profile
whose Argo-observed TCHP is 59 kJ/cm² higher than the RTOFS model estimate.
That is, the point where RTOFS most underestimated Milton's fuel on the day
the storm was forming.
"""
from __future__ import annotations

import argparse
import sys
from math import inf
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hhp_core import REF_TEMP_C, compute_d26, compute_tchp
from ml.paths import DATASETS_DIR, IBTRACS_CACHE_DIR, REPO_ROOT

PAIRS_CSV = DATASETS_DIR / "milton_pairs.csv"
PROFILES_PKL = DATASETS_DIR / "milton_pair_profiles.pkl"
TRACK_CSV = IBTRACS_CACHE_DIR / "ibtracs_MILTON_2024.csv"

OUT_DIR = REPO_ROOT / "docs"

ARGO_COLOR = "#0ea5e9"       # cyan
RTOFS_COLOR = "#f97316"      # orange
REF_COLOR = "#dc2626"        # red for 26 °C line
FUEL_THRESHOLD_KJ = 50.0     # Shay 2000 RI threshold


def _select_pair(pairs: pd.DataFrame, target_date: str | None, target_lat: float | None, target_lon: float | None) -> pd.Series:
    if target_date is not None and target_lat is not None and target_lon is not None:
        cand = pairs[pairs["obs_date"].astype(str) == target_date].copy()
        if cand.empty:
            raise SystemExit(f"No pairs on date {target_date}")
        cand["d"] = (cand["lat"] - target_lat) ** 2 + (cand["lon"] - target_lon) ** 2
        return cand.sort_values("d").iloc[0]
    # Fall back: pick the pair where the model most underestimates TCHP.
    signed = pairs.dropna(subset=["tchp_delta_kj_cm2"]).sort_values("tchp_delta_kj_cm2", ascending=False)
    return signed.iloc[0]


def _load_profile_arrays(profiles: pd.DataFrame, cast_id: str) -> dict:
    row = profiles[profiles["cast_id"] == cast_id]
    if row.empty:
        raise SystemExit(f"No profile arrays stored for cast {cast_id}")
    r = row.iloc[0]
    return {
        "obs_p": np.asarray(r["obs_pressure_dbar"], dtype=float),
        "obs_t": np.asarray(r["obs_temperature_c"], dtype=float),
        "obs_s": np.asarray(r["obs_salinity_psu"], dtype=float),
        "mod_z": np.asarray(r["model_depth_m"], dtype=float),
        "mod_t": np.asarray(r["model_temperature_c"], dtype=float),
        "mod_s": np.asarray(r["model_salinity_psu"], dtype=float),
    }


def _panel(ax, depth, temp, color, label, ref_temp=REF_TEMP_C, max_depth_plot=300.0):
    depth = np.asarray(depth)
    temp = np.asarray(temp)
    d26 = compute_d26(depth, temp, ref_temp)
    res = compute_tchp(depth, temp)

    # Restrict plot to upper ocean for readability.
    mask = depth <= max_depth_plot
    ax.plot(temp[mask], depth[mask], color=color, lw=2.2, label=label)

    # Shade the integration region: everything above D26 where T > ref_temp.
    if d26 is not None:
        shade_mask = depth <= d26
        d_shade = depth[shade_mask]
        t_shade = temp[shade_mask]
        # append the exact D26 intercept so the polygon closes at the 26°C line
        d_shade = np.concatenate([d_shade, [d26]])
        t_shade = np.concatenate([t_shade, [ref_temp]])
        ax.fill_betweenx(d_shade, ref_temp, t_shade, color=color, alpha=0.25,
                         where=(t_shade >= ref_temp))
        ax.axhline(d26, color=color, lw=0.8, ls=":", alpha=0.6)
        ax.text(ref_temp + 0.35, d26, f"D26 = {d26:.0f} m",
                color=color, fontsize=8, va="top")

    ax.axvline(ref_temp, color=REF_COLOR, lw=1.1, ls="--", alpha=0.8)
    ax.text(ref_temp + 0.05, max_depth_plot - 8, "26 °C", color=REF_COLOR,
            fontsize=9, rotation=0, va="bottom")

    ax.invert_yaxis()
    ax.set_xlim(23, 32)
    ax.set_ylim(max_depth_plot, 0)
    ax.set_xlabel("Temperature (°C)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    # Big TCHP label in panel corner.
    tchp_txt = f"TCHP = {res.tchp_kj_per_cm2:.0f} kJ/cm²" if res.tchp_kj_per_cm2 is not None else "TCHP: N/A"
    ax.text(0.04, 0.03, tchp_txt, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=color,
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.9, pad=4))
    return res


def _track_inset(fig, track_df, pair_row):
    ax = fig.add_axes([0.62, 0.60, 0.30, 0.28])
    lat = pd.to_numeric(track_df["latitude"], errors="coerce")
    lon = pd.to_numeric(track_df["longitude"], errors="coerce")
    wind = pd.to_numeric(track_df["usa_wind"], errors="coerce")

    # Gulf-of-Mexico sub-window of the track
    gom = (lat.between(15, 32)) & (lon.between(-100, -80))
    ax.plot(lon[gom], lat[gom], color="#475569", lw=1.0, alpha=0.6)
    scat = ax.scatter(lon[gom], lat[gom], c=wind[gom], cmap="YlOrRd",
                      s=18, edgecolor="#1e293b", linewidth=0.4, vmin=30, vmax=160)

    # Mark the pair location
    ax.plot(pair_row["lon"], pair_row["lat"], marker="*", markersize=16,
            markerfacecolor=ARGO_COLOR, markeredgecolor="#0f172a", markeredgewidth=1.0,
            zorder=5)

    ax.set_xlim(-99, -80)
    ax.set_ylim(17, 31)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title("Milton 2024 track (colored by wind, kt)", fontsize=9)
    cbar = fig.colorbar(scat, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    # Grid lines
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20241004", help="YYYYMMDD")
    parser.add_argument("--lat", type=float, default=25.47)
    parser.add_argument("--lon", type=float, default=-86.24)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "milton_leipper_figure.png")
    parser.add_argument("--max-depth-plot", type=float, default=250.0)
    args = parser.parse_args()

    pairs = pd.read_csv(PAIRS_CSV)
    profiles = pd.read_pickle(PROFILES_PKL)
    track = pd.read_csv(TRACK_CSV)

    pair = _select_pair(pairs, args.date, args.lat, args.lon)
    arrays = _load_profile_arrays(profiles, pair["cast_id"])

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.18,
                          left=0.08, right=0.55, top=0.88, bottom=0.12)
    ax_obs = fig.add_subplot(gs[0])
    ax_mod = fig.add_subplot(gs[1], sharey=ax_obs)

    _panel(ax_obs, arrays["obs_p"], arrays["obs_t"], ARGO_COLOR,
           "Argo observation", max_depth_plot=args.max_depth_plot)
    _panel(ax_mod, arrays["mod_z"], arrays["mod_t"], RTOFS_COLOR,
           "RTOFS model", max_depth_plot=args.max_depth_plot)
    ax_obs.set_ylabel("Depth (m)")
    ax_obs.set_title(f"Argo float {pair['platform']}", fontsize=11, color=ARGO_COLOR)
    ax_mod.set_title(f"RTOFS same-day column", fontsize=11, color=RTOFS_COLOR)

    # Storm / location banner
    fig.text(0.32, 0.95, "Hurricane Milton 2024 — Hurricane Heat Potential", ha="center",
             fontsize=15, fontweight="bold")
    obs_date_raw = str(pair["obs_date"])
    obs_date_str = f"{obs_date_raw[:4]}-{obs_date_raw[4:6]}-{obs_date_raw[6:]}"
    fig.text(0.32, 0.915,
             f"{obs_date_str}  •  ({pair['lat']:.2f} °N, {abs(pair['lon']):.2f} °W)  •  "
             f"central Gulf of Mexico",
             ha="center", fontsize=11, color="#475569")

    # Side panel: numeric summary + hurricane context
    side = fig.add_axes([0.62, 0.10, 0.30, 0.44])
    side.axis("off")
    obs_tchp = pair["obs_tchp_kj_cm2"]
    mod_tchp = pair["model_tchp_kj_cm2"]
    delta = pair["tchp_delta_kj_cm2"]
    obs_d26 = pair["obs_d26_m"]
    mod_d26 = pair["model_d26_m"]
    sst_obs = pair["obs_surface_t_c"]
    sst_mod = pair["model_surface_t_c"]

    summary_lines = [
        ("Argo observation",    ARGO_COLOR, [
            f"Surface T:  {sst_obs:.2f} °C",
            f"D26:        {obs_d26:.0f} m",
            f"TCHP:       {obs_tchp:.0f} kJ/cm²",
        ]),
        ("RTOFS model",         RTOFS_COLOR, [
            f"Surface T:  {sst_mod:.2f} °C",
            f"D26:        {mod_d26:.0f} m",
            f"TCHP:       {mod_tchp:.0f} kJ/cm²",
        ]),
    ]
    y_cursor = 0.98
    for name, color, lines in summary_lines:
        side.text(0.0, y_cursor, name, fontsize=12, fontweight="bold", color=color,
                  transform=side.transAxes, va="top")
        y_cursor -= 0.07
        for line in lines:
            side.text(0.05, y_cursor, line, fontsize=10.5, family="monospace",
                      color="#0f172a", transform=side.transAxes, va="top")
            y_cursor -= 0.055
        y_cursor -= 0.02

    # Delta + RI context
    side.axhline  # placeholder
    delta_color = "#059669" if delta > 0 else "#b91c1c"
    side.text(0.0, y_cursor, "Model gap", fontsize=12, fontweight="bold",
              color="#0f172a", transform=side.transAxes, va="top")
    y_cursor -= 0.07
    side.text(0.05, y_cursor,
              f"Observation − Model = {delta:+.0f} kJ/cm²",
              fontsize=11, family="monospace", color=delta_color,
              transform=side.transAxes, va="top")
    y_cursor -= 0.07
    thr = FUEL_THRESHOLD_KJ
    both_above = (obs_tchp is not None and mod_tchp is not None and obs_tchp >= thr and mod_tchp >= thr)
    both_below = (obs_tchp is not None and mod_tchp is not None and obs_tchp < thr and mod_tchp < thr)
    obs_above_only = (obs_tchp is not None and mod_tchp is not None and obs_tchp >= thr > mod_tchp)
    mod_above_only = (obs_tchp is not None and mod_tchp is not None and mod_tchp >= thr > obs_tchp)
    if both_above:
        ri_status = f"Both exceed the {thr:.0f} kJ/cm²\nrapid-intensification threshold"
    elif both_below:
        ri_status = f"Both below the {thr:.0f} kJ/cm²\nrapid-intensification threshold"
    elif obs_above_only:
        ri_status = f"Observation above {thr:.0f} kJ/cm²;\nmodel below RI threshold"
    elif mod_above_only:
        ri_status = f"Model above {thr:.0f} kJ/cm²;\nobservation below RI threshold"
    else:
        ri_status = "TCHP vs RI threshold: n/a"
    ri_note = (
        f"{ri_status}\n"
        f"(Shay et al. 2000). Milton\n"
        f"reached Cat 5 on Oct 7, 2024."
    )
    side.text(0.0, y_cursor, ri_note, fontsize=9.5, color="#475569",
              transform=side.transAxes, va="top", style="italic")

    _track_inset(fig, track, pair)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
