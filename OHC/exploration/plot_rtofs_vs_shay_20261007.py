"""RTOFS vs Nick Shay's satellite OHC product for the Gulf of Mexico on 2026-10-07.

Shay's product (UM Upper Ocean Dynamics Lab, run operationally at NOAA/NESDIS) estimates OHC (heat
above 26 C, the same quantity as our TCHP), D26, D20 and mixed-layer depth from satellite SST and
sea-surface-height anomaly. Source: NOAA CoastWatch ERDDAP dataset noaacwOHC14na (North Atlantic,
0.25 deg, daily, 2024-01-15 to present), downloaded to /data/suramya/shay_ohc/gulf_20261007.nc.

RTOFS (raw, 00Z run, 6-h forecast) is averaged onto Shay's 0.25-degree cells, and both are masked where
Shay has no value (land and water shallower than ~200 m), so the two are compared on the same cells.
No ML correction is applied here.

Output: OHC/output/shay_comparison_20261008/rtofs_vs_shay_gulf_20261007.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import DIVERGING, headline  # noqa: E402

RTOFS = Path("/data/suramya/rtofs_global_ohc_fields_2026/rtofs_tchp_20261007.nc")
SHAY = Path("/data/suramya/shay_ohc/gulf_20261007.nc")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/shay_comparison_20261008")
# colour scale approximating the one on Shay's maps, so the panels read like his figure
SHAY_CMAP = LinearSegmentedColormap.from_list("shay", [
    (0.0, "#000080"), (0.08, "#0000ff"), (0.18, "#2060ff"), (0.25, "#1a9a8a"), (0.33, "#1aa04a"),
    (0.42, "#7dc23a"), (0.5, "#ffff00"), (0.62, "#ffc000"), (0.75, "#ff8000"), (0.87, "#ff3000"), (1.0, "#8b0000")])


def to_shay_grid(lat, lon, v, glat, glon):
    """Average RTOFS cells into Shay's 0.25-degree cells (cell centres at glat, glon)."""
    ok = np.isfinite(v) & (lat >= glat[0] - 0.125) & (lat <= glat[-1] + 0.125) & \
         (lon >= glon[0] - 0.125) & (lon <= glon[-1] + 0.125)
    iy = np.clip(np.round((lat[ok] - glat[0]) / 0.25).astype(int), 0, len(glat) - 1)
    ix = np.clip(np.round((lon[ok] - glon[0]) / 0.25).astype(int), 0, len(glon) - 1)
    k = iy * len(glon) + ix
    s = np.bincount(k, v[ok], len(glat) * len(glon))
    c = np.bincount(k, None, len(glat) * len(glon))
    with np.errstate(invalid="ignore"):
        return (s / np.where(c > 0, c, np.nan)).reshape(len(glat), len(glon))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sh = xr.open_dataset(SHAY).isel(time=0)
    glat, glon = sh.latitude.values.astype(float), sh.longitude.values.astype(float)
    with xr.open_dataset(RTOFS) as ds:
        lat = ds.Latitude.values.astype(float)
        lon = ((ds.Longitude.values.astype(float) + 180) % 360) - 180
        sst = ds.surface_temp_c.values
        tchp = np.where(np.isfinite(sst) & (sst < 26) & ~np.isfinite(ds.tchp_kj_per_cm2.values), 0.0,
                        ds.tchp_kj_per_cm2.values)       # surface colder than 26 C: zero heat above 26 C
        d26 = ds.d26_m.values
    rows = [("tchp", "OHC / TCHP (kJ/cm²)", sh.ohc.values, to_shay_grid(lat, lon, tchp, glat, glon), 200, 60),
            ("d26", "D26 (m)", sh.iso26C.values, to_shay_grid(lat, lon, d26, glat, glon), 150, 50)]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.5))
    stats = {}
    for ax_row, (key, lab, s, r, vmax, dlim) in zip(axes, rows):
        both = np.isfinite(s) & np.isfinite(r)
        s_, r_ = np.where(both, s, np.nan), np.where(both, r, np.nan)
        dd = r_ - s_
        stats[key] = (np.nanmean(dd), np.nanmean(np.abs(dd)), np.corrcoef(s_[both], r_[both])[0, 1], int(both.sum()))
        for ax, v, ttl, cmap, norm in (
                (ax_row[0], s_, f"Shay / NOAA satellite {lab.split(' (')[0]}", SHAY_CMAP, plt.Normalize(0, vmax)),
                (ax_row[1], r_, f"RTOFS {lab.split(' (')[0]} (raw)", SHAY_CMAP, plt.Normalize(0, vmax)),
                (ax_row[2], dd, "RTOFS minus Shay", DIVERGING, TwoSlopeNorm(0, -dlim, dlim))):
            ax.set_facecolor("#1b1b1b")
            im = ax.pcolormesh(glon, glat, np.ma.masked_invalid(v), cmap=cmap, norm=norm, shading="nearest")
            ax.set_xlim(-100, -80)
            ax.set_ylim(14, 31)
            ax.set_aspect(1 / np.cos(np.deg2rad(22.5)))
            ax.grid(False)
            ax.set_xticks(range(-100, -79, 5))
            ax.set_xticklabels([f"{-x}°W" for x in range(-100, -79, 5)])
            ax.set_yticks(range(15, 31, 5))
            ax.set_yticklabels([f"{y}°N" for y in range(15, 31, 5)])
            ax.set_title(ttl, loc="left", fontsize=12.5)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.07,
                              extend="max" if ax is not ax_row[2] else "both")
            cb.set_label(lab if ax is not ax_row[2] else f"RTOFS − Shay, {lab.split('(')[1].rstrip(')')}"
                         + "  (red = RTOFS higher)")
    t, d = stats["tchp"], stats["d26"]
    headline(fig, "7 Oct 2026, Gulf of Mexico: RTOFS and Shay's satellite product place the same features, "
             "with different strengths",
             f"Compared on the same {t[3]:,} cells of 0.25° (land and water shallower than ~200 m left out, as in Shay's maps). "
             f"\nTCHP: RTOFS minus Shay averages {t[0]:+.1f} kJ/cm², typical difference {t[1]:.1f}, pattern correlation "
             f"{t[2]:.2f}. D26: average {d[0]:+.1f} m, typical difference {d[1]:.1f} m, correlation {d[2]:.2f}.\n"
             "Raw RTOFS, no correction. Shay's product comes from satellite SST and sea-surface height: independent of "
             "RTOFS's model physics, but not of altimetry.", top=0.86)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.06, hspace=0.2, wspace=0.12)
    fig.savefig(OUT / "rtofs_vs_shay_gulf_20261007.png", dpi=150)
    plt.close(fig)
    print({k: [round(float(x), 3) for x in v] for k, v in stats.items()})


if __name__ == "__main__":
    main()
