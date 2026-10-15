"""Maps of the GOFS and RTOFS errors against Argo, with the same interpolations as the earlier RTOFS maps.

Error = Argo minus model (positive = model too low), for TCHP and D26, warm rows only (both Argo and the
model have 26 C water), one profile per Argo cast, inside the GOFS download band (41.5 S - 47.5 N).

Two comparisons:
  A. Same floats and days (2024-01-31 .. 2024-09-04, the only period both models cover): GOFS error,
     RTOFS error and their difference (GOFS error minus RTOFS error = RTOFS minus GOFS).
  B. Each model's full record by season: GOFS Sep 2021 - Sep 2024, RTOFS Feb 2024 - Dec 2025, winter
     (Jan-Mar) and summer (Jul-Sep).
Interpolation exactly as render_ml_corrected_rtofs_2025.py: 0.25-degree grid, cells more than 100 km from
any profile left blank; Gaussian weighting (100 km length scale, 300 km cut-off), linear (Delaunay in
lon/lat) and Gaussian radial basis functions (100 km scale, 64 neighbours), all on the individual profiles.

Outputs: OHC/output/gofs_rtofs_error_maps_20261014/
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.seasonal_map_common import (  # noqa: E402
    add_land_overlay, gaussian_interpolate, linear_nd_interpolate, rbf_gaussian_interpolate)
from OHC.exploration.plot_meeting_followups_20261001 import DIVERGING, INK2, headline  # noqa: E402
from OHC.exploration.rtofs_followup_common import _merged  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/gofs_rtofs_error_maps_20261014")
GOFS = Path("/home/suramya/HHP-Prediction/OHC/output/gofs31_analysis_20261008/argo_gofs31a_collocated.parquet")
LAT_BAND = (-41.5, 47.5)
RES, MASK_KM = 0.25, 100.0
LIM = {"tchp": 30, "d26": 30}
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
METHODS = {"Gaussian weighting": "gaussian", "linear": "linear", "RBF (Gaussian kernel)": "rbf"}
SEASONS = {"winter (Jan-Mar)": [1, 2, 3], "summer (Jul-Sep)": [7, 8, 9]}
BOXES = {  # 20-degree boxes from build_hhp_density_scatter_diagnostics.NAMED_BOXES (lower-left corner)
    "Gulf of Mexico & NW Caribbean": (20, -100), "Atlantic MDR / E Caribbean": (0, -60), "Arabian Sea": (0, 60),
    "Bay of Bengal": (0, 80), "Philippine Sea / W Pacific warm pool": (0, 120), "Coral Sea": (-20, 140),
    "Central equatorial N Pacific": (0, -180), "SW Pacific (Fiji sector)": (-20, -180)}


def grid():
    lat = np.arange(LAT_BAND[0] + RES / 2, LAT_BAND[1], RES)
    lon = np.arange(-180 + RES / 2, 180, RES)
    gl, ga = np.meshgrid(lon, lat)
    return ga, gl


def interp(args):
    method, lat, lon, v = args
    ga, gl = grid()
    if method == "gaussian":
        return gaussian_interpolate(lat, lon, v, ga, gl, mask_distance_km=MASK_KM)
    if method == "linear":
        return linear_nd_interpolate(lat, lon, v, ga, gl, mask_distance_km=MASK_KM)
    return rbf_gaussian_interpolate(lat, lon, v, ga, gl, mask_distance_km=MASK_KM)


def load():
    g = pd.read_parquet(GOFS)
    g = g[g.argo_tchp_kj_per_cm2.notna() & g.gofs_tchp_kj_per_cm2.notna()].copy()
    g["err_tchp"], g["err_d26"] = g.gofs_delta_tchp, g.gofs_delta_d26
    r = _merged()
    r = r[r.argo_tchp_kj_per_cm2.notna() & r.model_interp_tchp_kj_per_cm2.notna() & r.lat.between(*LAT_BAND)].copy()
    r["err_tchp"] = r.argo_tchp_kj_per_cm2 - r.model_interp_tchp_kj_per_cm2
    r["err_d26"] = r.argo_d26_m - r.model_interp_d26_m
    for d in (g, r):
        d["date"] = d["date"].astype(str)
        d["month"] = d["date"].str[4:6].astype(int)
    both = g[["cast_id", "lat", "lon", "date", "err_tchp", "err_d26"]].merge(
        r[["cast_id", "err_tchp", "err_d26"]], on="cast_id", suffixes=("_gofs", "_rtofs"))
    for q in ("tchp", "d26"):
        both[f"err_{q}_diff"] = both[f"err_{q}_gofs"] - both[f"err_{q}_rtofs"]
    return g, r, both


ASPECT = (LAT_BAND[1] - LAT_BAND[0]) / 360.0


def layout(nrows, ncols, panel_w, left, right, top, bottom, wgap, hgap):
    """Figure with map panels placed exactly (inches), so the equal-aspect maps leave no empty bands."""
    ph = panel_w * ASPECT
    W = left + ncols * panel_w + (ncols - 1) * wgap + right
    H = top + nrows * ph + (nrows - 1) * hgap + bottom
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x0 = (left + c * (panel_w + wgap)) / W
            y0 = (H - top - (r + 1) * ph - r * hgap) / H
            axes[r, c] = fig.add_axes([x0, y0, panel_w / W, ph / H])
    cax = fig.add_axes([(W - right + 0.25) / W, bottom / H, 0.18 / W, (H - top - bottom) / H])
    return fig, axes, cax


def _style(ax, title):
    add_land_overlay(ax, zorder=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(*LAT_BAND)
    ax.set_facecolor("#f4f3f0")
    ax.grid(False)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.tick_params(labelsize=8.5)
    if title:
        ax.set_title(title, loc="left", fontsize=11.5)


def draw_map(ax, field, lim, title=None):
    ga, gl = grid()
    im = ax.pcolormesh(gl, ga, np.ma.masked_invalid(field), cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim),
                       shading="auto", rasterized=True)
    _style(ax, title)
    return im


def draw_points(ax, lat, lon, v, lim, title=None):
    im = ax.scatter(lon, lat, c=v, s=1.0, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), linewidths=0,
                    rasterized=True)
    _style(ax, title)
    return im


def row_label(fig, ax, text):
    b = ax.get_position()
    fig.text(b.x0 - 0.045 * (8 / fig.get_figwidth()) * 2.2, b.y0 + b.height / 2, text, ha="right", va="center",
             fontsize=10.5, color=INK2)


def box_stats(d, col):
    rows = {"global (41.5 S - 47.5 N)": d}
    for name, (la, lo) in BOXES.items():
        rows[name] = d[(d.lat >= la) & (d.lat < la + 20) & (d.lon >= lo) & (d.lon < lo + 20)]
    return {k: (float(v[col].mean()), float(v[col].abs().mean()), int(len(v))) for k, v in rows.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g, r, both = load()
    print(f"GOFS warm profiles {len(g):,}; RTOFS warm profiles {len(r):,}; same floats and days {len(both):,}", flush=True)
    jobs, keys = [], []
    for q in ("tchp", "d26"):
        for src in ("gofs", "rtofs", "diff"):
            for m in METHODS.values():
                keys.append(("overlap", q, src, m))
                jobs.append((m, both.lat.to_numpy(float), both.lon.to_numpy(float), both[f"err_{q}_{src}"].to_numpy(float)))
        for sname, months in SEASONS.items():
            for src, d in (("gofs", g), ("rtofs", r)):
                s = d[d.month.isin(months)]
                for m in METHODS.values():
                    keys.append(("season", q, f"{src}|{sname}", m))
                    jobs.append((m, s.lat.to_numpy(float), s.lon.to_numpy(float), s[f"err_{q}"].to_numpy(float)))
    if (OUT / "grids.npz").exists():
        z = np.load(OUT / "grids.npz")
        grids = {tuple(k.split("|", 3)) if k.count("|") == 3 else None: z[k] for k in z.files}
        grids = {}
        for k in z.files:
            parts = k.split("|")
            key = (parts[0], parts[1], "|".join(parts[2:-1]), parts[-1])
            grids[key] = z[k]
    else:
        with ProcessPoolExecutor(max_workers=8) as ex:
            grids = dict(zip(keys, ex.map(interp, jobs)))
        np.savez_compressed(OUT / "grids.npz", **{"|".join(k): v for k, v in grids.items()})
    print("interpolations done", flush=True)

    stats = []
    for q in ("tchp", "d26"):
        lim = LIM[q]
        # ---- A. same floats and days
        fig, axes, cax = layout(3, 4, 4.6, 2.5, 1.7, 1.45, 0.35, 0.3, 0.5)
        names = {"gofs": "GOFS error\n(Argo − GOFS)", "rtofs": "RTOFS error\n(Argo − RTOFS)",
                 "diff": "GOFS error minus\nRTOFS error\n(= RTOFS − GOFS)"}
        for i, src in enumerate(("gofs", "rtofs", "diff")):
            v = both[f"err_{q}_{src}"]
            im = draw_points(axes[i, 0], both.lat, both.lon, v, lim, "profiles" if i == 0 else None)
            for j, (mlabel, m) in enumerate(METHODS.items()):
                draw_map(axes[i, j + 1], grids[("overlap", q, src, m)], lim, mlabel if i == 0 else None)
            row_label(fig, axes[i, 0], f"{names[src]}\nmean {v.mean():+.1f}\nmean |·| {v.abs().mean():.1f}")
        gg, rr = grids[("overlap", q, "gofs", "gaussian")], grids[("overlap", q, "rtofs", "gaussian")]
        ok = np.isfinite(gg) & np.isfinite(rr)
        pc = np.corrcoef(gg[ok], rr[ok])[0, 1]
        cb = fig.colorbar(im, cax=cax, extend="both")
        cb.set_label(f"{UNIT[q]}; red = model too low\n(last row: red = GOFS lower than RTOFS)")
        eg, er = both[f"err_{q}_gofs"], both[f"err_{q}_rtofs"]
        headline(fig, f"{q.upper()}: GOFS and RTOFS errors against Argo on the same {len(both):,} floats and days "
                 "(31 Jan - 4 Sep 2024)",
                 f"Columns: the profiles themselves, then three interpolations (0.25° grid, blank beyond 100 km of a "
                 f"profile). Profile-by-profile correlation of the two errors {np.corrcoef(eg, er)[0, 1]:.2f}; correlation "
                 f"of the two\nGaussian-weighted maps {pc:.2f}. Blue in the last row = GOFS higher than RTOFS (it removes "
                 "more of RTOFS's low bias there); red = GOFS even lower than RTOFS.", top=0.87)
        fig.savefig(OUT / f"A_{q}_same_floats_gofs_rtofs_error_maps.png", dpi=140)
        plt.close(fig)
        for src, col in (("GOFS", f"err_{q}_gofs"), ("RTOFS", f"err_{q}_rtofs")):
            for box, (mean, mae, n) in box_stats(both, col).items():
                stats.append({"target": q, "comparison": "same floats 2024-01-31..09-04", "model": src, "region": box,
                              "mean_error_argo_minus_model": mean, "mean_abs_error": mae, "profiles": n})

        # ---- B. full records, one image per season (GOFS row above RTOFS row)
        period = {"gofs": "GOFS\nSep 2021 - Sep 2024", "rtofs": "RTOFS\nFeb 2024 - Dec 2025"}
        for sname in SEASONS:
            fig, axes, cax = layout(3, 2, 8.0, 2.7, 1.6, 1.75, 0.4, 0.45, 0.45)
            for j, src in enumerate(("gofs", "rtofs")):
                d = (g if src == "gofs" else r)
                s_ = d[d.month.isin(SEASONS[sname])][f"err_{q}"]
                head = period[src].replace("\n", ", ")
                axes[0, j].set_title(f"{head}\nmean {s_.mean():+.1f}, mean |·| {s_.abs().mean():.1f}, "
                                     f"{len(s_):,} profiles", loc="left", fontsize=12, fontweight="bold")
                for i, (mlabel, m) in enumerate(METHODS.items()):
                    im = draw_map(axes[i, j], grids[("season", q, f"{src}|{sname}", m)], lim)
            for i, mlabel in enumerate(METHODS):
                row_label(fig, axes[i, 0], mlabel.replace(" (", "\n("))
            a = grids[("season", q, f"gofs|{sname}", "gaussian")]
            b = grids[("season", q, f"rtofs|{sname}", "gaussian")]
            ok = np.isfinite(a) & np.isfinite(b)
            cr = np.corrcoef(a[ok], b[ok])[0, 1]
            cb = fig.colorbar(im, cax=cax, extend="both")
            cb.set_label(f"Argo − model ({UNIT[q]}); red = model too low")
            years = "2022-24" if sname.startswith("winter") else "2022-24"
            headline(fig, f"{q.upper()}, {sname}: where GOFS and RTOFS are too high or too low against Argo",
                     f"Left: GOFS, all {sname.split(' ')[0]}s in its record ({years}). Right: RTOFS ({sname.split(' ')[0]}s of "
                     f"2024 and 2025). Correlation of the two Gaussian-weighted maps {cr:.2f}.\nThe periods differ, so the "
                     "maps also contain year-to-year changes; the same-floats figure (A) is the like-for-like comparison. "
                     "Rows: three interpolation methods.", top=0.85)
            tag = "1_winter" if sname.startswith("winter") else "2_summer"
            fig.savefig(OUT / f"B{tag}_{q}_full_records.png", dpi=140)
            plt.close(fig)
        for src, d in (("GOFS 2021-24", g), ("RTOFS 2024-25", r)):
            for box, (mean, mae, n) in box_stats(d, f"err_{q}").items():
                stats.append({"target": q, "comparison": "full record", "model": src, "region": box,
                              "mean_error_argo_minus_model": mean, "mean_abs_error": mae, "profiles": n})
    S = pd.DataFrame(stats)
    S.to_csv(OUT / "regional_errors.csv", index=False)
    plot_regions(S)


def plot_regions(S: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=True)
    regions = list(dict.fromkeys(S.region))
    y = np.arange(len(regions))[::-1]
    cols = {"GOFS": "#2a78d6", "RTOFS": "#eb6834"}
    for i, q in enumerate(("tchp", "d26")):
        t = S[(S.target == q) & (S.comparison.str.startswith("same"))]
        for j, stat in enumerate(("mean_error_argo_minus_model", "mean_abs_error")):
            ax = axes[i, j]
            for k, mdl in enumerate(("GOFS", "RTOFS")):
                v = t[t.model == mdl].set_index("region").reindex(regions)[stat]
                ax.barh(y + (0.18 if k == 0 else -0.18), v.values, height=0.36, color=cols[mdl], label=mdl,
                        edgecolor="#fcfcfb")
            ax.axvline(0, color="#c3c2b7", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(regions, fontsize=10.5)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{q.upper()}: {'mean error (Argo − model; > 0 = model too low)' if j == 0 else 'mean absolute error'}"
                         f" ({UNIT[q]})", loc="left", fontsize=11.5)
            if i == 0 and j == 1:
                ax.legend(loc="lower right")
    headline(fig, "Both models are too low almost everywhere; GOFS is closer to Argo in most regions, "
             "but clearly worse in the Gulf of Mexico",
             "Same floats and days, 31 Jan - 4 Sep 2024 (25,066 profiles; 540-1,050 per box). 20° boxes as in the earlier "
             "RTOFS diagnostics. GOFS is also worse for TCHP in the Coral Sea;\nthe Atlantic hurricane region is equally "
             "low in both for TCHP.", top=0.88)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.05, hspace=0.25, wspace=0.08)
    fig.savefig(OUT / "C_regional_errors_same_floats.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
