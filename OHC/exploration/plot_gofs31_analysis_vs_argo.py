"""Figures for the GOFS 3.1 analysis download (2021-09 .. 2024-09) against Argo, and against RTOFS.

G1  monthly error of GOFS vs Argo over three years, with raw RTOFS (2024-25) for reference
G2  where each model is too high or too low: mean Argo-minus-model error in 2-degree boxes
G3  the transfer question: on the same floats and days (2024-01-31 .. 2024-09-04), does GOFS's error
    look like RTOFS's? If it does, three years of GOFS could help train the RTOFS correction.

Error = Argo minus model (positive = model too low). Warm rows only (both have 26 C water), one
profile per Argo cast. Inputs: build_gofs31_analysis_at_argo_points.py output and the clean RTOFS tables.
Outputs: OHC/output/gofs31_analysis_20261008/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, DIVERGING, INK2, MUTED, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import _merged  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/gofs31_analysis_20261008")
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
ENSO = [("2021-09-01", "2023-02-01", "La Niña", BLUE), ("2023-06-01", "2024-05-01", "El Niño", ORANGE),
        ("2024-12-01", "2025-04-01", "weak La Niña", BLUE)]   # approximate, from NOAA ONI


def load():
    g = pd.read_parquet(OUT / "argo_gofs31a_collocated.parquet")
    g = g[g.argo_tchp_kj_per_cm2.notna() & g.gofs_tchp_kj_per_cm2.notna()].copy()
    g["t"] = pd.to_datetime(g.date, format="%Y%m%d")
    r = _merged()
    r = r[r.argo_tchp_kj_per_cm2.notna() & r.model_interp_tchp_kj_per_cm2.notna()].copy()
    r["t"] = pd.to_datetime(r.date.astype(str), format="%Y%m%d")
    for tn, a, m in (("tchp", "argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2"), ("d26", "argo_d26_m", "model_interp_d26_m")):
        r[f"err_{tn}"] = r[a] - r[m]
        g[f"err_{tn}"] = g[a] - g[f"gofs_{tn}_kj_per_cm2" if tn == "tchp" else "gofs_d26_m"]
    return g, r


def fig_monthly(g, r) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), sharex=True)
    for row, tn in zip(axes, ("tchp", "d26")):
        for ax, stat in zip(row, ("mae", "bias")):
            for s, e, lab, col in ENSO:
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=col, alpha=0.07, lw=0)
                if tn == "tchp" and stat == "mae":
                    ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, 1.0, lab, ha="center",
                            va="bottom", transform=ax.get_xaxis_transform(), fontsize=9.5, color=INK2)
            for d, col, lab in ((g, BLUE, "GOFS 3.1 analysis"), (r, ORANGE, "raw RTOFS")):
                m = d.set_index("t")[f"err_{tn}"].resample("MS")
                v = m.apply(lambda x: x.abs().mean()) if stat == "mae" else m.mean()
                v = v[m.count() >= 200]
                ax.plot(v.index + pd.Timedelta(days=14), v.values, "-o", color=col, lw=2, ms=4, mec=SURF, label=lab)
            if stat == "bias":
                ax.axhline(0, color=AXIS, lw=1.2)
            ax.set_ylabel(f"{tn.upper()} {'mean absolute error' if stat == 'mae' else 'mean error (Argo − model)'}\n({UNIT[tn]})")
            if tn == "tchp" and stat == "mae":
                ax.legend(loc="upper right")
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    gm = {tn: (g[f"err_{tn}"].abs().mean(), g[f"err_{tn}"].mean()) for tn in ("tchp", "d26")}
    rm = {tn: (r[f"err_{tn}"].abs().mean(), r[f"err_{tn}"].mean()) for tn in ("tchp", "d26")}
    headline(fig, "GOFS is also too low against Argo, but less so than raw RTOFS, in every ENSO phase",
             f"Monthly error against Argo floats (months with at least 200 profiles; mean error > 0 = model too low). "
             f"GOFS 2021-24: TCHP error {gm['tchp'][0]:.1f}, mean {gm['tchp'][1]:+.1f};\nD26 {gm['d26'][0]:.1f}, mean "
             f"{gm['d26'][1]:+.1f}. Raw RTOFS 2024-25: TCHP {rm['tchp'][0]:.1f}, mean {rm['tchp'][1]:+.1f}; D26 "
             f"{rm['d26'][0]:.1f}, mean {rm['d26'][1]:+.1f}.\nBoth models assimilate Argo, so neither is independent of the "
             "floats. Shading: approximate ENSO phases (NOAA ONI).", top=0.83)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, hspace=0.12, wspace=0.2)
    fig.savefig(OUT / "G1_monthly_error_gofs_vs_rtofs.png", dpi=150)
    plt.close(fig)


def boxmap(ax, d, col, lim, title):
    b = d.assign(by=np.floor(d.lat / 2) * 2 + 1, bx=np.floor(d.lon / 2) * 2 + 1).groupby(["by", "bx"])
    m = b[col].mean()[b.size() >= 10].reset_index()
    sc = ax.scatter(m.bx, m.by, c=m[col], s=9, marker="s", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), linewidths=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-42, 48)
    ax.set_facecolor("#e9e8e3")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=12)
    return sc


def fig_maps(g, r) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.4))
    for row, tn in zip(axes, ("tchp", "d26")):
        lim = 30
        boxmap(row[0], g, f"err_{tn}", lim, f"{tn.upper()}: GOFS analysis, Sep 2021 - Sep 2024")
        sc = boxmap(row[1], r, f"err_{tn}", lim, f"{tn.upper()}: raw RTOFS, Feb 2024 - Dec 2025")
        cb = fig.colorbar(sc, ax=row, fraction=0.02, pad=0.01, extend="both")
        cb.set_label(f"Argo − model ({UNIT[tn]})\nred = model too low")
    headline(fig, "Where each model is too high or too low against Argo",
             "Average of Argo minus model in 2° boxes with at least 10 profiles. Red = model too low, blue = too high. "
             "The periods differ (GOFS ends Sep 2024, RTOFS starts Feb 2024);\nFigure G3 compares the two on the same "
             "floats and days.", top=0.86)
    fig.subplots_adjust(left=0.04, right=0.9, bottom=0.05, hspace=0.25, wspace=0.08)
    fig.savefig(OUT / "G2_error_maps_gofs_vs_rtofs.png", dpi=150)
    plt.close(fig)


def fig_same_floats(g, r) -> pd.DataFrame:
    m = g[["cast_id", "err_tchp", "err_d26", "lat", "lon"]].merge(r[["cast_id", "err_tchp", "err_d26"]], on="cast_id",
                                                                 suffixes=("_gofs", "_rtofs"))
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, tn in zip(axes, ("tchp", "d26")):
        x, y = m[f"err_{tn}_rtofs"].to_numpy(), m[f"err_{tn}_gofs"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        rr = np.corrcoef(x, y)[0, 1]
        lim = np.percentile(np.abs(np.r_[x, y]), 99)
        ax.hexbin(x, y, gridsize=70, extent=(-lim, lim, -lim, lim), bins="log", mincnt=1,
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#e8f0fb", "#86b6ef", "#1c5cab", "#0d366b"]))
        ax.plot([-lim, lim], [-lim, lim], color=ORANGE, ls="--", lw=1.4, label="same error in both")
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"RTOFS error, Argo − RTOFS ({UNIT[tn]})")
        ax.set_ylabel(f"GOFS error, Argo − GOFS ({UNIT[tn]})")
        ax.set_title(f"{tn.upper()}: {ok.sum():,} profiles, correlation {rr:.2f}", loc="left")
        ax.legend(loc="upper left")
        rows.append({"target": tn, "profiles": int(ok.sum()), "correlation": rr,
                     "rtofs_mae": float(np.abs(x).mean()), "rtofs_mean": float(x.mean()),
                     "gofs_mae": float(np.abs(y).mean()), "gofs_mean": float(y.mean())})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "G3_same_floats_summary.csv", index=False)
    t, d = S.set_index("target").loc["tchp"], S.set_index("target").loc["d26"]
    verdict = ("their errors are only weakly related" if max(t.correlation, d.correlation) < 0.4 else
               "their errors are partly related" if max(t.correlation, d.correlation) < 0.7 else
               "their errors are closely related")
    headline(fig, f"On the same floats and days, {verdict}",
             f"Jan 31 - Sep 4, 2024. TCHP: RTOFS error {t.rtofs_mae:.1f} (mean {t.rtofs_mean:+.1f}), GOFS {t.gofs_mae:.1f} "
             f"(mean {t.gofs_mean:+.1f}). D26: RTOFS {d.rtofs_mae:.1f} (mean {d.rtofs_mean:+.1f}), GOFS {d.gofs_mae:.1f} "
             f"(mean {d.gofs_mean:+.1f}).\nPoints near the dashed line have the same error in both models. A weak relation would "
             "mean a GOFS-trained correction cannot help RTOFS;\nthis moderate one means part of it could carry over, "
             "which is worth testing.", top=0.8)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.1, wspace=0.25)
    fig.savefig(OUT / "G3_same_floats_gofs_vs_rtofs_error.png", dpi=150)
    plt.close(fig)
    return S


if __name__ == "__main__":
    g, r = load()
    fig_monthly(g, r)
    fig_maps(g, r)
    print(fig_same_floats(g, r).round(3).to_string(index=False))
