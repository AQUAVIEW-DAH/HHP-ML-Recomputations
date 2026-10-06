"""Figure 5c, redrawn so it stands on its own: the candidate features tested in
run_missing_physics_search.py, each defined in plain words, with the actual numbers.

Reads OHC/output/missing_physics_20260925/{candidate_tests.csv, nino34_rtofs.csv}; no refitting.
Output: OHC/output/meeting_followups_figs_20261001/5c_candidate_features_explained.png
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)

SRC = Path("/home/suramya/HHP-Prediction/OHC/output/missing_physics_20260925")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/meeting_followups_figs_20261001/5c_candidate_features_explained.png")

ROWS = [  # (label in the csv starts with, short name, plain definition)
    ("+ cyclic longitude", "1. Longitude as sin/cos",
     "Two extra inputs, sin(lon) and cos(lon). They place longitude on a circle, so 179°E and 179°W "
     "become neighbours (see the drawing)."),
    ("+ Nino3.4 index", "2. El Niño index",
     "Our own daily index from RTOFS: average sea-surface temperature in the Niño 3.4 box "
     "(5°S-5°N, 170°W-120°W) minus its 2024-25 average for that calendar month. Positive = warmer than "
     "usual (El Niño-like). One number per day, the same for every profile that day. Not NOAA's official "
     "index, which uses a 30-year baseline and 3-month averages."),
    ("+ Nino3.4 + cyclic", "3. Both of the above", "Inputs 1 and 2 added together."),
    ("+ family averages", "4. Family averages",
     "Three extra inputs: the average of the TCHP, D26 and SST 1° anomalies; the average of their 1° "
     "spreads; the average of their gradients (each z-scored first)."),
    ("+ top screened", "5. Best 4 screened combinations*",
     "The four sums/products of existing inputs that best explained the model's leftover error. *Picked by "
     "looking at the test blocks themselves, so this score is optimistic; the honest version is figure 5a."),
]


def main() -> None:
    C = pd.read_csv(SRC / "candidate_tests.csv")
    base = C[C.variant.str.startswith("baseline")].set_index("target").mae
    fig = plt.figure(figsize=(17, 11))
    # --- results
    ax = fig.add_axes([0.2, 0.42, 0.37, 0.4])
    y = np.arange(len(ROWS))[::-1]
    ax.axvspan(-0.01, 0.01, color=NEUTRAL, zorder=0)
    ax.axvline(0, color=AXIS, lw=1.2)
    for tn, col, mk, dy in (("tchp", BLUE, "o", 0.17), ("d26", ORANGE, "D", -0.17)):
        t = C[C.target == tn]
        for yy, (key, name, _) in zip(y, ROWS):
            r = t[t.variant.str.startswith(key)].iloc[0]
            ax.errorbar(r.change_vs_baseline, yy + dy, xerr=r.seed_sd, fmt=mk, color=col, ms=8, mec=SURF,
                        elinewidth=1.4, label=tn.upper() if yy == y[0] else None, zorder=3)
            pos = r.change_vs_baseline >= 0
            ax.text(r.change_vs_baseline + (r.seed_sd + 0.003) * (1 if pos else -1), yy + dy,
                    f"{r.change_vs_baseline:+.3f}", ha="left" if pos else "right", va="center", fontsize=10,
                    color=col, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.08, 0.04)
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_xlabel("Change in error (MAE) when the input is added     ← better  |  worse →")
    ax.text(0, len(ROWS) - 0.45, "run-to-run noise", ha="center", fontsize=10, color=MUTED)
    ax.legend(loc="upper right")
    ax.set_title("What each candidate did", loc="left")
    # --- definitions
    d = fig.add_axes([0.6, 0.42, 0.38, 0.4])
    d.axis("off")
    d.set_title("What each candidate is", loc="left")
    yy = 0.98
    for _, name, txt in ROWS:
        d.text(0, yy, name, fontsize=11.5, fontweight="bold", color=INK, va="top", transform=d.transAxes)
        lines = textwrap.wrap(txt, 78)
        d.text(0, yy - 0.055, "\n".join(lines), fontsize=10.2, color=INK2, va="top", transform=d.transAxes,
               linespacing=1.35)
        yy -= 0.075 + 0.047 * len(lines)
    # --- the index itself
    n = pd.read_csv(SRC / "nino34_rtofs.csv", dtype={"date": str})
    n["t"] = pd.to_datetime(n.date)
    e = fig.add_axes([0.2, 0.07, 0.37, 0.24])
    e.plot(n.t, n.nino34_anom, color=BLUE, lw=1.4)
    for v in (0.5, -0.5):
        e.axhline(v, color=MUTED, lw=1, ls="--")
    e.axhline(0, color=AXIS, lw=1)
    e.text(n.t.iloc[-1], 0.58, "El Niño threshold +0.5", ha="right", va="bottom", fontsize=9.5, color=MUTED)
    e.text(n.t.iloc[-1], -0.58, "La Niña threshold −0.5", ha="right", va="top", fontsize=9.5, color=MUTED)
    e.set_ylabel("°C")
    e.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    e.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    e.set_title("Input 2, the El Niño index over our data: strong only in early 2024 (end of the 2023-24\n"
                "El Niño, training-only months) and a brief weak La Niña in early 2025", loc="left", fontsize=11.5)
    # --- the dateline drawing
    s = fig.add_axes([0.62, 0.2, 0.2, 0.07])
    s.set_xlim(-180, 180)
    s.set_ylim(0, 1)
    s.add_patch(Rectangle((-180, 0.15), 80, 0.7, color=ORANGE))
    s.add_patch(Rectangle((160, 0.15), 20, 0.7, color=ORANGE))
    s.set_yticks([])
    s.set_xticks([-180, -100, 0, 160, 180])
    s.set_xticklabels(["180°", "100°W", "0°", "160°E", ""], fontsize=9.5)
    s.grid(False)
    s.spines["left"].set_visible(False)
    s.set_title("Raw longitude: the East-Pacific region\n(160°E to 100°W) is two pieces, one at each end",
                loc="left", fontsize=11)
    c = fig.add_axes([0.85, 0.06, 0.12, 0.24])
    c.set_aspect("equal")
    c.add_patch(plt.Circle((0, 0), 1, fill=False, color=AXIS, lw=6))
    c.add_patch(Wedge((0, 0), 1.08, 160, 260, width=0.16, color=ORANGE))
    c.set_xlim(-1.35, 1.35)
    c.set_ylim(-1.35, 1.35)
    c.axis("off")
    c.text(0, 0, "sin/cos:\none piece", ha="center", va="center", fontsize=11, fontweight="bold", color=INK2)
    c.text(1.13, 0, "0°", va="center", fontsize=9.5, color=MUTED)
    c.text(-1.16, 0.15, "180°", ha="right", va="center", fontsize=9.5, color=MUTED)
    headline(fig, "Of the candidate inputs, only longitude as sin/cos is a clear, honest gain",
             f"Each candidate was added to the current recipe and scored on the same test blocks (3 runs each; bars = "
             f"spread between runs). For reference, the current recipe's error is {base['tchp']:.2f} kJ/cm² (TCHP)\n"
             f"and {base['d26']:.2f} m (D26), so −0.04 is about a 0.4% improvement. The El Niño index adds nothing, "
             "but the test periods held no real El Niño and only a brief, weak La Niña,\nso this cannot rule ENSO out.", top=0.84)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
