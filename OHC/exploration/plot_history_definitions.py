"""One image defining the RTOFS recent-history features, with the days each one uses and its results.

Reads the saved results (no refitting): OHC/output/rtofs_history_20261004/history_scores.csv and
OHC/output/rolling_year_windows_20261015/rolling_year_scores.csv.
Output: OHC/output/rtofs_history_20261004/0_recent_history_definitions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AQUA, AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)

O = Path("/home/suramya/HHP-Prediction/OHC/output")
OUT = O / "rtofs_history_20261004/0_recent_history_definitions.png"
ROWS = [("rtofs_{var}_mean_prev7d", "average over the 7 days before the profile day", "d−7 … d−1", range(1, 8), "mean"),
        ("rtofs_{var}_change_3d", "value on the profile day minus 3 days before", "d and d−3", (0, 3), "diff"),
        ("rtofs_{var}_change_7d", "value on the profile day minus 7 days before", "d and d−7", (0, 7), "diff"),
        ("rtofs_{var}_std_8d", "spread (standard deviation) over the 8 days", "d−7 … d", range(0, 8), "mean")]


def cell(ax, x, y, w, h, text, fc=SURF, bold=False, size=11.5, color=INK, ha="left"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=AXIS, lw=0.8))
    ax.text(x + (0.012 if ha == "left" else w / 2), y + h / 2, text, ha=ha, va="center", fontsize=size,
            fontweight="bold" if bold else None, color=color)


def main() -> None:
    H = pd.read_csv(O / "rtofs_history_20261004/history_scores.csv")
    R = pd.read_csv(O / "rolling_year_windows_20261015/rolling_year_scores.csv")
    fig = plt.figure(figsize=(16.5, 10.2))
    # ---- definitions table
    ax = fig.add_axes([0.02, 0.47, 0.6, 0.36])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cols = [(0.0, 0.33, "Feature (×3: TCHP, D26, SST)"), (0.33, 0.47, "Meaning"), (0.8, 0.2, "RTOFS days used")]
    h = 1 / (len(ROWS) + 1)
    for x, w, t in cols:
        cell(ax, x, 1 - h, w, h, t, fc=NEUTRAL, bold=True)
    for i, (name, meaning, days, _, _) in enumerate(ROWS):
        y = 1 - h * (i + 2)
        cell(ax, 0.0, y, 0.33, h, name, size=11.5)
        cell(ax, 0.33, y, 0.47, h, meaning)
        cell(ax, 0.8, y, 0.2, h, days, ha="center")
    ax.set_title("12 features = 4 kinds × 3 RTOFS variables ({var} = tchp, d26, sst)", loc="left", fontsize=12.5,
                 fontweight="bold")
    # ---- timeline
    tl = fig.add_axes([0.66, 0.47, 0.32, 0.36])
    tl.set_xlim(-10.4, 0.8)
    tl.set_ylim(-0.6, len(ROWS) - 0.4)
    for i, (name, _, _, days, kind) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        for dd in range(0, 8):
            used = dd in days
            col = (BLUE if kind == "mean" else ORANGE) if used else NEUTRAL
            tl.add_patch(FancyBboxPatch((-dd - 0.4, y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.02", fc=col,
                                        ec=SURF, lw=1.5))
        tl.text(-10.3, y, name.replace("rtofs_{var}_", ""), va="center", fontsize=10.5, color=INK2)
    tl.set_yticks([])
    tl.set_xticks(range(-7, 1))
    tl.set_xticklabels([f"d−{k}" if k else "d" for k in range(7, -1, -1)])
    tl.set_xlabel("RTOFS day (d = the profile's day; nothing after d is used)")
    for s in ("left", "right", "top"):
        tl.spines[s].set_visible(False)
    tl.grid(False)
    tl.set_title("Which days each feature reads", loc="left", fontsize=12.5, fontweight="bold")
    tl.text(0.0, -0.55, "blue: averaged   orange: differenced", fontsize=9.5, color=INK2, ha="right")
    # ---- results table
    rs = fig.add_axes([0.02, 0.06, 0.96, 0.31])
    rs.set_xlim(0, 1)
    rs.set_ylim(0, 1)
    rs.axis("off")
    rs.set_title("What adding the 12 features to the full recipe did (change in mean absolute error; negative = better; "
                 "chance level about ±0.03)", loc="left", fontsize=12.5, fontweight="bold")
    tests = [("forward test blocks (3 blocks)", lambda tn: H[(H.target == tn) & (H.design == "forward blocks")]),
             ("full-year tests (test 2024, test 2025)", lambda tn: H[(H.target == tn) & (H.design == "full-year tests")])]
    hdr = [(0.0, 0.34, "Test"), (0.34, 0.33, "TCHP (kJ/cm²)"), (0.67, 0.33, "D26 (m)")]
    h = 1 / 5
    for x, w, t in hdr:
        cell(rs, x, 1 - h, w, h, t, fc=NEUTRAL, bold=True)
    vals = {}
    for i, (lab, get) in enumerate(tests):
        y = 1 - h * (i + 2)
        cell(rs, 0, y, 0.34, h, lab)
        for j, tn in enumerate(("tchp", "d26")):
            g = get(tn)
            b = g[g.variant.str.startswith("baseline")].iloc[0]
            a = g[g.variant == "+ all 12 history features"].iloc[0]
            per = [a[c] - b[c] for c in g.columns if c.startswith("fold: ") and pd.notna(a[c])]
            txt = f"{a.mae - b.mae:+.2f} overall   (blocks: " + ", ".join(f"{v:+.2f}" for v in per) + ")"
            cell(rs, 0.34 + 0.33 * j, y, 0.33, h, txt)
            vals[(lab, tn)] = a.mae - b.mae
    y = 1 - h * 4
    cell(rs, 0, y, 0.34, h, "12-month windows starting Feb, May, Aug, Nov 2024, Jan 2025")
    for j, tn in enumerate(("tchp", "d26")):
        r = R[R.target == tn].pivot_table(index="window", columns="variant", values="mae")
        d = r["full recipe + RTOFS recent history"] - r["full recipe"]
        cell(rs, 0.34 + 0.33 * j, y, 0.33, h, f"{d.min():+.2f} to {d.max():+.2f} in all 5 windows")
    y = 1 - h * 5
    cell(rs, 0, y, 1.0, h, "Only RTOFS fields up to the profile day are used (no Argo, no later days), so the features "
         "exist in operation. Code: OHC/exploration/run_rtofs_history_features.py", size=10.5, color=INK2)
    headline(fig, "RTOFS recent history at the float: 12 extra features outside the 34/35-feature recipe",
             "For each profile, RTOFS TCHP, D26 and SST are read at the profile's nearest grid cell on the profile day "
             "and the 7 days before (cells with no 26 °C water count as TCHP = 0).\nNot yet checked: whether the "
             "profile-day field already contains this float's own measurement through data assimilation (test planned: "
             "the same features ending one or two days earlier).", top=0.89)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
