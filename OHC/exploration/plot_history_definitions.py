"""One image defining three groups of features used in the follow-up experiments, with their results:

  A. RTOFS recent history at the float (12 features outside the 34/35-feature recipe)
  B. Seasonal features: the 9 calendar features and the 10 seasonal RTOFS ("seasonal physical") features
     removed in the time-and-season experiment
  C. Pairwise combinations: sums, differences and products of z-scored features

Reads saved results only (no refitting):
  OHC/output/rtofs_history_20261004/history_scores.csv
  OHC/output/rolling_year_windows_20261015/rolling_year_scores.csv
  OHC/output/time_season_loyo_20261004/{time_season_scores,seasonal_share}.csv
  OHC/output/pairwise_decomposition_20261004/pairwise_family_scores.csv
Output: OHC/output/rtofs_history_20261004/0_feature_definitions_history_season_pairwise.png
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
    AXIS, BLUE, INK, INK2, NEUTRAL, ORANGE, SURF)

O = Path("/home/suramya/HHP-Prediction/OHC/output")
OUT = O / "rtofs_history_20261004/0_feature_definitions_history_season_pairwise.png"
W, H = 17.0, 22.6
HIST_ROWS = [("rtofs_{var}_mean_prev7d", "average over the 7 days before the profile day", "d−7 … d−1", range(1, 8), "mean"),
             ("rtofs_{var}_change_3d", "value on the profile day minus 3 days before", "d and d−3", (0, 3), "diff"),
             ("rtofs_{var}_change_7d", "value on the profile day minus 7 days before", "d and d−7", (0, 7), "diff"),
             ("rtofs_{var}_std_8d", "spread (standard deviation) over the 8 days", "d−7 … d", range(0, 8), "mean")]
CALENDAR = [("year", "calendar year of the profile"), ("month_int", "month number, 1-12"),
            ("month_sin, month_cos", "sin and cos of 2π · month / 12"),
            ("doy_sin, doy_cos", "sin and cos of 2π · day of year / 366"),
            ("is_winter_jfm", "1 in January-March, else 0"), ("is_summer_jas", "1 in July-September, else 0"),
            ("is_other", "1 in the other six months, else 0")]
SEASONAL = [("model_temp_excess_26c", "RTOFS surface temperature − 26 °C"),
            ("model_temp_excess_x_abs_lat", "the same × |lat| (TCHP recipe only)"),
            ("model_mixed_layer_thickness_m", "RTOFS mixed-layer thickness"),
            ("model_mlt_x_abs_lat", "mixed-layer thickness × |lat|"),
            ("model_surface_boundary_layer_thickness_m", "RTOFS boundary-layer thickness"),
            ("d26_minus_mlt_m", "RTOFS D26 − mixed-layer thickness"),
            ("d26_to_sblt_ratio", "RTOFS D26 / boundary-layer thickness"),
            ("model_sst_local_std_1deg", "spread of RTOFS SST within ~1°"),
            ("model_sst_grad_mag_per_100km", "RTOFS SST gradient at the float"),
            ("model_sst_anom_from_1deg_mean", "RTOFS SST minus its 1° average")]


def ax_at(top_in: float, height_in: float, left: float = 0.02, width: float = 0.96):
    return [left, 1 - (top_in + height_in) / H, width, height_in / H]


def heading(top_in: float, text: str, sub: str) -> None:
    y = 1 - top_in / H
    fig.text(0.02, y, text, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, y - 0.3 / H, sub, fontsize=11, color=INK2, va="top")
    fig.add_artist(plt.Line2D([0.02, 0.98], [y + 0.12 / H, y + 0.12 / H], color=AXIS, lw=1.2))


def table(ax, cols, rows, size=10.5, center_from=99, title=None):
    """cols: [(width, header)], rows: [[cell, ...]]; widths sum to 1."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    h = 1 / (len(rows) + 1)
    x = 0.0
    xs = []
    for w, t in cols:
        xs.append((x, w))
        ax.add_patch(Rectangle((x, 1 - h), w, h, fc=NEUTRAL, ec=AXIS, lw=0.8))
        ax.text(x + 0.008, 1 - h / 2, t, va="center", fontsize=size, fontweight="bold", color=INK)
        x += w
    for i, r in enumerate(rows):
        y = 1 - h * (i + 2)
        for j, ((x0, w), t) in enumerate(zip(xs, r)):
            ax.add_patch(Rectangle((x0, y), w, h, fc=SURF, ec=AXIS, lw=0.8))
            c = j >= center_from
            ax.text(x0 + (w / 2 if c else 0.008), y + h / 2, t, va="center", ha="center" if c else "left",
                    fontsize=size, color=INK)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)


def fmt(v: float) -> str:
    return f"{v:+.3f}"


def main() -> None:
    global fig
    Hs = pd.read_csv(O / "rtofs_history_20261004/history_scores.csv")
    RW = pd.read_csv(O / "rolling_year_windows_20261015/rolling_year_scores.csv")
    TS = pd.read_csv(O / "time_season_loyo_20261004/time_season_scores.csv")
    SS = pd.read_csv(O / "time_season_loyo_20261004/seasonal_share.csv")
    PF = pd.read_csv(O / "pairwise_decomposition_20261004/pairwise_family_scores.csv")
    fig = plt.figure(figsize=(W, H))
    fig.text(0.02, 1 - 0.3 / H, "Definitions of three feature groups used in the follow-up experiments",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.75 / H, "All changes are in mean absolute error against Argo (kJ/cm² for TCHP, m for D26); "
             "negative = better. Chance level from the noise-feature test: about ±0.03. Warm rows, one profile per "
             "Argo cast.", fontsize=11.5, color=INK2, va="top")

    # ================= A. recent history
    t = 1.5
    heading(t, "A. RTOFS recent history at the float: 12 extra features outside the 34/35-feature recipe",
            "For each profile, RTOFS TCHP, D26 and SST are read at the nearest grid cell on the profile day and the 7 "
            "days before (no 26 °C water counts as TCHP = 0). Only RTOFS up to the profile day is used.")
    table(fig.add_axes(ax_at(t + 1.0, 2.6, 0.02, 0.6)),
          [(0.36, "Feature (×3: TCHP, D26, SST)"), (0.44, "Meaning"), (0.20, "RTOFS days used")],
          [[n, m, d] for n, m, d, _, _ in HIST_ROWS], center_from=2)
    tl = fig.add_axes(ax_at(t + 1.15, 2.2, 0.66, 0.32))
    tl.set_xlim(-10.4, 0.8)
    tl.set_ylim(-0.6, len(HIST_ROWS) - 0.4)
    for i, (name, _, _, days, kind) in enumerate(HIST_ROWS):
        y = len(HIST_ROWS) - 1 - i
        for dd in range(8):
            col = (BLUE if kind == "mean" else ORANGE) if dd in days else NEUTRAL
            tl.add_patch(FancyBboxPatch((-dd - 0.4, y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.02", fc=col, ec=SURF))
        tl.text(-10.3, y, name.replace("rtofs_{var}_", ""), va="center", fontsize=10.5, color=INK2)
    tl.set_yticks([])
    tl.set_xticks(range(-7, 1))
    tl.set_xticklabels([f"d−{k}" if k else "d" for k in range(7, -1, -1)])
    tl.set_xlabel("RTOFS day (d = the profile's day)    blue: averaged, orange: differenced", fontsize=10.5)
    for s in ("left", "right", "top"):
        tl.spines[s].set_visible(False)
    tl.grid(False)
    rows = []
    for lab, des in (("forward test blocks", "forward blocks"), ("full-year tests (test 2024, test 2025)", "full-year tests")):
        r = [lab]
        for tn in ("tchp", "d26"):
            g = Hs[(Hs.target == tn) & (Hs.design == des)]
            b = g[g.variant.str.startswith("baseline")].iloc[0]
            a = g[g.variant == "+ all 12 history features"].iloc[0]
            per = [a[c] - b[c] for c in g.columns if c.startswith("fold: ") and pd.notna(a[c])]
            r.append(f"{a.mae - b.mae:+.2f}   (per block: " + ", ".join(f"{v:+.2f}" for v in per) + ")")
        rows.append(r)
    r = ["12-month windows starting Feb, May, Aug, Nov 2024, Jan 2025"]
    for tn in ("tchp", "d26"):
        p = RW[RW.target == tn].pivot_table(index="window", columns="variant", values="mae")
        d = p["full recipe + RTOFS recent history"] - p["full recipe"]
        r.append(f"{d.min():+.2f} to {d.max():+.2f} in all 5 windows")
    rows.append(r)
    table(fig.add_axes(ax_at(t + 3.85, 1.75)),
          [(0.36, "Adding the 12 features to the full recipe"), (0.32, "TCHP"), (0.32, "D26")], rows)

    # ================= B. seasonal features
    t = 7.95
    heading(t, "B. Seasonal features (time-and-season experiment)",
            "Two groups were removed from the full recipe, separately and together, to see whether the season reaches the "
            "model through the calendar or through RTOFS's own fields.")
    table(fig.add_axes(ax_at(t + 1.0, 4.2, 0.02, 0.36)), [(0.42, "Feature"), (0.58, "Definition")],
          [[a, b] for a, b in CALENDAR], title="Calendar features (9 values in 7 rows)")
    share = SS[SS.target == "tchp"].set_index("variable").seasonal_share_above_chance_pct
    table(fig.add_axes(ax_at(t + 1.0, 4.2, 0.40, 0.58)),
          [(0.47, "Feature"), (0.38, "Definition"), (0.15, "Seasonal share")],
          [[a, b, f"{share.get(a, float('nan')):.0f}%"] for a, b in SEASONAL], size=10, center_from=2,
          title="Seasonal physical features (10; 9 in the D26 recipe)")
    fig.text(0.40, 1 - (t + 5.35) / H, "Seasonal share: how much of the feature's variation the month explains once "
             "location (10° box) is known,\nminus chance. For comparison: the error itself is 5% seasonal (TCHP) and 9% (D26).",
             fontsize=10, color=INK2, va="top")
    rows = []
    for v, lab in (("remove calendar features", "Remove the calendar features"),
                   ("remove seasonal physical features", "Remove the seasonal physical features"),
                   ("remove both", "Remove both groups")):
        r = [lab]
        for tn in ("tchp", "d26"):
            for des in ("forward blocks (~5 months)", "full-year tests"):
                r.append(fmt(TS[(TS.target == tn) & (TS.variant == v) & (TS["test design"] == des)]["change mae"].iloc[0]))
        rows.append(r)
    table(fig.add_axes(ax_at(t + 5.85, 1.45)),
          [(0.36, "Change when removed (positive = removing hurts)"), (0.16, "TCHP, forward"), (0.16, "TCHP, full-year"),
           (0.16, "D26, forward"), (0.16, "D26, full-year")], rows, center_from=1)

    # ================= C. pairwise combinations
    t = 15.75
    heading(t, "C. Pairwise combinations",
            "New features made from two existing ones, added on top of the full recipe (calendar features are not combined).")
    steps = ("1.  Put every non-calendar feature on a common scale: z = (value − average) / spread, using training data "
             "only.\n2.  For every pair of features a, b add three new features: a + b,  a − b,  a × b.  Also add each "
             "feature squared, a².\n3.  TCHP: 25 features → 925 new ones;  D26: 26 features → 1,001 new ones.  Then "
             "retrain and score as usual.\n4.  Example: z(D26) − z(D26 1° anomaly),  z(SSH) − z(TCHP 1° anomaly),  "
             "z(lat) × z(mixed layer).")
    fig.text(0.03, 1 - (t + 1.0) / H, steps, fontsize=11, color=INK, va="top", linespacing=1.6)
    fam = [("+ location × location", "location × location", "lat, lon, |lat| with each other",
            "lat + lon,  lat − lon,  lat × lon", "9 / 9"),
           ("+ location × physics", "location × physics", "one location feature with one RTOFS feature",
            "|lat| − D26,  lat + SSH", "198 / 207"),
           ("+ physics × physics", "physics × physics", "two RTOFS features", "D26 − D26 anomaly,  SSH − TCHP anomaly",
            "693 / 759"),
           ("+ squares only", "squares", "each feature with itself", "SSH²,  lat²", "25 / 26"),
           ("+ all pairwise (reference)", "all of the above", "everything at once", "", "925 / 1,001")]
    rows = []
    for key, name, what, ex, n in fam:
        r = [name, what, ex, n]
        for tn in ("tchp", "d26"):
            for des in ("forward blocks", "full-year tests"):
                r.append(fmt(PF[(PF.target == tn) & (PF.design == des) & (PF.variant == key)].change_vs_baseline.iloc[0]))
        rows.append(r)
    table(fig.add_axes(ax_at(t + 2.45, 3.0)),
          [(0.14, "Family"), (0.2, "What is combined"), (0.2, "Examples"), (0.1, "Count TCHP / D26"),
           (0.09, "TCHP fwd"), (0.09, "TCHP full-yr"), (0.09, "D26 fwd"), (0.09, "D26 full-yr")],
          rows, size=10, center_from=3)
    fig.text(0.02, 1 - (t + 5.6) / H,
             "Why it helps: a tree splits one feature at a time, so a correction that depends on a difference (e.g. D26 "
             "minus its 1° anomaly) needs many small steps; giving it the difference saves them.\nThe step-by-step build-up "
             "refined this: pairwise differences of the 18 physical features alone give almost all of the gain (−0.3); "
             "products, sums and powers add nothing on top.", fontsize=10.5, color=INK2, va="top")
    fig.savefig(OUT, dpi=140, facecolor=SURF)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
