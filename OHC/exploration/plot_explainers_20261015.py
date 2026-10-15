"""Two explanation figures for the 2026-10-08 meeting questions.

E1  Why replacing features by squares or by distance from the median hurts the trees, while cube, exp,
    log and rank change nothing: one-to-one (order-keeping) transforms give a tree exactly the same
    possible splits; folding transforms put two different values on the same number. Illustrated with
    the two features that matter most: the TCHP 1-degree anomaly (its sign says which way to correct)
    and latitude (north and south differ).
E2  The "45 degree" question, read as: boundaries that run diagonally in lat/lon are awkward for trees,
    which split one axis at a time. Illustration, plus what adding diagonal combinations (lat+lon,
    lat-lon, lat x lon) and longitude as sin/cos actually did.

Outputs: OHC/output/explainers_20261015/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AQUA, AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, load_work  # noqa: E402

O = Path("/home/suramya/HHP-Prediction/OHC/output")
OUT = O / "explainers_20261015"


def e1() -> None:
    w = load_work("tchp")
    t = TARGET["tchp"]
    err = w[t.delta_col].to_numpy(float)
    E = pd.read_csv(O / "expanded_features_20261004/expanded_feature_scores.csv").set_index(["target", "variant"])
    ch = lambda tn, v: E.loc[(tn, f"every feature -> {v}"), "change_vs_baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    # A: one-to-one transforms keep the order
    ax = axes[0, 0]
    x = np.linspace(-2.5, 2.5, 200)
    for f, lab, col in ((lambda v: v ** 3 / 15.6, "cube", BLUE), (lambda v: (np.exp(v) - 1) / 11.2, "exponential", AQUA),
                        (lambda v: (np.searchsorted(np.sort(np.random.default_rng(0).standard_normal(5000)), v) / 5000 * 2 - 1),
                         "rank (quantile)", ORANGE)):
        ax.plot(x, f(x), color=col, lw=2.2, label=lab)
    t0 = 0.8
    ax.axvline(t0, color=INK2, ls="--", lw=1.2)
    ax.text(t0 - 0.08, -0.55, "a split\nx < 0.8", color=INK2, fontsize=10.5, ha="right")
    ax.annotate("", xy=(-2.4, 0.95), xytext=(t0 - 0.05, 0.95), arrowprops=dict(arrowstyle="<->", color=INK2))
    ax.text(-0.8, 1.0, "the same profiles fall below the split\nwhatever the transform", ha="center", va="bottom",
            fontsize=10, color=INK2)
    ax.set_xlabel("original feature value (standardised)")
    ax.set_ylabel("transformed value (rescaled)")
    ax.set_ylim(-1.15, 1.35)
    ax.legend(loc="lower right")
    ax.set_title(f"One-to-one transforms keep the order → the same tree\n(measured: cube {ch('tchp', 'cube'):+.3f}, "
                 f"exp {ch('tchp', 'exp of z-score'):+.3f}, rank {ch('tchp', 'rank (empirical quantile)'):+.3f} kJ/cm²)",
                 loc="left", fontsize=12)
    # B: folding transforms
    ax = axes[0, 1]
    ax.plot(x, x ** 2, color=BLUE, lw=2.2, label="square")
    ax.plot(x, np.abs(x - 0.3) * 2, color=ORANGE, lw=2.2, label="distance from the median (median = 0.3)")
    for a in (-1.6, 1.6):
        ax.plot(a, a ** 2, "o", color=BLUE, ms=10, mec=SURF)
        ax.vlines(a, 0, a ** 2, color=BLUE, ls=":", lw=1)
    ax.annotate("−1.6 and +1.6 both become 2.56:\nthe tree can no longer tell them apart", xy=(1.6, 2.56),
                xytext=(-1.3, 4.6), fontsize=10.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("original feature value (standardised)")
    ax.set_ylabel("transformed value")
    ax.set_ylim(-0.2, 6.6)
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title(f"Folding transforms put two different values on one number\n(measured: square {ch('tchp', 'square of z-score (not one-to-one)'):+.2f}, "
                 f"distance from median {ch('tchp', 'distance from median (not one-to-one)'):+.2f} kJ/cm²)",
                 loc="left", fontsize=12)
    # C: the anomaly's sign
    ax = axes[1, 0]
    P = w[["lat", "lon", "abs_lat"]].to_numpy(float)
    pos = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(P, err).predict(P)
    a = w["model_tchp_anom_from_1deg_mean"].to_numpy(float)
    r = err - pos
    ok = np.isfinite(a)
    a, r = a[ok], r[ok]
    bins = np.linspace(-15, 15, 31)
    c = 0.5 * (bins[1:] + bins[:-1])
    m = pd.Series(r).groupby(pd.cut(a, bins), observed=False).mean().to_numpy()
    ax.plot(c, m, "-o", color=BLUE, ms=5, mec=SURF, lw=2, label="average error correction needed, by anomaly")
    mf = pd.Series(r).groupby(pd.cut(np.abs(a), bins[15:]), observed=False).mean().to_numpy()
    ax.plot(c[15:], mf, "--", color=ORANGE, lw=2.2, label="what is left if only |anomaly| or anomaly² is known")
    ax.plot(-c[15:], mf, "--", color=ORANGE, lw=2.2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("RTOFS TCHP minus its 1° average at the float (kJ/cm²)")
    ax.set_ylabel("Argo − RTOFS (kJ/cm²),\nregional bias removed")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Example 1: the anomaly's sign says which way to correct\n(warmer spot → correct down, colder → up); "
                 "folding erases it", loc="left", fontsize=12)
    # D: latitude
    ax = axes[1, 1]
    lb = np.arange(0, 42, 2.0)
    cl = 0.5 * (lb[1:] + lb[:-1])
    lat = w["lat"].to_numpy(float)
    north = pd.Series(err[lat > 0]).groupby(pd.cut(lat[lat > 0], lb), observed=False).mean().to_numpy()
    south = pd.Series(err[lat < 0]).groupby(pd.cut(-lat[lat < 0], lb), observed=False).mean().to_numpy()
    both = pd.Series(err).groupby(pd.cut(np.abs(lat), lb), observed=False).mean().to_numpy()
    ax.plot(cl, north, "-o", color=BLUE, ms=5, mec=SURF, lw=2, label="northern hemisphere")
    ax.plot(cl, south, "-s", color=AQUA, ms=5, mec=SURF, lw=2, label="southern hemisphere")
    ax.plot(cl, both, "--", color=ORANGE, lw=2.2, label="what is left if only latitude² is known")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("distance from the equator (degrees)")
    ax.set_ylabel("Average Argo − RTOFS TCHP (kJ/cm²)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Example 2: north and south need different corrections;\nsquaring latitude merges them", loc="left",
                 fontsize=12)
    headline(fig, "Why squares and distance-from-median hurt the trees while cube, exp and rank change nothing",
             "A tree only asks 'is the value below a threshold?'. Any transform that keeps the order of the values (top "
             "left) allows exactly the same questions.\nSquares and distances fold the number line (top right), so values "
             "on opposite sides become identical. For our two most important features that throws away\nexactly what the "
             "correction depends on (bottom), hence +0.8 to +1.0 worse when every feature is folded.", top=0.86)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, hspace=0.38, wspace=0.18)
    fig.savefig(OUT / "E1_why_folding_transforms_hurt.png", dpi=150)
    plt.close(fig)


def e2() -> None:
    fig = plt.figure(figsize=(17, 7.2))
    rng = np.random.default_rng(1)
    pts = rng.uniform(-20, 20, (3000, 2))
    lab = (pts[:, 0] + pts[:, 1] > 0).astype(int)
    for k, (feats, title) in enumerate(((pts, "lat and lon only: depth-4 tree"),
                                        (np.c_[pts, pts[:, 0] + pts[:, 1]], "plus lat + lon: one split"))):
        ax = fig.add_axes([0.05 + 0.21 * k, 0.12, 0.18, 0.6])
        tr = DecisionTreeClassifier(max_depth=4 if k == 0 else 1).fit(feats, lab)
        gx, gy = np.meshgrid(np.linspace(-20, 20, 300), np.linspace(-20, 20, 300))
        G = np.c_[gx.ravel(), gy.ravel()]
        if k == 1:
            G = np.c_[G, G[:, 0] + G[:, 1]]
        z = tr.predict(G).reshape(gx.shape)
        ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=["#cde2fb", "#fbd5c4"])
        ax.plot([-20, 20], [20, -20], color=INK, lw=1.6, ls="--")
        ax.set_aspect("equal")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude" if k == 0 else "")
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.text(0.03, 0.03, "dashed: true 45° boundary", transform=ax.transAxes, fontsize=9.5, color=INK2)
    P = pd.read_csv(O / "pairwise_decomposition_20261004/pairwise_family_scores.csv")
    C = pd.read_csv(O / "missing_physics_20260925/candidate_tests.csv")
    N = pd.read_csv(O / "noise_feature_20261015/noise_summary.csv")
    ax = fig.add_axes([0.62, 0.12, 0.36, 0.6])
    rows = [("+ diagonal combinations\n(forward blocks)", "pair", "forward blocks"),
            ("+ diagonal combinations\n(full-year tests)", "pair", "full-year tests"),
            ("+ longitude sin/cos\n(forward blocks)", "sincos", None)]
    y = np.arange(len(rows))[::-1]
    band = 2 * N[N.design == "forward blocks"].val_mae_sd.mean()
    ax.axvspan(-band, band, color=NEUTRAL, zorder=0)
    ax.axvline(0, color=AXIS, lw=1)
    for (lab, kind, design), yy in zip(rows, y):
        for tn, col, dy in (("tchp", BLUE, 0.15), ("d26", ORANGE, -0.15)):
            if kind == "pair":
                v = P[(P.target == tn) & (P.design == design) & P.variant.str.contains("location × location")].change_vs_baseline.iloc[0]
            else:
                v = C[(C.target == tn) & C.variant.str.startswith("+ cyclic longitude")].change_vs_baseline.iloc[0]
            ax.barh(yy + dy, v, height=0.28, color=col, edgecolor=SURF, label=tn.upper() if yy == y[0] else None)
            ax.text(v + (0.002 if v >= 0 else -0.002), yy + dy, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=10, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.07, 0.04)
    ax.set_xlabel("Change in validation error (← better | worse →)")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, f"grey band: chance level ±{band:.3f}", transform=ax.transAxes, ha="right", fontsize=9.5,
            color=INK2)
    ax.set_title("What it did on our data", loc="left", fontsize=12)
    headline(fig, "Diagonal boundaries in lat/lon are awkward for trees, but ours did not need them",
             "Reading of the note 'only lat might have trouble at the 45° angle': a tree splits latitude or longitude "
             "one at a time, so a boundary running at 45° needs a staircase\nof splits (left), while a lat + lon feature "
             "draws it in one split (middle). On our data, adding the diagonal combinations lat + lon, lat − lon and "
             "lat × lon changed the\nerror within the chance level; longitude as sin/cos (which joins 180°E and 180°W) "
             "helped slightly, just above it.", top=0.84)
    fig.savefig(OUT / "E2_diagonal_boundaries_and_longitude.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    e1()
    e2()
