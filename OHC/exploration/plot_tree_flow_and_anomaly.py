"""Two explanatory figures for the 2026-09-24 meeting questions.

1. Tree flow: one real tree (the first of 300) of the TCHP model fitted on all
   clean rows, drawn as a flow chart with how many profiles take each branch.
   Every profile ends in exactly one leaf, so the leaf counts add back up to
   the total. The figure also states the three places where data are set
   aside or reduced: row/column sampling per tree, median filling of missing
   values, and thin leaves.
2. Anomaly slope: why `model_*_anom_from_1deg_mean` sits so high in the trees.
   x = RTOFS's departure from its own 1-degree neighbourhood mean at the
   float; y = Argo's departure from that same RTOFS neighbourhood mean, with
   the regional bias removed (a depth-6 tree on lat, lon, |lat| fitted to the
   error). Slope 1 would mean RTOFS's small-scale detail is right; slope 0
   that it carries no information. Checked separately well inside warm water
   and near the 26 C edge, where the neighbourhood mean is known to be biased.

Outputs: OHC/output/tree_flow_anomaly_20261004/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, BLUE_LIGHT, GRID, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, load_work, recipe  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/tree_flow_anomaly_20261004")
SHORT = lambda c: (c.replace("model_", "").replace("_kj_per_cm2", "").replace("_per_100km", "")
                   .replace("_anom_from_1deg_mean", " 1° anomaly").replace("interp_", "RTOFS "))


def tree_flow() -> None:
    tn = "tchp"
    t = TARGET[tn]
    work = load_work(tn)
    cols = recipe(tn, work)
    raw = work[cols].apply(pd.to_numeric, errors="coerce")
    imputed = raw.isna().mean() * 100
    X = raw.fillna(raw.median())
    err = work[t.delta_col].to_numpy(float)
    m = abl._xgb_model()
    m.fit(X.to_numpy(), err)
    T = m.get_booster().trees_to_dataframe()
    T = T[T.Tree == 0].set_index("Node")
    leaf_of = m.apply(X.to_numpy())[:, 0]
    count = pd.Series(leaf_of).value_counts().to_dict()
    mean_err = pd.Series(err).groupby(leaf_of).mean().to_dict()

    def n_of(node):
        r = T.loc[node]
        if r.Feature == "Leaf":
            return count.get(node, 0)
        return n_of(int(r.Yes.split("-")[1])) + n_of(int(r.No.split("-")[1]))

    pos, leaves = {}, []

    def layout(node, depth):
        r = T.loc[node]
        if r.Feature == "Leaf":
            leaves.append(node)
            pos[node] = (len(leaves) - 1, depth)
            return
        a, b = int(r.Yes.split("-")[1]), int(r.No.split("-")[1])
        layout(a, depth + 1)
        layout(b, depth + 1)
        pos[node] = ((pos[a][0] + pos[b][0]) / 2, depth)

    layout(0, 0)
    total = n_of(0)
    maxd = max(d for _, d in pos.values())
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_axes([0.01, 0.27, 0.98, 0.56])
    ax.axis("off")
    X0 = lambda x: x / max(len(leaves) - 1, 1)
    Y0 = lambda d: 1 - d / maxd
    for node, r in T.iterrows():
        if r.Feature == "Leaf":
            continue
        for child, side in ((int(r.Yes.split("-")[1]), "yes"), (int(r.No.split("-")[1]), "no")):
            n = n_of(child)
            ax.plot([X0(pos[node][0]), X0(pos[child][0])], [Y0(pos[node][1]), Y0(pos[child][1])],
                    color=BLUE_LIGHT, lw=1 + 18 * n / total, solid_capstyle="round", zorder=1)
    for node, r in T.iterrows():
        x, y = X0(pos[node][0]), Y0(pos[node][1])
        n = n_of(node)
        if r.Feature == "Leaf":
            txt = f"{n:,}\n{mean_err.get(node, np.nan):+.0f}"
            ax.text(x, y - 0.02, txt, ha="center", va="top", fontsize=9.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=NEUTRAL, ec=AXIS, lw=0.8))
        else:
            f = cols[int(r.Feature[1:])]
            ax.text(x, y, f"{SHORT(f)}\n< {r.Split:.3g}?\n{n:,} profiles", ha="center", va="center", fontsize=9.5,
                    color=INK, bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=BLUE, lw=1.2), zorder=3)
    ax.text(0, 1.07, "yes ← | → no at every question", fontsize=10.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.5, -0.1, f"Leaves (bottom): number of profiles, and their average error Argo − RTOFS (kJ/cm²). "
            f"The {len(leaves)} leaf counts add up to {sum(count.values()):,} = all {total:,} profiles.",
            ha="center", fontsize=11.5, color=INK2, transform=ax.transAxes)
    miss = imputed[imputed > 0].sort_values(ascending=False)
    notes = ("Where data are set aside or reduced, and why it is not lost:\n"
             f"•  Each tree trains on a random 80% of profiles and 80% of features (subsample, colsample = 0.8). "
             f"Over 300 trees each profile is used about {0.8 * 300:.0f} times, so all of it is used.\n"
             f"•  Missing values are filled with the median before the trees see them: the profile stays, but 'this was "
             f"missing' is lost. Largest: {', '.join(f'{SHORT(c)} {v:.1f}%' for c, v in miss.head(3).items())}.\n"
             "•  Deeper splits mean fewer profiles per leaf, so a deep leaf's estimate is noisier. That, not lost data, "
             "is the real cost of splitting.")
    fig.text(0.012, 0.03, notes, fontsize=11.5, color=INK2, va="bottom")
    headline(fig, "A tree split never drops a profile: it only sends it left or right",
             f"The first of the 300 trees in the TCHP model, applied to all {total:,} profiles. Line width = how many "
             "profiles take that branch; every profile ends in exactly one leaf.", top=0.84)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tree_flow_every_profile_lands_in_a_leaf.png", dpi=150)
    plt.close(fig)
    print("tree flow done", flush=True)


def anomaly_slope() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    slopes = []
    for ax, tn in zip(axes[:2], ("tchp", "d26")):
        t = TARGET[tn]
        work = load_work(tn)
        anom = work[f"model_{tn}_anom_from_1deg_mean"].to_numpy(float)
        err = work[t.delta_col].to_numpy(float)
        P = work[["lat", "lon", "abs_lat"]].to_numpy(float)
        pos = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(P, err).predict(P)
        y = (err - pos) + anom            # Argo's departure from the RTOFS 1-degree mean, regional bias removed
        ok = np.isfinite(anom) & np.isfinite(y)
        lo, hi = np.percentile(anom[ok], [1, 99])
        ok &= (anom >= lo) & (anom <= hi)
        x, yy = anom[ok], y[ok]
        groups = {"all profiles": np.ones(ok.sum(), bool)}
        tchp_m = work["model_interp_tchp_kj_per_cm2"].to_numpy(float)[ok]
        te = work["model_temp_excess_26c"].to_numpy(float)[ok]
        groups["well inside warm water"] = (tchp_m >= 40) & (te >= 1)
        groups["near the 26 °C edge"] = tchp_m < 15
        for g, mk in groups.items():
            s = np.polyfit(x[mk], yy[mk], 1)[0]
            slopes.append({"target": tn, "group": g, "slope": s, "n": int(mk.sum())})
        q = pd.qcut(x, 20, duplicates="drop")
        g = pd.DataFrame({"x": x, "y": yy}).groupby(q, observed=True)
        mx, my, se = g.x.mean(), g.y.mean(), g.y.std() / np.sqrt(g.size())
        ax.errorbar(mx, my, yerr=1.96 * se, fmt="o", color=BLUE, ms=6, mec=SURF, elinewidth=1.4, zorder=4,
                    label="average in 20 bins")
        lim = max(abs(lo), abs(hi))
        ax.plot([-lim, lim], [-lim, lim], color=ORANGE, lw=1.6, ls="--", label="slope 1: RTOFS detail fully right")
        ax.axhline(0, color=MUTED, lw=1.2, ls=":", label="slope 0: RTOFS detail carries nothing")
        s_all = [r["slope"] for r in slopes if r["target"] == tn and r["group"] == "all profiles"][0]
        ax.plot([-lim, lim], [-lim * s_all, lim * s_all], color=BLUE, lw=2, label=f"fitted slope {s_all:.2f}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        unit = "kJ/cm²" if tn == "tchp" else "m"
        ax.set_xlabel(f"RTOFS at the float minus RTOFS 1° mean ({unit})")
        ax.set_ylabel(f"Argo minus RTOFS 1° mean ({unit})")
        ax.set_title(tn.upper(), loc="left")
        ax.legend(loc="upper left", fontsize=9.5)
    S = pd.DataFrame(slopes)
    S.to_csv(OUT / "anomaly_slopes.csv", index=False)
    b = axes[2]
    gs = ["all profiles", "well inside warm water", "near the 26 °C edge"]
    for j, (tn, col) in enumerate((("tchp", BLUE), ("d26", ORANGE))):
        v = [S[(S.target == tn) & (S.group == g)].slope.iloc[0] for g in gs]
        yv = np.arange(len(gs))[::-1] + (0.18 if j == 0 else -0.18)
        b.barh(yv, v, height=0.34, color=col, edgecolor=SURF, lw=2, label=tn.upper())
        for yy_, vv in zip(yv, v):
            b.text(vv + 0.01, yy_, f"{vv:.2f}", va="center", fontsize=10.5, color=INK2)
    b.set_yticks(np.arange(len(gs))[::-1])
    b.set_yticklabels(["all profiles", "well inside\nwarm water", "near the\n26 °C edge"])
    b.set_xlim(0, 1)
    b.axvline(1, color=ORANGE, lw=1.2, ls="--")
    b.grid(axis="y", visible=False)
    b.set_xlabel("Fitted slope (1 = RTOFS detail fully right)")
    b.set_title("Not an edge effect", loc="left")
    b.legend(loc="lower right")
    s_t = S[(S.target == "tchp") & (S.group == "all profiles")].slope.iloc[0]
    s_in = S[(S.target == "tchp") & (S.group == "well inside warm water")].slope.iloc[0]
    headline(fig, f"Only about a quarter of RTOFS's small-scale detail at a float shows up in Argo",
             "When RTOFS shows a float's spot warmer (or deeper) than its own 1° surroundings, Argo agrees with only "
             f"about {100 * s_t:.0f}% of that difference ({100 * s_in:.0f}% well inside warm water for TCHP, so it is "
             "not an\nedge artifact). Points: averages in 20 bins with 95% intervals; regional bias removed. Undoing the "
             "rest is the model's main job after the regional bias, which is why\nthe 1° anomaly is asked so early in "
             "the trees. Likely cause: RTOFS's eddies and fronts are often slightly misplaced or mistimed.", top=0.78)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.12, wspace=0.42)
    fig.savefig(OUT / "anomaly_slope_rtofs_detail_vs_argo.png", dpi=150)
    plt.close(fig)
    print(S.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tree_flow()
    anomaly_slope()
