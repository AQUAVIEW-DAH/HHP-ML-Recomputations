"""Intuitive figures for the 2026-09-24 meeting follow-ups (one per item).

Reads the saved result tables where they exist; items 3/7 (seasonality),
5 (leaf sizes) and 6 (single-split strength) were first answered with
throwaway terminal snippets, so their computation is repeated here, with the
same rows (clean warm tables, one primary profile per cast) and the same
methods, so the figures and the numbers in the notes agree.

Figures (OHC/output/meeting_followups_figs_20261001/):
  01  items 1, 2, 8   removing grid distance / year / time features
  02  items 3, 7      seasons in RTOFS vs in its error
  03  item 4          how the data is split for testing
  04  item 5          rows per leaf: tree splits never drop data
  05  item 6          why the 1-degree anomaly is high in the trees
  06  items 9, 10     added features; the dateline seam
  07  item 10         El Nino could not be tested
  08  item 11         data coverage: RTOFS, GOFS, Argo
  09  item 11         GOFS vs RTOFS on the same day
  10  frozen recipe   current vs frozen, before the 2026 test
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, Wedge
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

ROOT = Path("/home/suramya/HHP-Prediction/OHC/output")
OUT = ROOT / "meeting_followups_figs_20261001"
GOFS_DIR = Path("/data/suramya/gofs31_analysis_fields")

# reference palette (dataviz skill, light mode)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, NEUTRAL = "#9ec5f4", "#f0efec"
TCOL = {"tchp": BLUE, "d26": ORANGE}
TMARK = {"tchp": "o", "d26": "D"}
TLAB = {"tchp": "TCHP (kJ/cm²)", "d26": "D26 (m)"}
DIVERGING = LinearSegmentedColormap.from_list("bluegrayred", ["#104281", "#3987e5", "#9ec5f4", NEUTRAL,
                                                            "#f4a3a2", "#e34948", "#9b1c1c"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 12.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def headline(fig, head: str, sub: str, top: float) -> None:
    fig.text(0.012, 0.985, head, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.012, 0.985 - 0.055 * (8 / fig.get_figheight()), sub, ha="left", va="top", fontsize=12, color=INK2)
    fig.subplots_adjust(top=top)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", name, flush=True)


@lru_cache(maxsize=1)
def _merged() -> pd.DataFrame:
    return _prepare_features(abl._merge_feature_tables())


def clean_rows(target) -> pd.DataFrame:
    df = _merged()
    w = df[df[target.obs_col].notna() & df[target.model_col].notna() & df["is_primary_profile"].astype(bool)].copy()
    w["err"] = w[target.obs_col] - w[target.model_col]
    return w.reset_index(drop=True)


def change_dots(ax, table: pd.DataFrame, labels: dict[str, str], noise: float, xlabel: str,
                legend_loc: str = "upper right") -> None:
    rows = [k for k in labels]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-noise, noise, color=NEUTRAL, zorder=0)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    for tn, dy in (("tchp", 0.17), ("d26", -0.17)):
        t = table[table.target == tn].set_index("variant")
        v = np.array([t.loc[r, "change_vs_baseline"] for r in rows])
        sd = np.array([t.loc[r, "seed_sd"] for r in rows])
        ax.errorbar(v, y + dy, xerr=sd, fmt=TMARK[tn], ms=8, color=TCOL[tn], ecolor=TCOL[tn], elinewidth=1.5,
                    capsize=0, mec=SURF, mew=1.5, label=tn.upper(), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[r] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.text(noise, y.max() + 0.62, "run-to-run noise", ha="right", va="bottom", fontsize=10.5, color=MUTED)
    ax.set_ylim(-0.7, y.max() + 0.9)
    ax.legend(loc=legend_loc, ncol=2)


# ------------------------------------------------------------------ 01
def fig_feature_removal() -> None:
    t = pd.read_csv(ROOT / "time_meta_ablation_20260925/time_meta_ablation.csv")
    labels = {
        "drop grid distance": "Remove grid distance",
        "drop year": "Remove year",
        "drop grid distance and year": "Remove both",
        "time = month_int only": "Time = month number (1-12) only",
        "time = month sin/cos only": "Time = month as sin/cos only",
        "time = day-of-year sin/cos only": "Time = day of year as sin/cos only",
        "month_int only, and drop grid distance": "Month number only, no grid distance",
        "no time features at all": "No time features at all",
    }
    noise = 2 * t[t.variant == "full recipe (baseline)"].seed_sd.max()
    fig, ax = plt.subplots(figsize=(12, 6.8))
    change_dots(ax, t, labels, noise, "Change in error (MAE) when the feature is removed     ← better  |  worse →")
    ax.set_xlim(-0.045, 0.045)
    ax.text(0.01, 0.02, "For scale: the model's error is about 10.8\n(raw RTOFS: 16.4 TCHP, 15.1 D26)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=MUTED)
    headline(fig, "Grid distance, year and the extra time features barely matter",
             "Change in error when each is removed or replaced (3 runs each; bars = spread between runs). "
             "Everything stays within ±0.03.", top=0.86)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.12)
    save(fig, "01_items1-2-8_removing_grid_year_time.png")


# ------------------------------------------------------------------ 02
def fig_seasons() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    notes = {}
    for row, target in zip(axes, TARGETS):
        w = clean_rows(target)
        g = w.groupby("month_int")
        ma, mr = g[target.obs_col].mean(), g[target.model_col].mean()
        q = g["err"].quantile([0.1, 0.5, 0.9]).unstack()
        me = g["err"].mean()
        swing, spread = me.max() - me.min(), g["err"].std().median()
        notes[target.name] = (ma.max() - ma.min(), mr.max() - mr.min(), swing, spread)
        m = ma.index.to_numpy()
        a = row[0]
        a.plot(m, ma, "-o", color=BLUE, lw=2, ms=6, mec=SURF, label="Argo (truth)")
        a.plot(m, mr, "-s", color=ORANGE, lw=2, ms=6, mec=SURF, label="RTOFS")
        a.text(m[-1] + 0.25, ma.iloc[-1], "Argo", color=INK2, va="center", fontsize=11)
        a.text(m[-1] + 0.25, mr.iloc[-1], "RTOFS", color=INK2, va="center", fontsize=11)
        a.set_title(f"{target.name.upper()}: monthly average", loc="left")
        a.set_ylabel(TLAB[target.name])
        a.set_ylim(min(ma.min(), mr.min()) - 1, max(ma.max(), mr.max()) + 0.25 * (ma.max() - mr.min()))
        a.legend(loc="upper left", ncol=2)
        b = row[1]
        b.fill_between(m, q[0.1], q[0.9], color=BLUE_LIGHT, alpha=0.55, lw=0, label="middle 80% of profiles")
        b.plot(m, me, "-o", color=BLUE, lw=2, ms=6, mec=SURF, label="average error")
        b.axhline(0, color=AXIS, lw=1.2)
        b.set_title(f"{target.name.upper()} error (Argo − RTOFS) by month", loc="left")
        b.set_ylabel(TLAB[target.name])
        b.set_ylim(q[0.1].min() - 3, q[0.9].max() * 1.3)
        b.legend(loc="upper left", ncol=2)
        b.text(0.99, 0.03, f"average moves {swing:.1f} over the year;\nprofiles scatter ±{spread:.0f} within any month",
               transform=b.transAxes, ha="right", va="bottom", fontsize=11, color=INK2)
        for ax in row:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(list("JFMAMJJASOND"))
    sw = notes["tchp"]
    headline(fig, "RTOFS already follows the seasons; its error hardly does",
             f"Left: TCHP swings {sw[0]:.1f} over the year in Argo and {sw[1]:.1f} in RTOFS, so the season reaches "
             "the model through the RTOFS value itself.\nRight: the error's monthly average moves little next to "
             "how much single profiles scatter, so a time feature has little left to explain.", top=0.87)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, hspace=0.32, wspace=0.27)
    save(fig, "02_items3-7_seasons_in_rtofs_vs_error.png")


# ------------------------------------------------------------------ 03
def fig_folds() -> None:
    target = TARGETS[0]
    w = clean_rows(target)
    ds = w["date"].dt.strftime("%Y%m%d")
    note = json.loads(abl.FOLD_PATH.read_text())
    folds = _build_forward_folds(sorted(ds.unique().tolist()), n_folds=note["n_folds"],
                                 embargo_dates=note["embargo_dates"])
    D = lambda s: pd.Timestamp(s)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for i, f in enumerate(folds):
        y = len(folds) - i + 0.6
        t0, t1 = D(f["train_dates"][0]), D(f["train_dates"][-1]) + pd.Timedelta(days=1)
        v0, v1 = D(f["val_dates"][0]), D(f["val_dates"][-1]) + pd.Timedelta(days=1)
        nval = int(ds.isin(set(f["val_dates"])).sum())
        ax.barh(y, (t1 - t0).days, left=t0, height=0.62, color=BLUE_LIGHT, edgecolor=SURF, lw=2)
        ax.barh(y, (v1 - v0).days, left=v0, height=0.62, color=ORANGE, edgecolor=SURF, lw=2)
        ax.text(t0 + (t1 - t0) / 2, y, f"train {(t1 - t0).days / 30.4:.1f} months", ha="center", va="center",
                fontsize=11, color=INK)
        ax.text(v0 + (v1 - v0) / 2, y, f"test {(v1 - v0).days / 30.4:.1f} mo\n{nval:,} profiles",
                ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
    h0, h1, e1 = D("2026-01-01"), D("2026-09-23"), D("2026-10-03")
    ax.barh(0.6, (h1 - h0).days, left=h0, height=0.62, color=MUTED, edgecolor=SURF, lw=2)
    ax.text(h0 + (h1 - h0) / 2, 0.6, "2026 holdout:\ntested once", ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    ax.barh(0.6, (e1 - h1).days, left=h1, height=0.62, color=AXIS, edgecolor=SURF, lw=2)
    ax.set_yticks([3.6, 2.6, 1.6, 0.6])
    ax.set_yticklabels(["Fold 1", "Fold 2", "Fold 3", "Final test"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(D("2024-01-01"), D("2026-11-01"))
    ax.axvline(D("2026-01-01"), color=AXIS, lw=1, ls=":")
    headline(fig, "Each test block is about 5 months, not a full year",
             "Every model trains only on earlier dates (light blue) and is tested on the next block (orange). "
             "The three blocks together cover\nSep 2024 – Dec 2025; Feb – Sep 2024 is only ever used for training. "
             "RTOFS data starts in Jan 2024, so two years leave room for only one full-year test.", top=0.8)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13)
    save(fig, "03_item4_how_the_data_is_split.png")


# ------------------------------------------------------------------ 04
def fig_leaves() -> None:
    target = TARGETS[0]
    w = clean_rows(target)
    cols = abl.FEATURE_SETS_BY_TARGET["tchp"]["global_pruned_plus_neighborhood"]
    X = w[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    m = abl._xgb_model()
    m.fit(X, w["err"].to_numpy())
    leaves = m.apply(X)
    sizes = np.concatenate([np.bincount(pd.factorize(leaves[:, t])[0]) for t in range(leaves.shape[1])])
    med, p5 = np.median(sizes), np.percentile(sizes, 5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.7, 1]})
    bins = np.logspace(0, np.log10(sizes.max() + 1), 45)
    a.hist(sizes, bins=bins, color=BLUE, edgecolor=SURF, lw=1)
    a.set_xscale("log")
    a.axvline(med, color=INK2, lw=1.5, ls="--")
    a.axvline(p5, color=ORANGE, lw=1.5, ls="--")
    ymax = a.get_ylim()[1]
    box = dict(boxstyle="round,pad=0.25", fc=SURF, ec="none")
    a.text(med * 1.1, ymax * 0.97, f"typical leaf\n{med:,.0f} profiles", color=INK2, fontsize=11, va="top", bbox=box)
    a.text(p5 * 1.1, ymax * 0.97, f"5% of leaves hold\n{p5:.0f} or fewer", color=ORANGE, fontsize=11, va="top",
           bbox=box)
    a.set_xlabel("Training profiles in a leaf (log scale)")
    a.set_ylabel("Number of leaves (all 300 trees)")
    a.set_title(f"{leaves.shape[0]:,} profiles × {leaves.shape[1]} trees: each profile sits in exactly one leaf "
                "of every tree", loc="left", fontsize=12.5)
    c = pd.read_csv(ROOT / "missing_physics_20260925/candidate_tests.csv")
    lab = {"min_child_weight = 20 (locked model allows 1)": "Leaves ≥ 20 profiles",
           "min_child_weight = 50 (locked model allows 1)": "Leaves ≥ 50 profiles",
           "min_child_weight = 100 (locked model allows 1)": "Leaves ≥ 100 profiles"}
    noise = 2 * c[c.variant.str.startswith("baseline")].seed_sd.max()
    change_dots(b, c, lab, noise, "Change in error   ← better | worse →", legend_loc="lower left")
    b.set_xlim(-0.04, 0.04)
    b.set_title("Forcing bigger leaves does not help", loc="left", fontsize=12.5)
    headline(fig, "Tree splits never drop data, but some leaves are small",
             "A split only sorts profiles into two groups; nothing is discarded. Most leaves rest on hundreds or "
             "thousands of profiles, a few on very few.", top=0.82)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, wspace=0.42)
    save(fig, "04_item5_tree_splits_keep_all_data.png")


# ------------------------------------------------------------------ 05
def fig_anomaly() -> None:
    target = TARGETS[0]
    w = clean_rows(target)
    cols = abl.FEATURE_SETS_BY_TARGET["tchp"]["global_pruned_plus_neighborhood"]
    X = w[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = w["err"].to_numpy()
    pos = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(X[["lat", "lon", "abs_lat"]], y)
    r = y - pos.predict(X[["lat", "lon", "abs_lat"]])

    def stump(c, tgt):
        t = DecisionTreeRegressor(max_depth=1).fit(X[[c]], tgt)
        return 100 * (np.var(tgt) - np.var(tgt - t.predict(X[[c]]))) / np.var(tgt)

    S = pd.DataFrame({"raw": {c: stump(c, y) for c in cols}, "after": {c: stump(c, r) for c in cols}})
    S["key"] = S[["raw", "after"]].max(axis=1)
    S = S.sort_values("key", ascending=False).head(12).sort_values("after")
    hl = "model_tchp_anom_from_1deg_mean"
    colors = [BLUE if c == hl else AXIS for c in S.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 7), sharey=True)
    for ax, col, ttl in ((a, "raw", "One split on the raw error"),
                         (b, "after", "One split after location (lat, lon, |lat|) is known")):
        ax.barh(np.arange(len(S)), S[col], color=colors, edgecolor=SURF, lw=2, height=0.72)
        for i, v in enumerate(S[col]):
            ax.text(v + 0.12, i, f"{v:.1f}%", va="center", fontsize=10.5, color=INK2)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("Share of TCHP error variance explained (%)" if col == "raw"
                      else "Share of the error left after location (%)")
        ax.grid(axis="y", visible=False)
    a.set_yticks(np.arange(len(S)))
    a.set_yticklabels([c.replace("model_", "").replace("_kj_per_cm2", "") for c in S.index])
    a.set_xlim(0, S.raw.max() * 1.25)
    b.set_xlim(0, S.raw.max() * 1.25)
    headline(fig, "The 1° anomaly is a weak first question but by far the best second one",
             "Trees first split on location (distance from the equator explains most). Within each region, how far "
             "RTOFS TCHP sits above or below\nits own 1° neighbourhood is the strongest remaining clue, which is "
             "why it appears high up in nearly every tree.", top=0.82)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.09, wspace=0.08)
    save(fig, "05_item6_why_the_anomaly_splits_early.png")


# ------------------------------------------------------------------ 06
def fig_new_features() -> None:
    c = pd.read_csv(ROOT / "missing_physics_20260925/candidate_tests.csv")
    c["variant"] = c.variant.where(~c.variant.str.startswith("+ top screened"), "+ top screened")
    c["variant"] = c.variant.where(~c.variant.str.startswith("+ family averages"), "+ family averages")
    lab = {"+ cyclic longitude (sin, cos)": "+ longitude as sin/cos",
           "+ Nino3.4 + cyclic longitude": "+ longitude sin/cos and El Niño index",
           "+ top screened": "+ best 4 screened sums/products*",
           "+ family averages": "+ family averages (anomaly, std, gradient)",
           "+ Nino3.4 index (from RTOFS SST)": "+ El Niño index (Niño 3.4)"}
    noise = 2 * c[c.variant.str.startswith("baseline")].seed_sd.max()
    fig = plt.figure(figsize=(15, 7))
    a = fig.add_axes([0.29, 0.12, 0.39, 0.66])
    change_dots(a, c, lab, noise, "Change in error   ← better  |  worse →", legend_loc="lower left")
    a.set_xlim(-0.08, 0.04)
    a.text(0, -0.17, "*picked using these same test blocks, so kept out of the final recipe",
           transform=a.transAxes, fontsize=10, color=MUTED)
    # the dateline seam, drawn
    s = fig.add_axes([0.71, 0.47, 0.27, 0.2])
    s.set_xlim(-180, 180)
    s.set_ylim(0, 1)
    s.add_patch(Rectangle((-180, 0.3), 80, 0.4, color=ORANGE))
    s.add_patch(Rectangle((160, 0.3), 20, 0.4, color=ORANGE))
    s.set_yticks([])
    s.set_xticks([-180, -100, 0, 180])
    s.set_xticklabels(["180°W", "100°W", "0°", "180°E"])
    s.text(170, 0.78, "160°E", ha="center", va="bottom", fontsize=10, color=MUTED)
    s.grid(False)
    for sp in ("left",):
        s.spines[sp].set_visible(False)
    s.set_title("Raw longitude: the East-Pacific region\nis two pieces at opposite ends", fontsize=11.5)
    cax = fig.add_axes([0.76, 0.06, 0.17, 0.3])
    cax.set_aspect("equal")
    cax.add_patch(plt.Circle((0, 0), 1, fill=False, color=AXIS, lw=6))
    cax.add_patch(Wedge((0, 0), 1.08, 160, 260, width=0.16, color=ORANGE))
    cax.set_xlim(-1.3, 1.3)
    cax.set_ylim(-1.3, 1.3)
    cax.axis("off")
    cax.text(0, 0, "sin/cos:\none piece", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK2)
    cax.text(1.12, 0, "0°", va="center", fontsize=10, color=MUTED)
    cax.text(-1.15, 0.12, "180°", ha="right", va="center", fontsize=10, color=MUTED)
    headline(fig, "Longitude as sin/cos is the one clear gain from new features",
             "Change in error when each candidate is added (3 runs each). Sin/cos lets the model treat 179°E and "
             "179°W as neighbours, so the\nEast-Pacific region, which crosses the dateline, is no longer split in two. "
             "The El Niño index and family averages did not help.", top=0.83)
    save(fig, "06_items9-10_new_features_and_dateline.png")


# ------------------------------------------------------------------ 07
def fig_enso() -> None:
    n = pd.read_csv(ROOT / "missing_physics_20260925/nino34_rtofs.csv", dtype={"date": str})
    n["t"] = pd.to_datetime(n.date)
    p = pd.read_csv(ROOT / "missing_physics_20260925/tchp_pacific_residual_by_month.csv", dtype={"month": str})
    p["t"] = pd.to_datetime(p.month, format="%Y%m") + pd.Timedelta(days=14)
    d = pd.read_csv(ROOT / "missing_physics_20260925/pacific_residual_drift.csv")
    d = d[(d.target == "tchp") & d.validates.notna()]
    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    for _, r in d.iterrows():
        v0, v1 = (pd.Timestamp(x) for x in r.validates.split("-"))
        for ax in (a, b):
            ax.axvspan(v0, v1, color=ORANGE, alpha=0.09, lw=0)
        a.text(v0 + (v1 - v0) / 2, 2.55, f"test block {int(r.fold)}\naverage {r.mean_nino34_anom:+.2f}",
               ha="center", va="top", fontsize=10.5, color=INK2)
    a.plot(n.t, n.nino34_anom, color=BLUE, lw=1.6)
    for v, txt in ((0.5, "El Niño threshold (+0.5)"), (-0.5, "La Niña threshold (−0.5)")):
        a.axhline(v, color=MUTED, lw=1, ls="--")
        a.text(n.t.iloc[-1], v + (0.08 if v > 0 else -0.08), txt, ha="right", va="bottom" if v > 0 else "top",
               fontsize=10, color=MUTED)
    a.axhline(0, color=AXIS, lw=1)
    a.set_ylabel("Niño 3.4 anomaly (°C)")
    a.set_title("El Niño index from RTOFS sea-surface temperature (relative to the 2024–25 average)", loc="left")
    a.set_ylim(min(-1.3, n.nino34_anom.min() - 0.15), 2.9)
    b.bar(p.t, p.east, width=24, color=[ORANGE if v > 0 else BLUE for v in p.east], edgecolor=SURF)
    b.axhline(0, color=AXIS, lw=1)
    b.set_ylabel("kJ/cm²")
    b.set_title("East tropical Pacific: corrected TCHP minus Argo, monthly average (above zero = over-corrected)",
                loc="left")
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    r_e = pd.read_csv(ROOT / "missing_physics_20260925/pacific_residual_drift.csv")
    r_e = float(r_e[(r_e.target == "tchp") & r_e.validates.isna()].east_pacific_mean_residual.iloc[0])
    headline(fig, "The El Niño idea cannot be tested yet: the test periods saw no real El Niño",
             "The only strong signal, early 2024 (end of the 2023–24 El Niño), falls in the training-only months; "
             "the test blocks hold only a brief, weak La Niña.\nThe east-Pacific over-correction appears in almost "
             f"every month and does not follow the index (monthly correlation {r_e:+.2f}), so something real is "
             "missing, but\ntwo years of RTOFS cannot say whether it is ENSO.", top=0.83)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, hspace=0.28)
    save(fig, "07_item10_el_nino_untestable.png")


# ------------------------------------------------------------------ 08
def fig_coverage() -> None:
    D = pd.Timestamp
    n_gofs = len([f for f in os.listdir(GOFS_DIR) if f.endswith(".nc")]) if GOFS_DIR.exists() else 0
    rows = [  # label, [(start, end, kind, text)]
        ("Argo floats", [(D("2014-01-01"), D("2026-10-03"), "avail", ""),
                         (D("2015-01-01"), D("2016-01-01"), "have", "2015"),
                         (D("2020-01-01"), D("2025-01-01"), "have", "2020–2024 downloaded"),
                         (D("2025-01-01"), D("2026-01-01"), "have", "2025"),
                         (D("2026-01-01"), D("2026-10-03"), "have", "2026")]),
        ("RTOFS (our target model)", [(D("2024-01-27"), D("2026-10-03"), "have", "Jan 2024 → now")]),
        ("GOFS 3.1 analysis", [(D("2014-07-01"), D("2024-09-05"), "avail", ""),
                               (D("2021-09-05"), D("2024-09-05"), "get", f"downloading: {n_gofs:,} of 1,096 days")]),
        ("GOFS 3.1 reanalysis", [(D("2014-01-01"), D("2016-01-01"), "avail", ""),
                                 (D("2015-01-01"), D("2016-01-01"), "have", "2015 pilot")]),
        ("ESPC-D-V02 (Navy, newer)", [(D("2024-08-10"), D("2026-10-03"), "avail", "")]),
    ]
    style = {"avail": dict(color=NEUTRAL, edgecolor=AXIS, lw=1), "have": dict(color=BLUE, edgecolor=SURF, lw=2),
             "get": dict(color=AQUA, edgecolor=SURF, lw=2)}
    fig, ax = plt.subplots(figsize=(15, 6.4))
    enso = [(D("2015-03-01"), D("2016-05-01"), "El Niño"), (D("2020-08-01"), D("2023-02-01"), "La Niña"),
            (D("2023-06-01"), D("2024-05-01"), "El Niño"), (D("2024-12-01"), D("2025-04-01"), "weak\nLa Niña")]
    for s, e, txt in enso:
        ax.axvspan(s, e, color=ORANGE if "Niño" in txt else BLUE, alpha=0.08, lw=0)
        ax.text(s + (e - s) / 2, len(rows) + 0.15, txt, ha="center", va="bottom", fontsize=10, color=INK2)
    ax.axvspan(D("2024-01-27"), D("2024-09-05"), color=AQUA, alpha=0.13, lw=0)
    ax.text(D("2024-05-15"), -0.65, "GOFS–RTOFS\noverlap", ha="center", va="top", fontsize=10, color=INK2)
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        for s, e, kind, txt in segs:
            ax.barh(y, (e - s).days, left=s, height=0.56, **style[kind])
            if txt:
                ax.text(s + (e - s) / 2, y, txt, ha="center", va="center", fontsize=10,
                        color="white" if kind != "avail" else INK2, fontweight="bold" if kind != "avail" else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(D("2014-01-01"), D("2026-12-31"))
    ax.set_ylim(-1.2, len(rows) + 0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(**style["have"], label="downloaded"), Patch(**style["get"], label="downloading now"),
                       Patch(**style["avail"], label="exists, not downloaded")],
              loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08))
    headline(fig, "RTOFS only exists from 2024; GOFS adds three earlier years with real El Niño and La Niña",
             "The GOFS download (Sep 2021 – Sep 2024) overlaps RTOFS for 7 months, so the two models' errors can be "
             "compared on the same floats.\nGOFS reanalysis runs 1994–2015; ENSO periods are approximate (NOAA ONI).",
             top=0.82)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17)
    save(fig, "08_item11_data_coverage_rtofs_gofs_argo.png")


# ------------------------------------------------------------------ 09
def fig_gofs_vs_rtofs(date: str = "20240615") -> None:
    g = xr.open_dataset(GOFS_DIR / f"gofs31a_fields_{date}.nc")
    r = xr.open_dataset(f"/data/suramya/rtofs_global_ohc_fields_2024/rtofs_tchp_{date}.nc")
    res = 0.5
    lat_e = np.arange(-42, 48 + res, res)
    lon_e = np.arange(-180, 180 + res, res)

    def binned(lat, lon, v):
        lat, lon, v = (np.asarray(x, float).ravel() for x in (lat, lon, v))
        lon = ((lon + 180) % 360) - 180
        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(v) & (lat >= lat_e[0]) & (lat < lat_e[-1])
        iy = ((lat[ok] - lat_e[0]) / res).astype(int)
        ix = np.clip(((lon[ok] + 180) / res).astype(int), 0, len(lon_e) - 2)
        k = iy * (len(lon_e) - 1) + ix
        sz = (len(lat_e) - 1) * (len(lon_e) - 1)
        s, c = np.bincount(k, v[ok], sz), np.bincount(k, None, sz)
        with np.errstate(invalid="ignore"):
            return (s / np.where(c > 0, c, np.nan)).reshape(len(lat_e) - 1, len(lon_e) - 1)

    G = binned(g.Latitude.values, g.Longitude.values, g.tchp_kj_per_cm2.values)
    R = binned(r.Latitude.values, r.Longitude.values, r.tchp_kj_per_cm2.values)
    dlt = G - R
    fig, ax = plt.subplots(figsize=(15, 6.2))
    lim = 40
    im = ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(dlt), cmap=DIVERGING,
                       norm=TwoSlopeNorm(0, -lim, lim), shading="flat", rasterized=True)
    ax.set_facecolor("#e9e8e3")
    ax.grid(False)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-40, 46)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="both")
    cb.set_label("GOFS − RTOFS TCHP (kJ/cm²)")
    ok = np.isfinite(dlt)
    headline(fig, "On the same day, GOFS and RTOFS disagree on TCHP by tens of kJ/cm² in places",
             f"GOFS minus RTOFS TCHP on 15 Jun 2024, 06Z, in 0.5° boxes (average {np.nanmean(dlt):+.1f} kJ/cm², "
             f"but both signs are common; red = GOFS higher; grey = no 26 °C water).\nAgainst Argo, RTOFS runs about "
             "11 too low and GOFS is close to unbiased; whether their errors resemble each other is what the overlap "
             "months will answer.",
             top=0.83)
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.11)
    save(fig, "09_item11_gofs_minus_rtofs_tchp_20240615.png")


# ------------------------------------------------------------------ 10
def fig_frozen() -> None:
    cur = pd.read_csv(ROOT / "moe_clean_20260925/clean_benchmark.csv")
    frz = pd.read_csv(ROOT / "frozen_recipe_dev_20260925/frozen_recipe_dev_scores.csv")
    pick = lambda t, tn, key: t[(t.target == tn) & t.model.str.contains(key, regex=False)].iloc[0]
    groups = [("TCHP\nall regions", "tchp", "mae"), ("TCHP\nGulf of Mexico", "tchp", "mae_gulf"),
              ("D26\nall regions", "d26", "mae"), ("D26\nGulf of Mexico", "d26", "mae_gulf")]
    series = [("Raw RTOFS", AXIS, lambda tn: pick(cur, tn, "raw RTOFS")),
              ("Current MoE", BLUE_LIGHT, lambda tn: pick(cur, tn, "MoE")),
              ("Frozen MoE (for the 2026 test)", BLUE, lambda tn: pick(frz, tn, "MoE"))]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    x = np.arange(len(groups))
    wd = 0.26
    for j, (lab, col, get) in enumerate(series):
        vals = [get(tn)[key] for _, tn, key in groups]
        bars = ax.bar(x + (j - 1) * wd, vals, width=wd - 0.02, color=col, edgecolor=SURF, lw=2, label=lab)
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Error (MAE): kJ/cm² for TCHP, m for D26")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 20.5)
    headline(fig, "The frozen recipe for the 2026 test scores as well as or better than the current one",
             "Same development test blocks. Frozen = no year, no grid distance, no deep-profile features, plus "
             "longitude as sin/cos.\nIt is fixed before 2026 is ever scored.", top=0.82)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12)
    save(fig, "10_frozen_recipe_vs_current.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for k, f in (("01", fig_feature_removal), ("02", fig_seasons), ("03", fig_folds), ("04", fig_leaves),
                 ("05", fig_anomaly), ("06", fig_new_features), ("07", fig_enso), ("08", fig_coverage),
                 ("09", fig_gofs_vs_rtofs), ("10", fig_frozen)):
        if not only or k in only:
            f()


if __name__ == "__main__":
    main()
