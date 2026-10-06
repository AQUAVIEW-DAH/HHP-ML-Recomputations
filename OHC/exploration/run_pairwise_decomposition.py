"""Where does the gain from adding all pairwise feature combinations come from?

run_expanded_features.py found that adding every pairwise product, sum and
difference of the z-scored features (plus squares) lowers the TCHP error
clearly, with no selection on test data. Candidate explanations:

  * geometry: sums and differences of lat, lon and |lat| give the trees
    diagonal boundaries, which axis-aligned splits on lat and lon alone can
    only approximate with many steps;
  * location x physics: a physical feature whose meaning changes with place;
  * physics x physics: genuine interactions between RTOFS fields.

Each family of candidates is added on its own, scored like the other
follow-ups (three seeds; forward blocks and full-year tests), and the model
with all candidates is inspected for which features carry its splits (total
gain per feature family, fitted on all rows).

Outputs: OHC/output/pairwise_decomposition_20261004/
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AQUA, AXIS, BLUE, INK2, ORANGE, SURF, change_dots, headline)
from OHC.exploration.rtofs_followup_common import (  # noqa: E402
    TARGET, TIME_COLS, evaluate, forward_folds, load_work, loyo_folds, median_X, recipe)
from OHC.exploration.run_expanded_features import base_matrix  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/pairwise_decomposition_20261004")
LOC = ["lat", "lon", "abs_lat"]
FAMILIES = {"location × location": "LL", "location × physics": "LP",
            "physics × physics": "PP", "squares only": "SQ", "all pairwise (reference)": "ALL"}


def family_of(a: str, b: str | None) -> str:
    if b is None:
        return "SQ"
    na, nb = a in LOC, b in LOC
    return "LL" if na and nb else "PP" if not (na or nb) else "LP"


def build(Z: np.ndarray, names: list[str], fam: str):
    out, nm, fm = [], [], []
    for i, j in itertools.combinations(range(Z.shape[1]), 2):
        f = family_of(names[i], names[j])
        if fam in (f, "ALL"):
            out += [Z[:, i] * Z[:, j], Z[:, i] + Z[:, j], Z[:, i] - Z[:, j]]
            nm += [f"{names[i]} * {names[j]}", f"{names[i]} + {names[j]}", f"{names[i]} - {names[j]}"]
            fm += [f] * 3
    if fam in ("SQ", "ALL"):
        for i in range(Z.shape[1]):
            out.append(Z[:, i] ** 2)
            nm.append(f"{names[i]}^2")
            fm.append("SQ")
    return np.column_stack(out).astype(np.float32), nm, fm


def family_X(cols: list[str], fam: str):
    phys = [c for c in cols if c not in TIME_COLS]

    def make(work, tr, va):
        Xtr, Xva = base_matrix(work, cols, tr, va)
        P = [cols.index(c) for c in phys]
        mu, sd = Xtr[:, P].mean(0), Xtr[:, P].std(0) + 1e-9
        Ctr, _, _ = build((Xtr[:, P] - mu) / sd, phys, fam)
        Cva, _, _ = build((Xva[:, P] - mu) / sd, phys, fam)
        return np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva])
    return make


def gain_by_family(work, tn, cols) -> pd.DataFrame:
    t = TARGET[tn]
    phys = [c for c in cols if c not in TIME_COLS]
    allm = np.ones(len(work), bool)
    X, _ = base_matrix(work, cols, allm, allm)
    P = [cols.index(c) for c in phys]
    Z = (X[:, P] - X[:, P].mean(0)) / (X[:, P].std(0) + 1e-9)
    C, nm, fm = build(Z, phys, "ALL")
    m = abl._xgb_model()
    m.fit(np.hstack([X, C]), work[t.delta_col].to_numpy(float))
    g = m.get_booster().get_score(importance_type="total_gain")
    names = cols + nm
    fams = ["original: location" if c in LOC else "original: calendar" if c in TIME_COLS else "original: physics"
            for c in cols] + fm
    rows = [{"feature": names[int(k[1:])], "family": fams[int(k[1:])], "total_gain": v} for k, v in g.items()]
    return pd.DataFrame(rows).sort_values("total_gain", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res, gains = [], []
    for tn in ("tchp", "d26"):
        work = load_work(tn)
        cols = recipe(tn, work)
        for design, folds in (("forward blocks", forward_folds(work)), ("full-year tests", loyo_folds(work))):
            base = evaluate(work, tn, folds, median_X(cols))
            res.append({"target": tn, "design": design, "variant": "baseline (full recipe)", **base})
            for label, fam in FAMILIES.items():
                r = evaluate(work, tn, folds, family_X(cols, fam))
                res.append({"target": tn, "design": design, "variant": "+ " + label, **r})
                print(tn, design, label, round(r["mae"] - base["mae"], 4), flush=True)
        g = gain_by_family(work, tn, cols)
        g["target"] = tn
        gains.append(g)
    R = pd.DataFrame(res)
    b = R[R.variant.str.startswith("baseline")].set_index(["target", "design"])["mae"]
    R["change_vs_baseline"] = R.apply(lambda r: r.mae - b[(r.target, r.design)], axis=1)
    R.to_csv(OUT / "pairwise_family_scores.csv", index=False)
    G = pd.concat(gains)
    G.to_csv(OUT / "all_pairwise_model_gain_by_feature.csv", index=False)
    plot(R, G)


def plot(R: pd.DataFrame, G: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(17, 10.5))
    labels = {"+ " + k: "+ " + k for k in FAMILIES}
    axes = [fig.add_axes([0.25, 0.47, 0.34, 0.33]), fig.add_axes([0.63, 0.47, 0.34, 0.33])]
    for ax, design in zip(axes, ("forward blocks", "full-year tests")):
        t = R[R.design == design]
        noise = 2 * t[t.variant.str.startswith("baseline")].seed_sd.max()
        change_dots(ax, t, labels, noise, "Change in error   ← better | worse →", legend_loc="upper left")
        ax.set_title({"forward blocks": "Forward test blocks", "full-year tests": "Full-year tests"}[design], loc="left")
    axes[1].set_yticklabels([])
    lo = min(-0.05, R.change_vs_baseline.min() - 0.02)
    for ax in axes:
        ax.set_xlim(lo, max(0.03, R.change_vs_baseline.max() + 0.01))
    # which families carry the all-candidates model
    fam_order = ["original: location", "original: physics", "original: calendar", "LL", "LP", "PP", "SQ"]
    fam_name = {"LL": "location × location", "LP": "location × physics", "PP": "physics × physics", "SQ": "squares"}
    for k, (tn, col) in enumerate((("tchp", BLUE), ("d26", ORANGE))):
        ax = fig.add_axes([0.25 + 0.38 * k, 0.07, 0.34, 0.27])
        g = G[G.target == tn].groupby("family").total_gain.sum()
        share = 100 * g.reindex(fam_order).fillna(0) / g.sum()
        y = np.arange(len(fam_order))[::-1]
        ax.barh(y, share.values, color=col, edgecolor=SURF, lw=2, height=0.7)
        for yy, v in zip(y, share.values):
            ax.text(v + 0.5, yy, f"{v:.0f}%", va="center", fontsize=10.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([fam_name.get(f, f) for f in fam_order] if k == 0 else [])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, share.max() * 1.25)
        ax.set_xlabel("Share of the model's split gain (%)")
        top = [f.replace("model_", "").replace("_kj_per_cm2", "").replace("_from_1deg_mean", "")
               for f in G[G.target == tn].head(3).feature]
        ax.set_title(f"{tn.upper()}: what the all-pairwise model splits on", loc="left", fontsize=11.5)
        ax.text(0.98, 0.97, "most used:\n" + "\n".join(top), transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=INK2)
    headline(fig, "Where the gain from pairwise combinations comes from",
             "Top: change in error when each family of combinations (products, sums, differences of z-scored features) "
             "is added on its own. Bottom: in the model given every\ncombination, the share of all split gain that each "
             "feature family carries (fitted on all rows). Diagonal location boundaries add nothing; the gain comes from "
             "combinations of RTOFS's physical fields,\nmostly differences such as D26 minus its 1° anomaly: straight-"
             "line combinations that trees otherwise have to build from many small steps.", top=0.86)
    fig.savefig(OUT / "pairwise_gain_sources.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(pd.read_csv(OUT / "pairwise_family_scores.csv"), pd.read_csv(OUT / "all_pairwise_model_gain_by_feature.csv"))
    else:
        main()
