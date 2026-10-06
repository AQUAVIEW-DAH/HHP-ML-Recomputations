"""Expanded feature sets the usual way, scored honestly.

Three questions from the 2026-09-24 meeting about functions and combinations of
the features:

  1. One-to-one transforms (powers, exp, log, rank) of each feature. A tree only
     uses the order of a feature's values, so these should change nothing;
     non-monotone ones (square of a centred value, distance from the median)
     can change the splits.
  2. Row statistics across all features: per profile, the mean, median,
     standard deviation, min, max and mode of its z-scored features (z-scored
     because the raw features mix metres, kJ/cm^2, degrees and months).
  3. Pairwise products, sums and differences of the z-scored features plus
     each squared z-score (~900-1,000 candidates). The earlier screen picked
     combinations on the same test blocks it was then scored on, which flatters
     it. Here the choice is made inside each fold from its training dates only:
     the baseline model is fitted on the first part of the training dates and
     its errors on the last part are screened (gain of one split on each
     candidate); the top k are then added and the fold scored as usual. All
     candidates at once is also scored.

Z-scores use training-fold means and standard deviations only.
Outputs: OHC/output/expanded_features_20261004/
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
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, INK2, NEUTRAL, ORANGE, SURF, change_dots, headline)
from OHC.exploration.rtofs_followup_common import (  # noqa: E402
    TARGET, TIME_COLS, evaluate, forward_folds, load_work, median_X, recipe)

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/expanded_features_20261004")
KS = (5, 10, 20, 50)
INNER_FRAC = 0.7     # first 70% of a fold's training dates fit the screening model, the last 30% are screened


def base_matrix(work, cols, tr, va):
    imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
    return imp.transform(work.loc[tr, cols]), imp.transform(work.loc[va, cols])


# ---------------------------------------------------------------- 1. transforms
def transform_X(cols: list[str], kind: str):
    def make(work, tr, va):
        Xtr, Xva = base_matrix(work, cols, tr, va)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        med = np.median(Xtr, 0)
        f = {"cube": lambda X: X ** 3,
             "exp of z-score": lambda X: np.exp(np.clip((X - mu) / sd, -30, 30)),
             "signed log": lambda X: np.sign(X) * np.log1p(np.abs(X)),
             "rank (empirical quantile)": lambda X: np.column_stack(
                 [np.searchsorted(np.sort(Xtr[:, j]), X[:, j]) / len(Xtr) for j in range(X.shape[1])]),
             "square of z-score (not one-to-one)": lambda X: ((X - mu) / sd) ** 2,
             "distance from median (not one-to-one)": lambda X: np.abs(X - med)}[kind]
        return f(Xtr), f(Xva)
    return make


# ---------------------------------------------------------------- 2. row statistics
def rowstats_X(cols: list[str]):
    phys = [c for c in cols if c not in TIME_COLS]

    def make(work, tr, va):
        Xtr, Xva = base_matrix(work, cols, tr, va)
        P = [cols.index(c) for c in phys]
        mu, sd = Xtr[:, P].mean(0), Xtr[:, P].std(0) + 1e-9

        def stats(X):
            Z = (X[:, P] - mu) / sd
            Zb = np.clip(np.round(Z * 2) / 2, -3, 3)           # half-sigma bins, so a mode exists
            mode = np.array([pd.Series(r).mode().iloc[0] for r in Zb])
            return np.column_stack([X, Z.mean(1), np.median(Z, 1), Z.std(1), Z.min(1), Z.max(1), mode])
        return stats(Xtr), stats(Xva)
    return make


# ---------------------------------------------------------------- 3. pairwise candidates
def candidates(Z: np.ndarray, names: list[str]):
    out, nm = [], []
    for i, j in itertools.combinations(range(Z.shape[1]), 2):
        out += [Z[:, i] * Z[:, j], Z[:, i] + Z[:, j], Z[:, i] - Z[:, j]]
        nm += [f"{names[i]} * {names[j]}", f"{names[i]} + {names[j]}", f"{names[i]} - {names[j]}"]
    for i in range(Z.shape[1]):
        out.append(Z[:, i] ** 2)
        nm.append(f"{names[i]}^2")
    return np.column_stack(out).astype(np.float32), nm


def pairwise_X(tn: str, cols: list[str], k: int | None, picks_log: list):
    phys = [c for c in cols if c not in TIME_COLS]
    t = TARGET[tn]

    def make(work, tr, va):
        Xtr, Xva = base_matrix(work, cols, tr, va)
        P = [cols.index(c) for c in phys]
        mu, sd = Xtr[:, P].mean(0), Xtr[:, P].std(0) + 1e-9
        Ctr, names = candidates((Xtr[:, P] - mu) / sd, phys)
        Cva, _ = candidates((Xva[:, P] - mu) / sd, phys)
        if k is None:
            return np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva])
        # honest screen: inner split of the TRAINING dates only
        dtr = work.loc[tr, "ds"].to_numpy()
        cut = np.sort(np.unique(dtr))[int(INNER_FRAC * len(np.unique(dtr)))]
        a, b = dtr < cut, dtr >= cut
        y = work.loc[tr, t.delta_col].to_numpy(float)
        m = abl._xgb_model()
        m.fit(Xtr[a], y[a])
        res = y[b] - m.predict(Xtr[b])
        vr = np.var(res)
        gain = np.empty(Ctr.shape[1])
        for j in range(Ctr.shape[1]):
            s = DecisionTreeRegressor(max_depth=1).fit(Ctr[b, j:j + 1], res)
            gain[j] = (vr - np.var(res - s.predict(Ctr[b, j:j + 1]))) / vr
        top = np.argsort(-gain)[:k]
        picks_log.append({"target": tn, "k": k, "picked": " | ".join(names[j] for j in top[:10])})
        return np.hstack([Xtr, Ctr[:, top]]), np.hstack([Xva, Cva[:, top]])
    return make


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res, picks = [], []
    for tn in ("tchp", "d26"):
        work = load_work(tn)
        cols = recipe(tn, work)
        folds = forward_folds(work)
        base = evaluate(work, tn, folds, median_X(cols))
        res.append({"target": tn, "family": "baseline", "variant": "baseline (full recipe)", **base})
        print(tn, "baseline", round(base["mae"], 3), flush=True)
        for kind in ("cube", "exp of z-score", "signed log", "rank (empirical quantile)",
                     "square of z-score (not one-to-one)", "distance from median (not one-to-one)"):
            r = evaluate(work, tn, folds, transform_X(cols, kind))
            res.append({"target": tn, "family": "transform", "variant": f"every feature -> {kind}", **r})
            print(tn, kind, round(r["mae"] - base["mae"], 4), flush=True)
        r = evaluate(work, tn, folds, rowstats_X(cols))
        res.append({"target": tn, "family": "row statistics",
                    "variant": "+ row mean, median, std, min, max, mode", **r})
        print(tn, "row stats", round(r["mae"] - base["mae"], 4), flush=True)
        for k in KS:
            r = evaluate(work, tn, folds, pairwise_X(tn, cols, k, picks))
            res.append({"target": tn, "family": "pairwise", "variant": f"+ top {k} pairwise (chosen in-fold)", **r})
            print(tn, "pairwise top", k, round(r["mae"] - base["mae"], 4), flush=True)
        r = evaluate(work, tn, folds, pairwise_X(tn, cols, None, picks))
        n = len([c for c in cols if c not in TIME_COLS])
        res.append({"target": tn, "family": "pairwise", "variant": "+ all pairwise candidates at once", **r,
                    "n_candidates": n * (n - 1) // 2 * 3 + n})
        print(tn, "pairwise all", round(r["mae"] - base["mae"], 4), flush=True)
    R = pd.DataFrame(res)
    b = R[R.family == "baseline"].set_index("target")["mae"]
    R["change_vs_baseline"] = R.apply(lambda r: r.mae - b[r.target], axis=1)
    R.to_csv(OUT / "expanded_feature_scores.csv", index=False)
    pd.DataFrame(picks).to_csv(OUT / "pairwise_picks.csv", index=False)
    plot(R)


def plot(R: pd.DataFrame) -> None:
    leaky = pd.read_csv(Path("/home/suramya/HHP-Prediction/OHC/output/missing_physics_20260925/candidate_tests.csv"))
    leaky = leaky[leaky.variant.str.startswith("+ top screened")].assign(
        variant="earlier screen, chosen on the test blocks (leaky)", family="pairwise")
    T = pd.concat([R, leaky], ignore_index=True)
    noise = 2 * R[R.family == "baseline"].seed_sd.max()
    fig = plt.figure(figsize=(16, 10.5))
    a = fig.add_axes([0.36, 0.56, 0.61, 0.27])
    b = fig.add_axes([0.36, 0.08, 0.61, 0.36])
    tl = {v: v.replace("every feature -> ", "every feature → ") for v in R[R.family == "transform"].variant.unique()}
    change_dots(a, T, tl, noise, "")
    a.set_title("1. Transform every feature: one-to-one transforms change nothing", loc="left")
    pl = {**{v: v for v in R[R.family == "row statistics"].variant.unique()},
          **{v: v for v in R[R.family == "pairwise"].variant.unique()},
          "earlier screen, chosen on the test blocks (leaky)": "earlier screen, chosen on the test blocks (leaky)"}
    change_dots(b, T, pl, noise, "Change in error vs the current recipe   ← better | worse →", legend_loc="lower right")
    b.set_title("2-3. Add row statistics or pairwise combinations (chosen honestly inside each fold)", loc="left")
    lo = min(-0.07, T.change_vs_baseline.min() - 0.01)
    hi = max(0.05, T.change_vs_baseline.max() + 0.01)
    a.set_xlim(-0.1, max(1.1, R[R.family == "transform"].change_vs_baseline.max() + 0.05))
    b.set_xlim(min(-0.2, T[T.family != "transform"].change_vs_baseline.min() - 0.02), 0.06)
    honest = R[R.family == "pairwise"].groupby("variant").change_vs_baseline.mean()
    allp = R[R.variant == "+ all pairwise candidates at once"].set_index("target").change_vs_baseline
    headline(fig, "Transforming single features changes nothing; adding all pairwise combinations does help",
             "Top: one-to-one transforms (cube, exp, log, rank) leave a tree unchanged; squares and distances throw away "
             "the sign and wreck it.\nBottom: the row mean/mode of all features does nothing; adding every product, sum "
             f"and difference of z-scored features at once (925 TCHP, 1,001 D26;\nnothing chosen on test data) lowers "
             f"the error by {abs(allp['tchp']):.2f} TCHP and {abs(allp['d26']):.2f} D26, more than any small hand-picked "
             "set. Three runs each.", top=0.87)
    fig.savefig(OUT / "expanded_features_effect.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(pd.read_csv(OUT / "expanded_feature_scores.csv"))
    else:
        main()
