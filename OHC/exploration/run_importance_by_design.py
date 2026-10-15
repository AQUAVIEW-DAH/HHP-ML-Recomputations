"""Does feature importance change between the forward test blocks and the full-year tests?

For each target (TCHP, D26) and input set (full recipe, Dr. Jacobs' 16), the model is fitted on the
training part of each test design (3 forward blocks; full year: train 2025 / test 2024 and train 2024 /
test 2025; 3 seeds each) and three importance measures are recorded:
  * scramble (permutation): rise in test-period error when one input's values are shuffled among the
    test profiles (5 shuffles), model not refitted;
  * split gain: the input's share of the model's total split gain;
  * first split: the share of the 300 trees whose first question is about that input.
Ranks are compared between designs (Spearman rank correlation of the scramble importances).

Outputs: OHC/output/importance_by_design_20261015/
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import INK, INK2, SURF, headline  # noqa: E402
from OHC.exploration.rtofs_followup_common import TARGET, forward_folds, loyo_folds, recipe  # noqa: E402
from OHC.exploration.run_noise_feature_test import JAC, prepare  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/importance_by_design_20261015")
SEEDS = (0, 1, 2)
N_SHUFFLE = 5
SETS = ("full recipe", "Dr. Jacobs' set")
DESIGNS = ["forward blocks", "full year: test 2024", "full year: test 2025"]


def task(args):
    tn, sname = args
    t = TARGET[tn]
    w = prepare(tn)
    cols = recipe(tn, w) if sname == "full recipe" else JAC
    X0 = w[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = w[t.delta_col].to_numpy(float)
    obs, mod = w[t.obs_col].to_numpy(float), w[t.model_col].to_numpy(float)
    folds = [("forward blocks", f) for f in forward_folds(w)] + [(f["name"], f) for f in loyo_folds(w)]
    rows, errs = [], []
    for design, f in folds:
        tr, va = w.ds.isin(set(f["train_dates"])).to_numpy(), w.ds.isin(set(f["val_dates"])).to_numpy()
        med = np.nanmedian(X0[tr], axis=0)
        X = np.where(np.isnan(X0), med, X0)
        for seed in SEEDS:
            m = abl._xgb_model()
            m.set_params(random_state=seed, n_jobs=8)
            m.fit(X[tr], y[tr])
            Xv = X[va]
            base = np.abs(mod[va] + m.predict(Xv) - obs[va]).mean()
            errs.append({"target": tn, "set": sname, "design": design, "fold": f["name"], "seed": seed, "mae": base,
                         "raw_rtofs_mae": float(np.abs(mod[va] - obs[va]).mean()), "test_profiles": int(va.sum())})
            booster = m.get_booster()
            gain = booster.get_score(importance_type="total_gain")
            tot = sum(gain.values())
            T = booster.trees_to_dataframe()
            roots = T[(T.Node == 0) & (T.Feature != "Leaf")].Feature.value_counts()
            rng = np.random.default_rng(seed)
            for j, c in enumerate(cols):
                inc = []
                for _ in range(N_SHUFFLE):
                    Xs = Xv.copy()
                    Xs[:, j] = Xv[rng.permutation(len(Xv)), j]
                    inc.append(np.abs(mod[va] + m.predict(Xs) - obs[va]).mean() - base)
                rows.append({"target": tn, "set": sname, "design": design, "fold": f["name"], "seed": seed,
                             "feature": c, "scramble": float(np.mean(inc)),
                             "gain_share": gain.get(f"f{j}", 0.0) / tot,
                             "first_split_share": roots.get(f"f{j}", 0) / 300})
        print(tn, sname, f["name"], "done", flush=True)
    return rows, errs


def short(f: str) -> str:
    return (f.replace("model_", "").replace("_kj_per_cm2", "").replace("_per_100km", "").replace("interp_", "RTOFS ")
            .replace("_from_1deg_mean", " (1°)").replace("rtofs0_", "RTOFS "))


def plots() -> None:
    P = pd.read_csv(OUT / "importance_runs.csv")
    A = P.groupby(["target", "set", "design", "feature"])[["scramble", "gain_share", "first_split_share"]].mean().reset_index()
    A.to_csv(OUT / "importance_by_design.csv", index=False)
    summary = []
    for sname in SETS:
        fig, axes = plt.subplots(1, 2, figsize=(17, 12 if sname == "full recipe" else 9))
        for ax, tn in zip(axes, ("tchp", "d26")):
            a = A[(A.target == tn) & (A.set == sname)]
            piv = a.pivot(index="feature", columns="design", values="scramble")[DESIGNS]
            fs = a.pivot(index="feature", columns="design", values="first_split_share")[DESIGNS]
            ranks = piv.rank(ascending=False)
            order = piv["forward blocks"].sort_values(ascending=False).index[:16]
            M = ranks.loc[order]
            im = ax.imshow(np.log1p(piv.loc[order].clip(lower=0).to_numpy()), cmap="Blues", aspect="auto")
            for i, f in enumerate(order):
                for j, d in enumerate(DESIGNS):
                    v, r, s = piv.loc[f, d], int(M.loc[f, d]), fs.loc[f, d]
                    ax.text(j, i, f"#{r}  {v:.2f}" + (f"\nfirst split {100 * s:.0f}%" if s >= 0.05 else ""),
                            ha="center", va="center", fontsize=9.5,
                            color="white" if np.log1p(max(v, 0)) > 0.6 * np.log1p(piv.max().max()) else INK)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["forward blocks\n(~5-month tests)", "full year:\ntest 2024", "full year:\ntest 2025"])
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([short(f) for f in order], fontsize=10)
            ax.grid(False)
            rho = {d: spearmanr(piv["forward blocks"], piv[d]).correlation for d in DESIGNS[1:]}
            r12 = spearmanr(piv[DESIGNS[1]], piv[DESIGNS[2]]).correlation
            ax.set_title(f"{tn.upper()}:  rank agreement with forward blocks\n{rho[DESIGNS[1]]:.2f} (test 2024),  "
                         f"{rho[DESIGNS[2]]:.2f} (test 2025)", loc="left", fontsize=11.5)
            for d in DESIGNS:
                top = piv[d].sort_values(ascending=False).index[:5].tolist()
                summary.append({"set": sname, "target": tn, "design": d, "top5": ", ".join(short(f) for f in top),
                                "rank_corr_with_forward": rho.get(d, 1.0), "rank_corr_2024_vs_2025": r12,
                                "top_first_split": short(fs[d].idxmax()), "top_first_split_share": float(fs[d].max())})
        headline(fig, f"{sname}: does input importance change when testing on full years?",
                 "Cells: rise in test-period error when the input is scrambled (#rank, then value in units of the "
                 "target), averaged over 3 seeds and the test blocks.\n'first split' = share of the 300 trees whose first "
                 "question is about that input (shown when 5% or more). Rows: the 16 most important inputs on the forward "
                 "blocks.\nRank agreement = rank correlation of the scramble importances over all inputs (1 = same order).",
                 top=0.82 if sname == "full recipe" else 0.78)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.06, wspace=0.42)
        fig.savefig(OUT / f"importance_by_design_{'full_recipe' if sname == 'full recipe' else 'jacobs_set'}.png", dpi=140)
        plt.close(fig)
    S = pd.DataFrame(summary)
    S.to_csv(OUT / "importance_summary.csv", index=False)
    print(S.to_string(index=False))
    E = pd.read_csv(OUT / "errors.csv").groupby(["target", "set", "design"])[["mae", "raw_rtofs_mae"]].mean().round(3)
    print(E.to_string())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(tn, s) for tn in ("tchp", "d26") for s in SETS]
    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r, e in ex.map(task, tasks):
            rows += r
            errs += e
    pd.DataFrame(rows).to_csv(OUT / "importance_runs.csv", index=False)
    pd.DataFrame(errs).to_csv(OUT / "errors.csv", index=False)
    plots()


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plots()
    else:
        main()
