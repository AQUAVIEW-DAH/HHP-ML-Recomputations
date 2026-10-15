"""The "nothing feature" test (Dr. Jacobs, meeting of 2026-10-08).

Add columns of pure Gaussian noise (normalised to mean 0, sd 1, a new draw for every repeat) to a model
and measure what they do to the training error and to the validation error. Noise carries no information
about the RTOFS error, so:
  * any drop in training error is the model memorising chance patterns (overfitting capacity);
  * the change in validation error is the chance level against which real features must be judged;
  * the share of split gain the trees give a noise column is the "usefulness floor": a real feature used
    less than noise is, to the model, no better than nothing.

Design: warm rows (26 C water in both Argo and RTOFS), one profile per Argo cast; locked XGBoost settings;
two feature sets (the full recipe, and Dr. Jacobs' 16 features); 0, 1, 5 and 20 noise columns (the 5 include
the 1, the 20 include the 5); 20 repeats, each with its own noise draw and its own random seed (the seed sets
the random 80% of rows and columns each tree sees, which plays the role of reshuffling the training data).
Every noise model is compared with the no-noise model of the same seed, so the difference is due to the noise
alone. Scored on the forward test blocks and on the full-year tests.
Measures: MAE of the corrected forecast (RTOFS + predicted error) against Argo, and the error variance
explained, 1 - var(error - prediction) / var(error), each on the training rows and the validation rows.

Outputs: OHC/output/noise_feature_20261015/
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, forward_folds, load_work, loyo_folds, recipe  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/noise_feature_20261015")
JAC_FEAT = Path("/home/suramya/HHP-Prediction/OHC/output/jacobs_feature_set_20261013/jacobs_features.parquet")
KS = (0, 1, 5, 20)
SEEDS = tuple(range(20))
SETS = ("full recipe", "Dr. Jacobs' set")
JAC = ["lat", "lon", "cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual",
       "rtofs0_tchp", "tchp_1deg_avg", "rtofs0_d26", "d26_1deg_avg", "rtofs0_ssh", "ssh_1deg_avg",
       "rtofs0_mlt", "mlt_1deg_avg", "rtofs0_sblt", "sblt_1deg_avg"]
UNIT = {"tchp": "kJ/cm²", "d26": "m"}


def prepare(tn: str) -> pd.DataFrame:
    w = load_work(tn)
    n = len(w)
    F = pd.read_parquet(JAC_FEAT)[["cast_id", "rtofs0_tchp", "rtofs0_d26", "tchp_1deg_avg", "d26_1deg_avg",
                                   "ssh_1deg_avg", "mlt_1deg_avg", "sblt_1deg_avg"]]
    w = w.merge(F, on="cast_id", how="left")
    assert len(w) == n
    f = (w["date"].dt.dayofyear - 0.5) / np.where(w["date"].dt.is_leap_year, 366.0, 365.0)
    for k, name in ((1, "annual"), (2, "semiannual")):
        w[f"cos_{name}"] = np.cos(2 * np.pi * k * f)
        w[f"sin_{name}"] = np.sin(2 * np.pi * k * f)
    w["rtofs0_ssh"] = w.model_ssh_m
    w["rtofs0_mlt"] = w.model_mixed_layer_thickness_m
    w["rtofs0_sblt"] = w.model_surface_boundary_layer_thickness_m
    return w


def run_task(task):
    tn, sname = task
    t = TARGET[tn]
    w = prepare(tn)
    cols = recipe(tn, w) if sname == "full recipe" else JAC
    X0 = w[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = w[t.delta_col].to_numpy(float)
    obs, mod = w[t.obs_col].to_numpy(float), w[t.model_col].to_numpy(float)
    rows, gains = [], []
    for design, folds in (("forward blocks", forward_folds(w)), ("full-year tests", loyo_folds(w))):
        masks = [(w.ds.isin(set(f["train_dates"])).to_numpy(), w.ds.isin(set(f["val_dates"])).to_numpy(), f["name"])
                 for f in folds]
        filled = []
        for tr, va, _ in masks:
            med = np.nanmedian(X0[tr], axis=0)
            filled.append((np.where(np.isnan(X0[tr]), med, X0[tr]), np.where(np.isnan(X0[va]), med, X0[va])))
        for seed in SEEDS:
            noise = np.random.default_rng(10_000 + seed).standard_normal((len(w), max(KS)))
            noise = (noise - noise.mean(0)) / noise.std(0)
            for k in KS:
                pv = np.full(len(w), np.nan)
                trs, nshare = [], []
                for (tr, va, fname), (Xtr, Xva) in zip(masks, filled):
                    A = np.hstack([Xtr, noise[tr, :k]]) if k else Xtr
                    B = np.hstack([Xva, noise[va, :k]]) if k else Xva
                    m = abl._xgb_model()
                    m.set_params(random_state=seed, n_jobs=8)
                    m.fit(A, y[tr])
                    ptr = m.predict(A)
                    pv[va] = m.predict(B)
                    trs.append((np.abs(mod[tr] + ptr - obs[tr]).mean(), 1 - np.var(y[tr] - ptr) / np.var(y[tr]), tr.sum()))
                    g = m.get_booster().get_score(importance_type="total_gain")
                    tot = sum(g.values())
                    nshare.append(sum(g.get(f"f{len(cols) + i}", 0.0) for i in range(k)) / tot)
                    if design == "forward blocks" and k == 1:
                        for i, nm in enumerate(cols + ["noise"]):
                            gains.append({"target": tn, "set": sname, "seed": seed, "fold": fname, "feature": nm,
                                          "share": g.get(f"f{i}", 0.0) / tot})
                v = np.isfinite(pv)
                n = np.array([s[2] for s in trs], float)
                rows.append({"target": tn, "set": sname, "design": design, "k": k, "seed": seed,
                             "n_features": len(cols),
                             "train_mae": float(np.average([s[0] for s in trs], weights=n)),
                             "train_evr": float(np.average([s[1] for s in trs], weights=n)),
                             "val_mae": float(np.abs(mod[v] + pv[v] - obs[v]).mean()),
                             "val_evr": float(1 - np.var(y[v] - pv[v]) / np.var(y[v])),
                             "noise_gain_share": float(np.mean(nshare))})
            print(tn, sname, design, "seed", seed, "done", flush=True)
    return rows, gains


def paired(R: pd.DataFrame) -> pd.DataFrame:
    base = R[R.k == 0].set_index(["target", "set", "design", "seed"])
    D = R[R.k > 0].copy()
    for c in ("train_mae", "train_evr", "val_mae", "val_evr"):
        D[f"d_{c}"] = D[c].to_numpy() - base.loc[list(zip(D.target, D.set, D.design, D.seed)), c].to_numpy()
    return D


# ------------------------------------------------------------------ figures
def plot_effect(D: pd.DataFrame, tn: str, fname: str) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), sharex=True)
    for j, sname in enumerate(SETS):
        for i, (metric, ylab, scale) in enumerate(
                (("mae", f"Change in error, MAE ({UNIT[tn]})\n← noise helped | noise hurt →", 1.0),
                 ("evr", "Change in error variance explained\n(percentage points; up = more explained)", 100.0))):
            ax = axes[i, j]
            ax.axhline(0, color=AXIS, lw=1.2)
            for split, col, mk, dx, lab in (("train", BLUE, "o", -0.16, "training rows (seen by the model)"),
                                           ("val", ORANGE, "D", 0.16, "validation rows (unseen test blocks)")):
                means = []
                for xi, k in enumerate((1, 5, 20)):
                    v = D[(D.target == tn) & (D.set == sname) & (D.design == "forward blocks") & (D.k == k)][
                        f"d_{split}_{metric}"].to_numpy() * scale
                    ax.plot(xi + dx + rng.uniform(-0.06, 0.06, len(v)), v, mk, color=col, ms=4, alpha=0.4, mec="none")
                    ax.plot(xi + dx, v.mean(), mk, color=col, ms=11, mec=INK, mew=1.2, zorder=4)
                    ax.text(xi + dx + (0.1 if split == "val" else -0.1), v.mean(), f"{v.mean():+.3f}" if metric == "mae"
                            else f"{v.mean():+.2f}", ha="left" if split == "val" else "right", va="center",
                            fontsize=9.5, color=col, fontweight="bold")
                    means.append(v.mean())
                ax.plot(np.arange(3) + dx, means, color=col, lw=1.6, zorder=3, label=lab)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["1 noise column", "5 noise columns", "20 noise columns"], fontsize=11.5)
            ax.set_xlim(-0.6, 2.6)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=11)
            if i == 0:
                n = int(D[(D.set == sname) & (D.target == tn)].n_features.iloc[0])
                ax.set_title(f"{sname} ({n} real features)", loc="left")
    axes[0, 0].legend(loc="upper left", fontsize=10)
    sub = D[(D.target == tn) & (D.design == "forward blocks")]
    d = sub[sub.k == 20].groupby("set")[["d_train_mae", "d_val_mae"]].mean()
    fr = d.loc["full recipe"]
    sd = sub.groupby(["set", "k"]).d_val_mae.std().mean()
    headline(fig, f"{tn.upper()}: noise columns barely change the training or the validation error",
             f"Forward test blocks. Small dots: 20 repeats, each a new noise draw and seed, compared with the same seed "
             f"without noise; large markers: averages.\nWith 20 noise columns the full recipe's training error changes by "
             f"{fr.d_train_mae:+.3f} and its validation error by {fr.d_val_mae:+.3f} {UNIT[tn]}: the trees almost never "
             f"split on noise.\nThe spread of the orange dots (± {sd:.3f}) is the chance level: a change in validation "
             f"error smaller than about ± {2 * sd:.2f} cannot be told apart from adding random numbers.", top=0.84)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, hspace=0.12, wspace=0.14)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_importance(G: pd.DataFrame, sname: str, fname: str) -> None:
    sub = G[G.set == sname]
    nfeat = sub.feature.nunique()
    H = max(7.5, 0.3 * nfeat + 3.4)
    fig, axes = plt.subplots(1, 2, figsize=(16, H))
    summary = []
    for ax, tn, col in zip(axes, ("tchp", "d26"), (BLUE, ORANGE)):
        g = sub[sub.target == tn]
        piv = g.pivot_table(index=["seed", "fold"], columns="feature", values="share") * 100
        noise = piv.pop("noise")
        below = piv.lt(noise, axis=0).mean()
        real = piv.mean().sort_values()
        lo, hi = np.percentile(noise, [5, 95])
        clip = 0.01
        vals = real.clip(lower=clip)
        colors = [MUTED if below[f] >= 0.5 else col for f in real.index]
        ax.barh(np.arange(len(real)), vals.values, color=colors, edgecolor=SURF, height=0.72)
        ax.axvspan(max(lo, clip), hi, color=NEUTRAL, zorder=0)
        ax.axvline(max(noise.mean(), clip), color=INK, ls="--", lw=1.4)
        ax.text(max(noise.mean(), clip) * 1.15, 0.015, f"noise column {noise.mean():.2f}%",
                transform=ax.get_xaxis_transform(), va="bottom", ha="left", fontsize=10, color=INK)
        ax.set_xscale("log")
        ax.set_yticks(np.arange(len(real)))
        ax.set_yticklabels([f.replace("model_", "").replace("_kj_per_cm2", "") for f in real.index], fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of the model's split gain (%, log scale)")
        nb = int((below >= 0.5).sum())
        ax.set_title(f"{tn.upper()}: {nb} of {len(real)} features used less than noise", loc="left")
        for f in real.index:
            summary.append({"target": tn, "set": sname, "feature": f, "gain_share_pct": real[f],
                            "fraction_of_fits_below_noise": below[f]})
    headline(fig, f"{sname}: which features the model uses no more than a column of random noise",
             "One noise column added, forward test blocks, 20 repeats x 3 blocks = 60 fitted models. Bars: each real "
             "feature's average share of split gain.\nDashed line and grey band: the noise column's average and its "
             "5-95% range. Grey bars: features used less than the noise column in at least half of the 60 models,\n"
             "so to the trees they carry no more usable information than random numbers.", top=1 - 1.45 / H)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.07, wspace=0.55)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return pd.DataFrame(summary)


def plot_designs(D: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    rows = [(s, k) for s in SETS for k in (1, 5, 20)]
    y = np.arange(len(rows))[::-1]
    for ax, tn in zip(axes, ("tchp", "d26")):
        ax.axvline(0, color=AXIS, lw=1.2)
        for design, mk, col, dy in (("forward blocks", "o", ORANGE, 0.14), ("full-year tests", "D", INK2, -0.14)):
            for yy, (s, k) in zip(y, rows):
                v = D[(D.target == tn) & (D.set == s) & (D.k == k) & (D.design == design)].d_val_mae
                ax.errorbar(v.mean(), yy + dy, xerr=v.std(), fmt=mk, color=col, ms=8, mec=SURF, elinewidth=1.5,
                            label=design if yy == y[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s}, {k} noise col." for s, k in rows])
        ax.grid(axis="y", visible=False)
        ax.set_title(tn.upper(), loc="left")
        ax.set_xlabel(f"Change in validation error, MAE ({UNIT[tn]})\naverage ± spread over 20 repeats")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.59, 0.0))
    headline(fig, "Forward blocks and full-year tests agree: noise changes validation error by about zero",
             "Markers: average change over the 20 repeats; bars: ± one standard deviation (the chance level).", top=0.83)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.22, wspace=0.08)
    fig.savefig(OUT / "5_forward_vs_full_year.png", dpi=150)
    plt.close(fig)


def plots() -> None:
    R = pd.read_csv(OUT / "noise_runs.csv")
    D = paired(R)
    G = pd.read_csv(OUT / "noise_gain_shares.csv")
    plot_effect(D, "tchp", "1_tchp_training_vs_validation.png")
    plot_effect(D, "d26", "2_d26_training_vs_validation.png")
    imp = pd.concat([plot_importance(G, "full recipe", "3_full_recipe_features_vs_noise.png"),
                     plot_importance(G, "Dr. Jacobs' set", "4_jacobs_set_features_vs_noise.png")])
    imp.to_csv(OUT / "features_vs_noise.csv", index=False)
    plot_designs(D)
    S = D.groupby(["target", "set", "design", "k"]).agg(
        train_mae_change=("d_train_mae", "mean"), train_mae_sd=("d_train_mae", "std"),
        val_mae_change=("d_val_mae", "mean"), val_mae_sd=("d_val_mae", "std"),
        train_evr_change_pp=("d_train_evr", lambda x: 100 * x.mean()),
        val_evr_change_pp=("d_val_evr", lambda x: 100 * x.mean()),
        noise_gain_share_pct=("noise_gain_share", lambda x: 100 * x.mean())).reset_index()
    base = R[R.k == 0].groupby(["target", "set", "design"])[["train_mae", "val_mae", "train_evr", "val_evr"]].agg(["mean", "std"])
    S.to_csv(OUT / "noise_summary.csv", index=False)
    base.to_csv(OUT / "baseline_run_to_run_spread.csv")
    print(S.round(4).to_string(index=False))
    print(base.round(4).to_string())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(tn, s) for tn in ("tchp", "d26") for s in SETS]
    rows, gains = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r, g in ex.map(run_task, tasks):
            rows += r
            gains += g
    pd.DataFrame(rows).to_csv(OUT / "noise_runs.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "noise_gain_shares.csv", index=False)
    plots()


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plots()
    else:
        main()
