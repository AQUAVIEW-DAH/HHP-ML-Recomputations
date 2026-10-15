"""Shuffle test for single features and for blocks of features (meeting of 2026-10-08).

Two complementary measurements, each judged against a chance level:

  1. Shuffle (permutation): fit the model once, then shuffle one feature's values among the validation
     profiles (it keeps its values but loses its link to each profile) and measure how much the
     validation error rises. 10 shuffles per feature, 3 seeds x 3 forward test blocks. One pure-noise
     column is included in every model; its shuffle effect is the chance level for this measurement.
     Blocks of features are also shuffled together (rows shuffled jointly, so the block's internal
     relations are kept), which avoids under-rating features that a correlated partner can stand in for.
  2. Remove and refit: drop a whole block and refit (3 seeds; forward blocks and full-year tests). The
     change in validation error is judged against the chance level of the noise-feature test (2 sd).

Both feature sets: the full recipe and Dr. Jacobs' 16 features. Warm rows, one profile per cast.
Outputs: OHC/output/shuffle_importance_20261015/
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
from OHC.exploration.rtofs_followup_common import TARGET, evaluate, forward_folds, loyo_folds, median_X, recipe  # noqa: E402
from OHC.exploration.run_noise_feature_test import JAC, prepare  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/shuffle_importance_20261015")
NOISE = Path("/home/suramya/HHP-Prediction/OHC/output/noise_feature_20261015/noise_summary.csv")
SEEDS = (0, 1, 2)
N_SHUFFLE = 10
SETS = ("full recipe", "Dr. Jacobs' set")
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
BLOCKS_FULL = {
    "location (lat, lon, |lat|)": ["lat", "lon", "abs_lat"],
    "calendar (9)": ["year", "month_int", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_winter_jfm",
                     "is_summer_jas", "is_other"],
    "grid distance": ["nearest_rtofs_grid_distance_km"],
    "RTOFS TCHP and D26 at the float": ["model_interp_tchp_kj_per_cm2", "model_interp_d26_m"],
    "surface and mixed-layer physics": ["model_ssh_m", "model_mixed_layer_thickness_m",
                                        "model_surface_boundary_layer_thickness_m", "model_temp_excess_26c",
                                        "d26_minus_mlt_m", "d26_to_sblt_ratio"],
    "latitude interactions (x |lat|)": ["model_ssh_x_abs_lat", "model_mlt_x_abs_lat", "model_temp_excess_x_abs_lat"],
    "TCHP neighbourhood": ["model_tchp_local_std_1deg", "model_tchp_local_std_2deg", "model_tchp_grad_mag_per_100km",
                           "model_tchp_anom_from_1deg_mean"],
    "D26 neighbourhood": ["model_d26_local_std_1deg", "model_d26_grad_mag_per_100km", "model_d26_anom_from_1deg_mean"],
    "SST neighbourhood": ["model_sst_local_std_1deg", "model_sst_grad_mag_per_100km", "model_sst_anom_from_1deg_mean"],
    "deep profile (steric, N²)": ["model_steric_1000_ref2000_m", "model_n2_max_upper200_s2", "model_n2_mean_to_d26_s2"],
}
BLOCKS_JAC = {
    "location (lat, lon)": ["lat", "lon"],
    "calendar (4)": ["cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"],
    "TCHP + 1° average": ["rtofs0_tchp", "tchp_1deg_avg"],
    "D26 + 1° average": ["rtofs0_d26", "d26_1deg_avg"],
    "SSH + 1° average": ["rtofs0_ssh", "ssh_1deg_avg"],
    "mixed layer + 1° average": ["rtofs0_mlt", "mlt_1deg_avg"],
    "boundary layer + 1° average": ["rtofs0_sblt", "sblt_1deg_avg"],
}


def setup(tn, sname):
    w = prepare(tn)
    cols = recipe(tn, w) if sname == "full recipe" else JAC
    blocks = BLOCKS_FULL if sname == "full recipe" else BLOCKS_JAC
    blocks = {k: [c for c in v if c in cols] for k, v in blocks.items()}
    return w, cols, {k: v for k, v in blocks.items() if v}


def permutation_task(task):
    tn, sname = task
    t = TARGET[tn]
    w, cols, blocks = setup(tn, sname)
    X0 = w[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = w[t.delta_col].to_numpy(float)
    obs, mod = w[t.obs_col].to_numpy(float), w[t.model_col].to_numpy(float)
    names = cols + ["noise column"]
    groups = {n: [i] for i, n in enumerate(names)}
    groups.update({f"[block] {k}": [cols.index(c) for c in v] for k, v in blocks.items()})
    rows = []
    for seed in SEEDS:
        noise = np.random.default_rng(20_000 + seed).standard_normal(len(w))
        noise = (noise - noise.mean()) / noise.std()
        for f in forward_folds(w):
            tr, va = w.ds.isin(set(f["train_dates"])).to_numpy(), w.ds.isin(set(f["val_dates"])).to_numpy()
            med = np.nanmedian(X0[tr], axis=0)
            X = np.column_stack([np.where(np.isnan(X0), med, X0), noise])
            m = abl._xgb_model()
            m.set_params(random_state=seed, n_jobs=8)
            m.fit(X[tr], y[tr])
            Xv = X[va]
            base = np.abs(mod[va] + m.predict(Xv) - obs[va]).mean()
            rng = np.random.default_rng(30_000 + seed)
            for gname, idx in groups.items():
                incs = []
                for _ in range(N_SHUFFLE):
                    Xs = Xv.copy()
                    Xs[:, idx] = Xv[rng.permutation(len(Xv))][:, idx]
                    incs.append(np.abs(mod[va] + m.predict(Xs) - obs[va]).mean() - base)
                rows.append({"target": tn, "set": sname, "seed": seed, "fold": f["name"], "feature": gname,
                             "is_block": gname.startswith("[block]"), "mae_increase": float(np.mean(incs)),
                             "shuffle_sd": float(np.std(incs))})
        print(tn, sname, "permutation seed", seed, flush=True)
    return rows


def dropblock_task(task):
    tn, sname, design = task
    w, cols, blocks = setup(tn, sname)
    folds = forward_folds(w) if design == "forward blocks" else loyo_folds(w)
    base = evaluate(w, tn, folds, median_X(cols))
    rows = [{"target": tn, "set": sname, "design": design, "block": "(none removed)", **base}]
    for b, bc in blocks.items():
        keep = [c for c in cols if c not in bc]
        r = evaluate(w, tn, folds, median_X(keep))
        rows.append({"target": tn, "set": sname, "design": design, "block": b, **r})
    out = pd.DataFrame(rows)
    out["change"] = out.mae - base["mae"]
    print(tn, sname, design, "drop-block done", flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(tn, s) for tn in ("tchp", "d26") for s in SETS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        perm = [r for rows in ex.map(permutation_task, tasks) for r in rows]
    pd.DataFrame(perm).to_csv(OUT / "permutation_runs.csv", index=False)
    dtasks = [(tn, s, d) for tn in ("tchp", "d26") for s in SETS for d in ("forward blocks", "full-year tests")]
    with ProcessPoolExecutor(max_workers=4) as ex:
        D = pd.concat(list(ex.map(dropblock_task, dtasks)), ignore_index=True)
    D.to_csv(OUT / "drop_block_scores.csv", index=False)
    plots()


def short(f: str) -> str:
    return (f.replace("model_", "").replace("_kj_per_cm2", "").replace("_per_100km", "").replace("interp_", "RTOFS ")
            .replace("_from_1deg_mean", " (1°)").replace("rtofs0_", "RTOFS "))


def plot_single(P: pd.DataFrame, sname: str, fname: str) -> pd.DataFrame:
    sub = P[(P.set == sname) & ~P.is_block]
    nfeat = sub.feature.nunique()
    H = max(7.0, 0.3 * nfeat + 3.2)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, H))
    summary = []
    for ax, tn, col in zip(axes, ("tchp", "d26"), (BLUE, ORANGE)):
        g = sub[sub.target == tn].groupby("feature").mae_increase
        m, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
        m = m.sort_values()
        nz = sub[(sub.target == tn) & (sub.feature == "noise column")].mae_increase
        chance = max(abs(nz.quantile(0.05)), abs(nz.quantile(0.95)), 1e-4)
        colors = [INK if f == "noise column" else (col if m[f] > chance else MUTED) for f in m.index]
        y = np.arange(len(m))
        ax.barh(y, m.values, color=colors, edgecolor=SURF, height=0.72)
        ax.hlines(y, lo[m.index].values, hi[m.index].values, color=INK2, lw=1)
        ax.axvspan(-chance, chance, color=NEUTRAL, zorder=0)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([short(f) for f in m.index], fontsize=9.5)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(-0.03, max(5, m.max() * 2))
        ax.set_xlabel(f"Rise in validation error when shuffled ({UNIT[tn]}, log scale)")
        ax.grid(axis="y", visible=False)
        nb = int((m.drop("noise column") <= chance).sum())
        ax.set_title(f"{tn.upper()}: {nb} of {len(m) - 1} features no better than the noise column", loc="left",
                     fontsize=12)
        for f in m.index:
            summary.append({"target": tn, "set": sname, "feature": f, "mae_increase": m[f], "p05": lo[f], "p95": hi[f],
                            "chance": chance, "beats_noise": bool(m[f] > chance)})
    headline(fig, f"{sname}: how much each feature matters when its link to the profiles is broken",
             "Each feature's values are shuffled among the validation profiles (10 shuffles x 3 seeds x 3 forward test "
             "blocks) and the rise in validation error is measured; the model is not refitted.\nBars: average rise; "
             "thin lines: 5-95% range. Black bar and grey band: a pure-noise column in the same models, the chance level. "
             "Grey bars: features no better than noise.", top=1 - 1.45 / H)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.07, wspace=0.6)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return pd.DataFrame(summary)


def plot_blocks(P: pd.DataFrame, D: pd.DataFrame, sname: str, fname: str) -> None:
    N = pd.read_csv(NOISE)
    band = {tn: 2 * float(N[(N.target == tn) & (N.design == "forward blocks")].val_mae_sd.mean()) for tn in ("tchp", "d26")}
    bl = list(dict.fromkeys(D[(D.set == sname) & (D.block != "(none removed)")].block))
    fig, axes = plt.subplots(2, 2, figsize=(17, 4.5 + 0.55 * len(bl) * 2), sharey="row")
    y = np.arange(len(bl))[::-1]
    for j, tn in enumerate(("tchp", "d26")):
        ax = axes[0, j]
        pb = P[(P.set == sname) & P.is_block & (P.target == tn)]
        g = pb.groupby("feature").mae_increase
        m = g.mean()
        lo, hi = g.quantile(0.05), g.quantile(0.95)
        keys = [f"[block] {b}" for b in bl]
        nz = P[(P.set == sname) & (P.target == tn) & (P.feature == "noise column")].mae_increase
        chance = max(abs(nz.quantile(0.05)), abs(nz.quantile(0.95)), 1e-4)
        ax.axvspan(-chance, chance, color=NEUTRAL, zorder=0)
        ax.barh(y, [m.get(k, np.nan) for k in keys], color=BLUE if tn == "tchp" else ORANGE, edgecolor=SURF, height=0.7)
        ax.hlines(y, [lo.get(k, np.nan) for k in keys], [hi.get(k, np.nan) for k in keys], color=INK2, lw=1)
        for yy, k in zip(y, keys):
            if np.isfinite(m.get(k, np.nan)):
                ax.text(max(hi.get(k, 0), m.get(k, 0)) * 1.15 + 0.004, yy, f"{m.get(k):+.2f}", va="center", fontsize=9.5,
                        color=INK2)
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-0.004, max(10, m.max() * 3))
        ax.set_yticks(y)
        ax.set_yticklabels(bl, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{tn.upper()}: shuffle the block (no refit)", loc="left", fontsize=12)
        ax.set_xlabel(f"Rise in validation error ({UNIT[tn]})")
        ax = axes[1, j]
        for design, mk, col, dy in (("forward blocks", "o", ORANGE, 0.15), ("full-year tests", "D", INK2, -0.15)):
            d = D[(D.set == sname) & (D.target == tn) & (D.design == design)].set_index("block").reindex(bl)
            ax.errorbar(d["change"].to_numpy(), y + dy, xerr=d["seed_sd"].to_numpy(), fmt=mk,
                        color=col, ms=8, mec=SURF, elinewidth=1.4, label=design)
        ax.axvspan(-band[tn], band[tn], color=NEUTRAL, zorder=0)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(bl, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{tn.upper()}: remove the block and refit", loc="left", fontsize=12)
        ax.set_xlabel(f"Change in validation error ({UNIT[tn]})\n← removing helps | removing hurts →")
        if j == 1:
            ax.legend(loc="lower right", fontsize=10)
    headline(fig, f"{sname}: which blocks of features carry information",
             "Top: the block's columns are shuffled together among the validation profiles (model not refitted); grey "
             "band = a noise column shuffled the same way.\nBottom: the block is removed and the model refitted; grey "
             "band = chance level from the noise-feature test (±2 sd).\nA block inside the bottom band can be dropped "
             "without a measurable loss, even if shuffling it hurts (another block can stand in for it).", top=0.87)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.07, hspace=0.3, wspace=0.08)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plots() -> None:
    P = pd.read_csv(OUT / "permutation_runs.csv")
    D = pd.read_csv(OUT / "drop_block_scores.csv")
    S = pd.concat([plot_single(P, "full recipe", "1_full_recipe_single_feature_shuffle.png"),
                   plot_single(P, "Dr. Jacobs' set", "2_jacobs_set_single_feature_shuffle.png")])
    S.to_csv(OUT / "single_feature_summary.csv", index=False)
    plot_blocks(P, D, "full recipe", "3_full_recipe_blocks.png")
    plot_blocks(P, D, "Dr. Jacobs' set", "4_jacobs_set_blocks.png")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plots()
    else:
        main()
