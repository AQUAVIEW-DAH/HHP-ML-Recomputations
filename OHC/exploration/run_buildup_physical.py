"""Step-by-step build-up of the feature set, physical features first (meeting of 2026-10-08).

Dr. Jacobs' suggested structure: start from location, add plain physical values, then add differences and
products of the physical values (which contain all the present differences, anomalies and interactions),
and at each step decide what to keep. Suramya's note: add sums, means and exponents on the physical
features only. Each step adds a block to the previous step:

  S1  lat, lon
  S2  + RTOFS values at the float: TCHP, D26, SSH, mixed layer, boundary layer, SST           (6)
  S3  + their 1-degree averages                                                               (6)
  S4  + local spread (1-degree std) and gradient magnitude of TCHP, D26, SST                  (6)
  S5  + pairwise differences of the 18 physical features (z-scored)                           (153)
  S6  + pairwise products                                                                     (153)
  S7  + pairwise sums and three family means                                                  (156)
  S8  + squares and exponentials of each physical feature                                     (36)
  S9  + |lat| and longitude as sin/cos                                                        (3)
  S10 + calendar: annual and semiannual sin/cos of the day of the year                        (4)

Keep rule: a step is kept when it lowers the validation error by more than the chance level measured by
the noise-feature test (2 x the spread of the validation change from adding pure noise). A final model is
built from the kept steps only. Z-scores use training-fold statistics. Warm rows, one profile per cast,
locked XGBoost settings, 3 seeds; forward test blocks and full-year tests; training error reported too.

Outputs: OHC/output/buildup_physical_20261015/
"""
from __future__ import annotations

import itertools
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
    AQUA, AXIS, BLUE, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, forward_folds, load_work, loyo_folds  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/buildup_physical_20261015")
JAC_FEAT = Path("/home/suramya/HHP-Prediction/OHC/output/jacobs_feature_set_20261013/jacobs_features.parquet")
NOISE = Path("/home/suramya/HHP-Prediction/OHC/output/noise_feature_20261015/noise_summary.csv")
SEEDS = (0, 1, 2)
POINT = {"TCHP": "model_interp_tchp_kj_per_cm2", "D26": "model_interp_d26_m", "SSH": "model_ssh_m",
         "MLT": "model_mixed_layer_thickness_m", "SBLT": "model_surface_boundary_layer_thickness_m",
         "SST": "model_surface_temp_c"}
AVG = {"TCHP": "model_tchp_local_mean_1deg", "D26": "model_d26_local_mean_1deg", "SSH": "ssh_1deg_avg",
       "MLT": "mlt_1deg_avg", "SBLT": "sblt_1deg_avg", "SST": "model_sst_local_mean_1deg"}
SPREAD = ["model_tchp_local_std_1deg", "model_d26_local_std_1deg", "model_sst_local_std_1deg",
          "model_tchp_grad_mag_per_100km", "model_d26_grad_mag_per_100km", "model_sst_grad_mag_per_100km"]
PHYS = list(POINT.values()) + list(AVG.values()) + SPREAD
STEPS = [("S1", "lat, lon"),
         ("S2", "+ RTOFS values at the float (6)"),
         ("S3", "+ their 1° averages (6)"),
         ("S4", "+ local spread and gradients (6)"),
         ("S5", "+ pairwise differences (153)"),
         ("S6", "+ pairwise products (153)"),
         ("S7", "+ pairwise sums, family means (156)"),
         ("S8", "+ squares, exponentials (36)"),
         ("S9", "+ |lat|, longitude sin/cos (3)"),
         ("S10", "+ calendar sin/cos (4)")]
UNIT = {"tchp": "kJ/cm²", "d26": "m"}


def prepare(tn: str) -> pd.DataFrame:
    w = load_work(tn)
    n = len(w)
    F = pd.read_parquet(JAC_FEAT)[["cast_id", "ssh_1deg_avg", "mlt_1deg_avg", "sblt_1deg_avg"]]
    w = w.merge(F, on="cast_id", how="left")
    assert len(w) == n
    f = (w["date"].dt.dayofyear - 0.5) / np.where(w["date"].dt.is_leap_year, 366.0, 365.0)
    for k, name in ((1, "annual"), (2, "semiannual")):
        w[f"cos_{name}"] = np.cos(2 * np.pi * k * f)
        w[f"sin_{name}"] = np.sin(2 * np.pi * k * f)
    w["lon_sin"] = np.sin(np.deg2rad(w["lon"]))
    w["lon_cos"] = np.cos(np.deg2rad(w["lon"]))
    return w


def blocks_for(steps: list[str]):
    """Raw columns and engineered blocks included when the given steps are active."""
    raw = []
    if "S1" in steps:
        raw += ["lat", "lon"]
    if "S2" in steps:
        raw += list(POINT.values())
    if "S3" in steps:
        raw += list(AVG.values())
    if "S4" in steps:
        raw += SPREAD
    if "S9" in steps:
        raw += ["abs_lat", "lon_sin", "lon_cos"]
    if "S10" in steps:
        raw += ["cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"]
    eng = [s for s in ("S5", "S6", "S7", "S8") if s in steps]
    return raw, eng


def engineered(Z: np.ndarray, eng: list[str]) -> np.ndarray:
    out = []
    pairs = list(itertools.combinations(range(Z.shape[1]), 2))
    if "S5" in eng:
        out += [Z[:, i] - Z[:, j] for i, j in pairs]
    if "S6" in eng:
        out += [Z[:, i] * Z[:, j] for i, j in pairs]
    if "S7" in eng:
        out += [Z[:, i] + Z[:, j] for i, j in pairs]
        out += [Z[:, 0:6].mean(1), Z[:, 6:12].mean(1), Z[:, 12:18].mean(1)]
    if "S8" in eng:
        out += [Z[:, i] ** 2 for i in range(Z.shape[1])]
        out += [np.exp(np.clip(Z[:, i], -20, 20)) for i in range(Z.shape[1])]
    return np.column_stack(out).astype(np.float32) if out else np.empty((len(Z), 0), np.float32)


def matrices(w: pd.DataFrame, steps: list[str], tr: np.ndarray, va: np.ndarray):
    raw, eng = blocks_for(steps)
    R = w[raw].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(R[tr], axis=0)
    R = np.where(np.isnan(R), med, R)
    parts = [R]
    if eng:
        P = w[PHYS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pmed = np.nanmedian(P[tr], axis=0)
        P = np.where(np.isnan(P), pmed, P)
        mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-9
        parts.append(engineered((P - mu) / sd, eng))
    X = np.hstack(parts).astype(np.float32)
    return X[tr], X[va], X.shape[1]


def run_one(args):
    tn, design, label, steps = args
    t = TARGET[tn]
    w = prepare(tn)
    folds = forward_folds(w) if design == "forward blocks" else loyo_folds(w)
    y = w[t.delta_col].to_numpy(float)
    obs, mod = w[t.obs_col].to_numpy(float), w[t.model_col].to_numpy(float)
    masks = [(w.ds.isin(set(f["train_dates"])).to_numpy(), w.ds.isin(set(f["val_dates"])).to_numpy()) for f in folds]
    mats = [matrices(w, steps, tr, va) for tr, va in masks]
    tr_mae, va_mae = [], []
    for seed in SEEDS:
        pv = np.full(len(w), np.nan)
        tr_err, tr_n = [], []
        for (tr, va), (Xtr, Xva, _) in zip(masks, mats):
            m = abl._xgb_model()
            m.set_params(random_state=seed, n_jobs=8)
            m.fit(Xtr, y[tr])
            ptr = m.predict(Xtr)
            pv[va] = m.predict(Xva)
            tr_err.append(np.abs(mod[tr] + ptr - obs[tr]).mean())
            tr_n.append(tr.sum())
        v = np.isfinite(pv)
        va_mae.append(np.abs(mod[v] + pv[v] - obs[v]).mean())
        tr_mae.append(np.average(tr_err, weights=tr_n))
    print(tn, design, label, round(float(np.mean(va_mae)), 3), flush=True)
    return {"target": tn, "design": design, "step": label, "n_features": mats[0][2],
            "val_mae": float(np.mean(va_mae)), "val_sd": float(np.std(va_mae)),
            "train_mae": float(np.mean(tr_mae)), "train_sd": float(np.std(tr_mae))}


def noise_band() -> dict:
    N = pd.read_csv(NOISE)
    N = N[N.design == "forward blocks"]
    return {tn: 2 * float(N[N.target == tn].val_mae_sd.mean()) for tn in ("tchp", "d26")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    codes = [s for s, _ in STEPS]
    jobs = [(tn, d, code, codes[:i + 1]) for tn in ("tchp", "d26") for d in ("forward blocks", "full-year tests")
            for i, code in enumerate(codes)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        R = pd.DataFrame(list(ex.map(run_one, jobs)))
    band = noise_band()
    kept_jobs = []
    for tn in ("tchp", "d26"):
        f = R[(R.target == tn) & (R.design == "forward blocks")].set_index("step").loc[codes]
        kept = ["S1"]
        best = f.loc["S1", "val_mae"]
        for code in codes[1:]:
            if best - f.loc[code, "val_mae"] > band[tn]:
                kept.append(code)
            best = min(best, f.loc[code, "val_mae"])
        R.loc[R.target == tn, "kept_steps"] = " ".join(kept)
        for d in ("forward blocks", "full-year tests"):
            kept_jobs.append((tn, d, "kept only", kept))
    with ProcessPoolExecutor(max_workers=4) as ex:
        K = pd.DataFrame(list(ex.map(run_one, kept_jobs)))
    R = pd.concat([R, K], ignore_index=True)
    R.to_csv(OUT / "buildup_scores.csv", index=False)
    plots()


def plots() -> None:
    R = pd.read_csv(OUT / "buildup_scores.csv")
    band = noise_band()
    ref = pd.read_csv(Path("/home/suramya/HHP-Prediction/OHC/output/noise_feature_20261015/noise_runs.csv"))
    ref = ref[ref.k == 0].groupby(["target", "set", "design"]).val_mae.mean()
    codes = [s for s, _ in STEPS]
    labels = dict(STEPS)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5), gridspec_kw={"height_ratios": [1.25, 1]})
    for j, tn in enumerate(("tchp", "d26")):
        r = R[R.target == tn]
        ax = axes[0, j]
        for d, col, mk in (("forward blocks", ORANGE, "o"), ("full-year tests", INK2, "D")):
            s = r[r.design == d].set_index("step").loc[codes]
            ax.errorbar(range(len(codes)), s.val_mae, yerr=s.val_sd, fmt=f"-{mk}", color=col, ms=7, mec=SURF,
                        lw=1.8, capsize=0, label=f"validation, {d}")
            k = r[(r.design == d) & (r.step == "kept only")]
            if len(k):
                ax.axhline(k.val_mae.iloc[0], color=col, ls=":", lw=1.5,
                           label=f"kept steps only, {d} ({int(k.n_features.iloc[0])} features): {k.val_mae.iloc[0]:.2f}")
        for sname, ls in (("full recipe", "--"), ("Dr. Jacobs' set", "-.")):
            v = ref.loc[(tn, sname, "forward blocks")]
            ax.axhline(v, color=MUTED, ls=ls, lw=1.2, label=f"{sname}, forward blocks: {v:.2f}")
        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels(codes)
        ax.set_ylabel(f"Validation error, MAE ({UNIT[tn]})")
        ax.set_title(tn.upper(), loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=9.5)
        ax = axes[1, j]
        s = r[r.design == "forward blocks"].set_index("step").loc[codes]
        delta = np.r_[np.nan, -np.diff(s.val_mae.to_numpy())]
        kept = set(str(r.kept_steps.dropna().iloc[0]).split())
        ax.axhspan(-band[tn], band[tn], color=NEUTRAL, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        for i, code in enumerate(codes[1:], start=1):
            ax.bar(i, delta[i], color=(AQUA if code in kept else AXIS), edgecolor=SURF, width=0.7)
            ax.text(i, delta[i] + (0.01 if delta[i] >= 0 else -0.01), f"{delta[i]:+.2f}", ha="center",
                    va="bottom" if delta[i] >= 0 else "top", fontsize=9.5, color=INK2)
        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels([f"{c}\n{labels[c]}" for c in codes], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(f"Improvement over the previous step\n({UNIT[tn]}; up = better)")
        ax.text(0.99, 0.97, f"grey band: chance level ±{band[tn]:.3f} (noise test)\nturquoise: kept, grey: not kept",
                transform=ax.transAxes, va="top", ha="right", fontsize=9.5, color=INK2)
        ax.set_ylim(min(-0.12, np.nanmin(delta) - 0.05), np.nanmax(delta) * 1.15)
        ax.grid(axis="x", visible=False)
    headline(fig, "Building the model step by step, physical features first",
             "Each step adds one block to the one before (forward test blocks; error bars: spread over 3 seeds). A step is "
             "kept only if it improves validation error by more than the chance level from the\nnoise-feature test. Dotted "
             "line: a model built from the kept steps only. Grey reference lines: the full recipe and Dr. Jacobs' 16 "
             "features on the same rows.", top=0.88)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, hspace=0.12, wspace=0.18)
    fig.savefig(OUT / "1_buildup_validation_error_by_step.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, tn in zip(axes, ("tchp", "d26")):
        s = R[(R.target == tn) & (R.design == "forward blocks")].set_index("step").loc[codes]
        ax.plot(range(len(codes)), s.train_mae, "-o", color=BLUE, ms=7, mec=SURF, label="training rows")
        ax.plot(range(len(codes)), s.val_mae, "-D", color=ORANGE, ms=7, mec=SURF, label="validation rows")
        for i, (a, b) in enumerate(zip(s.train_mae, s.val_mae)):
            ax.text(i, (a + b) / 2, f"{b - a:.2f}", ha="center", va="center", fontsize=9, color=INK2)
        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels(codes)
        ax.set_ylabel(f"MAE ({UNIT[tn]})")
        ax.set_title(f"{tn.upper()}: gap between training and validation error (number between the lines)", loc="left",
                     fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    headline(fig, "Does adding hundreds of combined features make the model memorise its training data?",
             "Training and validation error at each step (forward test blocks). A widening gap would mean the extra "
             "features are being used to fit noise in the training rows.", top=0.84)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.1, wspace=0.18)
    fig.savefig(OUT / "2_buildup_training_vs_validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plots()
    else:
        main()
