"""How much of RTOFS's local structure is real, as a function of scale?

Generalises run_correction_decomposition.py, which found that a bias map plus
ONE damping coefficient on RTOFS's 1-degree local anomaly recovers 88% (TCHP)
and 82% (D26) of the full model, keeping only ~12% / ~19% of that structure.

Here RTOFS's point value r is split into nested bands, like a Laplacian
pyramid, using the neighbourhood box means already computed on the native
1/12-degree grid (box half-widths of 6, 12 and 25 cells):

    r = m2 + (m1 - m2) + (m05 - m1) + (r - m05)
          4-deg mean   band3          band2          band1
                       ~220-460 km    ~110-220 km    finer than ~110 km

and the correction is modelled as
    delta ~ B(position, season) + b1*band1 + b2*band2 + b3*band3.
The corrected field is then B + m2 + sum_k (1 + b_k) * band_k, so (1 + b_k) is
the fraction of RTOFS's structure at scale k that the floats bear out.

Warm rows (both sides have 26 C water) and primary profiles only, after the
2026-09-24 rebuild. Robustness: repeated without near-edge rows, because the
box means exclude sub-threshold cells and so are biased at the 26 C edge.

Outputs: OHC/output/scale_decomposition_20260924/
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/scale_decomposition_20260924")
POS = ["lat", "lon", "abs_lat", "month_sin", "month_cos", "doy_sin", "doy_cos"]
FIELD = {"tchp": "tchp", "d26": "d26"}
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
BANDS = [("band1", "finer than ~110 km"), ("band2", "~110 to 220 km"), ("band3", "~220 to 460 km")]
rng = np.random.default_rng(0)


def date_boot(dates, fn, n=300):
    u = np.unique(dates)
    idx = {d: np.flatnonzero(dates == d) for d in u}
    out = []
    for _ in range(n):
        pick = rng.choice(u, len(u), replace=True)
        out.append(fn(np.concatenate([idx[d] for d in pick])))
    return np.array(out)


def ols(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fn = json.loads(abl.FOLD_PATH.read_text())
    df = abl._merge_feature_tables()
    if "is_primary_profile" not in df.columns:
        raise RuntimeError("run OHC/rebuild_tables_20260924.py first: is_primary_profile is missing")
    scores, coefs, corr_rows = [], [], []
    for t in TARGETS:
        tn, f = t.name, FIELD[t.name]
        w = df[pd.notna(df[t.obs_col]) & pd.notna(df[t.model_col]) & pd.notna(df[t.delta_col])
               & df["is_primary_profile"].astype(bool)].copy()
        w = _prepare_features(w).reset_index(drop=True)
        r = w[t.model_col].to_numpy(float)
        m05, m1, m2 = (w[f"model_{f}_local_mean_{s}"].to_numpy(float) for s in ("halfdeg", "1deg", "2deg"))
        ok = np.isfinite(m05) & np.isfinite(m1) & np.isfinite(m2)
        w, r, m05, m1, m2 = w[ok].reset_index(drop=True), r[ok], m05[ok], m1[ok], m2[ok]
        bands = np.column_stack([r - m05, m05 - m1, m1 - m2])
        anom1 = r - m1
        y = w[t.delta_col].to_numpy(float)
        P = w[POS].to_numpy(float)
        te = w["model_temp_excess_26c"].to_numpy(float)
        dates = w["date"].dt.strftime("%Y%m%d").to_numpy()
        folds = _build_forward_folds(sorted(np.unique(dates).tolist()),
                                     n_folds=fn["n_folds"], embargo_dates=fn["embargo_dates"])
        C = np.corrcoef(bands, rowvar=False)
        corr_rows.append({"target": tn, "r(b1,b2)": C[0, 1], "r(b1,b3)": C[0, 2], "r(b2,b3)": C[1, 2]})

        oof = {k: np.full(len(w), np.nan) for k in
               ("B only", "B + one 1-degree anomaly term", "B + three scale bands", "full recipe")}
        full_cols = list(abl.FEATURE_SETS_BY_TARGET[tn][
            {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}[tn]])
        from sklearn.impute import SimpleImputer
        F = SimpleImputer(strategy="median").fit_transform(w[full_cols].apply(pd.to_numeric, errors="coerce"))
        for fo in folds:
            tr, va = np.isin(dates, fo["train_dates"]), np.isin(dates, fo["val_dates"])
            m = abl._xgb_model(); m.fit(P[tr], y[tr]); oof["B only"][va] = m.predict(P[va])
            for key, Z in (("B + one 1-degree anomaly term", anom1[:, None]), ("B + three scale bands", bands)):
                beta = np.zeros(Z.shape[1])
                for _ in range(4):  # backfit B jointly with the linear band terms
                    mb = abl._xgb_model(); mb.fit(P[tr], y[tr] - Z[tr] @ beta)
                    beta = ols(Z[tr], y[tr] - mb.predict(P[tr]))
                oof[key][va] = mb.predict(P[va]) + Z[va] @ beta
            mf = abl._xgb_model(); mf.fit(F[tr], y[tr]); oof["full recipe"][va] = mf.predict(F[va])
            print(tn, "fold", fo["fold"], flush=True)

        v = np.isfinite(oof["full recipe"])
        raw = float(np.abs(y[v]).mean())
        full = float(np.abs(y[v] - oof["full recipe"][v]).mean())
        scores.append({"target": tn, "model": "raw RTOFS", "mae": raw, "pct_of_full_gain": 0.0, "rows": int(v.sum())})
        for k, p in oof.items():
            mae = float(np.abs(y[v] - p[v]).mean())
            scores.append({"target": tn, "model": k, "mae": mae,
                           "pct_of_full_gain": 100 * (raw - mae) / (raw - full), "rows": int(v.sum())})

        # band coefficients on HONEST (out-of-fold) bias-map residuals, jointly
        rB = y - oof["B only"]
        for lab, msk in (("all warm primary rows", v), ("excluding near-edge (temp excess < 1 C)", v & (te >= 1.0))):
            s = np.flatnonzero(msk)
            est = ols(bands[s], rB[s])
            bs = date_boot(dates[s], lambda rows, s=s: ols(bands[s][rows], rB[s][rows]))
            for k, (bk, desc) in enumerate(BANDS):
                coefs.append({"target": tn, "subset": lab, "band": bk, "scale": desc,
                              "beta": float(est[k]), "ci_lo": float(np.percentile(bs[:, k], 2.5)),
                              "ci_hi": float(np.percentile(bs[:, k], 97.5)), "retained": float(1 + est[k]),
                              "band_std": float(bands[s, k].std()), "n": int(len(s))})
            print(tn, lab, "betas", np.round(est, 3), flush=True)

    S = pd.DataFrame(scores); S.to_csv(OUT / "scale_scores.csv", index=False)
    K = pd.DataFrame(coefs); K.to_csv(OUT / "retained_fraction_by_scale.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "band_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    xs = np.arange(3)
    for ax, tn in zip(axes, ("tchp", "d26")):
        for lab, c, off in (("all warm primary rows", "#2563eb", -0.07),
                            ("excluding near-edge (temp excess < 1 C)", "#94a3b8", 0.07)):
            q = K[(K.target == tn) & (K.subset == lab)].set_index("band").loc[[b for b, _ in BANDS]]
            ax.errorbar(xs + off, q.retained, yerr=[q.retained - (1 + q.ci_lo), (1 + q.ci_hi) - q.retained],
                        fmt="o-", color=c, capsize=4, lw=2, ms=7, label=lab)
        ax.axhline(0, color="#dc2626", ls=":", lw=1.2)
        ax.axhline(1, color="#166534", ls=":", lw=1.2)
        ax.text(2.35, 1.0, " all of it real", color="#166534", va="center", fontsize=8.5)
        ax.text(2.35, 0.0, " none of it real", color="#dc2626", va="center", fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([d for _, d in BANDS], fontsize=9.5)
        ax.set_xlim(-0.4, 2.9)
        ax.set_xlabel("scale of the structure RTOFS shows")
        ax.set_ylabel("fraction of it the floats bear out (1 + β)")
        ax.set_title(tn.upper(), fontsize=12)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("How much of RTOFS's local structure is real, at each scale?\n"
                 "each band is RTOFS's structure between two neighbourhood-average sizes; primary Argo profiles, warm rows",
                 fontsize=13)
    fig.savefig(OUT / "retained_fraction_by_scale.png", dpi=160)
    plt.close(fig)
    print(S.to_string(index=False))
    print(K.to_string(index=False))
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
