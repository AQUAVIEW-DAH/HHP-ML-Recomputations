"""Is the learned correction two physical operations? (decomposition test)

Hypothesis:  correction ~ B(position, season) + beta * anomaly
  B       : a smooth large-scale bias map (position and season only)
  anomaly : RTOFS's own local departure from its 1-degree neighbourhood mean
  beta<0  : the correction damps local structure; 1+beta is the fraction of
            RTOFS's local structure that is kept.

Test 1 - how much of the full model's improvement do these two pieces recover?
Test 2 - amplitude or position? Both predict beta<0. A pure amplitude error
  (model feature = a x true feature) gives beta = (1-a)/a, a constant that does
  not depend on how active the ocean is. A displacement error gives a beta that
  depends on feature size relative to the displacement, and leftover error that
  grows with the local gradient (delta ~ -d . grad T). So: beta by eddy-activity
  quintile, and leftover error versus gradient controlling for |anomaly| and
  activity.

Warm rows only (both sides have 26 C water), so the 2026-09 interpolation bug
cannot enter; same locked folds as the benchmark. Robustness: beta re-estimated
without near-edge rows (temperature excess < 1 C), because the neighbourhood
anomaly is biased at the 26 C edge.

Outputs: OHC/output/correction_decomposition_20260924/
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import OHC.run_locked_xgb_physics_semi_ablation as abl
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features
from sklearn.impute import SimpleImputer

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/correction_decomposition_20260924")
RECIPES = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
POS = ["lat", "lon", "abs_lat", "month_sin", "month_cos", "doy_sin", "doy_cos"]
COLS = {"tchp": ("model_tchp_anom_from_1deg_mean", "model_tchp_local_std_1deg", "model_tchp_grad_mag_per_100km"),
        "d26": ("model_d26_anom_from_1deg_mean", "model_d26_local_std_1deg", "model_d26_grad_mag_per_100km")}
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
MOE = {"tchp": 11.189, "d26": 10.553}
NB = 1000
rng = np.random.default_rng(0)


def xgb():
    return abl._xgb_model()


def date_boot(dates, fn, n=NB):
    """Date-block bootstrap: resample dates with replacement, keep multiplicity."""
    u = np.unique(dates); idx = {d: np.flatnonzero(dates == d) for d in u}
    out = []
    for _ in range(n):
        pick = rng.choice(u, len(u), replace=True)
        rows = np.concatenate([idx[d] for d in pick])
        out.append(fn(rows))
    return np.array(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fn = json.loads(abl.FOLD_PATH.read_text())
    df = abl._merge_feature_tables()
    score_rows, beta_rows, grad_rows, shape = [], [], [], {}
    for t in TARGETS:
        tn = t.name
        acol, scol, gcol = COLS[tn]
        full_cols = list(abl.FEATURE_SETS_BY_TARGET[tn][RECIPES[tn]])
        w = df[pd.notna(df[t.obs_col]) & pd.notna(df[t.model_col]) & pd.notna(df[t.delta_col])].copy()
        w = _prepare_features(w).reset_index(drop=True)
        y = w[t.delta_col].to_numpy(float)
        a = w[acol].fillna(0.0).to_numpy(float)
        act = w[scol].fillna(w[scol].median()).to_numpy(float)
        grad = w[gcol].fillna(w[gcol].median()).to_numpy(float)
        te = w["model_temp_excess_26c"].to_numpy(float)
        P = w[POS].to_numpy(float)
        F = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(
            w[full_cols].apply(pd.to_numeric, errors="coerce")), columns=full_cols).to_numpy()
        dates = w["date"].dt.strftime("%Y%m%d").to_numpy()
        folds = _build_forward_folds(sorted(np.unique(dates).tolist()),
                                     n_folds=fn["n_folds"], embargo_dates=fn["embargo_dates"])
        oof = {k: np.full(len(w), np.nan) for k in
               ("B only", "B + beta*anomaly", "B + beta(activity)*anomaly", "trees on position+season+anomaly", "full recipe")}
        betas = []
        for f in folds:
            tr = np.isin(dates, f["train_dates"]); va = np.isin(dates, f["val_dates"])
            m = xgb(); m.fit(P[tr], y[tr]); oof["B only"][va] = m.predict(P[va])
            beta = 0.0
            for _ in range(4):  # backfitting: B and beta jointly
                mb = xgb(); mb.fit(P[tr], y[tr] - beta * a[tr])
                r = y[tr] - mb.predict(P[tr])
                beta = float(a[tr] @ r / (a[tr] @ a[tr]))
            betas.append(beta)
            Bva = mb.predict(P[va])
            oof["B + beta*anomaly"][va] = Bva + beta * a[va]
            edges = np.quantile(act[tr], [0.2, 0.4, 0.6, 0.8])
            btr, bva = np.digitize(act[tr], edges), np.digitize(act[va], edges)
            bk = np.array([a[tr][btr == k] @ r[btr == k] / max(a[tr][btr == k] @ a[tr][btr == k], 1e-9) for k in range(5)])
            oof["B + beta(activity)*anomaly"][va] = Bva + bk[bva] * a[va]
            m2 = xgb(); m2.fit(np.column_stack([P[tr], a[tr]]), y[tr])
            oof["trees on position+season+anomaly"][va] = m2.predict(np.column_stack([P[va], a[va]]))
            m3 = xgb(); m3.fit(F[tr], y[tr]); oof["full recipe"][va] = m3.predict(F[va])
            print(tn, "fold", f["fold"], "beta", round(beta, 3), flush=True)

        v = np.isfinite(oof["full recipe"])
        raw = float(np.abs(y[v]).mean()); full = float(np.abs(y[v] - oof["full recipe"][v]).mean())
        score_rows.append({"target": tn, "model": "raw RTOFS", "mae": raw, "pct_recovered": 0.0, "n_params_beyond_B": ""})
        for k, p in oof.items():
            mae = float(np.abs(y[v] - p[v]).mean())
            score_rows.append({"target": tn, "model": k, "mae": mae,
                               "pct_recovered": 100 * (raw - mae) / (raw - full),
                               "n_params_beyond_B": {"B + beta*anomaly": 1, "B + beta(activity)*anomaly": 5}.get(k, "")})
        score_rows.append({"target": tn, "model": "MoE blend (recommended)", "mae": MOE[tn],
                           "pct_recovered": 100 * (raw - MOE[tn]) / (raw - full), "n_params_beyond_B": ""})

        # ---- test 2a: beta by eddy-activity quintile, on HONEST (out-of-fold) B residuals
        rB = y - oof["B only"]
        vv = v & np.isfinite(rB)
        q = np.quantile(act[vv], [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(act, q)
        for k in range(5):
            sel = np.flatnonzero(vv & (bins == k))
            est = float(a[sel] @ rB[sel] / (a[sel] @ a[sel]))
            bs = date_boot(dates[sel], lambda rows, s=sel: a[s][rows] @ rB[s][rows] / (a[s][rows] @ a[s][rows]), n=300)
            beta_rows.append({"target": tn, "activity_quintile": k + 1, "activity_median": float(np.median(act[sel])),
                              "beta": est, "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                              "retained_fraction": 1 + est, "n": int(len(sel))})
        # pooled beta, and robustness without near-edge rows
        for lab, msk in (("all warm rows", vv), ("excluding near-edge (temp excess < 1 C)", vv & (te >= 1.0))):
            s = np.flatnonzero(msk)
            est = float(a[s] @ rB[s] / (a[s] @ a[s]))
            bs = date_boot(dates[s], lambda rows, s=s: a[s][rows] @ rB[s][rows] / (a[s][rows] @ a[s][rows]), n=300)
            beta_rows.append({"target": tn, "activity_quintile": lab, "beta": est,
                              "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                              "retained_fraction": 1 + est, "n": int(len(s))})
            print(tn, lab, "beta", round(est, 3), flush=True)
        # asymmetry: warm bumps vs cold dips
        for lab, msk in (("warm bumps (anomaly > 0)", vv & (a > 0)), ("cold dips (anomaly < 0)", vv & (a < 0))):
            s = np.flatnonzero(msk)
            est = float(a[s] @ rB[s] / (a[s] @ a[s]))
            beta_rows.append({"target": tn, "activity_quintile": lab, "beta": est,
                              "retained_fraction": 1 + est, "n": int(len(s))})
        # shape of rB vs anomaly
        qa = np.quantile(a[vv], np.linspace(0.02, 0.98, 25))
        ba = np.digitize(a, qa)
        shape[tn] = [(float(np.median(a[vv & (ba == k)])), float(np.mean(rB[vv & (ba == k)])))
                     for k in range(1, len(qa)) if (vv & (ba == k)).sum() > 50]

        # ---- test 2b: leftover error vs gradient, controlling for |anomaly| and activity
        e = y - oof["B + beta*anomaly"]
        gq = np.quantile(grad[vv], [0.2, 0.4, 0.6, 0.8]); gb = np.digitize(grad, gq)
        for k in range(5):
            sel = vv & (gb == k)
            grad_rows.append({"target": tn, "gradient_quintile": k + 1, "gradient_median": float(np.median(grad[sel])),
                              "rms_leftover": float(np.sqrt(np.mean(e[sel] ** 2))),
                              "mean_abs_leftover": float(np.mean(np.abs(e[sel])))})
        s = np.flatnonzero(vv)
        Z = lambda x: (x - x[s].mean()) / x[s].std()
        X = np.column_stack([np.ones(len(w)), Z(np.abs(a)), Z(act), Z(grad)])
        def coef(rows, s=s):
            c, *_ = np.linalg.lstsq(X[s][rows], np.abs(e[s][rows]), rcond=None)
            return c[3]
        c_est = coef(np.arange(len(s)))
        bs = date_boot(dates[s], coef, n=200)
        grad_rows.append({"target": tn, "gradient_quintile": "partial effect of gradient on |leftover| (per sd)",
                          "rms_leftover": float(c_est), "mean_abs_leftover": np.nan,
                          "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))})
        print(tn, "gradient partial effect", round(c_est, 3), [round(np.percentile(bs, p), 3) for p in (2.5, 97.5)], flush=True)

    S = pd.DataFrame(score_rows); S.to_csv(OUT / "decomposition_scores.csv", index=False)
    Bt = pd.DataFrame(beta_rows); Bt.to_csv(OUT / "beta_by_activity.csv", index=False)
    G = pd.DataFrame(grad_rows); G.to_csv(OUT / "leftover_vs_gradient.csv", index=False)

    # ---- figure 1: how much skill do two operations recover?
    order = ["raw RTOFS", "B only", "B + beta*anomaly", "B + beta(activity)*anomaly",
             "trees on position+season+anomaly", "full recipe", "MoE blend (recommended)"]
    lab = {"raw RTOFS": "raw RTOFS", "B only": "B: bias map\n(position + season)",
           "B + beta*anomaly": "B + β·anomaly\n(ONE extra number)", "B + beta(activity)*anomaly": "B + β(activity)·anomaly\n(five numbers)",
           "trees on position+season+anomaly": "trees on the same\ntwo ingredients",
           "full recipe": "full model\n(34/35 inputs)", "MoE blend (recommended)": "MoE blend\n(recommended)"}
    col = ["#dc2626", "#94a3b8", "#2563eb", "#1d4ed8", "#7c3aed", "#166534", "#15803d"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), constrained_layout=True)
    for ax, tn in zip(axes, ("tchp", "d26")):
        s = S[S.target == tn].set_index("model").loc[order]
        yy = np.arange(len(order))[::-1]
        ax.barh(yy, s.mae, color=col)
        for i, (k, r) in zip(yy, s.iterrows()):
            txt = f"{r.mae:.2f}" + (f"   ({r.pct_recovered:.0f}% of the full model's gain)" if k not in ("raw RTOFS",) else "")
            ax.text(r.mae + 0.05, i, txt, va="center", fontsize=8.5)
        ax.set_yticks(yy); ax.set_yticklabels([lab[k] for k in order], fontsize=8.8)
        ax.set_xlim(0, s.mae.max() * 1.55); ax.set_xlabel(f"out-of-fold MAE ({UNIT[tn]})")
        ax.set_title(tn.upper(), fontsize=12); ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Is the learned correction just two physical operations?\n"
                 "a static bias map, plus shrinking the local structure RTOFS claims toward its neighbourhood mean", fontsize=13)
    fig.savefig(OUT / "decomposition_skill.png", dpi=160); plt.close(fig)

    # ---- figure 2: shape, and beta by activity
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for j, tn in enumerate(("tchp", "d26")):
        ax = axes[0, j]
        xs, ys = zip(*shape[tn])
        pooled = Bt[(Bt.target == tn) & (Bt.activity_quintile == "all warm rows")].iloc[0]
        ax.plot(xs, ys, "o", color="#2563eb", label="binned mean of (error − bias map)")
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, pooled.beta * xx, "-", color="#dc2626", lw=2, label=f"one straight line, β = {pooled.beta:.2f}")
        ax.axhline(0, color="#94a3b8", lw=0.8); ax.axvline(0, color="#94a3b8", lw=0.8)
        ax.set_xlabel(f"local anomaly RTOFS claims ({UNIT[tn]})"); ax.set_ylabel(f"error left after the bias map ({UNIT[tn]})")
        ax.set_title(f"{tn.upper()}: is the damping a straight line?"); ax.grid(alpha=0.15); ax.legend(fontsize=9)
        ax = axes[1, j]
        q = Bt[(Bt.target == tn) & Bt.activity_quintile.apply(lambda v: isinstance(v, (int, np.integer)))]
        ax.errorbar(q.activity_median, q.beta, yerr=[q.beta - q.ci_lo, q.ci_hi - q.beta], fmt="o-",
                    color="#2563eb", capsize=4, lw=2)
        ax.axhline(pooled.beta, color="#dc2626", ls="--", lw=1.2, label=f"pooled β = {pooled.beta:.2f}")
        ax.set_xscale("log"); ax.set_xlabel(f"eddy activity: local standard deviation ({UNIT[tn]}, log)")
        ax.set_ylabel("damping coefficient β")
        ax2 = ax.secondary_yaxis("right", functions=(lambda b: 1 + b, lambda r: r - 1))
        ax2.set_ylabel("fraction of RTOFS local structure kept (1+β)")
        ax.set_title(f"{tn.upper()}: flat = amplitude error · sloped = position error", fontsize=11)
        ax.grid(alpha=0.15); ax.legend(fontsize=9)
    fig.suptitle("Amplitude or position? Both predict β < 0; only a position error makes β depend on eddy activity", fontsize=13)
    fig.savefig(OUT / "damping_amplitude_or_position.png", dpi=160); plt.close(fig)

    # ---- figure 3: leftover vs gradient
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for tn, c in (("tchp", "#2563eb"), ("d26", "#16a34a")):
        g = G[(G.target == tn) & G.gradient_quintile.apply(lambda v: isinstance(v, (int, np.integer)))]
        pe = G[(G.target == tn) & ~G.gradient_quintile.apply(lambda v: isinstance(v, (int, np.integer)))].iloc[0]
        ax.plot(g.gradient_median, g.mean_abs_leftover, "o-", color=c,
                label=f"{tn.upper()}  (gradient effect with |anomaly| and activity held fixed: "
                      f"{pe.rms_leftover:+.2f} per s.d., 95% [{pe.ci_lo:+.2f}, {pe.ci_hi:+.2f}])")
    ax.set_xscale("log"); ax.set_xlabel("local gradient of the RTOFS field (per 100 km, log)")
    ax.set_ylabel("mean |error left after bias map + damping|")
    ax.set_title("A position error leaves behind error that grows with the gradient\n"
                 "(δ ≈ −d·∇T); a pure amplitude error does not", fontsize=11.5)
    ax.grid(alpha=0.15); ax.legend(fontsize=8)
    fig.savefig(OUT / "leftover_vs_gradient.png", dpi=160); plt.close(fig)
    print(S.to_string(index=False)); print(Bt.to_string(index=False)); print(G.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
