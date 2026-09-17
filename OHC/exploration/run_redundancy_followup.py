"""Follow-up to the redundancy analysis (Dr. Jacobs' original ask, completed).

Part 1 — the correlation matrix he actually asked for: the N x N cross
correlation of features EXTENDED with the target errors as extra rows and
columns, so relatedness-to-target and relatedness-to-each-other are readable
from one picture. Both targets appear in both matrices, signed and absolute.

Part 2 — which directions actually carry the predictive information:
  (a) loadings of the first 8 principal components (what they physically are)
  (b) the boosted model's importance per component against eigenvalue rank
  (c) the decisive test: principal components reordered by relevance to the
      error instead of by variance, rescored
  (d) partial least squares, which builds components to maximise covariance
      with the target directly
  (e) which features the boosted trees split on in their first two levels

Outputs: OHC/output/redundancy_followup_20260916/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/redundancy_followup_20260916")
RECIPES = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
M_SWEEP = [2, 5, 8, 12, 20]
REF = {"tchp": 11.397, "d26": 10.755}
HIGHLIGHT = "model_temp_excess_26c"
ERR_COLS = ["delta_tchp_kj_per_cm2", "delta_d26_m"]
ERR_LABEL = {"delta_tchp_kj_per_cm2": "ERROR: TCHP (Argo-RTOFS)",
             "delta_d26_m": "ERROR: D26 (Argo-RTOFS)",
             "abs_delta_tchp_kj_per_cm2": "|ERROR|: TCHP",
             "abs_delta_d26_m": "|ERROR|: D26"}


def prepare(target):
    df = abl._merge_feature_tables()
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col])
              & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work).reset_index(drop=True)
    cols = list(abl.FEATURE_SETS_BY_TARGET[target.name][RECIPES[target.name]])
    X = work[cols].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=cols)
    return work, cols, X


def part1_matrix(work, cols, X, tname):
    err = pd.DataFrame(index=X.index)
    for c in ERR_COLS:
        v = pd.to_numeric(work[c], errors="coerce")
        err[c] = v
        err["abs_" + c] = v.abs()
    err = err.fillna(err.median())
    extra = list(err.columns)
    full = pd.concat([X, err], axis=1)
    R = np.nan_to_num(spearmanr(full.to_numpy(), axis=0).correlation)
    A = np.abs(R)
    nf = len(cols)
    Afeat = A[:nf, :nf].copy()
    np.fill_diagonal(Afeat, 1.0)
    d = np.clip(1.0 - Afeat, 0, None)
    np.fill_diagonal(d, 0.0)
    order = list(leaves_list(linkage(squareform(d, checks=False), method="average")))
    idx = list(range(nf, nf + len(extra))) + order
    labels = [ERR_LABEL[extra[i - nf]] for i in range(nf, nf + len(extra))] + [cols[i] for i in order]
    M = A[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(15.5, 13.5), constrained_layout=True)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text().startswith(("ERROR", "|ERROR|")):
            t.set_color("#dc2626"); t.set_fontweight("bold")
        elif t.get_text() == HIGHLIGHT:
            t.set_color("#2563eb"); t.set_fontweight("bold")
    k = len(extra)
    ax.add_patch(Rectangle((-0.5, -0.5), n, k, fill=False, edgecolor="#dc2626", lw=2.0))
    ax.add_patch(Rectangle((-0.5, -0.5), k, n, fill=False, edgecolor="#dc2626", lw=2.0))
    if HIGHLIGHT in labels:
        h = labels.index(HIGHLIGHT)
        ax.add_patch(Rectangle((-0.5, h - 0.5), n, 1, fill=False, edgecolor="#2563eb", lw=1.6))
        ax.add_patch(Rectangle((h - 0.5, -0.5), 1, n, fill=False, edgecolor="#2563eb", lw=1.6))
    fig.colorbar(im, ax=ax, shrink=0.65).set_label("|Spearman correlation|")
    ax.set_title(f"{tname.upper()} recipe: features against each other AND against the forecast error\n"
                 f"red strip = the two target errors, signed and absolute (the rows Dr. Jacobs asked for)\n"
                 f"bright blocks away from the red strip are duplicates that predict nothing; "
                 f"blue row is the boundary-case detector, correlation near zero yet highly valuable",
                 fontsize=11.5)
    fig.savefig(OUT / f"{tname}_correlation_matrix_with_error.png", dpi=150)
    plt.close(fig)

    rows = []
    for i, c in enumerate(cols):
        rows.append({"target": tname, "feature": c,
                     **{ERR_LABEL[e]: float(A[i, nf + j]) for j, e in enumerate(extra)},
                     "max_abs_corr_to_other_features": float(np.max(np.delete(A[i, :nf], i))),
                     "mean_abs_corr_to_other_features": float(np.mean(np.delete(A[i, :nf], i)))})
    return pd.DataFrame(rows)


def part2(work, cols, X, target, fold_note):
    tname = target.name
    y = work[target.delta_col].to_numpy(float)
    y_obs = work[target.obs_col].to_numpy(float)
    y_mod = work[target.model_col].to_numpy(float)
    date_str = work["date"].dt.strftime("%Y%m%d")
    folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                 n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
    d = len(cols)
    Xv = X.to_numpy()
    rows, imps, splits = [], [], defaultdict(float)

    def score(oof):
        v = np.isfinite(oof)
        return float(np.abs(y_mod + oof - y_obs)[v].mean())

    for m in M_SWEEP:
        for mode in ("pca_relevance", "pls"):
            oof = np.full(len(work), np.nan)
            for fold in folds:
                tr = date_str.isin(set(fold["train_dates"])).to_numpy()
                va = date_str.isin(set(fold["val_dates"])).to_numpy()
                sc = StandardScaler().fit(Xv[tr])
                Ztr, Zva = sc.transform(Xv[tr]), sc.transform(Xv[va])
                if mode == "pca_relevance":
                    p = PCA(n_components=d, random_state=0).fit(Ztr)
                    Ttr, Tva = p.transform(Ztr), p.transform(Zva)
                    rel = np.array([abs(np.corrcoef(Ttr[:, j], y[tr])[0, 1]) for j in range(d)])
                    keep = np.argsort(rel)[::-1][:m]
                    Ttr, Tva = Ttr[:, keep], Tva[:, keep]
                else:
                    p = PLSRegression(n_components=m, scale=False).fit(Ztr, y[tr])
                    Ttr, Tva = p.transform(Ztr), p.transform(Zva)
                mod = abl._xgb_model(); mod.fit(Ttr, y[tr])
                oof[va] = mod.predict(Tva)
            rows.append({"target": tname, "method": mode, "n_components": m, "mae": score(oof)})
            print(tname, mode, m, round(rows[-1]["mae"], 3), flush=True)

    # component importance on the full PCA basis, and tree first splits on raw features
    for fold in folds:
        tr = date_str.isin(set(fold["train_dates"])).to_numpy()
        sc = StandardScaler().fit(Xv[tr])
        p = PCA(n_components=d, random_state=0).fit(sc.transform(Xv[tr]))
        mod = abl._xgb_model(); mod.fit(p.transform(sc.transform(Xv[tr])), y[tr])
        imps.append(mod.feature_importances_)
        raw = abl._xgb_model(); raw.fit(pd.DataFrame(Xv[tr], columns=cols), y[tr])
        td = raw.get_booster().trees_to_dataframe()
        top = td[(td.Feature != "Leaf") & (td.ID.str.split("-").str[1].astype(int) < 3)]
        for f, g in top.groupby("Feature")["Gain"].sum().items():
            splits[f] += float(g)
    imp = np.mean(np.stack(imps), axis=0)

    # loadings of the first 8 components from the final training block
    tr = date_str.isin(set(folds[-1]["train_dates"])).to_numpy()
    sc = StandardScaler().fit(Xv[tr])
    p8 = PCA(n_components=8, random_state=0).fit(sc.transform(Xv[tr]))
    L = p8.components_
    ev = p8.explained_variance_ratio_
    ordf = np.argsort(-np.abs(L).max(axis=0))
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    im = ax.imshow(L[:, ordf], cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"PC{i+1}  ({100*ev[i]:.0f}% of variance)" for i in range(8)], fontsize=9)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([cols[i] for i in ordf], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("loading")
    ax.set_title(f"{tname.upper()}: what the first eight principal components are made of\n"
                 "each row is one component; colour is how strongly each input contributes to it", fontsize=12)
    fig.savefig(OUT / f"{tname}_pc_loadings.png", dpi=150)
    plt.close(fig)

    return pd.DataFrame(rows), imp, dict(splits)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    var_curve = pd.read_csv("/home/suramya/HHP-Prediction/OHC/output/feature_redundancy_20260909/skill_vs_components.csv")
    corr_rows, curves, imps, splits = [], [], {}, {}
    for target in TARGETS:
        t = target.name
        work, cols, X = prepare(target)
        corr_rows.append(part1_matrix(work, cols, X, t))
        c, imp, sp = part2(work, cols, X, target, fold_note)
        curves.append(c); imps[t] = imp; splits[t] = sp
    corr = pd.concat(corr_rows); corr.to_csv(OUT / "feature_vs_error_correlations.csv", index=False)
    cur = pd.concat(curves); cur.to_csv(OUT / "skill_vs_components_three_ways.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        v = var_curve[(var_curve.target == t) & (var_curve.n_components.isin(M_SWEEP))]
        ax.plot(v.n_components, v.mae, "o-", color="#94a3b8", label="principal components, ordered by variance")
        for mode, c, lab in (("pca_relevance", "#2563eb", "principal components, ordered by relevance to the error"),
                             ("pls", "#16a34a", "partial least squares (built against the error)")):
            s = cur[(cur.target == t) & (cur.method == mode)]
            ax.plot(s.n_components, s.mae, "o-", color=c, label=lab)
        ax.axhline(REF[t], color="#166534", ls="--", lw=1.2, label=f"all raw inputs ({REF[t]:.2f})")
        ax.set_xlabel("number of components kept"); ax.set_ylabel("out-of-fold MAE")
        ax.set_title(t.upper()); ax.grid(alpha=0.15); ax.legend(fontsize=8.5)
    fig.suptitle("How many independent directions of USEFUL information do the inputs carry?\n"
                 "ordering components by variance is not the same as ordering them by what predicts the error", fontsize=13)
    fig.savefig(OUT / "skill_vs_components_three_ways.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        y = imps[t]
        ax.bar(np.arange(1, len(y) + 1), y, color="#2563eb")
        ax.set_xlabel("principal component, ordered by variance (1 = largest)")
        ax.set_ylabel("share of the model's splitting gain")
        ax.set_title(f"{t.upper()}: which components the model actually uses"); ax.grid(alpha=0.15, axis="y")
    fig.suptitle("If the information lived in the high-variance directions, these bars would fall off to the right", fontsize=13)
    fig.savefig(OUT / "component_importance_vs_rank.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        s = pd.Series(splits[t]).sort_values(ascending=True).tail(14)
        s = s / s.sum()
        ax.barh(s.index, s.values, color="#f59e0b")
        ax.set_xlabel("share of gain in the first three levels of the trees")
        ax.set_title(t.upper()); ax.tick_params(axis="y", labelsize=8); ax.grid(alpha=0.15, axis="x")
    fig.suptitle("What the boosted trees split on first, before anything else", fontsize=13)
    fig.savefig(OUT / "first_splits.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(splits).to_csv(OUT / "first_split_gain.csv")
    print(cur.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
