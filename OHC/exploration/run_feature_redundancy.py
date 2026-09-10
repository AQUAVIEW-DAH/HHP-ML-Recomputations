"""Formal redundancy analysis of the full feature recipes (Dr. Jacobs, 2026-09).

Three tiers:
1. His score: s_j = |rho(feature_j, error)| - sum_k |rho(feature_j, feature_k)|
   (Spearman), sorted; plus a mean-normalised variant so large families are
   not penalised by size alone. Clustered |rho| heatmap.
2. Effective dimension: eigenvalue spectrum of the feature correlation matrix,
   participation ratio, and the number of components for 90/95/99% variance.
3. Skill vs dimension: locked XGBoost on the top-m principal components,
   m in {2, 5, 8, 12, 20, all}. If a small m reproduces the full-recipe
   score, the 35 features carry only ~m dimensions of usable information.

Outputs: OHC/output/feature_redundancy_20260909/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/feature_redundancy_20260909")
RECIPES = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
M_SWEEP = [2, 5, 8, 12, 20]
REF = {"tchp": 11.397, "d26": 10.755}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = abl._merge_feature_tables()
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    score_rows, dim_rows, skill_rows = [], [], []
    for target in TARGETS:
        t = target.name
        cols = abl.FEATURE_SETS_BY_TARGET[t][RECIPES[t]]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        X = work[cols].apply(pd.to_numeric, errors="coerce")
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=cols)
        y = work[target.delta_col].to_numpy(float)

        # --- tier 1: Spearman correlations and his score ---
        R = spearmanr(X.to_numpy(), axis=0).correlation
        R = np.nan_to_num(R)
        r_y = np.array([spearmanr(X[c], y).correlation for c in cols])
        absR = np.abs(R); np.fill_diagonal(absR, 0.0)
        d = len(cols)
        for j, c in enumerate(cols):
            score_rows.append({"target": t, "feature": c, "abs_corr_to_error": abs(r_y[j]),
                               "sum_abs_corr_to_others": absR[j].sum(),
                               "mean_abs_corr_to_others": absR[j].sum() / (d - 1),
                               "max_abs_corr_to_others": absR[j].max(),
                               "jacobs_score": abs(r_y[j]) - absR[j].sum(),
                               "normalised_score": abs(r_y[j]) - absR[j].sum() / (d - 1)})
        order = leaves_list(linkage(1 - np.abs(R), method="average"))
        fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
        im = ax.imshow(np.abs(R)[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(d)); ax.set_yticks(range(d))
        ax.set_xticklabels([cols[i] for i in order], rotation=90, fontsize=7.5)
        ax.set_yticklabels([cols[i] for i in order], fontsize=7.5)
        ax.set_title(f"{t.upper()} recipe: |Spearman correlation| between features, clustered\n(bright blocks = families of near-duplicates)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.savefig(OUT / f"{t}_feature_correlation_heatmap.png", dpi=150)
        plt.close(fig)

        # --- tier 2: eigen spectrum ---
        Z = StandardScaler().fit_transform(X)
        C = np.corrcoef(Z, rowvar=False)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        ev = np.clip(ev, 0, None)
        frac = np.cumsum(ev) / ev.sum()
        pr = ev.sum() ** 2 / (ev ** 2).sum()
        dim_rows.append({"target": t, "n_features": d, "participation_ratio": float(pr),
                         "n_for_90pct": int(np.searchsorted(frac, 0.90) + 1),
                         "n_for_95pct": int(np.searchsorted(frac, 0.95) + 1),
                         "n_for_99pct": int(np.searchsorted(frac, 0.99) + 1),
                         "n_eig_gt_1": int((ev > 1).sum())})

        # --- tier 3: skill vs number of principal components ---
        unique_dates = sorted(pd.Series(work["date"].dt.strftime("%Y%m%d").unique()).tolist())
        folds = _build_forward_folds(unique_dates, n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        date_str = work["date"].dt.strftime("%Y%m%d")
        y_obs = work[target.obs_col].to_numpy(float); y_mod = work[target.model_col].to_numpy(float)
        for m in M_SWEEP + [d]:
            oof = np.full(len(work), np.nan)
            for fold in folds:
                tr = date_str.isin(set(fold["train_dates"])).to_numpy()
                va = date_str.isin(set(fold["val_dates"])).to_numpy()
                sc = StandardScaler().fit(X.to_numpy()[tr])
                Ztr, Zva = sc.transform(X.to_numpy()[tr]), sc.transform(X.to_numpy()[va])
                if m < d:
                    pca = PCA(n_components=m, random_state=0).fit(Ztr)
                    Ztr, Zva = pca.transform(Ztr), pca.transform(Zva)
                model = abl._xgb_model(); model.fit(Ztr, y[tr])
                oof[va] = model.predict(Zva)
            v = np.isfinite(oof)
            skill_rows.append({"target": t, "n_components": m, "mae": float(np.abs(y_mod + oof - y_obs)[v].mean())})
            print(t, "m =", m, "MAE", skill_rows[-1]["mae"], flush=True)

    scores = pd.DataFrame(score_rows).sort_values(["target", "jacobs_score"], ascending=[True, False])
    scores.to_csv(OUT / "jacobs_feature_scores.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(OUT / "effective_dimension.csv", index=False)
    sk = pd.DataFrame(skill_rows); sk.to_csv(OUT / "skill_vs_components.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        s = sk[sk.target == t]
        ax.plot(s.n_components, s.mae, "o-", color="#2563eb", label="XGBoost on top-m principal components")
        ax.axhline(REF[t], color="#166534", linestyle="--", label=f"full recipe, raw features ({REF[t]:.2f})")
        ax.set_xlabel("number of principal components kept (m)"); ax.set_ylabel("out-of-fold MAE")
        ax.set_title(f"{t.upper()}: skill vs effective dimension"); ax.grid(alpha=0.15); ax.legend()
    fig.suptitle("How many independent directions do the 35 features really carry?", fontsize=14)
    fig.savefig(OUT / "skill_vs_components.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        s = scores[scores.target == t].sort_values("jacobs_score")
        ax.barh(s.feature, s.jacobs_score, color=np.where(s.jacobs_score > s.jacobs_score.median(), "#2563eb", "#b9c3d8"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("|corr to error| − Σ|corr to other features|  (Spearman)")
        ax.set_title(f"{t.upper()}: Dr. Jacobs' redundancy score (higher = more unique and relevant)")
        ax.tick_params(axis="y", labelsize=7.5)
    fig.savefig(OUT / "jacobs_feature_scores.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(dim_rows).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
