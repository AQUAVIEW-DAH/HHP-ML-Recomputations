"""Use PCA to CHOOSE raw features, instead of using it to transform them.

Rotating the inputs costs ~0.61 MAE because trees split one axis at a time
(see run_rotation_invariance_test.py). But the loadings still tell us which
features move together, so we can use them to pick a raw feature subset and
keep the original axes.

Two selection rules are compared:
  * "PC_k cluster"  : the features loading most strongly on one component.
    These co-vary by construction, so the set is deliberately redundant.
  * "one per PC"    : the top loader of each of the first m components, which
    gives a compact set spanning m decorrelated directions.

Every subset is also compared against RANDOM subsets of identical size, since
any smaller set loses ground simply for being smaller; only a gap over the
random baseline is evidence that the PCA guidance did any work.

Outputs: OHC/output/redundancy_followup_20260916/pc_guided_selection.{png,csv}
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import OHC.run_locked_xgb_physics_semi_ablation as abl
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/redundancy_followup_20260916")
RECIPES = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
TOP_PER_PC = 5
N_PCS = 8
N_RANDOM = 8
SEED = 0

def main() -> None:
    fn = json.loads(abl.FOLD_PATH.read_text())
    df = abl._merge_feature_tables()
    rows = []
    for t in TARGETS:
        cols = list(abl.FEATURE_SETS_BY_TARGET[t.name][RECIPES[t.name]])
        w = df[pd.notna(df[t.obs_col]) & pd.notna(df[t.model_col]) & pd.notna(df[t.delta_col])].copy()
        w = _prepare_features(w).reset_index(drop=True)
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(
            w[cols].apply(pd.to_numeric, errors="coerce")), columns=cols)
        y = w[t.delta_col].to_numpy(float)
        yo, ym = w[t.obs_col].to_numpy(float), w[t.model_col].to_numpy(float)
        ds = w["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(ds.unique().tolist()),
                                     n_folds=fn["n_folds"], embargo_dates=fn["embargo_dates"])
        def score(sub):
            oof = np.full(len(w), np.nan)
            for f in folds:
                tr = ds.isin(set(f["train_dates"])).to_numpy(); va = ds.isin(set(f["val_dates"])).to_numpy()
                m = abl._xgb_model(); m.fit(X.loc[tr, sub], y[tr]); oof[va] = m.predict(X.loc[va, sub])
            v = np.isfinite(oof); return float(np.abs(ym + oof - yo)[v].mean())

        # loadings from the EARLIEST training block only (no validation leakage)
        tr0 = ds.isin(set(folds[0]["train_dates"])).to_numpy()
        sc = StandardScaler().fit(X.to_numpy()[tr0])
        pca = PCA(n_components=N_PCS, random_state=0).fit(sc.transform(X.to_numpy()[tr0]))
        L, ev = pca.components_, pca.explained_variance_ratio_

        full = score(cols)
        rows.append({"target": t.name, "set": "full recipe", "kind": "reference",
                     "n": len(cols), "mae": full, "features": "|".join(cols)})
        print(f"{t.name} full recipe ({len(cols)}): {full:.3f}", flush=True)

        rng = np.random.default_rng(SEED)
        sizes = {}
        for k in range(N_PCS):
            sub = [cols[i] for i in np.argsort(-np.abs(L[k]))[:TOP_PER_PC]]
            mae = score(sub)
            rows.append({"target": t.name, "set": f"PC{k+1} cluster", "kind": "pc_cluster",
                         "n": len(sub), "mae": mae, "variance_pct": 100*ev[k],
                         "features": "|".join(sub)})
            sizes.setdefault(len(sub), [])
            print(f"  PC{k+1} cluster ({len(sub)}): {mae:.3f}   {sub}", flush=True)
        for m in (4, 8, 12):
            sub = list(dict.fromkeys(cols[int(np.argmax(np.abs(L[k])))] for k in range(min(m, N_PCS))))
            if m > N_PCS:
                pca_m = PCA(n_components=m, random_state=0).fit(sc.transform(X.to_numpy()[tr0]))
                sub = list(dict.fromkeys(cols[int(np.argmax(np.abs(pca_m.components_[k])))] for k in range(m)))
            mae = score(sub)
            rows.append({"target": t.name, "set": f"one per PC, first {m}", "kind": "one_per_pc",
                         "n": len(sub), "mae": mae, "features": "|".join(sub)})
            sizes.setdefault(len(sub), [])
            print(f"  one per PC (first {m}) -> {len(sub)} features: {mae:.3f}   {sub}", flush=True)
        for n in sorted(sizes):
            vals = []
            for _ in range(N_RANDOM):
                sub = list(rng.choice(cols, size=n, replace=False))
                vals.append(score(sub))
            rows.append({"target": t.name, "set": f"random {n} features", "kind": "random",
                         "n": n, "mae": float(np.mean(vals)), "mae_best": float(np.min(vals)),
                         "mae_worst": float(np.max(vals)), "features": ""})
            print(f"  random {n} features: mean {np.mean(vals):.3f}  best {np.min(vals):.3f}  worst {np.max(vals):.3f}", flush=True)

    out = pd.DataFrame(rows); out.to_csv(OUT / "pc_guided_selection.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7), constrained_layout=True)
    for ax, tname in zip(axes, ("tchp", "d26")):
        s = out[out.target == tname]
        full = float(s[s.kind == "reference"].mae.iloc[0])
        show = s[s.kind.isin(["pc_cluster", "one_per_pc"])].sort_values("mae")
        rnd = s[s.kind == "random"].set_index("n")
        yy = np.arange(len(show))
        colors = ["#2563eb" if k == "one_per_pc" else "#94a3b8" for k in show.kind]
        ax.barh(yy, show.mae, color=colors)
        for i, (_, r) in enumerate(show.iterrows()):
            if r.n in rnd.index:
                b = rnd.loc[r.n]
                ax.plot([b.mae_best, b.mae_worst], [i, i], color="#dc2626", lw=2.2, zorder=4)
                ax.plot(b.mae, i, "o", color="#dc2626", ms=5, zorder=5)
            ax.text(r.mae + 0.02, i, f"{r.mae:.2f}  (n={int(r.n)})", va="center", fontsize=8)
        ax.axvline(full, color="#166534", ls="--", lw=1.4)
        ax.text(full, len(show) - 0.3, f" full recipe {full:.2f}", color="#166534", fontsize=9)
        ax.set_yticks(yy); ax.set_yticklabels(show.set, fontsize=9)
        ax.set_xlabel("out-of-fold MAE"); ax.set_title(tname.upper(), fontsize=12)
        ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Using PCA to CHOOSE raw features instead of rotating them\n"
                 "blue = one representative per component · grey = all top loaders of one component\n"
                 "red bars are random subsets of the SAME size: a set only earns its keep by beating them",
                 fontsize=12.5)
    fig.savefig(OUT / "pc_guided_selection.png", dpi=160)
    print("wrote", OUT / "pc_guided_selection.png")

if __name__ == "__main__":
    main()
