"""Gulf of Mexico attribution analysis (mentor request, meeting 2026-07-21).

Quantifies HOW the Gulf-local model differs from the global model, using the
same blocked-forward folds as the local-vs-global experiment:

1. SHAP feature contributions (exact TreeSHAP via XGBoost pred_contribs),
   computed out-of-fold on identical Gulf rows for both models:
   - mean |SHAP| ranking, global vs local, side by side
   - dependence curves (feature value vs contribution) for the top features
2. Backward feature elimination on the Gulf-local model: greedy removal,
   reporting the out-of-fold MAE cost of each feature ("skill ladder").
3. Observed vs raw vs corrected distributions inside the Gulf (per target).

Outputs to a dated directory for trackability.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)

RUN_DATE = "2026-07-22"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/gom_attribution_{RUN_DATE.replace('-', '')}")
RECIPE = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}
UNITS = {"tchp": "kJ/cm²", "d26": "m"}
SHORT = {"tchp": "TCHP", "d26": "D26"}
GOM = {"lat_min": 18.0, "lat_max": 31.0, "lon_min": -98.0, "lon_max": -80.0}
TOP_N_BARS = 12
TOP_N_DEPENDENCE = 4


def _in_gom(df: pd.DataFrame) -> pd.Series:
    return (
        (df["lat"] >= GOM["lat_min"]) & (df["lat"] <= GOM["lat_max"])
        & (df["lon"] >= GOM["lon_min"]) & (df["lon"] <= GOM["lon_max"])
    )


def _short_name(col: str) -> str:
    out = col
    if out.startswith("model_"):
        out = out[len("model_"):]
    for suffix in ("_kj_per_cm2", "_s2", "_m"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out.replace("interp_", "raw ").replace("_", " ")


def _fold_models_and_shap(work, gom, folds, cols, target):
    """Train per fold (global on all rows, local on Gulf rows); return OOF
    predictions and OOF SHAP matrices for the Gulf validation rows."""
    date_all = work["date"].dt.strftime("%Y%m%d")
    date_gom = gom["date"].dt.strftime("%Y%m%d")
    n, f = len(gom), len(cols)
    shap_global = np.full((n, f), np.nan)
    shap_local = np.full((n, f), np.nan)
    pred_global = np.full(n, np.nan)
    pred_local = np.full(n, np.nan)

    for fold in folds:
        tr_dates, va_dates = set(fold["train_dates"]), set(fold["val_dates"])
        va_idx = np.where(date_gom.isin(va_dates).to_numpy())[0]
        if va_idx.size == 0:
            continue
        for scope, train_df, shap_out, pred_out in [
            ("global", work[date_all.isin(tr_dates)], shap_global, pred_global),
            ("local", gom[date_gom.isin(tr_dates)], shap_local, pred_local),
        ]:
            if train_df.empty:
                continue
            pre = _make_preprocessor(cols)
            X_tr = pre.fit_transform(train_df[cols])
            X_va = pre.transform(gom.iloc[va_idx][cols])
            model = _xgb_model()
            model.fit(X_tr, train_df[target.delta_col].to_numpy(float))
            contribs = model.get_booster().predict(xgb.DMatrix(X_va), pred_contribs=True)
            shap_out[va_idx] = contribs[:, :f]
            pred_out[va_idx] = gom.iloc[va_idx][target.model_col].to_numpy(float) + model.predict(X_va)
    return shap_global, shap_local, pred_global, pred_local


def _plot_mean_shap(tname, cols, shap_global, shap_local, out_path):
    mg = np.nanmean(np.abs(shap_global), axis=0)
    ml = np.nanmean(np.abs(shap_local), axis=0)
    order = np.argsort(ml)[::-1][:TOP_N_BARS]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.barh(y - 0.2, mg[order], 0.4, color="#94a3b8", label="global model")
    ax.barh(y + 0.2, ml[order], 0.4, color="#2563eb", label="Gulf-local model")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(cols[i]) for i in order])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean |SHAP contribution| ({UNITS[tname]})")
    ax.set_title(f"{SHORT[tname]}: what each model relies on inside the Gulf of Mexico\n(top {TOP_N_BARS} features of the local model)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [cols[i] for i in order[:TOP_N_DEPENDENCE]]


def _plot_dependence(tname, gom, cols, top_cols, shap_global, shap_local, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for ax, col in zip(axes.ravel(), top_cols):
        j = cols.index(col)
        x = pd.to_numeric(gom[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(shap_local[:, j])
        ax.scatter(x[ok], shap_global[ok, j], s=7, alpha=0.35, color="#94a3b8", label="global model")
        ax.scatter(x[ok], shap_local[ok, j], s=7, alpha=0.35, color="#2563eb", label="Gulf-local model")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(_short_name(col))
        ax.set_ylabel(f"SHAP contribution ({UNITS[tname]})")
        ax.grid(True, alpha=0.15)
    axes[0, 0].legend(markerscale=2.5)
    fig.suptitle(f"{SHORT[tname]}: how the top features act in the Gulf (contribution vs feature value)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _backward_elimination(gom, folds, cols, target):
    date_gom = gom["date"].dt.strftime("%Y%m%d")
    y_obs = gom[target.obs_col].to_numpy(float)

    def oof_mae(active):
        oof = np.full(len(gom), np.nan)
        for fold in folds:
            tr = gom[date_gom.isin(set(fold["train_dates"]))]
            va_idx = np.where(date_gom.isin(set(fold["val_dates"])).to_numpy())[0]
            if tr.empty or va_idx.size == 0:
                continue
            pre = _make_preprocessor(active)
            model = _xgb_model()
            model.fit(pre.fit_transform(tr[active]), tr[target.delta_col].to_numpy(float))
            oof[va_idx] = gom.iloc[va_idx][target.model_col].to_numpy(float) + model.predict(pre.transform(gom.iloc[va_idx][active]))
        ok = np.isfinite(oof)
        return float(np.abs(oof[ok] - y_obs[ok]).mean())

    active = list(cols)
    ladder = [{"step": 0, "removed": "(full recipe)", "mae": oof_mae(active), "n_features": len(active)}]
    step = 0
    while len(active) > 4:
        step += 1
        candidates = [(oof_mae([c for c in active if c != rm]), rm) for rm in active]
        best_mae, best_rm = min(candidates)
        active.remove(best_rm)
        ladder.append({"step": step, "removed": best_rm, "mae": best_mae, "n_features": len(active)})
    return pd.DataFrame(ladder)


def _plot_ladder(tname, ladder, out_path):
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.plot(ladder["step"], ladder["mae"], marker="o", color="#2563eb", linewidth=2)
    ax.set_xticks(ladder["step"])
    ax.set_xticklabels([_short_name(r) if r != "(full recipe)" else "full" for r in ladder["removed"]], rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("feature removed at each step (cheapest first)")
    ax.set_ylabel(f"Gulf out-of-fold MAE ({UNITS[tname]})")
    ax.set_title(f"{SHORT[tname]}: backward elimination on the Gulf-local model\nA jump means the removed feature was carrying real skill")
    ax.grid(True, alpha=0.15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_distribution(tname, target, gom, pred_global, pred_local, out_path):
    obs = gom[target.obs_col].to_numpy(float)
    raw = gom[target.model_col].to_numpy(float)
    lo = float(np.nanquantile(obs, 0.005))
    hi = float(np.nanquantile(obs, 0.995))
    bins = np.linspace(lo, hi, 45)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for vals, color, label, lw in [
        (obs, "black", "observed (Argo)", 2.8),
        (raw, "#dc2626", "raw RTOFS", 2.0),
        (pred_global, "#94a3b8", "corrected, global model", 2.0),
        (pred_local, "#2563eb", "corrected, Gulf-local model", 2.0),
    ]:
        ax.hist(vals[np.isfinite(vals)], bins=bins, density=True, histtype="step", linewidth=lw, color=color, label=label)
    ax.set_xlabel(f"{SHORT[tname]} ({UNITS[tname]})")
    ax.set_ylabel("probability density")
    ax.set_title(f"{SHORT[tname]} distributions inside the Gulf of Mexico (out-of-fold)")
    ax.legend()
    ax.grid(True, alpha=0.15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    manifest = {"run_date": RUN_DATE, "targets": {}}

    for target in TARGETS:
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        gom = work[_in_gom(work)].copy().reset_index(drop=True)
        cols = [c for c in FEATURE_SETS_BY_TARGET[target.name][RECIPE[target.name]] if c in work.columns]
        gom_dates = sorted(gom["date"].dt.strftime("%Y%m%d").unique().tolist())
        folds = _build_forward_folds(gom_dates, n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

        shap_g, shap_l, pred_g, pred_l = _fold_models_and_shap(work, gom, folds, cols, target)

        p1 = OUT_DIR / f"{target.name}_gom_mean_shap.png"
        top_cols = _plot_mean_shap(target.name, cols, shap_g, shap_l, p1)
        p2 = OUT_DIR / f"{target.name}_gom_shap_dependence.png"
        _plot_dependence(target.name, gom, cols, top_cols, shap_g, shap_l, p2)

        ladder = _backward_elimination(gom, folds, cols, target)
        ladder.to_csv(OUT_DIR / f"{target.name}_gom_backward_elimination.csv", index=False)
        p3 = OUT_DIR / f"{target.name}_gom_backward_elimination.png"
        _plot_ladder(target.name, ladder, p3)

        p4 = OUT_DIR / f"{target.name}_gom_distributions.png"
        _plot_distribution(target.name, target, gom, pred_g, pred_l, p4)

        shap_df = pd.DataFrame({
            "feature": cols,
            "mean_abs_shap_global": np.nanmean(np.abs(shap_g), axis=0),
            "mean_abs_shap_local": np.nanmean(np.abs(shap_l), axis=0),
        }).sort_values("mean_abs_shap_local", ascending=False)
        shap_df.to_csv(OUT_DIR / f"{target.name}_gom_mean_shap.csv", index=False)

        manifest["targets"][target.name] = {
            "rows_gom": int(len(gom)),
            "figures": [str(p) for p in (p1, p2, p3, p4)],
            "top_local_features": [str(c) for c in shap_df.head(5)["feature"]],
        }
        print(f"{target.name}: done ({len(gom)} Gulf rows)")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
