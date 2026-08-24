"""Is data the limit? Skill vs training-set size (meeting item, 2026-08).

Within each locked blocked-forward fold, subsample the TRAINING dates to a
fraction f (validation rows untouched), retrain the best single-model recipe,
and record out-of-fold MAE. If skill still improves at f=1.0, more data
(gliders, HYCOM years) is justified; if flat, architecture matters more.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/learning_curve_20260824")
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEED = 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        y = work[target.obs_col].to_numpy(float)
        r = work[target.model_col].to_numpy(float)
        gm = _in_gom(work).to_numpy()
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        for f in FRACTIONS:
            oof = np.full(len(work), np.nan)
            n_train_rows = 0
            for fold in folds:
                tr_dates = sorted(fold["train_dates"])
                keep = sorted(rng.choice(tr_dates, size=max(2, int(round(f * len(tr_dates)))), replace=False)) if f < 1.0 else tr_dates
                tr_mask = date_str.isin(set(keep)).to_numpy()
                va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
                if not tr_mask.any() or not va_mask.any():
                    continue
                n_train_rows += int(tr_mask.sum())
                pre = _make_preprocessor(cols)
                model = _xgb_model()
                model.fit(pre.fit_transform(work.loc[tr_mask, cols]), work.loc[tr_mask, target.delta_col].to_numpy(float))
                va_idx = np.where(va_mask)[0]
                oof[va_idx] = r[va_idx] + model.predict(pre.transform(work.iloc[va_idx][cols]))
            ok = np.isfinite(oof)
            rows.append({"target": tname, "fraction": f, "train_rows_total": n_train_rows,
                         "mae": float(np.abs(oof[ok] - y[ok]).mean()),
                         "mae_gom": float(np.abs(oof[ok & gm] - y[ok & gm]).mean())})
            print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)
    for ax, tname in zip(axes, ("tchp", "d26")):
        s = res[res.target == tname]
        ax.plot(s["train_rows_total"], s["mae"], marker="o", linewidth=2, color="#2563eb", label="global")
        ax.plot(s["train_rows_total"], s["mae_gom"], marker="s", linewidth=2, color="#dc2626", label="Gulf of Mexico")
        ax.set_xlabel("training rows used (summed over folds)")
        ax.set_ylabel(f"out-of-fold MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: skill vs training-set size")
        ax.grid(True, alpha=0.15)
        ax.legend()
    fig.suptitle("Learning curve: still improving at full data = more data will help")
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
