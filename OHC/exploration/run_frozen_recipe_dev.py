"""Score the proposed frozen recipe on the development folds, before the 2026 test.

The 2026 holdout may be evaluated exactly once, so the configuration has to be
fixed first. Changes from the recommended recipes, each chosen on principle
rather than by chasing a score, because every choice made on these same folds
adds optimism:

  * remove `year`: no training fold can ever contain 2026, so a tree can
    only extrapolate on it;
  * remove the three deep-profile features (D26 recipe): they need the 3D
    archives, deleted after each day is processed, so they cannot be built
    for 2026 (and are 89% imputed in 2024-2025);
  * remove `nearest_rtofs_grid_distance_km`: no effect in any test;
  * add sin/cos of longitude: raw longitude puts an artificial seam at the
    dateline, which splits the East-Pacific expert's region in two
    (-0.041 on both targets in run_missing_physics_search.py).

Feature combinations from the residual screen are deliberately left out: they
were selected against residuals on these folds and disagree between targets.

Clean tables, warm rows, one primary profile per cast, locked folds.
Outputs: OHC/output/frozen_recipe_dev_20260925/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402
from OHC.exploration.run_moe_v2_tuning import _run_geographic, _run_regime  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/frozen_recipe_dev_20260925")
WINNERS = {"tchp": {"alpha": 0.75, "k": 6, "w": 0.05}, "d26": {"alpha": 0.50, "k": 12, "w": 0.05}}
DROP = {"year", "nearest_rtofs_grid_distance_km", "model_steric_1000_ref2000_m",
        "model_n2_max_upper200_s2", "model_n2_mean_to_d26_s2"}
ADD = ["lon_sin", "lon_cos"]


def frozen_recipe(target_name: str) -> list[str]:
    base = abl.FEATURE_SETS_BY_TARGET[target_name][RECIPE[target_name]]
    return [c for c in base if c not in DROP] + ADD


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = abl._merge_feature_tables()
    df = df[df["is_primary_profile"].astype(bool)].reset_index(drop=True)
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    rows, recipes = [], {}
    for target in TARGETS:
        tn, cfg = target.name, WINNERS[target.name]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["lon_sin"] = np.sin(np.deg2rad(work["lon"]))
        work["lon_cos"] = np.cos(np.deg2rad(work["lon"]))
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = frozen_recipe(tn)
        recipes[tn] = cols
        ds = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(ds.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        y, r = work[target.obs_col].to_numpy(float), work[target.model_col].to_numpy(float)
        glob = np.full(len(work), np.nan)
        for f in folds:
            tr, va = ds.isin(set(f["train_dates"])).to_numpy(), ds.isin(set(f["val_dates"])).to_numpy()
            imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
            m = abl._xgb_model()
            m.fit(imp.transform(work.loc[tr, cols]), work.loc[tr, target.delta_col].to_numpy(float))
            glob[va] = r[va] + m.predict(imp.transform(work.loc[va, cols]))
        geo = _run_geographic(work, target, cols, folds, cfg["w"])
        reg = _run_regime(work, target, cols, folds, cfg["w"], cfg["k"])
        moe = cfg["alpha"] * geo + (1 - cfg["alpha"]) * reg
        v = np.isfinite(moe) & np.isfinite(glob)
        gm = ((work["lat"] >= 18) & (work["lat"] <= 31) & (work["lon"] >= -98) & (work["lon"] <= -80)).to_numpy()[v]
        for name, p in (("raw RTOFS", r[v]), ("single global model, frozen recipe", glob[v]),
                        ("MoE blend, frozen recipe", moe[v])):
            e = p - y[v]
            rows.append({"target": tn, "model": name, "n_features": len(cols), "mae": float(np.abs(e).mean()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                         "mae_gulf": float(np.abs(e[gm]).mean()), "rows": int(v.sum())})
        print(tn, "done", rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "frozen_recipe_dev_scores.csv", index=False)
    (OUT / "frozen_recipes.json").write_text(json.dumps(recipes, indent=2))
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
