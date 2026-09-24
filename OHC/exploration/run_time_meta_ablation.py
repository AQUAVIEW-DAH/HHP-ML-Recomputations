"""Time and metadata feature ablations (meeting notes, 2026-09-24).

Questions: should `nearest_rtofs_grid_distance_km` and `year` be in the model?
Why do the time features barely matter when TCHP itself has a strong seasonal
cycle? Does keeping only `month_int` do as well as the full set of eight
calendar encodings?

Each variant is scored with three random seeds so a difference can be read
against seed-to-seed noise, and per fold, because fold 1 trains only on
February to September 2024 and so has never seen October to January, nor the
year 2025, when it is scored.

Clean tables (warm rows, one primary profile per Argo cast), locked folds.
Outputs: OHC/output/time_meta_ablation_20260925/
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

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/time_meta_ablation_20260925")
CAL = ["year", "month_int", "month_sin", "month_cos", "doy_sin", "doy_cos",
       "is_winter_jfm", "is_summer_jas", "is_other"]
GRID = "nearest_rtofs_grid_distance_km"
SEEDS = (0, 1, 2)


def variants(cols: list[str]) -> dict[str, list[str]]:
    base = list(cols)
    no_cal = [c for c in base if c not in CAL]
    return {
        "full recipe (baseline)": base,
        "drop grid distance": [c for c in base if c != GRID],
        "drop year": [c for c in base if c != "year"],
        "drop grid distance and year": [c for c in base if c not in (GRID, "year")],
        "time = month_int only": no_cal + ["month_int"],
        "time = month sin/cos only": no_cal + ["month_sin", "month_cos"],
        "time = day-of-year sin/cos only": no_cal + ["doy_sin", "doy_cos"],
        "no time features at all": no_cal,
        "month_int only, and drop grid distance": [c for c in no_cal if c != GRID] + ["month_int"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = abl._merge_feature_tables()
    df = df[df["is_primary_profile"].astype(bool)].reset_index(drop=True)
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    rows = []
    for target in TARGETS:
        tn = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        cols = [c for c in abl.FEATURE_SETS_BY_TARGET[tn][RECIPE[tn]] if c in work.columns]
        ds = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(ds.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        y = work[target.delta_col].to_numpy(float)
        obs, mod = work[target.obs_col].to_numpy(float), work[target.model_col].to_numpy(float)
        for vname, vc in variants(cols).items():
            per_seed, per_fold = [], {f["fold"]: [] for f in folds}
            for seed in SEEDS:
                oof = np.full(len(work), np.nan)
                for f in folds:
                    tr, va = ds.isin(set(f["train_dates"])).to_numpy(), ds.isin(set(f["val_dates"])).to_numpy()
                    imp = SimpleImputer(strategy="median").fit(work.loc[tr, vc])
                    m = abl._xgb_model(); m.set_params(random_state=seed)
                    m.fit(imp.transform(work.loc[tr, vc]), y[tr])
                    oof[va] = m.predict(imp.transform(work.loc[va, vc]))
                    per_fold[f["fold"]].append(float(np.abs(mod[va] + oof[va] - obs[va]).mean()))
                v = np.isfinite(oof)
                per_seed.append(float(np.abs(mod[v] + oof[v] - obs[v]).mean()))
            rows.append({"target": tn, "variant": vname, "n_features": len(vc),
                         "mae": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed)),
                         **{f"fold{k}_mae": float(np.mean(val)) for k, val in per_fold.items()}})
            print(tn, vname, round(rows[-1]["mae"], 3), "+/-", round(rows[-1]["seed_sd"], 3), flush=True)
    T = pd.DataFrame(rows)
    base = T[T.variant == "full recipe (baseline)"].set_index("target")["mae"]
    T["change_vs_baseline"] = T.apply(lambda r: r.mae - base[r.target], axis=1)
    T.to_csv(OUT / "time_meta_ablation.csv", index=False)
    print(T.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
