"""Shared pieces for the 2026-10-04 RTOFS follow-up experiments.

Same rows, model and scoring as the earlier ablations (run_time_meta_ablation.py):
the clean tables (warm rows, one primary profile per Argo cast), the locked
XGBoost settings, the recommended recipe per target, three seeds, and the
locked blocked-forward folds. Adds leave-one-year-out folds (full-year test
blocks) with a one-week buffer on each side of the year boundary.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE  # noqa: E402

SEEDS = (0, 1, 2)
TIME_COLS = ["year", "month_int", "month_sin", "month_cos", "doy_sin", "doy_cos",
             "is_winter_jfm", "is_summer_jas", "is_other"]
LOYO_BUFFER_DAYS = 7
TARGET = {t.name: t for t in TARGETS}


@lru_cache(maxsize=1)
def _merged() -> pd.DataFrame:
    df = abl._merge_feature_tables()
    return df[df["is_primary_profile"].astype(bool)].reset_index(drop=True)


def load_work(tn: str) -> pd.DataFrame:
    """Rows scored for this target, features prepared, with a 'ds' (YYYYMMDD) column."""
    t = TARGET[tn]
    df = _merged()
    w = df[pd.notna(df[t.obs_col]) & pd.notna(df[t.model_col]) & pd.notna(df[t.delta_col])].copy()
    w = _prepare_features(w).reset_index(drop=True)
    w["ds"] = w["date"].dt.strftime("%Y%m%d")
    return w


def recipe(tn: str, work: pd.DataFrame) -> list[str]:
    return [c for c in abl.FEATURE_SETS_BY_TARGET[tn][RECIPE[tn]] if c in work.columns]


def forward_folds(work: pd.DataFrame) -> list[dict]:
    note = json.loads(abl.FOLD_PATH.read_text())
    folds = _build_forward_folds(sorted(work["ds"].unique().tolist()), n_folds=note["n_folds"],
                                 embargo_dates=note["embargo_dates"])
    return [{"name": f"forward {f['fold']}", "train_dates": f["train_dates"], "val_dates": f["val_dates"]}
            for f in folds]


def loyo_folds(work: pd.DataFrame) -> list[dict]:
    """Leave one year out: each year is the test block once, the other year trains; a week each side of
    1 Jan 2025 is left out of both, so neighbouring days cannot leak across the boundary."""
    d = pd.to_datetime(pd.Series(sorted(work["ds"].unique())), format="%Y%m%d")
    cut = pd.Timestamp("2025-01-01")
    lo, hi = cut - pd.Timedelta(days=LOYO_BUFFER_DAYS), cut + pd.Timedelta(days=LOYO_BUFFER_DAYS)
    y24 = d[d < lo].dt.strftime("%Y%m%d").tolist()
    y25 = d[d >= hi].dt.strftime("%Y%m%d").tolist()
    return [{"name": "full year: test 2024", "train_dates": y25, "val_dates": y24},
            {"name": "full year: test 2025", "train_dates": y24, "val_dates": y25}]


def median_X(cols: list[str]) -> Callable:
    def make(work: pd.DataFrame, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
        return imp.transform(work.loc[tr, cols]), imp.transform(work.loc[va, cols])
    return make


def evaluate(work: pd.DataFrame, tn: str, folds: list[dict], make_X: Callable, seeds=SEEDS) -> dict:
    """Out-of-fold MAE of RTOFS + predicted delta, mean over seeds, plus per-fold MAE and the seed spread."""
    t = TARGET[tn]
    y = work[t.delta_col].to_numpy(float)
    obs, mod = work[t.obs_col].to_numpy(float), work[t.model_col].to_numpy(float)
    masks = [(work["ds"].isin(set(f["train_dates"])).to_numpy(), work["ds"].isin(set(f["val_dates"])).to_numpy())
             for f in folds]
    Xs = [make_X(work, tr, va) for tr, va in masks]
    per_seed, per_fold = [], {f["name"]: [] for f in folds}
    for seed in seeds:
        oof = np.full(len(work), np.nan)
        for f, (tr, va), (Xtr, Xva) in zip(folds, masks, Xs):
            m = abl._xgb_model()
            m.set_params(random_state=seed)
            m.fit(Xtr, y[tr])
            oof[va] = m.predict(Xva)
            per_fold[f["name"]].append(float(np.abs(mod[va] + oof[va] - obs[va]).mean()))
        v = np.isfinite(oof)
        per_seed.append(float(np.abs(mod[v] + oof[v] - obs[v]).mean()))
    return {"mae": float(np.mean(per_seed)), "seed_sd": float(np.std(per_seed)),
            **{f"fold: {k}": float(np.mean(v)) for k, v in per_fold.items()}}
