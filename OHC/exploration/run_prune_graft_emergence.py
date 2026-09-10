"""Pruning, grafting, and emergent-feature tests (meeting notes, 2026-09).

Part 1 — prune/graft in both directions on the locked recipes (warm rows):
  * graft-single : MAE(core + one feature)      -> what it adds alone
  * prune-single : MAE(full - one feature)      -> what is lost without it
  * graft-family : MAE(core + whole family)
  * prune-family : MAE(full - whole family)
  A feature that helps when grafted but costs nothing when pruned is
  redundant-but-useful. Emergence is super-additivity: a family gain larger
  than the sum (or the best) of its members' individual gains.

Part 2 — the ~8,000 boundary cases (one side has 26 C water, the other does
  not; targets zero-filled per Dr. Jacobs). Only 23% of the D26-derived
  features exist there, but SSH / MLT / SBLT / temperature-excess do, so we
  add feature groups one at a time and score separately on boundary rows,
  warm rows, and everything. If temperature excess collapses the boundary
  error while doing little elsewhere, resolving the 26 C edge is a genuine
  interaction effect rather than a global gain.

Outputs: OHC/output/prune_graft_20260910/
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
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/prune_graft_20260910")
RECIPES = {"tchp": "global_pruned_plus_neighborhood", "d26": "drop_both_lat_interactions_plus_neighborhood"}

CORE = {"tchp": ["model_interp_tchp_kj_per_cm2", "model_ssh_m", "model_temp_excess_26c",
                 "model_mixed_layer_thickness_m", "lat", "lon", "month_sin", "month_cos"],
        "d26": ["model_interp_d26_m", "model_ssh_m", "model_temp_excess_26c",
                "model_mixed_layer_thickness_m", "lat", "lon", "month_sin", "month_cos"]}


def family_of(c: str) -> str:
    if c in ("lat", "lon", "abs_lat"):
        return "location"
    if c.startswith(("month", "doy", "is_")) or c == "year":
        return "calendar"
    if c.endswith("_x_abs_lat"):
        return "lat interactions"
    if c.startswith("model_interp"):
        return "raw model value"
    if "local_std" in c or "grad_mag" in c or "anom_from" in c:
        return "neighborhood context"
    if "steric" in c or "n2" in c:
        return "deep steric / stratification"
    if c == "nearest_rtofs_grid_distance_km":
        return "collocation geometry"
    return "model physics"


def fit_score(work, cols, target, folds, date_str, extra_masks=None):
    y = work[target.delta_col].to_numpy(float)
    y_obs = work[target.obs_col].to_numpy(float)
    y_mod = work[target.model_col].to_numpy(float)
    oof = np.full(len(work), np.nan)
    for fold in folds:
        tr = date_str.isin(set(fold["train_dates"])).to_numpy()
        va = date_str.isin(set(fold["val_dates"])).to_numpy()
        imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
        m = abl._xgb_model()
        m.fit(imp.transform(work.loc[tr, cols]), y[tr])
        oof[va] = m.predict(imp.transform(work.loc[va, cols]))
    v = np.isfinite(oof)
    err = np.abs(y_mod + oof - y_obs)
    out = {"mae": float(err[v].mean())}
    for name, mask in (extra_masks or {}).items():
        out[f"mae_{name}"] = float(err[v & mask].mean()) if (v & mask).any() else np.nan
    return out


def part1(df, fold_note):
    rows = []
    for target in TARGETS:
        t = target.name
        full = list(abl.FEATURE_SETS_BY_TARGET[t][RECIPES[t]])
        core = [c for c in CORE[t] if c in full]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col])
                  & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        base = {"core": fit_score(work, core, target, folds, date_str)["mae"],
                "full": fit_score(work, full, target, folds, date_str)["mae"]}
        rows.append({"target": t, "kind": "reference", "name": "core only", "n_features": len(core), "mae": base["core"]})
        rows.append({"target": t, "kind": "reference", "name": "full recipe", "n_features": len(full), "mae": base["full"]})
        print(t, "core", base["core"], "full", base["full"], flush=True)

        fams = sorted({family_of(c) for c in full})
        for fam in fams:
            members = [c for c in full if family_of(c) == fam]
            g = [c for c in core + members if c in full]
            g = list(dict.fromkeys(g))
            p = [c for c in full if c not in members]
            rows.append({"target": t, "kind": "graft-family", "name": fam, "n_features": len(g),
                         "mae": fit_score(work, g, target, folds, date_str)["mae"], "members": len(members)})
            if p:
                rows.append({"target": t, "kind": "prune-family", "name": fam, "n_features": len(p),
                             "mae": fit_score(work, p, target, folds, date_str)["mae"], "members": len(members)})
            print(t, "family", fam, "done", flush=True)

        for c in full:
            if c not in core:
                rows.append({"target": t, "kind": "graft-single", "name": c, "n_features": len(core) + 1,
                             "mae": fit_score(work, core + [c], target, folds, date_str)["mae"],
                             "family": family_of(c)})
            rows.append({"target": t, "kind": "prune-single", "name": c, "n_features": len(full) - 1,
                         "mae": fit_score(work, [x for x in full if x != c], target, folds, date_str)["mae"],
                         "family": family_of(c)})
        print(t, "singles done", flush=True)
    return pd.DataFrame(rows)


def part2(df, fold_note):
    groups = [
        ("position", ["lat", "lon", "abs_lat"]),
        ("+ calendar", ["month_sin", "month_cos", "doy_sin", "doy_cos", "is_winter_jfm", "is_summer_jas", "is_other"]),
        ("+ raw model value (0 where no 26C)", ["model_interp_tchp_kj_per_cm2", "model_interp_d26_m"]),
        ("+ SSH / MLT / SBLT", ["model_ssh_m", "model_mixed_layer_thickness_m", "model_surface_boundary_layer_thickness_m"]),
        ("+ temperature excess above 26C", ["model_temp_excess_26c"]),
        ("+ SST neighbourhood", ["model_sst_local_std_1deg", "model_sst_anom_from_1deg_mean", "model_sst_grad_mag_per_100km"]),
    ]
    rows = []
    work = _prepare_features(df).reset_index(drop=True)
    date_str = work["date"].dt.strftime("%Y%m%d")
    folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                 n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
    for target in TARGETS:
        t = target.name
        obs0 = work[target.obs_col].fillna(0.0)
        mod0 = work[target.model_col].fillna(0.0)
        w2 = work.copy()
        w2[target.obs_col] = obs0
        w2[target.model_col] = mod0
        w2[target.delta_col] = obs0 - mod0
        for c in ("model_interp_tchp_kj_per_cm2", "model_interp_d26_m"):
            w2[c] = w2[c].fillna(0.0)
        warm = (df[target.obs_col].notna() & df[target.model_col].notna()).to_numpy()
        bnd = (df[target.obs_col].notna() != df[target.model_col].notna()).to_numpy()
        masks = {"warm": warm, "boundary": bnd, "cold": ~warm & ~bnd}
        cols = []
        for label, add in groups:
            cols = list(dict.fromkeys(cols + [c for c in add if c in w2.columns]))
            s = fit_score(w2, cols, target, folds, date_str, masks)
            rows.append({"target": t, "step": label, "n_features": len(cols), **s})
            print(t, label, s, flush=True)
        raw = np.abs(mod0 - obs0).to_numpy()
        rows.append({"target": t, "step": "raw RTOFS", "n_features": 0, "mae": float(raw.mean()),
                     **{f"mae_{k}": float(raw[m].mean()) for k, m in masks.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = abl._merge_feature_tables()
    fold_note = json.loads(abl.FOLD_PATH.read_text())

    p1 = part1(df, fold_note)
    p1.to_csv(OUT / "prune_graft_results.csv", index=False)

    summ = []
    for t in p1.target.unique():
        s = p1[p1.target == t]
        core = float(s[(s.kind == "reference") & (s.name == "core only")].mae.iloc[0])
        full = float(s[(s.kind == "reference") & (s.name == "full recipe")].mae.iloc[0])
        for fam in s[s.kind == "graft-family"].name:
            gf = float(s[(s.kind == "graft-family") & (s.name == fam)].mae.iloc[0])
            pf = s[(s.kind == "prune-family") & (s.name == fam)].mae
            singles = s[(s.kind == "graft-single") & (s.family == fam)]
            gains = core - singles.mae if len(singles) else pd.Series(dtype=float)
            summ.append({"target": t, "family": fam,
                         "graft_family_gain": core - gf,
                         "prune_family_cost": (float(pf.iloc[0]) - full) if len(pf) else np.nan,
                         "best_single_gain": float(gains.max()) if len(gains) else np.nan,
                         "sum_single_gains": float(gains.sum()) if len(gains) else np.nan,
                         "emergence_vs_best": (core - gf) - (float(gains.max()) if len(gains) else 0.0),
                         "superadditivity": (core - gf) - (float(gains.sum()) if len(gains) else 0.0)})
    summary = pd.DataFrame(summ)
    summary.to_csv(OUT / "family_emergence_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, t in zip(axes, ["tchp", "d26"]):
        s = summary[summary.target == t].sort_values("graft_family_gain")
        yy = np.arange(len(s)); h = 0.38
        ax.barh(yy + h/2, s.graft_family_gain, h, color="#2563eb", label="grafted onto core (gain)")
        ax.barh(yy - h/2, s.prune_family_cost, h, color="#dc2626", label="pruned from full recipe (cost)")
        ax.plot(s.best_single_gain, yy + h/2, "kd", ms=6, label="best single member alone")
        ax.set_yticks(yy); ax.set_yticklabels(s.family, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("MAE improvement"); ax.set_title(t.upper()); ax.grid(alpha=0.15, axis="x"); ax.legend(fontsize=8)
    fig.suptitle("Feature families: what they add when grafted vs what they cost when pruned\n"
                 "gain far above the best single member = emergent (members only work together)", fontsize=13)
    fig.savefig(OUT / "family_prune_graft.png", dpi=160)
    plt.close(fig)

    p2 = part2(df, fold_note)
    p2.to_csv(OUT / "boundary_emergence.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, t in zip(axes, ["tchp", "d26"]):
        s = p2[(p2.target == t) & (p2.step != "raw RTOFS")]
        x = np.arange(len(s))
        for col, lab, c in (("mae_boundary", "boundary rows (one side has no 26 °C water)", "#dc2626"),
                            ("mae_warm", "warm rows (both sides)", "#2563eb"),
                            ("mae", "all rows", "#94a3b8")):
            ax.plot(x, s[col], "o-", color=c, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(s.step, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("MAE"); ax.set_title(t.upper()); ax.set_yscale("log"); ax.grid(alpha=0.15); ax.legend(fontsize=8)
    fig.suptitle("Do the ~8,000 boundary cases need feature combinations?\n"
                 "features added cumulatively; a drop only in the red line is an interaction effect", fontsize=13)
    fig.savefig(OUT / "boundary_emergence.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(p2.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
