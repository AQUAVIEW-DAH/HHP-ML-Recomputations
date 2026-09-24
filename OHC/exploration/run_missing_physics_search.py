"""What is the model missing? (meeting notes, 2026-09-24)

The recommended MoE's leftover error is not spatially random: it is too high
across the central and eastern tropical Pacific and too low in the western
warm pool. That east-west Pacific shape suggests the RTOFS bias shifts with
the El Nino / La Nina state, which a correction learned from earlier months
cannot anticipate. This script tests that, and searches for other missing
information.

Note on what can help a tree at all: splitting on f and on any one-to-one
monotone transform of f (log f, f^2, ...) separates the same rows, so such
transforms can never add information. Only these can:
  * new information (an ENSO index built from RTOFS's own surface temperature)
  * non-monotone transforms (sin/cos of longitude, which also removes the
    artificial seam at the dateline, in the middle of the Pacific)
  * combinations of features (products, sums, ratios, family averages)

Parts:
  A. Nino 3.4 index from RTOFS SST (5S-5N, 170W-120W), anomaly from the
     calendar-month mean. Does the MoE's Pacific residual change between
     folds, and does it track the index?
  B. Screen candidate features against the single global model's out-of-fold
     residual, then add the promising ones to the model and rescore, three
     seeds each, on the locked folds.
  C. min_child_weight: the locked model allows leaves holding a single row.

Clean tables (warm rows, one primary profile per cast).
Outputs: OHC/output/missing_physics_20260925/
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/missing_physics_20260925")
FIELDS = {2024: Path("/data/suramya/rtofs_global_ohc_fields_2024"),
          2025: Path("/data/suramya/rtofs_global_ohc_fields_2025")}
MOE = Path("/home/suramya/HHP-Prediction/OHC/output/moe_clean_20260925")
PHYS = ["model_interp_tchp_kj_per_cm2", "model_interp_d26_m", "model_ssh_m", "model_mixed_layer_thickness_m",
        "model_temp_excess_26c", "model_tchp_anom_from_1deg_mean", "model_d26_anom_from_1deg_mean",
        "model_tchp_local_std_1deg", "abs_lat"]
SEEDS = (0, 1, 2)


def nino34() -> pd.DataFrame:
    cache = OUT / "nino34_rtofs.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"date": str})
    files = sorted(p for d in FIELDS.values() for p in d.glob("rtofs_tchp_*.nc"))
    with xr.open_dataset(files[0]) as ds:
        la, lo = ds["Latitude"].values, ds["Longitude"].values
    box = (la >= -5) & (la <= 5) & (lo >= -170) & (lo <= -120)
    rows = []
    for p in files:
        with xr.open_dataset(p) as ds:
            sst = ds["surface_temp_c"].values
        rows.append({"date": p.stem.split("_")[-1], "nino34_sst": float(np.nanmean(sst[box]))})
    t = pd.DataFrame(rows)
    t["month"] = t["date"].str[4:6].astype(int)
    t["nino34_anom"] = t["nino34_sst"] - t.groupby("month")["nino34_sst"].transform("mean")
    t.to_csv(cache, index=False)
    return t


def oof(work, cols, y, folds, ds, seed=0, mcw=None):
    pred = np.full(len(work), np.nan)
    for f in folds:
        tr, va = ds.isin(set(f["train_dates"])).to_numpy(), ds.isin(set(f["val_dates"])).to_numpy()
        imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
        m = abl._xgb_model(); m.set_params(random_state=seed)
        if mcw is not None:
            m.set_params(min_child_weight=mcw)
        m.fit(imp.transform(work.loc[tr, cols]), y[tr])
        pred[va] = m.predict(imp.transform(work.loc[va, cols]))
    return pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = nino34()
    print("nino34 built:", len(idx), "dates; anomaly range", round(idx.nino34_anom.min(), 2), round(idx.nino34_anom.max(), 2), flush=True)
    df = abl._merge_feature_tables()
    df = df[df["is_primary_profile"].astype(bool)].reset_index(drop=True)
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    screen_rows, test_rows, drift_rows = [], [], []

    for target in TARGETS:
        tn = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        ds = work["date"].dt.strftime("%Y%m%d")
        work["nino34_anom"] = ds.map(idx.set_index("date")["nino34_anom"]).to_numpy(float)
        work["lon_sin"] = np.sin(np.deg2rad(work["lon"]))
        work["lon_cos"] = np.cos(np.deg2rad(work["lon"]))
        cols = [c for c in abl.FEATURE_SETS_BY_TARGET[tn][RECIPE[tn]] if c in work.columns]
        folds = _build_forward_folds(sorted(ds.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        y = work[target.delta_col].to_numpy(float)
        obs, mod = work[target.obs_col].to_numpy(float), work[target.model_col].to_numpy(float)

        # ---- A. does the MoE's Pacific residual drift between folds?
        m = pd.read_parquet(MOE / f"moe_predictions_{tn}.parquet")
        m["res"] = m["pred_obs__moe_blend"] - m[target.obs_col]
        m = m.merge(idx[["date", "nino34_anom"]], on="date", how="left")
        east = (m.lat.abs() <= 15) & (m.lon >= -160) & (m.lon <= -90)
        west = (m.lat.abs() <= 15) & (m.lon >= 120) & (m.lon <= 170)
        for f in folds:
            va = m["date"].isin(f["val_dates"])
            drift_rows.append({"target": tn, "fold": f["fold"], "validates": f"{f['val_dates'][0]}-{f['val_dates'][-1]}",
                               "east_pacific_mean_residual": float(m.loc[va & east, "res"].mean()),
                               "west_pacific_mean_residual": float(m.loc[va & west, "res"].mean()),
                               "mean_nino34_anom": float(m.loc[va, "nino34_anom"].mean())})
        mo = m.assign(month=m["date"].str[:6])
        g = mo[east | west].assign(side=np.where(east[east | west], "east", "west")).groupby(["month", "side"])["res"].mean().unstack()
        g = g.join(mo.groupby("month")["nino34_anom"].mean())
        g.to_csv(OUT / f"{tn}_pacific_residual_by_month.csv")
        c_e = g[["east", "nino34_anom"]].dropna().corr().iloc[0, 1]
        c_w = g[["west", "nino34_anom"]].dropna().corr().iloc[0, 1]
        print(tn, f"monthly Pacific residual vs Nino3.4: east r={c_e:+.2f}, west r={c_w:+.2f}", flush=True)
        drift_rows.append({"target": tn, "fold": "monthly correlation with Nino3.4",
                           "east_pacific_mean_residual": c_e, "west_pacific_mean_residual": c_w})

        # ---- B. screening on the single global model's out-of-fold residual
        base = oof(work, cols, y, folds, ds)
        v = np.isfinite(base)
        res = y - base
        X = work[cols + PHYS + ["nino34_anom", "lon_sin", "lon_cos"]].apply(pd.to_numeric, errors="coerce")
        X = X.loc[:, ~X.columns.duplicated()]
        Z = (X - X.mean()) / X.std()
        cand = {"nino34_anom": X["nino34_anom"], "lon_sin": X["lon_sin"], "lon_cos": X["lon_cos"]}
        for a, b in itertools.combinations([p for p in PHYS if p in X.columns], 2):
            cand[f"{a} * {b}"] = Z[a] * Z[b]
            cand[f"{a} + {b}"] = Z[a] + Z[b]
            cand[f"{a} / {b}"] = X[a] / X[b].replace(0, np.nan)
        fam = {"neighbourhood anomaly mean": ["model_tchp_anom_from_1deg_mean", "model_d26_anom_from_1deg_mean", "model_sst_anom_from_1deg_mean"],
               "neighbourhood std mean": ["model_tchp_local_std_1deg", "model_d26_local_std_1deg", "model_sst_local_std_1deg"],
               "gradient mean": ["model_tchp_grad_mag_per_100km", "model_d26_grad_mag_per_100km", "model_sst_grad_mag_per_100km"]}
        for name, mem in fam.items():
            mem = [c for c in mem if c in work.columns]
            zz = work[mem].apply(pd.to_numeric, errors="coerce")
            zz = (zz - zz.mean()) / zz.std()
            cand[name] = zz.mean(axis=1)
            cand[name.replace("mean", "max")] = zz.max(axis=1)
        vr = np.var(res[v])
        for name, s in cand.items():
            s = s.to_numpy(float)
            ok = v & np.isfinite(s)
            t = DecisionTreeRegressor(max_depth=2, min_samples_leaf=200).fit(s[ok, None], res[ok])
            screen_rows.append({"target": tn, "candidate": name,
                                "pct_residual_variance_explained": 100 * (vr - np.var(res[ok] - t.predict(s[ok, None]))) / vr})
        S = pd.DataFrame([r for r in screen_rows if r["target"] == tn]).sort_values("pct_residual_variance_explained", ascending=False)
        top = [c for c in S["candidate"].head(6) if c not in ("lon_sin", "lon_cos", "nino34_anom")][:4]
        print(tn, "top screened:", S.head(8).to_string(index=False), flush=True)

        # ---- real tests: add to the model and rescore
        def rescore(label, extra_cols, extra_frame=None, mcw=None):
            w2 = work if extra_frame is None else pd.concat([work, extra_frame], axis=1)
            maes = []
            for s_ in SEEDS:
                p = oof(w2, cols + extra_cols, y, folds, ds, seed=s_, mcw=mcw)
                vv = np.isfinite(p)
                maes.append(float(np.abs(mod[vv] + p[vv] - obs[vv]).mean()))
            test_rows.append({"target": tn, "variant": label, "mae": float(np.mean(maes)), "seed_sd": float(np.std(maes))})
            print(tn, label, round(np.mean(maes), 3), "+/-", round(np.std(maes), 3), flush=True)

        rescore("baseline (full recipe)", [])
        rescore("+ Nino3.4 index (from RTOFS SST)", ["nino34_anom"])
        rescore("+ cyclic longitude (sin, cos)", ["lon_sin", "lon_cos"])
        rescore("+ Nino3.4 + cyclic longitude", ["nino34_anom", "lon_sin", "lon_cos"])
        if top:
            ef = pd.DataFrame({f"cand_{i}": cand[n].to_numpy(float) for i, n in enumerate(top)})
            rescore("+ top screened combinations: " + "; ".join(top), list(ef.columns), ef)
        agg = pd.DataFrame({k.replace(" ", "_"): cand[k].to_numpy(float) for k in fam})
        rescore("+ family averages (anomaly, std, gradient)", list(agg.columns), agg)
        for mcw in (20, 50, 100):
            rescore(f"min_child_weight = {mcw} (locked model allows 1)", [], mcw=mcw)

    pd.DataFrame(screen_rows).sort_values(["target", "pct_residual_variance_explained"], ascending=[True, False]) \
        .to_csv(OUT / "candidate_screen.csv", index=False)
    T = pd.DataFrame(test_rows)
    base = T[T.variant == "baseline (full recipe)"].set_index("target")["mae"]
    T["change_vs_baseline"] = T.apply(lambda r: r.mae - base[r.target], axis=1)
    T.to_csv(OUT / "candidate_tests.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUT / "pacific_residual_drift.csv", index=False)
    print(T.to_string(index=False))
    print(pd.DataFrame(drift_rows).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
