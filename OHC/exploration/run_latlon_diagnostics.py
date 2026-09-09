"""Diagnostics batch on the lat/lon-only models (meeting notes, 2026-09).

Covers: cached OOF + grid predictions (infrastructure), MAE+RMSE tables,
date-block bootstrap of paired model differences ("is the ranking noise?"),
RF leaf-size sweep with train-vs-holdout curves (overfitting), RF bootstrap
stability map vs GP posterior std (is RF detail noise?), seam analysis for
the GoM discontinuity question, and smooth-region variance summaries.

Outputs: OHC/output/latlon_diagnostics_20260908/
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
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import _build_forward_folds  # noqa: E402
from OHC.seasonal_map_common import EARTH_R_KM, add_land_overlay, latlon_to_xyz  # noqa: E402

SRC = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/latlon_diagnostics_20260908")
TARGETS = {"tchp": ("argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "kJ/cm²"),
           "d26": ("argo_d26_m", "model_interp_d26_m", "m")}
SEED = 42
LEAF_SWEEP = [5, 10, 25, 50, 100, 200]
BOOT_MODELS = 12
BOOT_TREES = 100
N_BOOT_DATES = 1000
GOM = dict(lat0=18, lat1=31, lon0=-98, lon1=-80)


def build_grid():
    lats = np.arange(-70, 70.001, 0.5)
    lons = np.arange(-180, 180.001, 0.5)
    glon, glat = np.meshgrid(lons, lats)
    return glat, glon, np.column_stack([glat.ravel(), glon.ravel()])


def gp_predict_chunked(gpr, X, chunk=20000, std=False):
    outs, stds = [], []
    for i in range(0, len(X), chunk):
        if std:
            m, s = gpr.predict(X[i:i + chunk], return_std=True)
            outs.append(m); stds.append(s)
        else:
            outs.append(gpr.predict(X[i:i + chunk]))
    return (np.concatenate(outs), np.concatenate(stds)) if std else np.concatenate(outs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(SRC)
    dates = pd.to_datetime(df["date"].astype(str))
    df = df.assign(_date=dates.dt.strftime("%Y%m%d")).sort_values("_date").reset_index(drop=True)
    folds = _build_forward_folds(sorted(df["_date"].unique()), n_folds=3, embargo_dates=1)
    X_all = df[["lat", "lon"]].to_numpy(float)
    glat, glon, Xg = build_grid()

    metric_rows, sweep_rows, boot_rows, seam_rows, summary = [], [], [], [], {}
    for tname, (obs_c, mod_c, unit) in TARGETS.items():
        obs0 = df[obs_c].fillna(0.0).to_numpy(float)
        mod0 = df[mod_c].fillna(0.0).to_numpy(float)
        delta0 = obs0 - mod0
        warm = (df[obs_c].notna() & df[mod_c].notna()).to_numpy()

        oof = {m: np.full(len(df), np.nan) for m in ("rf", "svr_rbf", "gpr", "xgb")}
        grid_pred, gpr_final, sc_final = {}, None, None
        for fold in folds:
            tr = df["_date"].isin(set(fold["train_dates"])).to_numpy()
            va = df["_date"].isin(set(fold["val_dates"])).to_numpy()
            Xtr, ytr = X_all[tr], delta0[tr]

            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=50, n_jobs=16, random_state=SEED)
            rf.fit(Xtr, ytr); oof["rf"][va] = rf.predict(X_all[va])

            xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                               n_jobs=16, random_state=SEED)
            xgb.fit(Xtr, ytr); oof["xgb"][va] = xgb.predict(X_all[va])

            idx = rng.choice(np.flatnonzero(tr), size=min(20000, tr.sum()), replace=False)
            sc = StandardScaler().fit(X_all[idx])
            svr = SVR(kernel="rbf", C=50.0, epsilon=1.0, gamma="scale", cache_size=1000)
            svr.fit(sc.transform(X_all[idx]), delta0[idx])
            oof["svr_rbf"][va] = svr.predict(sc.transform(X_all[va]))

            gidx = rng.choice(np.flatnonzero(tr), size=min(3000, tr.sum()), replace=False)
            kernel = (ConstantKernel(10.0, (1e-2, 1e4)) * RBF([10.0, 20.0], (1.0, 90.0))
                      + WhiteKernel(50.0, (1e-1, 1e4)))
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=SEED)
            gpr.fit(X_all[gidx], delta0[gidx]); oof["gpr"][va] = gpr.predict(X_all[va])

            if fold["fold"] == len(folds):
                grid_pred["rf"] = rf.predict(Xg).reshape(glat.shape)
                grid_pred["xgb"] = xgb.predict(Xg).reshape(glat.shape)
                grid_pred["svr_rbf"] = svr.predict(sc.transform(Xg)).reshape(glat.shape)
                gm, gs = gp_predict_chunked(gpr, Xg, std=True)
                grid_pred["gpr"] = gm.reshape(glat.shape)
                grid_pred["gpr_std"] = gs.reshape(glat.shape)
                gpr_final, tr_final = gpr, tr
            print(tname, "fold", fold["fold"], "base models done", flush=True)

        # ---- cache OOF predictions ----
        v = np.isfinite(oof["rf"])
        cache = pd.DataFrame({"date": df["_date"], "lat": df["lat"], "lon": df["lon"],
                              "obs0": obs0, "mod0": mod0, "warm": warm})
        for m, p in oof.items():
            cache[f"pred_{m}"] = p
        cache[v].to_parquet(OUT / f"{tname}_oof_predictions.parquet", index=False)

        # ---- item 2: MAE + RMSE ----
        for m, p in oof.items():
            for scope, mask in (("all rows", v), ("warm", v & warm)):
                e = (mod0 + p - obs0)[mask]
                metric_rows.append({"target": tname, "model": m, "scope": scope,
                                    "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                    "rows": int(mask.sum())})
        for scope, mask in (("all rows", v), ("warm", v & warm)):
            e = (mod0 - obs0)[mask]
            metric_rows.append({"target": tname, "model": "raw", "scope": scope,
                                "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                "rows": int(mask.sum())})

        # ---- item 3a: date-block bootstrap of paired differences ----
        vdates = df.loc[v, "_date"].to_numpy()
        uniq = np.unique(vdates)
        per_date = {}
        for m, p in oof.items():
            e = np.abs((mod0 + p - obs0))[v]
            s = pd.DataFrame({"d": vdates, "e": e}).groupby("d")["e"].agg(["sum", "count"])
            per_date[m] = s
        for a, b in (("rf", "gpr"), ("rf", "xgb"), ("gpr", "xgb"), ("rf", "svr_rbf")):
            diffs = []
            for _ in range(N_BOOT_DATES):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                sa, sb = per_date[a].loc[pick], per_date[b].loc[pick]
                diffs.append(sa["sum"].sum() / sa["count"].sum() - sb["sum"].sum() / sb["count"].sum())
            diffs = np.array(diffs)
            boot_rows.append({"target": tname, "pair": f"{a} - {b}",
                              "mean_diff": float(diffs.mean()),
                              "ci_lo": float(np.percentile(diffs, 2.5)),
                              "ci_hi": float(np.percentile(diffs, 97.5)),
                              "significant": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0)})

        # ---- item 4a: leaf-size sweep (train vs OOF) ----
        for leaf in LEAF_SWEEP:
            tr_maes, oof_pred = [], np.full(len(df), np.nan)
            for fold in folds:
                tr = df["_date"].isin(set(fold["train_dates"])).to_numpy()
                va = df["_date"].isin(set(fold["val_dates"])).to_numpy()
                rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=leaf, n_jobs=16, random_state=SEED)
                rf.fit(X_all[tr], delta0[tr])
                tr_maes.append(np.abs(mod0[tr] + rf.predict(X_all[tr]) - obs0[tr]).mean())
                oof_pred[va] = rf.predict(X_all[va])
            vv = np.isfinite(oof_pred)
            sweep_rows.append({"target": tname, "leaf": leaf,
                               "train_mae": float(np.mean(tr_maes)),
                               "oof_mae": float(np.abs(mod0 + oof_pred - obs0)[vv].mean()),
                               "oof_mae_warm": float(np.abs(mod0 + oof_pred - obs0)[vv & warm].mean())})
            print(tname, "leaf", leaf, "done", flush=True)

        # ---- item 4b: RF bootstrap stability map ----
        tr = tr_final
        tr_dates = df.loc[tr, "_date"].to_numpy()
        uniq_tr = np.unique(tr_dates)
        boots = []
        for b in range(BOOT_MODELS):
            pick = set(rng.choice(uniq_tr, size=len(uniq_tr), replace=True))
            m2 = tr & df["_date"].isin(pick).to_numpy()
            rf_b = RandomForestRegressor(n_estimators=BOOT_TREES, min_samples_leaf=50, n_jobs=16,
                                         random_state=SEED + b)
            rf_b.fit(X_all[m2], delta0[m2])
            boots.append(rf_b.predict(Xg))
            print(tname, "boot", b + 1, "/", BOOT_MODELS, flush=True)
        rf_boot_std = np.std(np.stack(boots), axis=0).reshape(glat.shape)

        np.savez_compressed(OUT / f"{tname}_grid_predictions.npz", glat=glat, glon=glon,
                            rf=grid_pred["rf"], xgb=grid_pred["xgb"], svr_rbf=grid_pred["svr_rbf"],
                            gpr=grid_pred["gpr"], gpr_std=grid_pred["gpr_std"], rf_boot_std=rf_boot_std)

        # ---- item 1: seam analysis ----
        gy, gx = np.gradient(grid_pred["rf"])
        seam = np.hypot(gy, gx)
        gpy, gpx = np.gradient(grid_pred["gpr"])
        gp_grad = np.hypot(gpy, gpx)
        tree = cKDTree(latlon_to_xyz(df.loc[tr, "lat"].to_numpy(float), df.loc[tr, "lon"].to_numpy(float)))
        counts = tree.query_ball_point(latlon_to_xyz(glat.ravel(), glon.ravel()),
                                       r=2.0 * np.sin(np.deg2rad(1.0) / 2.0) * 2.0, workers=-1,
                                       return_length=True).reshape(glat.shape)
        sel = seam > 1.0
        in_gom = ((glat >= GOM["lat0"]) & (glat <= GOM["lat1"]) &
                  (glon >= GOM["lon0"]) & (glon <= GOM["lon1"]))
        for name, mask in (("global", sel), ("gom", sel & in_gom)):
            seam_rows.append({"target": tname, "scope": name, "n_seams": int(mask.sum()),
                              "median_seam": float(np.median(seam[mask])) if mask.any() else np.nan,
                              "median_density": float(np.median(counts[mask])) if mask.any() else np.nan,
                              "corr_seam_gpgrad": float(np.corrcoef(seam[mask], gp_grad[mask])[0, 1]) if mask.sum() > 10 else np.nan,
                              "corr_seam_invsqrtdens": float(np.corrcoef(seam[mask], 1.0 / np.sqrt(np.maximum(counts[mask], 1)))[0, 1]) if mask.sum() > 10 else np.nan})

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        ax = axes[0]
        pm = ax.pcolormesh(glon, glat, seam, cmap="inferno", norm=PowerNorm(0.45, vmin=0, vmax=np.percentile(seam, 98)))
        add_land_overlay(ax, facecolor="#c9c9c9", zorder=5)
        ax.set_xlim(GOM["lon0"], GOM["lon1"]); ax.set_ylim(GOM["lat0"], GOM["lat1"])
        s = df.loc[tr]
        gsel = (s["lat"] >= GOM["lat0"]) & (s["lat"] <= GOM["lat1"]) & (s["lon"] >= GOM["lon0"]) & (s["lon"] <= GOM["lon1"])
        ax.scatter(s.loc[gsel, "lon"], s.loc[gsel, "lat"], s=2, c="#3bd66f", alpha=0.5, zorder=6, label="training profiles")
        ax.legend(loc="lower left"); ax.set_title(f"{tname.upper()} Gulf zoom: RF seams + data coverage")
        fig.colorbar(pm, ax=ax, shrink=0.9).set_label(f"jump ({unit})")
        ax = axes[1]
        sub = rng.choice(np.flatnonzero(sel.ravel()), size=min(4000, int(sel.sum())), replace=False)
        sc2 = ax.scatter(counts.ravel()[sub], seam.ravel()[sub], c=gp_grad.ravel()[sub], s=6, cmap="viridis", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training profiles within ~1° (density)")
        ax.set_ylabel(f"seam size ({unit})")
        ax.set_title("seam size vs data density, colored by the GP's real gradient")
        fig.colorbar(sc2, ax=ax, shrink=0.9).set_label(f"GP field gradient ({unit}/0.5°)")
        fig.suptitle(f"{tname.upper()}: are the seams real gradient or small-sample noise?", fontsize=14)
        fig.savefig(OUT / f"{tname}_seam_analysis.png", dpi=160)
        plt.close(fig)

        # ---- item 4b/10 figure: RF instability vs GP uncertainty ----
        fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
        vmax = np.percentile(rf_boot_std, 98)
        for ax, (fld, title) in zip(axes, ((rf_boot_std, "RF bootstrap instability: std of the prediction over 12 refits on resampled dates"),
                                           (grid_pred["gpr_std"], "GP posterior standard deviation (its own stated uncertainty)"))):
            pm = ax.pcolormesh(glon, glat, fld, cmap="magma", vmin=0, vmax=vmax)
            add_land_overlay(ax, facecolor="#c9c9c9", zorder=5)
            ax.set_xlim(-180, 180); ax.set_ylim(-70, 70)
            ax.set_title(f"{tname.upper()}: {title}", fontsize=12)
            fig.colorbar(pm, ax=ax, shrink=0.85).set_label(unit)
        fig.savefig(OUT / f"{tname}_stability_maps.png", dpi=160)
        plt.close(fig)

        # ---- item 10: variance in smooth regions ----
        smooth = gp_grad < np.percentile(gp_grad, 50)
        summary[tname] = {
            "rf_texture_std_smooth_regions": float(np.std(grid_pred["rf"][smooth])),
            "gpr_texture_std_smooth_regions": float(np.std(grid_pred["gpr"][smooth])),
            "rf_boot_std_median_smooth": float(np.median(rf_boot_std[smooth])),
            "gpr_std_median_smooth": float(np.median(grid_pred["gpr_std"][smooth])),
            "gpr_kernel": str(gpr_final.kernel_),
        }
        print(tname, "target complete", flush=True)

    pd.DataFrame(metric_rows).to_csv(OUT / "mae_rmse_table.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_ranking.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(OUT / "rf_leaf_sweep.csv", index=False)
    pd.DataFrame(seam_rows).to_csv(OUT / "seam_stats.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    sw = pd.DataFrame(sweep_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, tname in zip(axes, TARGETS):
        s = sw[sw.target == tname]
        ax.plot(s.leaf, s.train_mae, "o-", label="training error", color="#94a3b8")
        ax.plot(s.leaf, s.oof_mae, "o-", label="held-out error", color="#2563eb")
        ax.set_xscale("log"); ax.set_xlabel("minimum samples per leaf"); ax.set_ylabel("MAE (zero-filled scale)")
        ax.set_title(f"{tname.upper()}: RF overfitting check"); ax.legend(); ax.grid(alpha=0.15)
    fig.suptitle("Gap between curves = overfitting; where the blue curve bottoms out = right leaf size")
    fig.savefig(OUT / "rf_leaf_sweep.png", dpi=160)
    plt.close(fig)

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.DataFrame(boot_rows).to_string(index=False))
    print(pd.DataFrame(seam_rows).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
