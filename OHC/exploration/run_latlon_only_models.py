"""Fit the TCHP/D26 error with latitude and longitude only (Dr. Jacobs, 2026-09).

Spec from his email:
- Include ALL collocated rows. Where a side has no 26 C water, set its TCHP
  and D26 to 0 (physically real for TCHP; convention for D26), so coverage
  spans all latitudes. Mixed cases (one side warm, other not) zero-fill the
  missing side and compute the error normally.
- Features: lat, lon only. Models: random forest (show the lat/lon partition
  and its discontinuities), SVR with an RBF kernel, and Gaussian process
  regression with an anisotropic kernel (separate lat/lon length scales).
- Locked blocked-forward protocol (3 folds, 1-date embargo) for the scores.

Outputs in OHC/output/latlon_only_20260903/: MAE table, predicted-correction
maps per model, a single-tree partition-box figure, and an RF discontinuity
(gradient magnitude) map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import _build_forward_folds  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402

RUN_DATE = "20260903"
SRC = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT = Path(f"/home/suramya/HHP-Prediction/OHC/output/latlon_only_{RUN_DATE}")
TARGETS = {
    "tchp": ("argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "kJ/cm²"),
    "d26": ("argo_d26_m", "model_interp_d26_m", "m"),
}
SEED = 42
SVR_TRAIN_MAX = 20000
GPR_TRAIN_MAX = 3000


def grid_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.arange(-70, 70.001, 0.5)
    lons = np.arange(-180, 180.001, 0.5)
    glon, glat = np.meshgrid(lons, lats)
    X = np.column_stack([glat.ravel(), glon.ravel()])
    return glat, glon, X


def tree_boxes(tree: DecisionTreeRegressor, bounds=(-90.0, 90.0, -180.0, 180.0)):
    t = tree.tree_
    boxes = []

    def walk(node, lat0, lat1, lon0, lon1):
        if t.children_left[node] == -1:
            boxes.append((lat0, lat1, lon0, lon1, t.value[node][0][0], t.n_node_samples[node]))
            return
        f, thr = t.feature[node], t.threshold[node]
        if f == 0:
            walk(t.children_left[node], lat0, min(lat1, thr), lon0, lon1)
            walk(t.children_right[node], max(lat0, thr), lat1, lon0, lon1)
        else:
            walk(t.children_left[node], lat0, lat1, lon0, min(lon1, thr))
            walk(t.children_right[node], lat0, lat1, max(lon0, thr), lon1)

    walk(0, *bounds)
    return boxes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(SRC)
    dates = pd.to_datetime(df["date"].astype(str))
    df = df.assign(_date=dates.dt.strftime("%Y%m%d")).sort_values("_date").reset_index(drop=True)
    unique_dates = sorted(df["_date"].unique())
    folds = _build_forward_folds(unique_dates, n_folds=3, embargo_dates=1)

    rows = []
    for tname, (obs_c, mod_c, unit) in TARGETS.items():
        obs0 = df[obs_c].fillna(0.0).to_numpy(float)
        mod0 = df[mod_c].fillna(0.0).to_numpy(float)
        delta0 = obs0 - mod0
        both_warm = (df[obs_c].notna() & df[mod_c].notna()).to_numpy()
        X_all = df[["lat", "lon"]].to_numpy(float)

        models_pred_grid: dict[str, np.ndarray] = {}
        glat, glon, Xg = grid_features()

        oof_pred = {m: np.full(len(df), np.nan) for m in ("rf", "svr_rbf", "gpr")}
        for fold in folds:
            tr = df["_date"].isin(set(fold["train_dates"])).to_numpy()
            va = df["_date"].isin(set(fold["val_dates"])).to_numpy()
            Xtr, ytr = X_all[tr], delta0[tr]

            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=SEED)
            rf.fit(Xtr, ytr)
            oof_pred["rf"][va] = rf.predict(X_all[va])

            idx = rng.choice(np.flatnonzero(tr), size=min(SVR_TRAIN_MAX, tr.sum()), replace=False)
            sc = StandardScaler().fit(X_all[idx])
            svr = SVR(kernel="rbf", C=50.0, epsilon=1.0, gamma="scale", cache_size=1000)
            svr.fit(sc.transform(X_all[idx]), delta0[idx])
            oof_pred["svr_rbf"][va] = svr.predict(sc.transform(X_all[va]))

            gidx = rng.choice(np.flatnonzero(tr), size=min(GPR_TRAIN_MAX, tr.sum()), replace=False)
            kernel = (ConstantKernel(10.0, (1e-2, 1e4))
                      * RBF(length_scale=[10.0, 20.0], length_scale_bounds=(1.0, 90.0))
                      + WhiteKernel(noise_level=50.0, noise_level_bounds=(1e-1, 1e4)))
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=SEED)
            gpr.fit(X_all[gidx], delta0[gidx])
            oof_pred["gpr"][va] = gpr.predict(X_all[va])
            if fold["fold"] == len(folds):
                models_pred_grid["rf"] = rf.predict(Xg).reshape(glat.shape)
                models_pred_grid["svr_rbf"] = svr.predict(sc.transform(Xg)).reshape(glat.shape)
                models_pred_grid["gpr"] = gpr.predict(Xg).reshape(glat.shape)
                print(tname, "fitted GPR kernel:", gpr.kernel_)

        val_mask = np.isfinite(oof_pred["rf"])
        raw_mae = np.abs(delta0[val_mask]).mean()
        rows.append({"target": tname, "model": "raw RTOFS (zero-filled scale)", "scope": "all rows",
                     "mae": raw_mae, "rows": int(val_mask.sum())})
        for m, pred in oof_pred.items():
            err = (mod0 + pred - obs0)[val_mask]
            rows.append({"target": tname, "model": f"{m} lat/lon only", "scope": "all rows",
                         "mae": float(np.abs(err).mean()), "rows": int(val_mask.sum())})
            bw = val_mask & both_warm
            errb = (mod0 + pred - obs0)[bw]
            rows.append({"target": tname, "model": f"{m} lat/lon only", "scope": "both sides warm",
                         "mae": float(np.abs(errb).mean()), "rows": int(bw.sum())})
        rows.append({"target": tname, "model": "raw RTOFS", "scope": "both sides warm",
                     "mae": float(np.abs(delta0[val_mask & both_warm]).mean()),
                     "rows": int((val_mask & both_warm).sum())})

        vmax = float(np.nanpercentile(np.abs(delta0), 98))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        fig, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
        labels = {"rf": "Random forest", "svr_rbf": "SVR, RBF kernel", "gpr": "Gaussian process (anisotropic RBF)"}
        for ax, m in zip(axes, ("rf", "svr_rbf", "gpr")):
            pm = ax.pcolormesh(glon, glat, models_pred_grid[m], shading="auto", cmap="RdBu_r", norm=norm, zorder=1)
            add_land_overlay(ax, zorder=5)
            ax.set_xlim(-180, 180); ax.set_ylim(-70, 70)
            ax.set_title(f"{labels[m]}: predicted correction from (lat, lon) alone", fontsize=13)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
        fig.suptitle(f"{tname.upper()} correction learned from position only (zero-filled targets, final training block)", fontsize=15)
        fig.colorbar(pm, ax=axes, shrink=0.7, pad=0.02).set_label(f"predicted Argo − RTOFS ({unit})")
        fig.savefig(OUT / f"{tname}_latlon_prediction_maps.png", dpi=160)
        plt.close(fig)

        gy, gx = np.gradient(models_pred_grid["rf"])
        fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
        pm = ax.pcolormesh(glon, glat, np.hypot(gy, gx), shading="auto", cmap="magma", zorder=1)
        add_land_overlay(ax, zorder=5)
        ax.set_xlim(-180, 180); ax.set_ylim(-70, 70)
        ax.set_title(f"{tname.upper()}: random-forest discontinuities (gradient magnitude of the predicted correction)", fontsize=13)
        fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02).set_label(f"|∇ prediction| ({unit}/0.5°)")
        fig.savefig(OUT / f"{tname}_rf_discontinuity_map.png", dpi=160)
        plt.close(fig)

        tr = df["_date"].isin(set(folds[-1]["train_dates"])).to_numpy()
        dt = DecisionTreeRegressor(max_leaf_nodes=64, min_samples_leaf=200, random_state=SEED)
        dt.fit(X_all[tr], delta0[tr])
        boxes = tree_boxes(dt)
        fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
        bnorm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")
        for lat0, lat1, lon0, lon1, val, nrows in boxes:
            lat0, lat1 = max(lat0, -70), min(lat1, 70)
            lon0, lon1 = max(lon0, -180), min(lon1, 180)
            ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                   facecolor=cmap(bnorm(val)), edgecolor="black", linewidth=0.6, zorder=1))
        add_land_overlay(ax, zorder=5)
        ax.set_xlim(-180, 180); ax.set_ylim(-70, 70)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"{tname.upper()}: how a single tree (64 leaves) partitions the globe\n"
                     f"each box is one leaf; color = the correction it predicts ({unit})", fontsize=13)
        sm = plt.cm.ScalarMappable(norm=bnorm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02).set_label(f"leaf prediction ({unit})")
        fig.savefig(OUT / f"{tname}_tree_partition_boxes.png", dpi=160)
        plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "latlon_only_mae.csv", index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
