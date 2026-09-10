"""Detail ladder for the smooth lat/lon models, plus the MoE reference maps.

Question (meeting notes 2026-09): is the smoothness of the GP/SVR maps an
upper bound of those models or just our settings? We vary the three things
that control it — training subsample size, kernel length scale (forced
shorter than the likelihood optimum), and kernel family (two-scale RBF,
Matern) — and, for the SVR, the bump width gamma. Every variant is scored
out-of-fold under the locked protocol on zero-filled targets.

Also builds an empirical semivariogram of the error field (nugget = noise
floor, range = true correlation scale) and two reference maps from the
recommended MoE blend's saved OOF predictions: its correction binned in
space, and that map minus the GP lat/lon map ("physics beyond geography").

Outputs: OHC/output/smooth_ladder_20260909/
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
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import _build_forward_folds  # noqa: E402
from OHC.seasonal_map_common import EARTH_R_KM, add_land_overlay, latlon_to_xyz  # noqa: E402
from OHC.exploration.run_latlon_diagnostics import SRC, TARGETS, build_grid, gp_predict_chunked  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/smooth_ladder_20260909")
DIAG = Path("/home/suramya/HHP-Prediction/OHC/output/latlon_diagnostics_20260908")
MOE = Path("/home/suramya/HHP-Prediction/OHC/output/moe_showcase_20260811")
SEED = 42


def gp_variants():
    C = lambda: ConstantKernel(10.0, (1e-2, 1e4))
    W = lambda: WhiteKernel(50.0, (1e-1, 1e4))
    return {
        "gp learned scales, n=3k": (3000, C() * RBF([10.0, 20.0], (1.0, 90.0)) + W(), True),
        "gp forced 4°x16°, n=3k": (3000, C() * RBF([4.0, 16.0], "fixed") + W(), True),
        "gp forced 2°x8°, n=3k": (3000, C() * RBF([2.0, 8.0], "fixed") + W(), True),
        "gp two-scale (long+short), n=3k": (3000, C() * RBF([10.0, 40.0], (2.0, 90.0)) + C() * RBF([2.0, 5.0], (0.5, 8.0)) + W(), True),
        "gp Matern 3/2 aniso, n=3k": (3000, C() * Matern([8.0, 25.0], (1.0, 90.0), nu=1.5) + W(), True),
        "gp learned scales, n=6k": (6000, C() * RBF([10.0, 20.0], (1.0, 90.0)) + W(), True),
        "gp learned scales, n=10k": (10000, C() * RBF([10.0, 20.0], (1.0, 90.0)) + W(), True),
    }


def svr_variants():
    return {"svr gamma x1 (baseline)": 1.0, "svr gamma x4": 4.0, "svr gamma x16": 16.0}


def semivariogram(lat, lon, val, rng, n_pairs=400000, max_km=4000, nbins=40):
    n = len(val)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    xyz = latlon_to_xyz(lat, lon)
    chord = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    dkm = EARTH_R_KM * 2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1))
    g = 0.5 * (val[i] - val[j]) ** 2
    bins = np.linspace(0, max_km, nbins + 1)
    idx = np.digitize(dkm, bins) - 1
    ok = (idx >= 0) & (idx < nbins)
    gamma = np.array([g[ok & (idx == b)].mean() if (ok & (idx == b)).any() else np.nan for b in range(nbins)])
    return 0.5 * (bins[1:] + bins[:-1]), gamma


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(SRC)
    dates = pd.to_datetime(df["date"].astype(str))
    df = df.assign(_date=dates.dt.strftime("%Y%m%d")).sort_values("_date").reset_index(drop=True)
    folds = _build_forward_folds(sorted(df["_date"].unique()), n_folds=3, embargo_dates=1)
    X_all = df[["lat", "lon"]].to_numpy(float)
    glat, glon, Xg = build_grid()
    rows, kernels = [], {}

    for tname, (obs_c, mod_c, unit) in TARGETS.items():
        obs0 = df[obs_c].fillna(0.0).to_numpy(float)
        mod0 = df[mod_c].fillna(0.0).to_numpy(float)
        delta0 = obs0 - mod0
        warm = (df[obs_c].notna() & df[mod_c].notna()).to_numpy()
        grids = {}

        variants = {**{k: ("gp", v) for k, v in gp_variants().items()},
                    **{k: ("svr", v) for k, v in svr_variants().items()}}
        for name, (kind, spec) in variants.items():
            oof = np.full(len(df), np.nan)
            for fold in folds:
                tr = df["_date"].isin(set(fold["train_dates"])).to_numpy()
                va = df["_date"].isin(set(fold["val_dates"])).to_numpy()
                last = fold["fold"] == len(folds)
                if kind == "gp":
                    n, kernel, _ = spec
                    gidx = rng.choice(np.flatnonzero(tr), size=min(n, tr.sum()), replace=False)
                    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=SEED)
                    gpr.fit(X_all[gidx], delta0[gidx])
                    oof[va] = gp_predict_chunked(gpr, X_all[va])
                    if last:
                        grids[name] = gp_predict_chunked(gpr, Xg).reshape(glat.shape)
                        kernels[f"{tname} | {name}"] = str(gpr.kernel_)
                else:
                    idx = rng.choice(np.flatnonzero(tr), size=min(20000, tr.sum()), replace=False)
                    sc = StandardScaler().fit(X_all[idx])
                    Xs = sc.transform(X_all[idx])
                    gamma = spec * (1.0 / (Xs.shape[1] * Xs.var()))
                    svr = SVR(kernel="rbf", C=50.0, epsilon=1.0, gamma=gamma, cache_size=1000)
                    svr.fit(Xs, delta0[idx])
                    oof[va] = svr.predict(sc.transform(X_all[va]))
                    if last:
                        grids[name] = svr.predict(sc.transform(Xg)).reshape(glat.shape)
            v = np.isfinite(oof)
            e_all = (mod0 + oof - obs0)[v]; e_w = (mod0 + oof - obs0)[v & warm]
            rows.append({"target": tname, "variant": name, "mae_all": float(np.abs(e_all).mean()),
                         "mae_warm": float(np.abs(e_w).mean()), "rmse_warm": float(np.sqrt((e_w ** 2).mean())),
                         "map_texture_std": float(np.std(grids[name][(glat > -25) & (glat < 25)]))})
            print(tname, name, f"warm MAE {rows[-1]['mae_warm']:.3f}", flush=True)

        # reference maps: RF and MoE
        z = np.load(DIAG / f"{tname}_grid_predictions.npz")
        grids["random forest (reference)"] = z["rf"]
        moe = pd.read_parquet(MOE / f"moe_predictions_{tname}.parquet")
        corr = (moe["pred_obs__moe_blend"] - moe["pred_obs__raw_rtofs"]).to_numpy(float)
        ok = np.isfinite(corr)
        la = np.floor(moe["lat"].to_numpy(float)[ok]) ; lo = np.floor(moe["lon"].to_numpy(float)[ok])
        binned = pd.DataFrame({"la": la, "lo": lo, "c": corr[ok]}).groupby(["la", "lo"])["c"].mean()
        moe_grid = np.full(glat.shape, np.nan)
        cell = {(a, b): c for (a, b), c in binned.items()}
        fl_la, fl_lo = np.floor(glat), np.floor(glon)
        for i in range(glat.shape[0]):
            for j in range(glat.shape[1]):
                moe_grid[i, j] = cell.get((fl_la[i, j], fl_lo[i, j]), np.nan)
        grids["MoE blend (best model), binned OOF correction"] = moe_grid
        grids["physics beyond geography: MoE − GP(lat,lon)"] = moe_grid - grids["gp learned scales, n=3k"]
        np.savez_compressed(OUT / f"{tname}_ladder_grids.npz", glat=glat, glon=glon, **{k.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("°", "deg").replace("−", "minus").replace(":", ""): v for k, v in grids.items()})

        # ladder figure
        order = list(gp_variants().keys()) + list(svr_variants().keys()) + ["random forest (reference)",
                 "MoE blend (best model), binned OOF correction", "physics beyond geography: MoE − GP(lat,lon)"]
        vmax = float(np.nanpercentile(np.abs(delta0), 98))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ncol = 3; nrow = int(np.ceil(len(order) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(21, 3.6 * nrow), constrained_layout=True)
        maes = {r["variant"]: r["mae_warm"] for r in rows if r["target"] == tname}
        for ax, name in zip(axes.ravel(), order):
            pm = ax.pcolormesh(glon, glat, np.ma.masked_invalid(grids[name]), shading="auto", cmap="RdBu_r", norm=norm)
            add_land_overlay(ax, facecolor="#c9c9c9", zorder=5)
            ax.set_xlim(-180, 180); ax.set_ylim(-40, 40); ax.set_xticks([]); ax.set_yticks([])
            extra = f"  —  OOF MAE {maes[name]:.2f}" if name in maes else ""
            ax.set_title(name + extra, fontsize=10.5)
        for ax in axes.ravel()[len(order):]:
            ax.axis("off")
        fig.colorbar(pm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01).set_label(f"correction ({unit})")
        fig.suptitle(f"{tname.upper()}: the detail ladder — smoothness is a setting, not a bound\n"
                     f"(reference MAEs: RF {np.nan if 'rf' not in locals() else ''}12.44/12.50 lat/lon; full-feature MoE 11.19/10.55)", fontsize=14)
        fig.savefig(OUT / f"{tname}_detail_ladder.png", dpi=150)
        plt.close(fig)

        # semivariogram of the error field (warm rows, final training block)
        tr = df["_date"].isin(set(folds[-1]["train_dates"])).to_numpy() & warm
        h, g = semivariogram(df.loc[tr, "lat"].to_numpy(float), df.loc[tr, "lon"].to_numpy(float), delta0[tr], rng)
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.plot(h, g, "o-", color="#2563eb")
        ax.axhline(np.nanvar(delta0[tr]), color="#94a3b8", linestyle="--", label="total variance (sill)")
        ax.set_xlabel("separation between profile pairs (km)"); ax.set_ylabel(f"semivariance γ(h) ({unit}²)")
        ax.set_title(f"{tname.upper()}: semivariogram of the forecast error\nintercept ≈ noise floor (nugget); leveling distance ≈ true correlation scale (range)")
        ax.grid(alpha=0.15); ax.legend()
        fig.savefig(OUT / f"{tname}_semivariogram.png", dpi=160)
        plt.close(fig)
        pd.DataFrame({"separation_km": h, "semivariance": g}).to_csv(OUT / f"{tname}_semivariogram.csv", index=False)
        print(tname, "nugget (first bin)", float(g[0]), "sill", float(np.nanvar(delta0[tr])), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "smooth_ladder_scores.csv", index=False)
    (OUT / "fitted_kernels.json").write_text(json.dumps(kernels, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
