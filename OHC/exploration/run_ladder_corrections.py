"""Corrections to the smooth-model ladder (audit, 2026-09-23).

1. Same-date semivariogram. The original pooled pairs across dates, so its
   intercept mixed spatial and temporal variability and the "noise floor" it
   implied was not a floor for any model that knows the date.
2. Extend the SVR bump-width sweep. x16 was the largest value tested and won
   monotonically, so the reported optimum sat on the edge of the search range.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import OHC.run_locked_xgb_physics_semi_ablation as abl
from OHC.benchmark_rtofs_argo_tabular_models import _build_forward_folds
from OHC.seasonal_map_common import EARTH_R_KM, latlon_to_xyz
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/smooth_ladder_20260909")
TARGETS = {"tchp": ("argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "kJ/cm²"),
           "d26": ("argo_d26_m", "model_interp_d26_m", "m")}
GAMMA = [1.0, 4.0, 16.0, 32.0, 64.0, 128.0]

def variogram(lat, lon, err, dates, same_date, bins):
    chord = lambda km: 2*np.sin(km/EARTH_R_KM/2)
    num = np.zeros(len(bins)-1); cnt = np.zeros(len(bins)-1)
    if same_date:
        for d in np.unique(dates):
            m = dates == d
            if m.sum() < 2: continue
            xyz = latlon_to_xyz(lat[m], lon[m]); e = err[m]; t = cKDTree(xyz)
            for i, j in t.query_pairs(chord(bins[-1])):
                dk = EARTH_R_KM*2*np.arcsin(np.clip(np.linalg.norm(xyz[i]-xyz[j])/2, 0, 1))
                b = np.searchsorted(bins, dk) - 1
                if 0 <= b < len(num): num[b] += 0.5*(e[i]-e[j])**2; cnt[b] += 1
    else:
        rng = np.random.default_rng(0); xyz = latlon_to_xyz(lat, lon)
        i = rng.integers(0, len(err), 4_000_000); j = rng.integers(0, len(err), 4_000_000)
        k = i != j; i, j = i[k], j[k]
        dk = EARTH_R_KM*2*np.arcsin(np.clip(np.linalg.norm(xyz[i]-xyz[j], axis=1)/2, 0, 1))
        g = 0.5*(err[i]-err[j])**2; b = np.searchsorted(bins, dk) - 1
        ok = (b >= 0) & (b < len(num))
        np.add.at(num, b[ok], g[ok]); np.add.at(cnt, b[ok], 1)
    return np.where(cnt > 0, num/np.maximum(cnt, 1), np.nan), cnt

def main() -> None:
    df = abl._merge_feature_tables()
    rows = []
    bins = np.arange(0, 1001, 50.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    for ax, (t, (oc, mc, unit)) in zip(axes, TARGETS.items()):
        w = df[df[oc].notna() & df[mc].notna()].copy()
        err = (w[oc] - w[mc]).to_numpy(float)
        lat, lon = w["lat"].to_numpy(float), w["lon"].to_numpy(float)
        dates = w["date"].astype(str).to_numpy()
        g_any, _ = variogram(lat, lon, err, dates, False, bins)
        g_same, c_same = variogram(lat, lon, err, dates, True, bins)
        h = 0.5*(bins[1:] + bins[:-1])
        ax.plot(h, g_any, "o-", color="#94a3b8", label="all pairs, any date (what we used before)")
        ax.plot(h, g_same, "o-", color="#2563eb", label="pairs from the SAME day (spatial only)")
        ax.axhline(np.var(err), color="#64748b", ls="--", lw=1, label="total variance")
        f_any, f_same = np.sqrt(g_any[0]), np.sqrt(g_same[0])
        ax.set_title(f"{t.upper()}: floor {f_any:.1f} pooled vs {f_same:.1f} same-day ({unit})", fontsize=11.5)
        ax.set_xlabel("separation (km)"); ax.set_ylabel(f"semivariance ({unit}²)")
        ax.grid(alpha=0.15); ax.legend(fontsize=8.5)
        rows.append({"target": t, "floor_rmse_pooled": float(f_any), "floor_rmse_same_day": float(f_same),
                     "temporal_share_pct": float(100*(g_any[0]-g_same[0])/g_any[0]),
                     "same_day_pairs_first_bin": int(c_same[0]), "total_variance": float(np.var(err))})
        print(rows[-1], flush=True)
    fig.suptitle("Correction: most of the apparent 'noise floor' was temporal, not spatial\n"
                 "a model that knows the date faces the blue floor, not the grey one", fontsize=13)
    fig.savefig(OUT / "semivariogram_same_date_correction.png", dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "noise_floor_corrected.csv", index=False)

    # --- SVR bump-width sweep, now bracketed ---
    srows = []
    for t, (oc, mc, unit) in TARGETS.items():
        obs0 = df[oc].fillna(0.0).to_numpy(float); mod0 = df[mc].fillna(0.0).to_numpy(float)
        delta0 = obs0 - mod0; warm = (df[oc].notna() & df[mc].notna()).to_numpy()
        X = df[["lat", "lon"]].to_numpy(float)
        ds = pd.to_datetime(df["date"].astype(str)).dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(ds.unique().tolist()), n_folds=3, embargo_dates=1)
        rng = np.random.default_rng(42)
        for gmul in GAMMA:
            oof = np.full(len(df), np.nan)
            for f in folds:
                tr = ds.isin(set(f["train_dates"])).to_numpy(); va = ds.isin(set(f["val_dates"])).to_numpy()
                idx = rng.choice(np.flatnonzero(tr), size=min(20000, tr.sum()), replace=False)
                sc = StandardScaler().fit(X[idx]); Xs = sc.transform(X[idx])
                gamma = gmul * (1.0/(Xs.shape[1]*Xs.var()))
                m = SVR(kernel="rbf", C=50.0, epsilon=1.0, gamma=gamma, cache_size=1000).fit(Xs, delta0[idx])
                oof[va] = m.predict(sc.transform(X[va]))
            v = np.isfinite(oof)
            L = 1/np.sqrt(2*gamma)
            srows.append({"target": t, "gamma_multiplier": gmul,
                          "implied_length_deg_lat": float(L*df['lat'].std()),
                          "implied_length_deg_lon": float(L*df['lon'].std()),
                          "mae_warm": float(np.abs(mod0+oof-obs0)[v & warm].mean())})
            print(srows[-1], flush=True)
    sd = pd.DataFrame(srows); sd.to_csv(OUT / "svr_gamma_sweep_bracketed.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for t, c, ref in (("tchp", "#2563eb", 12.441), ("d26", "#16a34a", 12.503)):
        s = sd[sd.target == t]
        ax.plot(s.gamma_multiplier, s.mae_warm, "o-", color=c, label=f"{t.upper()} (random forest {ref:.2f})")
        ax.axhline(ref, color=c, ls=":", lw=1)
        b = s.loc[s.mae_warm.idxmin()]
        ax.plot(b.gamma_multiplier, b.mae_warm, "*", ms=16, color=c)
    ax.set_xscale("log", base=2); ax.set_xlabel("bump-width multiplier (larger = narrower bumps)")
    ax.set_ylabel("out-of-fold MAE, warm rows"); ax.grid(alpha=0.15); ax.legend(fontsize=9)
    ax.set_title("SVR bump width, now bracketed\nthe earlier sweep stopped at 16, which was its own best value", fontsize=12)
    fig.savefig(OUT / "svr_gamma_sweep_bracketed.png", dpi=160); plt.close(fig)
    print(sd.to_string(index=False))

if __name__ == "__main__":
    main()
