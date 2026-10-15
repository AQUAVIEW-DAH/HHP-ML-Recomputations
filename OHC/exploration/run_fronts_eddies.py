"""Is RTOFS's small-scale error consistent with misplaced eddies and fronts? (meeting of 2026-10-08)

Figure 3a showed that only ~24% of RTOFS's departure from its own 1-degree average at a float shows up in
Argo. Dr. Jacobs asked why Argo's departure would be smaller, and whether this is what misplaced fronts and
eddies would do. Three checks:

  1. The full scatter behind 3a: Argo's departure is not smaller (std 15 vs 6 kJ/cm2) but barely
     correlated with RTOFS's (r ~ 0.1); the 0.24 slope is that weak correlation, not a smooth Argo field.
  2. If the cause is misplacement, the slope should be lowest where gradients are steep (fronts, eddy
     edges): there a small shift turns a warm-side value into a cold-side one. Slope by gradient quintile,
     with 95% intervals from resampling whole days.
  3. Smoothing and shifting RTOFS around the float, using the RTOFS values in a +/-12-cell window
     (~1 degree) around each float's nearest grid cell:
       - error spread vs the size of the averaging box, by gradient quintile: misplacement predicts that
         averaging helps most at fronts;
       - error spread for RTOFS values taken a few cells away in each direction (13 x 13 shifts): a
         systematic offset (e.g. eddies drifting west faster or slower than in reality) would move the
         minimum away from the centre; random misplacement makes the surface flat near the centre.
Error spread = standard deviation of (Argo - RTOFS), which ignores the overall bias.

Warm rows, one profile per cast. Outputs: OHC/output/fronts_eddies_20261015/
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.build_rtofs_neighborhood_features_2024_2025 as nbhd  # noqa: E402
from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, load_work  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/fronts_eddies_20261015")
WIN = OUT / "rtofs_windows.npz"
HW = 12
SH = 6
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
VAR = {"tchp": "tchp_kj_per_cm2", "d26": "d26_m"}


# ------------------------------------------------------------------ sampling
def field_path(date: str) -> Path:
    return nbhd.RTOFS_DAILY_DIR[int(date[:4])] / f"rtofs_tchp_{date}.nc"


def sample_date(args):
    date, idx, y0, x0 = args
    off = np.arange(-HW, HW + 1)
    with xr.open_dataset(field_path(date)) as ds:
        arrs = {q: ds[v].values.astype(np.float32) for q, v in VAR.items()}
    ny, nx = arrs["tchp"].shape
    Y = y0[:, None] + off[None, :]
    okY = (Y >= 0) & (Y < ny)
    Yc = np.clip(Y, 0, ny - 1)
    X = (x0[:, None] + off[None, :]) % nx
    out = {}
    for q, a in arrs.items():
        w = a[Yc[:, :, None], X[:, None, :]]
        w[~okY] = np.nan
        out[q] = w.astype(np.float16)
    return idx, out


def build_windows() -> None:
    w = load_work("tchp")
    d = load_work("d26")
    rows = pd.concat([w[["cast_id", "ds", "lat", "lon"]], d[["cast_id", "ds", "lat", "lon"]]]).drop_duplicates("cast_id")
    rows = rows.reset_index(drop=True)
    tree, yi, xi, lat2d, lon2d = nbhd._build_grid_lookup(field_path(sorted(rows.ds)[0]))
    _, k = tree.query(latlon_to_xyz(rows.lat.to_numpy(float), rows.lon.to_numpy(float)).astype(np.float32))
    y0, x0 = yi[k].astype(np.int64), xi[k].astype(np.int64)
    north = np.nanmedian(lat2d[np.clip(y0 + 1, 0, lat2d.shape[0] - 1), x0] - lat2d[y0, x0]) > 0
    east = np.nanmedian(((lon2d[y0, (x0 + 1) % lon2d.shape[1]] - lon2d[y0, x0] + 180) % 360) - 180) > 0
    jobs = [(dt, np.asarray(ix), y0[ix], x0[ix]) for dt, ix in rows.groupby("ds").indices.items()]
    W = {q: np.full((len(rows), 2 * HW + 1, 2 * HW + 1), np.nan, np.float16) for q in VAR}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (ix, out) in enumerate(ex.map(sample_date, jobs, chunksize=4)):
            for q in VAR:
                W[q][ix] = out[q]
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} dates sampled", flush=True)
    np.savez_compressed(WIN, cast_id=rows.cast_id.to_numpy(str), tchp=W["tchp"], d26=W["d26"],
                        north_is_up=np.array(north), east_is_right=np.array(east))


# ------------------------------------------------------------------ analyses
def slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def anomaly_frame(q: str) -> pd.DataFrame:
    t = TARGET[q]
    w = load_work(q)
    err = w[t.delta_col].to_numpy(float)
    P = w[["lat", "lon", "abs_lat"]].to_numpy(float)
    pos = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(P, err).predict(P)
    x = w[f"model_{q}_anom_from_1deg_mean"].to_numpy(float)
    return pd.DataFrame({"cast_id": w.cast_id, "ds": w.ds, "x": x, "y": err - pos + x,
                         "grad": w[f"model_{q}_grad_mag_per_100km"].to_numpy(float),
                         "obs": w[t.obs_col].to_numpy(float)}).dropna(subset=["x", "y"])


def fig_scatter(frames) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    cm = LinearSegmentedColormap.from_list("b", ["#eef4fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    for ax, q in zip(axes, ("tchp", "d26")):
        f = frames[q]
        lo, hi = np.percentile(f.x, [0.5, 99.5])
        lim = max(abs(lo), abs(hi))
        ylim = np.percentile(np.abs(f.y), 99)
        hb = ax.hexbin(f.x, f.y, gridsize=70, extent=(-lim, lim, -ylim, ylim), bins="log", mincnt=1, cmap=cm,
                       linewidths=0)
        q20 = pd.qcut(f.x.clip(-lim, lim), 20, duplicates="drop")
        g = f.groupby(q20, observed=True)
        ax.plot(g.x.mean(), g.y.mean(), "o", color=ORANGE, ms=6, mec=SURF, label="average in 20 bins", zorder=4)
        s = slope(f.x.to_numpy(), f.y.to_numpy())
        xx = np.array([-lim, lim])
        ax.plot(xx, xx, "--", color=INK2, lw=1.3, label="slope 1 (RTOFS detail fully right)")
        ax.plot(xx, s * xx, "-", color=ORANGE, lw=2, label=f"fitted slope {s:.2f}")
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-ylim, ylim)
        r = np.corrcoef(f.x, f.y)[0, 1]
        ax.text(0.02, 0.03, f"spread of RTOFS departure (x): {f.x.std():.1f}\nspread of Argo departure (y): {f.y.std():.1f}\n"
                f"correlation: {r:.2f}\nslope = correlation x spread(y) / spread(x)\n      = {r:.2f} x {f.y.std():.1f} / "
                f"{f.x.std():.1f} = {r * f.y.std() / f.x.std():.2f}", transform=ax.transAxes, fontsize=10, color=INK2,
                va="bottom", bbox=dict(fc=SURF, ec=AXIS, boxstyle="round,pad=0.4"))
        ax.set_xlabel(f"RTOFS at the float minus RTOFS 1° average ({UNIT[q]})")
        ax.set_ylabel(f"Argo minus RTOFS 1° average ({UNIT[q]}), regional bias removed")
        ax.set_title(f"{q.upper()}: {len(f):,} profiles", loc="left")
        ax.legend(loc="upper left", fontsize=9.5)
        cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("profiles per cell (log)")
    headline(fig, "Argo's departure from the local average is larger than RTOFS's, but barely lines up with it",
             "Full scatter behind figure 3a. Argo scatters widely around RTOFS's 1° average (its own small-scale "
             "structure plus measurement and sampling noise),\nand that scatter is nearly unrelated to the structure "
             "RTOFS puts at the float. The low slope comes from the weak correlation, not from Argo being smooth.\n"
             "Features in slightly wrong places would do this, but so would small-scale structure RTOFS does not resolve; "
             "figures 2 and 3 test which.", top=0.82)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.09, wspace=0.35)
    fig.savefig(OUT / "1_anomaly_full_scatter.png", dpi=150)
    plt.close(fig)


def boot_slope(f, n=200, seed=0):
    rng = np.random.default_rng(seed)
    days = f.ds.unique()
    by = f.groupby("ds").indices
    vals = []
    for _ in range(n):
        pick = rng.choice(days, len(days), replace=True)
        ix = np.concatenate([by[d] for d in pick])
        vals.append(slope(f.x.to_numpy()[ix], f.y.to_numpy()[ix]))
    return np.percentile(vals, [2.5, 97.5])


def box_mean(W: np.ndarray, r: int) -> np.ndarray:
    c = HW
    b = W[:, c - r:c + r + 1, c - r:c + r + 1].astype(np.float32)
    n = np.isfinite(b).sum((1, 2))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(b, axis=(1, 2))
    return np.where(n >= 0.5 * (2 * r + 1) ** 2, m, np.nan)


def fig_gradient_smoothing(frames, Z) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    rows = []
    for j, q in enumerate(("tchp", "d26")):
        f = frames[q].copy()
        f["gq"] = pd.qcut(f.grad, 5, labels=False)
        ax = axes[0, j]
        meds, sl, lo, hi = [], [], [], []
        for k in range(5):
            s = f[f.gq == k]
            meds.append(s.grad.median())
            sl.append(slope(s.x.to_numpy(), s.y.to_numpy()))
            a, b = boot_slope(s, seed=k)
            lo.append(a)
            hi.append(b)
            rows.append({"target": q, "check": "slope by gradient quintile", "quintile": k + 1,
                         "median_gradient": meds[-1], "value": sl[-1], "lo95": a, "hi95": b, "profiles": len(s)})
        ax.errorbar(range(1, 6), sl, yerr=[np.array(sl) - lo, np.array(hi) - sl], fmt="-o", color=BLUE if q == "tchp"
                    else ORANGE, ms=8, mec=SURF, capsize=3, lw=1.8)
        ax.axhline(1, color=INK2, ls="--", lw=1)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"Q{k}\n{m:.0f}" for k, m in zip(range(1, 6), meds)])
        ax.set_xlabel(f"RTOFS gradient quintile (median, {UNIT[q]} per 100 km)   weak → steep")
        ax.set_ylabel("Slope: share of RTOFS's small-scale\ndetail that shows up in Argo")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{q.upper()}: slope by how steep the RTOFS field is at the float (95% intervals)", loc="left",
                     fontsize=12)
        ax.grid(axis="x", visible=False)
        # smoothing curves
        ax = axes[1, j]
        W = Z[q]
        idx = pd.Series(np.arange(len(Z["cast_id"])), index=Z["cast_id"])
        ii = idx.reindex(f.cast_id.to_numpy()).to_numpy()
        ok = ~np.isnan(ii)
        Wq = W[ii[ok].astype(int)]
        obs = f.obs.to_numpy()[ok]
        gq = f.gq.to_numpy()[ok]
        radii = list(range(0, HW + 1, 2))
        ests = {r: box_mean(Wq, r) for r in radii}
        good = np.all([np.isfinite(ests[r]) for r in radii], axis=0)
        cols = plt.cm.Blues(np.linspace(0.35, 0.95, 5)) if q == "tchp" else plt.cm.Oranges(np.linspace(0.35, 0.95, 5))
        for k in range(5):
            mk = good & (gq == k)
            base = np.std(obs[mk] - ests[0][mk])
            rel = [100 * (np.std(obs[mk] - ests[r][mk]) / base - 1) for r in radii]
            ax.plot([2 * r + 1 for r in radii], rel, "-o", color=cols[k], ms=5, mec=SURF, lw=1.8,
                    label=f"Q{k + 1}" + (" (weakest gradients)" if k == 0 else " (steepest)" if k == 4 else ""))
            for r, v in zip(radii, rel):
                rows.append({"target": q, "check": "smoothing box", "quintile": k + 1, "box_cells": 2 * r + 1,
                             "value": v, "profiles": int(mk.sum())})
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks([2 * r + 1 for r in radii])
        ax.set_xlabel("Box width (grid cells; 1 cell ≈ 6-9 km, 25 cells ≈ 1°)")
        ax.set_ylabel("Change in error spread vs the\nsingle nearest cell (%; down = closer to Argo)")
        ax.set_title(f"{q.upper()}: averaging RTOFS around the float, by gradient quintile", loc="left", fontsize=12)
        ax.legend(loc="upper right", fontsize=9, ncol=3)
        ax.grid(axis="x", visible=False)
    headline(fig, "Averaging RTOFS helps most at fronts, but the share of its detail that is real does not drop there",
             "Top: the slope from figure 1 for five groups of profiles, weakest to steepest RTOFS gradients (95% intervals "
             "from resampling whole days). It does not fall at fronts (for D26 it rises),\nso the low slope is mostly "
             "weak, noisy detail in calm water rather than misplaced fronts. Bottom: error spread (standard deviation of "
             "Argo − RTOFS) when RTOFS is averaged over a\nbox around the float. Averaging over ~13-17 cells helps at every "
             "gradient level and about twice as much at the steepest, where a small misplacement changes the value most.",
             top=0.86)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, hspace=0.38, wspace=0.2)
    fig.savefig(OUT / "2_slope_and_smoothing_by_gradient.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_shift(frames, Z) -> pd.DataFrame:
    north, east = bool(Z["north_is_up"]), bool(Z["east_is_right"])
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.5))
    rows = []
    c = HW
    offs = np.arange(-SH, SH + 1)
    for j, q in enumerate(("tchp", "d26")):
        f = frames[q].copy()
        f["gq"] = pd.qcut(f.grad, 5, labels=False)
        idx = pd.Series(np.arange(len(Z["cast_id"])), index=Z["cast_id"])
        ii = idx.reindex(f.cast_id.to_numpy()).to_numpy()
        ok = ~np.isnan(ii)
        Wq = Z[q][ii[ok].astype(int)][:, c - SH:c + SH + 1, c - SH:c + SH + 1].astype(np.float32)
        obs = f.obs.to_numpy()[ok]
        gq = f.gq.to_numpy()[ok]
        full = np.isfinite(Wq).all((1, 2))
        for i, (name, mk) in enumerate((("all profiles", full), ("steepest 20% of gradients", full & (gq == 4)))):
            ax = axes[i, j]
            E = np.std(obs[mk, None, None] - Wq[mk], axis=0)
            rel = 100 * (E / E[SH, SH] - 1)
            if not north:
                rel = rel[::-1]
            if not east:
                rel = rel[:, ::-1]
            vmax = np.percentile(np.abs(rel), 98)
            im = ax.imshow(rel, origin="lower", extent=(-SH - 0.5, SH + 0.5, -SH - 0.5, SH + 0.5), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            iy, ix = np.unravel_index(np.argmin(rel), rel.shape)
            ax.plot(offs[ix], offs[iy], "*", color=INK, ms=16, mec="white")
            ax.plot(0, 0, "+", color=INK, ms=14, mew=2)
            ax.set_xlabel("shift east (+) / west (−), grid cells")
            ax.set_ylabel("shift north (+) / south (−), grid cells")
            ring1 = rel[SH - 1:SH + 2, SH - 1:SH + 2]
            ax.set_title(f"{q.upper()}, {name} ({int(mk.sum()):,} profiles)\nbest shift ({offs[ix]:+d}, {offs[iy]:+d}) "
                         f"cells: {rel.min():+.1f}%;  one cell away: {np.delete(ring1.ravel(), 4).mean():+.1f}% on average",
                         loc="left", fontsize=11)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("error spread vs no shift (%)")
            rows.append({"target": q, "group": name, "best_shift_east": int(offs[ix]), "best_shift_north": int(offs[iy]),
                         "best_change_pct": float(rel.min()), "one_cell_mean_change_pct":
                         float(np.delete(ring1.ravel(), 4).mean()), "profiles": int(mk.sum())})
    headline(fig, "Shift test: no systematic offset; the best shift is at most 2 cells and gains under 0.4%",
             "Error spread (standard deviation of Argo − RTOFS) when the RTOFS value is taken up to 6 cells away "
             "(~40-55 km) from the float,\nrelative to the nearest cell (+). Star: the best shift. A systematic offset "
             "(e.g. eddies drifting at the wrong speed) would put a clear minimum\naway from the centre. Instead the bowl "
             "is centred, and steeper at fronts.", top=0.86)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.05, hspace=0.34, wspace=0.35)
    fig.savefig(OUT / "3_shift_test.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not WIN.exists():
        build_windows()
    Z = dict(np.load(WIN, allow_pickle=False))
    frames = {q: anomaly_frame(q) for q in ("tchp", "d26")}
    fig_scatter(frames)
    G = fig_gradient_smoothing(frames, Z)
    S = fig_shift(frames, Z)
    G.to_csv(OUT / "slope_and_smoothing.csv", index=False)
    S.to_csv(OUT / "shift_test.csv", index=False)
    print(S.to_string(index=False))
    print(G[G.check == "slope by gradient quintile"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
