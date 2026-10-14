"""Dr. Jacobs' hand-picked feature set vs the full recipe (meeting of 2026-10-08).

His set ("one oceanographer's features that would be expected to be physically related"):
  lat, lon, (|lat|, optional: run with and without),
  cos/sin(2 pi doy) (annual) and cos/sin(4 pi doy) (semiannual), doy as a fraction of the year,
  RTOFS TCHP, D26, SSH, mixed-layer thickness and surface-boundary-layer thickness, each with its
  1-degree average.
and "use all data: if a profile (Argo or RTOFS) has no 26 C water, set TCHP and D26 to 0".

Data rule, applied the same way to both sides:
  * Argo: TCHP/D26 missing and the surface below 26 C -> 0. Missing with the surface at or above 26 C
    means the profile never cools to 26 C (D26 deeper than the profile): unknown, row dropped.
  * RTOFS: the daily TCHP and D26 fields are set to 0 in ocean cells whose top layer is below 26 C;
    cells warm all the way to the bottom stay unknown. The value at the float is then re-sampled from
    these fields with the usual 8-neighbour inverse-distance weighting, so cold neighbours count as 0
    (the old tables skipped them).
  * 1-degree average: mean of the valid cells in the same window as the existing 1-degree features
    (+/- 12 grid cells around the nearest cell, longitude wraps), required to be at least 25% valid.
    SSH, mixed layer and boundary layer come from rtofs_glo_2ds_f006_diag.nc of the same day.
One primary profile per Argo cast, as everywhere else.

Comparison: same rows, targets (error = Argo - RTOFS, with the zeros), XGBoost settings, three seeds,
on the locked forward test blocks and on full-year tests. The full recipe is the recommended one per
target (34 features for TCHP, 35 for D26), with RTOFS TCHP/D26 replaced by the zero-filled values and
the two features derived from D26 recomputed from them. Scores are reported on all rows, on rows where
either side has 26 C water, and on rows where both do (the population of all earlier results).

Outputs: OHC/output/jacobs_feature_set_20261013/
"""
from __future__ import annotations

import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import _prepare_features  # noqa: E402
from OHC.build_rtofs_at_argo_points_multiyear import (  # noqa: E402
    K_NEIGHBORS, _build_grid_lookup, _interpolate_neighbor_values)
from OHC.seasonal_map_common import latlon_to_xyz  # noqa: E402
from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AQUA, AXIS, BLUE, BLUE_LIGHT, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import _merged, forward_folds, loyo_folds  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/jacobs_feature_set_20261013")
FEAT = OUT / "jacobs_features.parquet"
FIELD_DIR = {2024: Path("/data/suramya/rtofs_global_ohc_fields_2024"), 2025: Path("/data/suramya/rtofs_global_ohc_fields_2025")}
DIAG = "/data/suramya/rtofs_global_cache/rtofs.{d}/rtofs_glo_2ds_f006_diag.nc"
ARGO_CACHE = {2024: "/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2020_2024",
              2025: "/data/suramya/argo_cache_hhp/global_argo_tchp_d26_2025"}
HW, MIN_FRAC, REF = 12, 0.25, 26.0
SEEDS = (0, 1, 2)
PHYS = ["tchp", "d26", "ssh", "mlt", "sblt"]
TARGETS = {"tchp": "kJ/cm²", "d26": "m"}


# ------------------------------------------------------------------ building the features
def box_mean(field: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mean of finite cells in the (2*HW+1)^2 window around (y, x); longitude wraps; NaN below MIN_FRAC valid."""
    ny, nx = field.shape
    ok = np.isfinite(field)
    v = np.pad(np.where(ok, field, 0.0).astype(np.float64), ((0, 0), (HW, HW)), mode="wrap")
    c = np.pad(ok.astype(np.float64), ((0, 0), (HW, HW)), mode="wrap")
    S = np.zeros((ny + 1, nx + 2 * HW + 1))
    C = np.zeros_like(S)
    S[1:, 1:] = v.cumsum(0).cumsum(1)
    C[1:, 1:] = c.cumsum(0).cumsum(1)
    y0, y1 = np.clip(y - HW, 0, ny), np.clip(y + HW + 1, 0, ny)
    x0, x1 = x, x + 2 * HW + 1                     # padded coordinates: column x sits at x + HW
    tot = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
    cnt = C[y1, x1] - C[y0, x1] - C[y1, x0] + C[y0, x0]
    area = (y1 - y0) * (2 * HW + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= MIN_FRAC * area, tot / cnt, np.nan)


def one_date(args) -> pd.DataFrame:
    date, rows, yk, xk, dk = args
    with xr.open_dataset(FIELD_DIR[int(date[:4])] / f"rtofs_tchp_{date}.nc") as ds:
        sst = ds["surface_temp_c"].values.astype(np.float64)
        tchp = ds["tchp_kj_per_cm2"].values.astype(np.float64)
        d26 = ds["d26_m"].values.astype(np.float64)
    cold = np.isfinite(sst) & (sst < REF)
    tchp = np.where(cold, 0.0, tchp)
    d26 = np.where(cold, 0.0, d26)
    with xr.open_dataset(DIAG.format(d=date)) as dg:
        ssh = dg["ssh"].isel(MT=0).values.astype(np.float64)
        mlt = dg["mixed_layer_thickness"].isel(MT=0).values.astype(np.float64)
        sblt = dg["surface_boundary_layer_thickness"].isel(MT=0).values.astype(np.float64)
    y0, x0 = yk[:, 0], xk[:, 0]
    out = pd.DataFrame({"cast_id": rows})
    out["rtofs0_tchp"] = _interpolate_neighbor_values(tchp.astype(np.float32), yk, xk, dk)
    out["rtofs0_d26"] = _interpolate_neighbor_values(d26.astype(np.float32), yk, xk, dk)
    for name, f in (("tchp", tchp), ("d26", d26), ("ssh", ssh), ("mlt", mlt), ("sblt", sblt)):
        out[f"{name}_1deg_avg"] = box_mean(f, y0, x0)
    return out


def argo_surface(p: pd.DataFrame) -> pd.Series:
    """Argo surface temperature per row (min and max over rows of the cache matching cast, date and position)."""
    parts = []
    for year, path in ARGO_CACHE.items():
        for f in sorted(glob.glob(f"{path}/*.parquet")):
            x = pd.read_parquet(f).reindex(columns=["cast_id", "date", "lat", "lon", "surface_t_c", "error"])
            x = x[x.error.isna() & x.surface_t_c.notna() & x.date.astype(str).str.startswith(str(year))]
            parts.append(x.drop(columns="error"))
    a = pd.concat(parts, ignore_index=True)
    key = lambda d: d.cast_id.astype(str) + "|" + d.date.astype(str) + "|" + d.lat.round(4).astype(str) + "|" + d.lon.round(4).astype(str)
    g = a.assign(k=key(a)).groupby("k").surface_t_c.agg(["min", "max"])
    m = g.reindex(key(p).to_numpy())
    return pd.DataFrame({"argo_sst_min": m["min"].to_numpy(), "argo_sst_max": m["max"].to_numpy()}, index=p.index)


def build_features() -> pd.DataFrame:
    p = _merged().reset_index(drop=True)
    p["date"] = p["date"].astype(str)
    sample = FIELD_DIR[2024] / f"rtofs_tchp_{sorted(p.date)[0]}.nc"
    tree, all_y, all_x = _build_grid_lookup(sample)
    chord, flat = tree.query(latlon_to_xyz(p.lat.to_numpy(float), p.lon.to_numpy(float)).astype(np.float32),
                             k=K_NEIGHBORS, workers=-1)
    yk, xk = all_y[flat], all_x[flat]
    dk = 6371.0 * 2.0 * np.arcsin(np.clip(chord.astype(np.float64) / 2.0, 0.0, 1.0))
    jobs = [(d, p.loc[idx, "cast_id"].to_numpy(), yk[idx], xk[idx], dk[idx]) for d, idx in p.groupby("date").indices.items()]
    stage = OUT / "rtofs_zero_filled_and_1deg.parquet"
    if stage.exists():
        F = pd.read_parquet(stage)
    else:
        with ProcessPoolExecutor(max_workers=8) as ex:
            parts = []
            for i, r in enumerate(ex.map(one_date, jobs, chunksize=4)):
                parts.append(r)
                if i % 100 == 0:
                    print(f"{i}/{len(jobs)} dates", flush=True)
        F = pd.concat(parts, ignore_index=True)
        F.to_parquet(stage, index=False)
    s = argo_surface(p)
    F = F.merge(pd.concat([p[["cast_id"]], s], axis=1), on="cast_id", how="left")
    F.to_parquet(FEAT, index=False)
    return F


# ------------------------------------------------------------------ the comparison
def load_rows() -> pd.DataFrame:
    p = _prepare_features(_merged().copy()).reset_index(drop=True)
    F = pd.read_parquet(FEAT)
    w = p.merge(F, on="cast_id", how="inner")
    w["ds"] = w["date"].dt.strftime("%Y%m%d")
    n0 = len(w)
    for q in ("tchp", "d26"):
        a = w[f"argo_tchp_kj_per_cm2" if q == "tchp" else "argo_d26_m"]
        cold = a.isna() & (w.argo_sst_max < REF)
        w[f"argo0_{q}"] = np.where(cold, 0.0, a)
    keep = w.argo0_tchp.notna() & w.argo0_d26.notna() & w.rtofs0_tchp.notna() & w.rtofs0_d26.notna()
    w = w[keep].reset_index(drop=True)
    print(f"rows: {n0:,} one-per-cast profiles, {len(w):,} kept ({n0 - len(w):,} with an unknown D26 or no RTOFS value)")
    f = (w["date"].dt.dayofyear - 0.5) / np.where(w["date"].dt.is_leap_year, 366.0, 365.0)
    for k, name in ((1, "annual"), (2, "semiannual")):
        w[f"cos_{name}"] = np.cos(2 * np.pi * k * f)
        w[f"sin_{name}"] = np.sin(2 * np.pi * k * f)
    w["rtofs0_ssh"], w["rtofs0_mlt"], w["rtofs0_sblt"] = w.model_ssh_m, w.model_mixed_layer_thickness_m, \
        w.model_surface_boundary_layer_thickness_m
    for q in ("tchp", "d26"):
        w[f"err_{q}"] = w[f"argo0_{q}"] - w[f"rtofs0_{q}"]
    w["warm_both"] = (w.argo0_tchp > 0) & (w.rtofs0_tchp > 0)
    w["warm_either"] = (w.argo0_tchp > 0) | (w.rtofs0_tchp > 0)
    return w


def feature_sets(tn: str, w: pd.DataFrame) -> dict[str, list[str]]:
    jac = ["lat", "lon", "cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"] + \
          [c for q in PHYS for c in (f"rtofs0_{q}", f"{q}_1deg_avg")]
    full = [c for c in abl.FEATURE_SETS_BY_TARGET[tn][RECIPE[tn]] if c in w.columns]
    return {f"Dr. Jacobs' set ({len(jac)})": jac,
            f"Dr. Jacobs' set + |lat| ({len(jac) + 1})": jac + ["abs_lat"],
            f"full recipe ({len(full)})": full}


def full_recipe_frame(w: pd.DataFrame) -> pd.DataFrame:
    """The full recipe on the same rows: RTOFS TCHP/D26 replaced by the zero-filled values."""
    x = w.copy()
    x["model_interp_tchp_kj_per_cm2"] = x.rtofs0_tchp
    x["model_interp_d26_m"] = x.rtofs0_d26
    x["d26_minus_mlt_m"] = x.rtofs0_d26 - x.model_mixed_layer_thickness_m
    x["d26_to_sblt_ratio"] = x.rtofs0_d26 / x.model_surface_boundary_layer_thickness_m
    return x


def run() -> pd.DataFrame:
    w = full_recipe_frame(load_rows())
    pops = {"all profiles": np.ones(len(w), bool), "26 °C water in Argo or RTOFS": w.warm_either.to_numpy(),
            "26 °C water in both (earlier results)": w.warm_both.to_numpy()}
    rows, gains = [], []
    for tn in ("tchp", "d26"):
        y = w[f"err_{tn}"].to_numpy(float)
        obs, mod = w[f"argo0_{tn}"].to_numpy(float), w[f"rtofs0_{tn}"].to_numpy(float)
        for design, folds in (("forward blocks", forward_folds(w)), ("full-year tests", loyo_folds(w))):
            masks = [(w.ds.isin(set(f["train_dates"])).to_numpy(), w.ds.isin(set(f["val_dates"])).to_numpy(), f["name"])
                     for f in folds]
            scored = np.zeros(len(w), bool)
            for _, va, _ in masks:
                scored |= va
            for pop, pm in pops.items():
                m = scored & pm
                rows.append({"target": tn, "design": design, "features": "raw RTOFS (no correction)", "population": pop,
                             "mae": float(np.abs(mod[m] - obs[m]).mean()), "seed_sd": 0.0, "n": int(m.sum()),
                             **{f"block: {nm}": float(np.abs(mod[va & pm] - obs[va & pm]).mean()) for _, va, nm in masks}})
            for fname, cols in feature_sets(tn, w).items():
                X = w[cols].apply(pd.to_numeric, errors="coerce")
                per = {pop: [] for pop in pops}
                blk = {(pop, nm): [] for pop in pops for _, _, nm in masks}
                for seed in SEEDS:
                    oof = np.full(len(w), np.nan)
                    for tr, va, nm in masks:
                        med = X[tr].median()
                        mdl = abl._xgb_model()
                        mdl.set_params(random_state=seed)
                        mdl.fit(X[tr].fillna(med).to_numpy(), y[tr])
                        oof[va] = mdl.predict(X[va].fillna(med).to_numpy())
                    e = np.abs(mod + oof - obs)
                    for pop, pm in pops.items():
                        per[pop].append(float(e[scored & pm].mean()))
                        for _, va, nm in masks:
                            blk[(pop, nm)].append(float(e[va & pm].mean()))
                for pop in pops:
                    rows.append({"target": tn, "design": design, "features": fname, "population": pop,
                                 "mae": float(np.mean(per[pop])), "seed_sd": float(np.std(per[pop])),
                                 "n": int((scored & pops[pop]).sum()),
                                 **{f"block: {nm}": float(np.mean(blk[(pop, nm)])) for _, _, nm in masks}})
                print(tn, design, fname, {k: round(np.mean(v), 3) for k, v in per.items()}, flush=True)
                if design == "forward blocks" and fname.startswith("Dr. Jacobs' set ("):
                    mdl = abl._xgb_model()
                    mdl.fit(X.fillna(X.median()).to_numpy(), y)
                    g = mdl.get_booster().get_score(importance_type="total_gain")
                    gains += [{"target": tn, "feature": cols[int(k[1:])], "total_gain": v} for k, v in g.items()]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "jacobs_vs_full_scores.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "jacobs_set_feature_gain.csv", index=False)
    return R


# ------------------------------------------------------------------ figures
ORDER = ["raw RTOFS (no correction)", "full recipe", "Dr. Jacobs' set + |lat|", "Dr. Jacobs' set ("]
COL = {"raw RTOFS (no correction)": AXIS, "full recipe": BLUE, "Dr. Jacobs' set + |lat|": AQUA, "Dr. Jacobs' set (": ORANGE}


def key_of(f: str) -> str:
    return next(k for k in ORDER if f.startswith(k))


def plot_scores(R: pd.DataFrame, design: str, fname: str) -> None:
    pops = list(dict.fromkeys(R.population))
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
    for i, tn in enumerate(("tchp", "d26")):
        for j, pop in enumerate(pops):
            ax = axes[i, j]
            t = R[(R.target == tn) & (R.design == design) & (R.population == pop)].copy()
            t["k"] = t.features.map(key_of)
            t = t.set_index("k").loc[ORDER]
            yy = np.arange(len(t))[::-1]
            ax.barh(yy, t.mae, color=[COL[k] for k in t.index], edgecolor=SURF, lw=2, height=0.7)
            for y_, (k, r) in zip(yy, t.iterrows()):
                ax.text(r.mae, y_, f" {r.mae:.2f}", va="center", fontsize=10.5, color=INK2)
            ax.set_yticks(yy)
            ax.set_yticklabels([f.replace("Dr. Jacobs' set", "Jacobs set") for f in t.features] if j == 0 else [])
            ax.set_xlim(0, t.mae.max() * 1.22)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(f"{pop}\n({int(t.n.iloc[0]):,} profiles scored)", loc="left", fontsize=12)
            if i == 1:
                ax.set_xlabel(f"Mean absolute error (lower = better)")
            if j == 0:
                ax.text(-0.66, 0.5, f"{tn.upper()}\n({TARGETS[tn]})", transform=ax.transAxes, ha="center",
                        va="center", fontsize=12.5, fontweight="bold")
    s = R[(R.design == design) & (R.population == pops[2])].copy()
    s["k"] = s.features.map(key_of)
    s = s.set_index(["target", "k"]).mae
    J, F, RAW = "Dr. Jacobs' set (", "full recipe", "raw RTOFS (no correction)"
    v = {(tn, k): s[(tn, k)] for tn in ("tchp", "d26") for k in (J, F, RAW)}
    share = {tn: 100 * (v[(tn, RAW)] - v[(tn, J)]) / (v[(tn, RAW)] - v[(tn, F)]) for tn in ("tchp", "d26")}
    sub = (f"{design.capitalize()}; same rows, model and settings, 3 runs each. 'All data': profiles without 26 °C water "
           f"count as TCHP = D26 = 0. On the rows of all earlier results (right column):\nraw RTOFS {v[('tchp', RAW)]:.2f} "
           f"(TCHP) and {v[('d26', RAW)]:.2f} (D26), full recipe {v[('tchp', F)]:.2f} and {v[('d26', F)]:.2f}, Jacobs set "
           f"{v[('tchp', J)]:.2f} and {v[('d26', J)]:.2f}. With half the features his set gets {share['tchp']:.0f}% (TCHP) "
           f"and {share['d26']:.0f}% (D26) of the full recipe's improvement.")
    headline(fig, "Dr. Jacobs' 16 features come close to the full recipe, which stays slightly better overall",
             sub, top=0.82)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.08, hspace=0.18, wspace=0.12)
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_blocks(R: pd.DataFrame) -> None:
    """Jacobs set minus full recipe, per test block: consistent or not?"""
    pops = list(dict.fromkeys(R.population))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True)
    for ax, tn in zip(axes, ("tchp", "d26")):
        rows, labels = [], []
        for design in ("forward blocks", "full-year tests"):
            t = R[(R.target == tn) & (R.design == design)].copy()
            t["k"] = t.features.map(key_of)
            bcols = [c for c in t.columns if c.startswith("block: ") and t[c].notna().any()]
            for c in bcols + ["mae"]:
                rows.append((design, c))
                labels.append(("ALL " + design) if c == "mae" else c.replace("block: ", ""))
        yy = np.arange(len(rows))[::-1]
        ax.axvline(0, color=AXIS, lw=1.2)
        for k, (pop, col, mk) in enumerate(zip(pops, (MUTED, AQUA, ORANGE), ("s", "D", "o"))):
            for y_, (design, c) in zip(yy, rows):
                t = R[(R.target == tn) & (R.design == design) & (R.population == pop)].copy()
                t["k"] = t.features.map(key_of)
                t = t.set_index("k")
                d = t.loc["Dr. Jacobs' set (", c] - t.loc["full recipe", c]
                ax.plot(d, y_ + (1 - k) * 0.22, mk, color=col, ms=8, mec=SURF, label=pop if y_ == yy[0] else None)
        for y_, (design, c) in zip(yy, rows):
            if c == "mae":
                ax.axhspan(y_ - 0.45, y_ + 0.45, color=NEUTRAL, zorder=0)
        ax.set_yticks(yy)
        ax.set_yticklabels(labels)
        ax.grid(axis="y", visible=False)
        ax.set_title(tn.upper(), loc="left")
        ax.set_xlabel(f"Jacobs set minus full recipe ({TARGETS[tn]})\n← Jacobs set better  |  full recipe better →")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0))
    headline(fig, "The full recipe is better in 27 of 30 test-block comparisons; all 3 exceptions are TCHP block 3",
             "Each row is one test block; shaded rows are overall scores. Left of zero means Dr. Jacobs' 16 features "
             "did better than the full recipe on that block. The exceptions are all TCHP\nforward block 3 (Aug-Dec 2025), "
             "where his set is slightly better.", top=0.83)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, wspace=0.08)
    fig.savefig(OUT / "3_difference_by_test_block.png", dpi=150)
    plt.close(fig)


def plot_gain(G: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, tn, col in zip(axes, ("tchp", "d26"), (BLUE, ORANGE)):
        g = G[G.target == tn].set_index("feature").total_gain
        g = (100 * g / g.sum()).sort_values()
        ax.barh(np.arange(len(g)), g.values, color=col, edgecolor=SURF, lw=2, height=0.72)
        for i, v in enumerate(g.values):
            ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=10, color=INK2)
        ax.set_yticks(np.arange(len(g)))
        nice = {"sin_annual": "sin(2π·day of year)", "cos_annual": "cos(2π·day of year)",
                "sin_semiannual": "sin(4π·day of year)", "cos_semiannual": "cos(4π·day of year)"}
        ax.set_yticklabels([nice.get(s, s.replace("rtofs0_", "RTOFS ").replace("_1deg_avg", " 1° average")
                                     .replace("mlt", "mixed layer").replace("sblt", "boundary layer")
                                     .replace("tchp", "TCHP").replace("d26", "D26").replace("ssh", "SSH"))
                            for s in g.index])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, g.max() * 1.2)
        ax.set_title(f"{tn.upper()} model", loc="left")
        ax.set_xlabel("Share of the model's total split gain (%)")
    headline(fig, "With all data, the 1° average of D26 carries most of the model, then latitude",
             "Share of all split gain carried by each feature, in a model fitted on all rows with his set (no |lat|). "
             "With cold water set to 0,\nthe 1° average of D26 also says how much warm water "
             "surrounds the float, i.e. how close it is to the 26 °C edge. The semiannual terms and the boundary layer are "
             "hardly used.", top=0.84)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.09, wspace=0.55)
    fig.savefig(OUT / "4_jacobs_set_feature_use.png", dpi=150)
    plt.close(fig)


def plots() -> None:
    R = pd.read_csv(OUT / "jacobs_vs_full_scores.csv")
    plot_scores(R, "forward blocks", "1_scores_forward_blocks.png")
    plot_scores(R, "full-year tests", "2_scores_full_year_tests.png")
    plot_blocks(R)
    plot_gain(pd.read_csv(OUT / "jacobs_set_feature_gain.csv"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot" not in sys.argv:
        if not FEAT.exists():
            build_features()
        run()
    plots()
