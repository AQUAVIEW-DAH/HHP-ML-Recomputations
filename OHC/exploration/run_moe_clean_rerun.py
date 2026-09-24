"""Re-run the recommended MoE on the rebuilt, deduplicated tables, and map its errors.

The MoE's published scores (TCHP 11.19, D26 10.55) were computed on tables
that counted 12% of Argo casts two or more times. After the 2026-09-24
rebuild the population changed, so those numbers can no longer be compared
with anything new. This re-establishes the benchmark on the clean population
(warm rows, one primary profile per cast): raw RTOFS, the single global model
and the MoE blend, all fitted fresh on identical rows and identical folds.

It also draws the maps requested at the 2026-09-24 meeting: where the error
sits geographically, before correction (raw RTOFS) and after (MoE), as
profile points and as a smooth interpolated field.

Outputs: OHC/output/moe_clean_20260925/
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
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE  # noqa: E402
from OHC.exploration.run_moe_regions import _region_of  # noqa: E402
from OHC.exploration.run_moe_v2_tuning import _run_geographic, _run_regime  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay, build_global_grid, gaussian_interpolate  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/moe_clean_20260925")
WINNERS = {"tchp": {"alpha": 0.75, "k": 6, "w": 0.05}, "d26": {"alpha": 0.50, "k": 12, "w": 0.05}}
UNIT = {"tchp": "kJ/cm²", "d26": "m"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = abl._merge_feature_tables()
    df = df[df["is_primary_profile"].astype(bool)].reset_index(drop=True)
    fold_note = json.loads(abl.FOLD_PATH.read_text())
    scores = []
    for target in TARGETS:
        tn, cfg = target.name, WINNERS[target.name]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["region"] = _region_of(work["lat"].to_numpy(float), work["lon"].to_numpy(float))
        cols = [c for c in abl.FEATURE_SETS_BY_TARGET[tn][RECIPE[tn]] if c in work.columns]
        ds = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(ds.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        y = work[target.obs_col].to_numpy(float)
        r = work[target.model_col].to_numpy(float)

        # single global model, fresh on the same rows
        glob = np.full(len(work), np.nan)
        for f in folds:
            tr, va = ds.isin(set(f["train_dates"])).to_numpy(), ds.isin(set(f["val_dates"])).to_numpy()
            imp = SimpleImputer(strategy="median").fit(work.loc[tr, cols])
            m = abl._xgb_model()
            m.fit(imp.transform(work.loc[tr, cols]), work.loc[tr, target.delta_col].to_numpy(float))
            glob[va] = r[va] + m.predict(imp.transform(work.loc[va, cols]))
        print(tn, "global done", flush=True)
        geo = _run_geographic(work, target, cols, folds, cfg["w"])
        reg = _run_regime(work, target, cols, folds, cfg["w"], cfg["k"])
        moe = cfg["alpha"] * geo + (1 - cfg["alpha"]) * reg
        print(tn, "MoE done", flush=True)

        v = np.isfinite(moe) & np.isfinite(glob)
        out = work.loc[v, ["date", "lat", "lon", "platform", "cast_id", target.obs_col, target.model_col]].copy()
        out["date"] = out["date"].dt.strftime("%Y%m%d")
        out["pred_obs__raw_rtofs"] = r[v]
        out["pred_obs__global_best"] = glob[v]
        out["pred_obs__moe_blend"] = moe[v]
        out.to_parquet(OUT / f"moe_predictions_{tn}.parquet", index=False)
        gm = ((work["lat"] >= 18) & (work["lat"] <= 31) & (work["lon"] >= -98) & (work["lon"] <= -80)).to_numpy()[v]
        for name, p in (("raw RTOFS", r[v]), ("single global model", glob[v]), ("MoE blend (recommended)", moe[v])):
            e = p - y[v]
            scores.append({"target": tn, "model": name, "mae": float(np.abs(e).mean()),
                           "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                           "mae_gulf": float(np.abs(e[gm]).mean()), "rows": int(v.sum())})

        # ---- error maps: before (raw) and after (MoE), points and interpolated
        e_raw, e_moe = r[v] - y[v], moe[v] - y[v]
        lat, lon = work.loc[v, "lat"].to_numpy(float), work.loc[v, "lon"].to_numpy(float)
        vmax = float(np.nanpercentile(np.abs(e_raw), 97))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        glat, glon = build_global_grid(0.5)
        fig, axes = plt.subplots(2, 2, figsize=(20, 10.5), constrained_layout=True)
        for j, (lab, e) in enumerate((("raw RTOFS", e_raw), ("MoE blend (recommended)", e_moe))):
            ax = axes[0, j]
            ax.scatter(lon, lat, c=e, s=3, cmap="RdBu_r", norm=norm, linewidths=0, zorder=2)
            ax.set_title(f"{lab}: error at each Argo profile (MAE {np.abs(e).mean():.2f}, bias {e.mean():+.2f})", fontsize=12)
            ax = axes[1, j]
            grid = gaussian_interpolate(lat, lon, e, glat, glon, length_scale_km=250.0,
                                        truncation_radius_km=600.0, mask_distance_km=200.0)
            pm = ax.pcolormesh(glon, glat, np.ma.masked_invalid(grid), shading="auto", cmap="RdBu_r", norm=norm, zorder=1)
            ax.set_title(f"{lab}: interpolated (Gaussian, 250 km scale, blank beyond 200 km of a float)", fontsize=12)
            for ax in (axes[0, j], axes[1, j]):
                add_land_overlay(ax, zorder=5)
                ax.set_xlim(-180, 180); ax.set_ylim(-45, 50)
                ax.grid(alpha=0.15)
        fig.colorbar(pm, ax=axes, shrink=0.6, pad=0.01).set_label(f"model minus Argo ({UNIT[tn]}); blue = model too low")
        fig.suptitle(f"{tn.upper()}: where the error is, before and after the correction\n"
                     f"out-of-fold predictions, Sep 2024 - Dec 2025, one profile per Argo cast, {int(v.sum()):,} profiles",
                     fontsize=14)
        fig.savefig(OUT / f"{tn}_error_maps_before_after.png", dpi=140)
        plt.close(fig)
        print(tn, "maps done", flush=True)

    S = pd.DataFrame(scores)
    S.to_csv(OUT / "clean_benchmark.csv", index=False)
    print(S.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
