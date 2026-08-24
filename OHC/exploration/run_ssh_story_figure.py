"""The sea-surface-height story in one figure (closes the meeting-note thread).

Left panel: what the anomaly is — the January monthly-mean RTOFS SSH map from
the 701-day climatology (the reference the anomaly subtracts).
Right panels: locked out-of-fold MAE per target for four recipes:
  best (raw SSH)  |  + SSH monthly anomaly  |  + steric monthly anomaly  |  + both
The steric anomaly is built here: profile steric height (0/1000 dbar) minus a
5-degree x month binned mean (cells -> annual cell -> global month fallback).
Caveat printed on the figure: both climatologies average the full 2024-2025
period, the same convention as using WOA.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.titlesize": 16})
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET, FOLD_PATH, _make_preprocessor, _merge_feature_tables, _xgb_model,
)
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/expert_cross_eval_20260824")
CLIM_PATH = Path("/data/suramya/rtofs_ssh_climatology/monthly_mean_ssh.nc")
STERIC = "model_steric_0_1000_m"
BIN_DEG = 5.0


def _steric_anom(df: pd.DataFrame) -> np.ndarray:
    s = pd.to_numeric(df[STERIC], errors="coerce")
    lat_b = np.floor(df["lat"].to_numpy(float) / BIN_DEG) * BIN_DEG
    lon_b = np.floor(((df["lon"].to_numpy(float) + 180.0) % 360.0 - 180.0) / BIN_DEG) * BIN_DEG
    month = df["month"].astype(int).to_numpy()
    frame = pd.DataFrame({"s": s, "la": lat_b, "lo": lon_b, "m": month})
    cell_m = frame.groupby(["la", "lo", "m"])["s"].transform("mean")
    cell_y = frame.groupby(["la", "lo"])["s"].transform("mean")
    glob_m = frame.groupby(["m"])["s"].transform("mean")
    clim = cell_m.fillna(cell_y).fillna(glob_m)
    return (s - clim).to_numpy(float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    results = []
    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        work["steric_anom_m"] = _steric_anom(work)
        base_cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        variants = {
            "best recipe (raw SSH)": base_cols,
            "+ SSH monthly anomaly": base_cols + ["ssh_anom_monthly_m"],
            "+ steric monthly anomaly": base_cols + ["steric_anom_m"],
            "+ both anomalies": base_cols + ["ssh_anom_monthly_m", "steric_anom_m"],
        }
        y = work[target.obs_col].to_numpy(float)
        r = work[target.model_col].to_numpy(float)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        for label, cols in variants.items():
            oof = np.full(len(work), np.nan)
            for fold in folds:
                tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
                va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
                if not tr_mask.any() or not va_mask.any():
                    continue
                pre = _make_preprocessor(cols)
                m = _xgb_model()
                m.fit(pre.fit_transform(work.loc[tr_mask, cols]), work.loc[tr_mask, target.delta_col].to_numpy(float))
                va_idx = np.where(va_mask)[0]
                oof[va_idx] = r[va_idx] + m.predict(pre.transform(work.iloc[va_idx][cols]))
            ok = np.isfinite(oof)
            results.append({"target": tname, "variant": label,
                            "mae": float(np.abs(oof[ok] - y[ok]).mean()),
                            "bias": float((oof[ok] - y[ok]).mean())})
            print(results[-1])
    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "ssh_story_recipes.csv", index=False)

    fig = plt.figure(figsize=(22, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    axm = fig.add_subplot(gs[0])
    with xr.open_dataset(CLIM_PATH) as ds:
        jan = np.asarray(ds["mean_ssh"].isel(month=0).values)
        lat2d = np.asarray(ds["Latitude"].values)
        lon2d = np.asarray(ds["Longitude"].values)
    step = 4
    lon_n = ((lon2d[::step, ::step] + 180.0) % 360.0) - 180.0
    lat_s = lat2d[::step, ::step]
    jan_s = jan[::step, ::step]
    ok = np.isfinite(jan_s)
    pc = axm.scatter(lon_n[ok], lat_s[ok], c=jan_s[ok], s=1.2, cmap="RdBu_r",
                     vmin=-1.5, vmax=1.5, rasterized=True)
    axm.set_xlim(-180, 180)
    axm.set_ylim(-70, 70)
    axm.set_xlabel("Longitude")
    axm.set_ylabel("Latitude")
    axm.set_title("The reference being subtracted: January mean RTOFS SSH\n(from all 701 cached days; one of 12 monthly maps)")
    plt.colorbar(pc, ax=axm, shrink=0.85).set_label("mean SSH (m)")

    order = ["best recipe (raw SSH)", "+ SSH monthly anomaly", "+ steric monthly anomaly", "+ both anomalies"]
    colors = ["#94a3b8", "#2563eb", "#16a34a", "#7c3aed"]
    for gi, tname in enumerate(("tchp", "d26")):
        ax = fig.add_subplot(gs[gi + 1])
        sub = res[res.target == tname].set_index("variant").loc[order]
        bars = ax.bar(range(len(order)), sub["mae"], color=colors)
        for b, v in zip(bars, sub["mae"]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["raw SSH\n(best recipe)", "+ SSH\nanomaly", "+ steric\nanomaly", "+ both"], fontsize=10)
        ax.set_ylabel(f"locked OOF MAE ({UNITS[tname]})")
        ax.set_title(SHORT[tname])
        ax.set_ylim(bottom=float(sub["mae"].min()) * 0.985)
        ax.grid(True, axis="y", alpha=0.15)
    fig.suptitle('Sorting out sea surface height: "ssh" is the RTOFS model SSH; anomalies are deviations from (lat, lon, month) means\n'
                 "Raw SSH stays in every recipe (swapping it out hurts); anomalies are tested as additions. "
                 "Climatologies average the full 2024-2025 period, the same convention as a WOA background.",
                 fontsize=13)
    fig.savefig(OUT_DIR / "ssh_story.png", dpi=180)
    plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
