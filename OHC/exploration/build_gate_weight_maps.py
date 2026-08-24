"""Gate-weight maps for the recommended MoE (meeting item, 2026-08).

For the learned-regime branch (winning K per target), computes the soft gate
out-of-fold for every collocated point and maps:
- the dominant regime (which expert owns each location), and
- the dominant expert's gate weight (how decisively it owns it).

No expert training needed: only clustering and distances.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _build_forward_folds, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import FOLD_PATH, _merge_feature_tables  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import SHORT  # noqa: E402
from OHC.exploration.run_moe_regions import GATE_TEMPERATURE, REGIME_FEATURES  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/gate_weight_maps_20260824")
K_WINNER = {"tchp": 6, "d26": 12}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    for target in TARGETS:
        tname = target.name
        k = K_WINNER[tname]
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])
        dominant = np.full(len(work), -1)
        top_w = np.full(len(work), np.nan)
        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            imput = work.loc[tr_mask, REGIME_FEATURES].apply(pd.to_numeric, errors="coerce")
            med = imput.median()
            scaler = StandardScaler().fit(imput.fillna(med))
            km = KMeans(n_clusters=k, n_init=8, random_state=0).fit(scaler.transform(imput.fillna(med)))
            Z_va = scaler.transform(work.loc[va_mask, REGIME_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(med))
            d = np.linalg.norm(Z_va[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
            T = GATE_TEMPERATURE * np.median(d)
            g = np.exp(-d / max(T, 1e-9))
            g = g / g.sum(axis=1, keepdims=True)
            va_idx = np.where(va_mask)[0]
            dominant[va_idx] = np.argmax(g, axis=1)
            top_w[va_idx] = np.max(g, axis=1)

        keep = dominant >= 0
        lon = work["lon"].to_numpy(float)
        lat = work["lat"].to_numpy(float)
        fig, axes = plt.subplots(2, 1, figsize=(16, 13.5), constrained_layout=True)
        sc = axes[0].scatter(lon[keep], lat[keep], c=dominant[keep], s=2.5,
                             cmap="tab20" if k > 10 else "tab10", vmin=-0.5, vmax=(19.5 if k > 10 else 9.5),
                             rasterized=True)
        plt.colorbar(sc, ax=axes[0], shrink=0.85, ticks=range(k)).set_label("dominant regime expert")
        axes[0].set_title(f"{SHORT[tname]} (K={k}): which regime expert owns each location")
        sc = axes[1].scatter(lon[keep], lat[keep], c=top_w[keep], s=2.5, cmap="magma",
                             vmin=1.0 / k, vmax=float(np.nanquantile(top_w[keep], 0.98)), rasterized=True)
        plt.colorbar(sc, ax=axes[1], shrink=0.85).set_label("dominant expert's gate weight")
        axes[1].set_title(f"How decisively it owns it (uniform gate would give {1.0 / k:.2f})")
        for ax in axes:
            add_land_overlay(ax, zorder=2)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-70, 70)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        fig.suptitle(f"{SHORT[tname]}: the learned-regime gate, out-of-fold (state-space clustering, no coordinates)")
        fig.savefig(OUT_DIR / f"{tname}_gate_weight_maps.png", dpi=180)
        plt.close(fig)
        print(f"{tname}: K={k}, mapped {int(keep.sum())} rows, median top weight {np.nanmedian(top_w[keep]):.2f}")


if __name__ == "__main__":
    main()
