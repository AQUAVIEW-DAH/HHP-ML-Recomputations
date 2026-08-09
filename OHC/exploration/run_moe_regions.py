"""Mixture of experts over regions and regimes (meeting item C1).

Variants, all on the locked blocked-forward folds, full global table,
best+neighborhood recipe, evaluated out-of-fold on the observed scale:

- global            one model for everything (baseline; matches the production recipe)
- moe_region        five geographic experts (Gulf of Mexico, Atlantic, Indian,
                    West Pacific, East/Central Pacific+rest), hard-gated by
                    position. Each expert trains on ALL rows with sample
                    weights: own-region rows 1.0, elsewhere GLOBAL_PRIOR_W —
                    a local specialist with a global prior.
- moe_regime        K learned regimes: k-means (fit on training folds only) in
                    standardised physics-state space (no lat/lon), one expert
                    per regime with the same prior weighting, soft-gated at
                    prediction time by softmax over centroid distances, so the
                    gate is defined everywhere including float deserts.

Reported globally, per named 20-degree box, and for the Gulf specifically
(where the hand-built local model set the target to beat).
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
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    FOLD_PATH,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)
from OHC.build_hhp_density_scatter_diagnostics import NAMED_BOXES, _augment_regions_and_patches, _named_box_rows  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import RECIPE, SHORT, UNITS, _in_gom  # noqa: E402

RUN_DATE = "2026-08-11"
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/moe_regions_{RUN_DATE.replace('-', '')}")
GLOBAL_PRIOR_W = 0.15
N_REGIMES = 6
GATE_TEMPERATURE = 1.0  # in units of the median centroid distance
REGIME_FEATURES = [
    "model_ssh_m",
    "model_temp_excess_26c",
    "model_mixed_layer_thickness_m",
    "model_tchp_local_std_1deg",
    "model_tchp_anom_from_1deg_mean",
    "abs_lat",
]


def _region_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.empty(len(lat), dtype=object)
    gom = (lat >= 18) & (lat <= 31) & (lon >= -98) & (lon <= -80)
    atlantic = (~gom) & (lon >= -100) & (lon < 20)
    indian = (lon >= 20) & (lon < 100)
    wpac = (lon >= 100) & (lon < 160)
    out[:] = "epac_other"
    out[wpac] = "west_pacific"
    out[indian] = "indian"
    out[atlantic] = "atlantic"
    out[gom] = "gulf_of_mexico"
    return out


def _fit_expert(train_df, cols, delta_col, weights):
    pre = _make_preprocessor(cols)
    X = pre.fit_transform(train_df[cols])
    model = _xgb_model()
    model.fit(X, train_df[delta_col].to_numpy(float), sample_weight=weights)
    return pre, model


def _predict(pre, model, df, cols):
    return model.predict(pre.transform(df[cols]))


def _metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    e = p[ok] - y[ok]
    return {"rows": int(ok.sum()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()
    fold_note = json.loads(FOLD_PATH.read_text())
    all_rows = []
    regime_map_saved = False

    for target in TARGETS:
        tname = target.name
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        cols = [c for c in FEATURE_SETS_BY_TARGET[tname][RECIPE[tname]] if c in work.columns]
        lat = work["lat"].to_numpy(float)
        lon = work["lon"].to_numpy(float)
        work["region"] = _region_of(lat, lon)
        y = work[target.obs_col].to_numpy(float)
        r = work[target.model_col].to_numpy(float)
        date_str = work["date"].dt.strftime("%Y%m%d")
        folds = _build_forward_folds(sorted(date_str.unique().tolist()),
                                     n_folds=fold_note["n_folds"], embargo_dates=fold_note["embargo_dates"])

        oof = {"global": np.full(len(work), np.nan),
               "moe_region": np.full(len(work), np.nan),
               "moe_regime": np.full(len(work), np.nan)}
        regime_label_oof = np.full(len(work), -1)

        for fold in folds:
            tr_mask = date_str.isin(set(fold["train_dates"])).to_numpy()
            va_mask = date_str.isin(set(fold["val_dates"])).to_numpy()
            if not tr_mask.any() or not va_mask.any():
                continue
            train_df = work[tr_mask]
            val_df = work[va_mask]
            va_idx = np.where(va_mask)[0]

            # --- global baseline
            pre_g, model_g = _fit_expert(train_df, cols, target.delta_col, None)
            oof["global"][va_idx] = r[va_idx] + _predict(pre_g, model_g, val_df, cols)

            # --- geographic experts with global prior
            region_pred = np.full(len(val_df), np.nan)
            for region in pd.unique(work["region"]):
                w = np.where(train_df["region"].to_numpy() == region, 1.0, GLOBAL_PRIOR_W)
                pre_e, model_e = _fit_expert(train_df, cols, target.delta_col, w)
                sel = val_df["region"].to_numpy() == region
                if sel.any():
                    region_pred[sel] = _predict(pre_e, model_e, val_df[sel], cols)
            oof["moe_region"][va_idx] = r[va_idx] + region_pred

            # --- learned regimes with soft distance gate
            imput = train_df[REGIME_FEATURES].apply(pd.to_numeric, errors="coerce")
            med = imput.median()
            scaler = StandardScaler().fit(imput.fillna(med))
            Z_tr = scaler.transform(imput.fillna(med))
            km = KMeans(n_clusters=N_REGIMES, n_init=8, random_state=0).fit(Z_tr)
            lab_tr = km.labels_

            Z_va = scaler.transform(val_df[REGIME_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(med))
            d_va = np.linalg.norm(Z_va[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
            T = GATE_TEMPERATURE * np.median(d_va)
            gate = np.exp(-d_va / max(T, 1e-9))
            gate = gate / gate.sum(axis=1, keepdims=True)
            regime_label_oof[va_idx] = np.argmin(d_va, axis=1)

            regime_preds = np.zeros((len(val_df), N_REGIMES))
            for k in range(N_REGIMES):
                w = np.where(lab_tr == k, 1.0, GLOBAL_PRIOR_W)
                pre_k, model_k = _fit_expert(train_df, cols, target.delta_col, w)
                regime_preds[:, k] = _predict(pre_k, model_k, val_df, cols)
            oof["moe_regime"][va_idx] = r[va_idx] + (gate * regime_preds).sum(axis=1)

        # ---- metrics: global, per named box, Gulf
        aug = _augment_regions_and_patches(work.assign(date=work["date"].dt.strftime("%Y-%m-%d")))
        for variant, pred in [("raw_rtofs", r)] + list(oof.items()):
            all_rows.append({"target": tname, "variant": variant, "scope": "global", "subset": "all", **_metrics(y, pred)})
            gm = _in_gom(work).to_numpy()
            all_rows.append({"target": tname, "variant": variant, "scope": "region", "subset": "gulf_of_mexico_true", **_metrics(y[gm], pred[gm])})
            for box in NAMED_BOXES:
                sel = ((aug["patch_lat0"] == box.lat0) & (aug["patch_lon0"] == box.lon0)).to_numpy()
                all_rows.append({"target": tname, "variant": variant, "scope": "named_box", "subset": box.key, **_metrics(y[sel], pred[sel])})
        print(f"{tname}: global MAE — global {_metrics(y, oof['global'])['mae']:.3f}, "
              f"moe_region {_metrics(y, oof['moe_region'])['mae']:.3f}, "
              f"moe_regime {_metrics(y, oof['moe_regime'])['mae']:.3f}")

        # ---- regime interpretability map (once; regimes are target-independent enough)
        if not regime_map_saved:
            fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
            keep = regime_label_oof >= 0
            sc = ax.scatter(lon[keep], lat[keep], c=regime_label_oof[keep], s=2.5, cmap="tab10", vmin=-0.5, vmax=9.5, rasterized=True)
            add_land_overlay(ax, zorder=2)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-70, 70)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(
                f"Learned regimes (k-means, K={N_REGIMES}, physics-state space, no lat/lon input)\n"
                "If clusters form coherent geography, the ocean's regimes are recoverable from state alone"
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, ticks=range(N_REGIMES))
            cbar.set_label("regime id")
            fig.savefig(OUT_DIR / "learned_regime_map.png", dpi=180)
            plt.close(fig)
            regime_map_saved = True

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_DIR / "moe_summary.csv", index=False)

    # ---- comparison figure: per named box MAE, global vs the two MoE variants
    fig, axes = plt.subplots(2, 1, figsize=(16, 13), constrained_layout=True)
    width = 0.26
    for ax, tname in zip(axes, ("tchp", "d26")):
        sub = res[(res.target == tname) & (res.scope == "named_box")]
        x = np.arange(len(NAMED_BOXES))
        for i, (variant, color, label) in enumerate([
            ("global", "#94a3b8", "global model"),
            ("moe_region", "#2563eb", "MoE: geographic experts"),
            ("moe_regime", "#16a34a", "MoE: learned regimes"),
        ]):
            vals = [float(sub[(sub.variant == variant) & (sub.subset == b.key)]["mae"].iloc[0]) for b in NAMED_BOXES]
            ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([b.display for b in NAMED_BOXES], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(f"MAE ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: out-of-fold MAE per named 20° box")
        ax.grid(True, axis="y", alpha=0.15)
    axes[0].legend()
    fig.suptitle("Mixture of experts vs the single global model")
    fig.savefig(OUT_DIR / "moe_named_box_comparison.png", dpi=180)
    plt.close(fig)

    print(res[res.scope != "named_box"].to_string(index=False))


if __name__ == "__main__":
    main()
