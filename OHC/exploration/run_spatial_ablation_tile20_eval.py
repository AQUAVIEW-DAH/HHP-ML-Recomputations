"""20-degree-box holdout ablation of the coordinate features, all oceans.

Companion experiment to the presentation spatial-ablation figures: instead of
the locked date-forward split, group rows into 20-degree boxes over the whole
ocean and run GroupKFold(5) so every validation row sits in a box never seen
in training. Compares, per target:

- raw_rtofs
- best recipe without any coordinate features (no lat/lon/abs_lat/x|lat|)
- best recipe (with coordinate features)
- best recipe + Tier-0 neighborhood stencils

Run twice on two box grids (offset 0 and a 10-degree diagonal shift) to show
the conclusion does not depend on where the box edges fall.

Also renders, for the offset-0 grid, a per-box map of the OOF MAE difference
(no-coordinates minus with-coordinates), showing WHERE coordinate features
help or hurt when the box itself is held out.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.benchmark_rtofs_argo_tabular_models import TARGETS, _prepare_features  # noqa: E402
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402
from OHC.run_locked_xgb_physics_semi_ablation import (  # noqa: E402
    FEATURE_SETS_BY_TARGET,
    NEIGHBORHOOD_CORE,
    _make_preprocessor,
    _merge_feature_tables,
    _xgb_model,
)

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/spatial_tiers")
TILE_DEG = 20
N_SPLITS = 5
MAP_MIN_ROWS = 50
RECIPES = {
    "tchp": {
        "no_spatial": "global_pruned_no_spatial",
        "best": "global_pruned",
        "best_plus_neighborhood": "global_pruned_plus_neighborhood",
    },
    "d26": {
        "no_spatial": "drop_both_lat_interactions_no_spatial",
        "best": "drop_both_lat_interactions",
        "best_plus_neighborhood": "drop_both_lat_interactions_plus_neighborhood",
    },
}


def _tile_id(lat: np.ndarray, lon: np.ndarray, offset: float) -> pd.Series:
    lat_bin = np.floor((lat + 90.0 + offset) / TILE_DEG).astype(int)
    lon_bin = np.floor(((lon + 180.0 + offset) % 360.0) / TILE_DEG).astype(int)
    return pd.Series([f"t{a:02d}_{b:02d}" for a, b in zip(lat_bin, lon_bin)])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "rows": int(len(y_true)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))),
    }


def _tile_oof(work: pd.DataFrame, target, feature_cols: list[str], groups: np.ndarray) -> np.ndarray:
    oof = np.full(len(work), np.nan)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr_idx, va_idx in gkf.split(work, groups=groups):
        train_df, val_df = work.iloc[tr_idx], work.iloc[va_idx]
        pre = _make_preprocessor(feature_cols)
        X_tr = pre.fit_transform(train_df[feature_cols])
        X_va = pre.transform(val_df[feature_cols])
        model = _xgb_model()
        model.fit(X_tr, train_df[target.delta_col].to_numpy(float))
        oof[va_idx] = val_df[target.model_col].to_numpy(float) + model.predict(X_va)
    return oof


def _plot_tile_improvement_map(target, work: pd.DataFrame, oof_no_spatial: np.ndarray, oof_best: np.ndarray, out_path: Path) -> None:
    y = work[target.obs_col].to_numpy(float)
    err_ns = np.abs(oof_no_spatial - y)
    err_b = np.abs(oof_best - y)
    frame = pd.DataFrame({
        "tile": work["tile_group_off0"].to_numpy(),
        "lat0": (np.floor((work["lat"].to_numpy(float) + 90.0) / TILE_DEG) * TILE_DEG - 90.0),
        "lon0": (np.floor((work["lon"].to_numpy(float) + 180.0) / TILE_DEG) * TILE_DEG - 180.0),
        "err_ns": err_ns,
        "err_b": err_b,
    })
    agg = frame.groupby(["tile", "lat0", "lon0"]).agg(rows=("err_ns", "size"), mae_ns=("err_ns", "mean"), mae_b=("err_b", "mean")).reset_index()
    agg = agg[agg["rows"] >= MAP_MIN_ROWS].copy()
    agg["improvement"] = agg["mae_ns"] - agg["mae_b"]

    vlim = float(np.nanquantile(np.abs(agg["improvement"]), 0.95))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    add_land_overlay(ax, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 70)
    for _, row in agg.iterrows():
        ax.add_patch(Rectangle(
            (row["lon0"], row["lat0"]), TILE_DEG, TILE_DEG,
            facecolor=cmap(norm(row["improvement"])), edgecolor="white", linewidth=0.4, alpha=0.9, zorder=1,
        ))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"MAE(no coordinates) - MAE(best recipe) ({target.name}, {'kJ/cm²' if target.name == 'tchp' else 'm'})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{target.name.upper()}: where coordinate features help when the 20° box is HELD OUT\n"
        f"Red = coordinates still helped out-of-box, blue = coordinates hurt; boxes with ≥{MAP_MIN_ROWS} rows"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _merge_feature_tables()

    rows: list[dict] = []
    for target in TARGETS:
        work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
        work = _prepare_features(work).reset_index(drop=True)
        lat = work["lat"].to_numpy(float)
        lon = work["lon"].to_numpy(float)
        work["tile_group_off0"] = _tile_id(lat, lon, 0.0).to_numpy()
        work["tile_group_off10"] = _tile_id(lat, lon, 10.0).to_numpy()

        y_true = work[target.obs_col].to_numpy(float)
        keep_for_map: dict[str, np.ndarray] = {}
        for offset_name in ["off0", "off10"]:
            groups = work[f"tile_group_{offset_name}"].to_numpy()
            evaluation = f"tile{TILE_DEG}_{offset_name}_group_oof"
            rows.append({
                "target": target.name, "model": "raw_rtofs", "evaluation": evaluation,
                **_metrics(y_true, work[target.model_col].to_numpy(float)),
            })
            for variant, recipe in RECIPES[target.name].items():
                cols = FEATURE_SETS_BY_TARGET[target.name][recipe]
                oof = _tile_oof(work, target, cols, groups)
                rows.append({
                    "target": target.name, "model": variant, "evaluation": evaluation,
                    **_metrics(y_true, oof),
                })
                if offset_name == "off0" and variant in ("no_spatial", "best"):
                    keep_for_map[variant] = oof
            print(f"{target.name} {offset_name}: done ({work[f'tile_group_{offset_name}'].nunique()} boxes, {len(work)} rows)")

        map_path = OUT_DIR / f"{target.name}_tile20_coordinate_improvement_map.png"
        _plot_tile_improvement_map(target, work, keep_for_map["no_spatial"], keep_for_map["best"], map_path)

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "spatial_ablation_tile20_oof_summary.csv"
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
