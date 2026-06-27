"""Correlation and redundancy diagnostics for global RTOFS-only physics features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:  # pragma: no cover - optional dependency
    variance_inflation_factor = None


DATA_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")

BASE_FEATURES = [
    "lat",
    "lon",
    "nearest_rtofs_grid_distance_km",
    "model_interp_tchp_kj_per_cm2",
    "model_interp_d26_m",
]

PHYSICS_FEATURES = [
    "model_surface_temp_c",
    "model_ssh_m",
    "model_mixed_layer_thickness_m",
    "model_surface_boundary_layer_thickness_m",
    "model_temp_excess_26c",
    "d26_minus_mlt_m",
    "d26_minus_sblt_m",
    "d26_to_mlt_ratio",
    "d26_to_sblt_ratio",
    "warm_layer_thickness_positive_m",
    "model_ssh_x_abs_lat",
    "model_mlt_x_abs_lat",
    "model_temp_excess_x_abs_lat",
]

TARGETS = {
    "tchp": "delta_tchp_kj_per_cm2",
    "d26": "delta_d26_m",
}


def _corr_table(df: pd.DataFrame, target_col: str, features: list[str]) -> pd.DataFrame:
    rows = []
    cols = features + [target_col]
    sub = df[cols].dropna().copy()
    for feature in features:
        pearson = sub[[feature, target_col]].corr(method="pearson", numeric_only=True).iloc[0, 1]
        spearman = sub[[feature, target_col]].corr(method="spearman", numeric_only=True).iloc[0, 1]
        rows.append(
            {
                "feature": feature,
                "rows": int(len(sub)),
                "pearson_r": float(pearson),
                "spearman_r": float(spearman),
                "abs_pearson_r": float(abs(pearson)),
                "abs_spearman_r": float(abs(spearman)),
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False)


def _vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sub = df[features].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature", "vif", "rows"])
    x = sub.to_numpy(dtype=float)
    rows = []
    if variance_inflation_factor is not None:
        for i, feature in enumerate(features):
            vif = variance_inflation_factor(x, i)
            rows.append({"feature": feature, "vif": float(vif), "rows": int(len(sub))})
        return pd.DataFrame(rows).sort_values("vif", ascending=False)

    for i, feature in enumerate(features):
        y = x[:, i]
        x_other = np.delete(x, i, axis=1)
        model = LinearRegression()
        model.fit(x_other, y)
        r2 = model.score(x_other, y)
        vif = np.inf if r2 >= 0.999999 else 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": feature, "vif": float(vif), "rows": int(len(sub))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).copy()
    df["abs_lat"] = np.abs(df["lat"].to_numpy(float))

    combined_features = BASE_FEATURES + ["abs_lat"] + PHYSICS_FEATURES
    feature_corr = df[combined_features].corr(numeric_only=True)
    feature_corr.to_csv(OUT_DIR / "rtofs_global_physics_feature_feature_corr.csv")

    summary = {
        "data_path": str(DATA_PATH),
        "rows_total": int(len(df)),
        "feature_feature_corr_csv": str(OUT_DIR / "rtofs_global_physics_feature_feature_corr.csv"),
        "targets": {},
    }

    for target_name, target_col in TARGETS.items():
        corr_df = _corr_table(df, target_col, combined_features)
        corr_path = OUT_DIR / f"rtofs_global_physics_corr_{target_name}.csv"
        corr_df.to_csv(corr_path, index=False)

        vif_df = _vif_table(df, combined_features)
        vif_path = OUT_DIR / f"rtofs_global_physics_vif_{target_name}.csv"
        vif_df.to_csv(vif_path, index=False)

        summary["targets"][target_name] = {
            "target_col": target_col,
            "corr_csv": str(corr_path),
            "vif_csv": str(vif_path),
            "top_corr_features": corr_df.head(10).to_dict(orient="records"),
            "top_vif_features": vif_df.head(10).to_dict(orient="records"),
        }

    out_json = OUT_DIR / "rtofs_global_physics_analysis_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
