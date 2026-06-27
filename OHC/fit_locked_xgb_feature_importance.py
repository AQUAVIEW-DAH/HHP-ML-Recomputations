"""Fit locked XGBoost models on all eligible rows and export feature importance."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBRegressor

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.benchmark_rtofs_argo_tabular_models import (  # noqa: E402
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGETS,
    _prepare_features,
    _tree_preprocessor,
)


OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")
FIG_DIR = OUT_DIR / "figures"


def _xgb_model():
    return XGBRegressor(
        random_state=0,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=8,
    )


def _fit_target(target) -> dict:
    df = pd.read_parquet(DATA_PATH)
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = _prepare_features(work)

    pre = _tree_preprocessor()
    x = pre.fit_transform(work[FEATURE_COLUMNS])
    y = work[target.delta_col].to_numpy(float)

    model = _xgb_model()
    model.fit(x, y)

    booster = model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    score_weight = booster.get_score(importance_type="weight")
    rows = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        key = f"num__x{idx}" if f"num__x{idx}" in score_gain or f"num__x{idx}" in score_weight else f"f{idx}"
        rows.append(
            {
                "target": target.name,
                "feature": feature,
                "gain": float(score_gain.get(key, 0.0)),
                "weight": float(score_weight.get(key, 0.0)),
            }
        )
    imp_df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df.to_csv(OUT_DIR / f"locked_xgb_feature_importance_{target.name}.csv", index=False)

    top = imp_df.head(15).sort_values("gain")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["gain"])
    plt.title(f"Locked XGB feature importance ({target.name})")
    plt.xlabel("Gain")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"locked_xgb_feature_importance_{target.name}.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    result = {
        "target": target.name,
        "rows": int(len(work)),
        "feature_importance_csv": str(OUT_DIR / f"locked_xgb_feature_importance_{target.name}.csv"),
        "feature_importance_png": str(fig_path),
        "top_features": imp_df.head(10).to_dict(orient="records"),
    }

    if shap is not None:
        sample_n = min(2000, x.shape[0])
        sample_idx = np.linspace(0, x.shape[0] - 1, sample_n, dtype=int)
        x_sample = x[sample_idx]
        booster = model.get_booster()
        shap_values = booster.predict(DMatrix(x_sample, feature_names=[f"f{i}" for i in range(x_sample.shape[1])]), pred_contribs=True)
        shap_values = shap_values[:, :-1]
        shap_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": shap_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
        shap_csv = OUT_DIR / f"locked_xgb_shap_importance_{target.name}.csv"
        shap_df.to_csv(shap_csv, index=False)
        try:
            shap.summary_plot(shap_values, features=x_sample, feature_names=FEATURE_COLUMNS, show=False, max_display=15)
            shap_fig = FIG_DIR / f"locked_xgb_shap_summary_{target.name}.png"
            plt.tight_layout()
            plt.savefig(shap_fig, dpi=180, bbox_inches="tight")
            plt.close()
            result["shap_summary_png"] = str(shap_fig)
        except Exception:
            plt.close("all")
        result["shap_importance_csv"] = str(shap_csv)
        result["top_shap_features"] = shap_df.head(10).to_dict(orient="records")

    return result


def main() -> None:
    results = [_fit_target(target) for target in TARGETS]
    out = {"results": results, "shap_available": shap is not None}
    (OUT_DIR / "locked_xgb_feature_importance_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
