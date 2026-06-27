"""Analyze locked XGBoost OOF results by year, season, and coarse basin."""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks")


def _rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((y_true - y_pred) ** 2))


def _assign_region(lat: float, lon: float) -> str:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    if lat < -40:
        return "southern_high_lat"
    if lat > 45:
        return "northern_high_lat"
    if -100 <= lon < 20:
        return "atlantic"
    if 20 <= lon < 147:
        return "indian"
    return "pacific"


def _season_from_month(month: int) -> str:
    if month in (1, 2, 3):
        return "winter_jfm"
    if month in (7, 8, 9):
        return "summer_jas"
    return "other"


def _metric_rows(df: pd.DataFrame, *, target: str, obs_col: str, pred_cols: list[str], group_col: str, group_name: str) -> list[dict]:
    rows: list[dict] = []
    for key, gdf in df.groupby(group_col):
        y_true = gdf[obs_col].to_numpy(float)
        for pred_col in pred_cols:
            y_pred = gdf[pred_col].to_numpy(float)
            model = pred_col.replace("pred_obs__", "")
            rows.append(
                {
                    "target": target,
                    "grouping": group_name,
                    "group": str(key),
                    "model": model,
                    "rows": int(len(gdf)),
                    "dates": int(gdf["date"].nunique()),
                    "mae": float(np.mean(np.abs(y_true - y_pred))),
                    "rmse": float(_rmse(y_true, y_pred)),
                    "bias": float(np.mean(y_pred - y_true)),
                    "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))) if len(gdf) > 1 else np.nan,
                }
            )
    return rows


def analyze_target(name: str, obs_col: str) -> pd.DataFrame:
    path = OUT_DIR / f"locked_xgb_oof_predictions_{name}.parquet"
    df = pd.read_parquet(path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_group"] = df["year"].astype(int).astype(str)
    df["month_int"] = df["month"].astype(int)
    df["season_group"] = df["month_int"].map(_season_from_month)
    df["region_group"] = [_assign_region(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]

    pred_cols = ["pred_obs__raw_rtofs", "pred_obs__xgb_delta_locked"]
    rows: list[dict] = []
    rows.extend(_metric_rows(df, target=name, obs_col=obs_col, pred_cols=pred_cols, group_col="year_group", group_name="year"))
    rows.extend(_metric_rows(df, target=name, obs_col=obs_col, pred_cols=pred_cols, group_col="season_group", group_name="season"))
    rows.extend(_metric_rows(df, target=name, obs_col=obs_col, pred_cols=pred_cols, group_col="region_group", group_name="region"))
    return pd.DataFrame(rows)


def main() -> None:
    tchp = analyze_target("tchp", "argo_tchp_kj_per_cm2")
    d26 = analyze_target("d26", "argo_d26_m")
    out = pd.concat([tchp, d26], ignore_index=True)
    out.to_csv(OUT_DIR / "locked_xgb_grouped_metrics.csv", index=False)

    delta = out.copy()
    pivot = (
        delta.pivot_table(
            index=["target", "grouping", "group", "rows", "dates"],
            columns="model",
            values=["mae", "rmse", "bias", "corr"],
        )
        .reset_index()
    )
    pivot.columns = ["__".join(col).strip("_") if isinstance(col, tuple) else col for col in pivot.columns.values]
    if "mae__raw_rtofs" in pivot.columns and "mae__xgb_delta_locked" in pivot.columns:
        pivot["mae_improvement_vs_raw"] = pivot["mae__raw_rtofs"] - pivot["mae__xgb_delta_locked"]
    if "rmse__raw_rtofs" in pivot.columns and "rmse__xgb_delta_locked" in pivot.columns:
        pivot["rmse_improvement_vs_raw"] = pivot["rmse__raw_rtofs"] - pivot["rmse__xgb_delta_locked"]
    pivot.to_csv(OUT_DIR / "locked_xgb_grouped_comparison.csv", index=False)

    print(pivot.sort_values(["target", "grouping", "group"]).to_string(index=False))


if __name__ == "__main__":
    main()
