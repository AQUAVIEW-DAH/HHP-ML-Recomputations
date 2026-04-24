"""Feature builders for HHP model training and inference."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "model_tchp_kj_cm2",
    "model_d26_m",
    "model_surface_t_c",
    "model_grid_distance_km",
    "lat",
    "lon",
    "day_sin",
    "day_cos",
]


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy of ``df`` with cyclic day-of-year features added."""
    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    dayofyear = ts.dt.dayofyear.astype(float)
    out["day_sin"] = np.sin(2.0 * np.pi * dayofyear / 366.0)
    out["day_cos"] = np.cos(2.0 * np.pi * dayofyear / 366.0)
    return out


def build_training_feature_frame(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from the Milton pairs table."""
    featured = add_time_features(pairs_df, "obs_time")
    return featured[FEATURE_COLUMNS].copy()


def build_inference_feature_row(
    *,
    query_time: str,
    lat: float,
    lon: float,
    model_tchp_kj_cm2: float | None,
    model_d26_m: float | None,
    model_surface_t_c: float | None,
    model_grid_distance_km: float,
) -> pd.DataFrame:
    """Build a one-row feature frame for query-time inference."""
    row = pd.DataFrame([{
        "obs_time": query_time,
        "lat": lat,
        "lon": lon,
        "model_tchp_kj_cm2": model_tchp_kj_cm2,
        "model_d26_m": model_d26_m,
        "model_surface_t_c": model_surface_t_c,
        "model_grid_distance_km": model_grid_distance_km,
    }])
    row = add_time_features(row, "obs_time")
    return row[FEATURE_COLUMNS].copy()


def safe_float(value: Any) -> float | None:
    """Convert scalar-like values to float, preserving missing values as None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
