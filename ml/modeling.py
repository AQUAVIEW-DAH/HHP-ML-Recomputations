"""Model artifact helpers for HHP correction models."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from ml.paths import MODELS_DIR

DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "HHP_MODEL_PATH",
        str(MODELS_DIR / "argo_gdac_teos10_2024_allstorms_rf_delta_model.pkl"),
    )
)


def save_model_bundle(bundle: dict[str, Any], path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_model_bundle(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)
