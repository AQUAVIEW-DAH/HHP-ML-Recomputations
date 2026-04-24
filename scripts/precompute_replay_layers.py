#!/usr/bin/env python3
"""Precompute replay map layers to disk for fast frontend loads."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
from ml.paths import LAYERS_DIR
import asyncio

STRIDE = 10
LAYER_KINDS = ["tchp", "corrected_tchp", "correction", "observations"]


def _layer_path(date_yyyymmdd: str, layer: str, stride: int) -> Path:
    return LAYERS_DIR / f"{date_yyyymmdd}_{layer}_s{stride}.json"


async def main() -> None:
    cm = api.lifespan(api.app)
    await cm.__aenter__()
    try:
        for date_yyyymmdd in api.available_dates:
            iso_time = f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:]}T12:00:00Z"
            ds = api._get_rtofs_ds(date_yyyymmdd)
            for layer in LAYER_KINDS:
                path = _layer_path(date_yyyymmdd, layer, STRIDE)
                if layer == "observations":
                    day = api.pairs_df[api.pairs_df["obs_date"] == date_yyyymmdd]
                    payload = {
                        "layer": "observations",
                        "time": date_yyyymmdd,
                        "points": [{
                            "lat": round(float(r["lat"]), 4),
                            "lon": round(float(r["lon"]), 4),
                            "value": round(float(r["obs_tchp_kj_cm2"]), 2) if r["obs_tchp_kj_cm2"] == r["obs_tchp_kj_cm2"] else None,
                            "model_value": round(float(r["model_tchp_kj_cm2"]), 2) if r["model_tchp_kj_cm2"] == r["model_tchp_kj_cm2"] else None,
                            "cast_id": r["cast_id"],
                            "platform": str(r["platform"]),
                            "obs_time": r["obs_time"],
                        } for _, r in day.iterrows()],
                        "point_count": int(len(day)),
                    }
                else:
                    payload = api._build_layer(ds, layer, STRIDE, iso_time)
                path.write_text(json.dumps(payload), encoding="utf-8")
                print(f"wrote {path}")
    finally:
        await cm.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
