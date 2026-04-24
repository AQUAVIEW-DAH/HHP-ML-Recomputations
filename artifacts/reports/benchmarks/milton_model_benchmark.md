# Milton HHP Baseline Benchmark

This benchmark compares simple grouped-cross-validation baselines on the Milton
2024 Argo/RTOFS pair table. Groups are Argo `platform` IDs so the model does
not train and test on the same float.

## Best simple model

- Model: `gbr_delta`
- MAE: `15.292` kJ/cm²
- RMSE: `19.687` kJ/cm²
- R²: `0.852`
- Bias: `-0.905` kJ/cm²

## Raw RTOFS baseline

- MAE: `18.402` kJ/cm²
- RMSE: `23.445` kJ/cm²
- R²: `0.791`
- Bias: `-5.458` kJ/cm²

## Improvement over raw RTOFS

- MAE improvement: `3.110` kJ/cm²
- RMSE improvement: `3.758` kJ/cm²

## Frozen prototype artifact

```json
{
  "model_name": "gbr_delta",
  "target_name": "tchp_delta_kj_cm2",
  "feature_columns": [
    "model_tchp_kj_cm2",
    "model_d26_m",
    "model_surface_t_c",
    "model_grid_distance_km",
    "lat",
    "lon",
    "day_sin",
    "day_cos"
  ],
  "train_rows": 65,
  "train_platforms": 35,
  "train_event": "MILTON_2024",
  "train_date_min": "2024-09-29T01:22:48Z",
  "train_date_max": "2024-10-11T21:47:12Z"
}
```
