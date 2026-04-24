# Multi-Storm HHP Benchmark

This report benchmarks simple correction baselines across the supplied storm
pair tables and then runs leave-one-storm-out tests to estimate event transfer.

## Combined grouped cross-validation

- Best model: `gbr_delta`
- MAE: `17.465` kJ/cm²
- RMSE: `21.834` kJ/cm²
- R²: `0.822`
- Bias: `-0.413` kJ/cm²

## Combined raw RTOFS baseline

- MAE: `19.174` kJ/cm²
- RMSE: `24.567` kJ/cm²
- R²: `0.775`
- Bias: `-5.507` kJ/cm²

## Combined improvement over raw RTOFS

- MAE improvement: `1.709` kJ/cm²
- RMSE improvement: `2.733` kJ/cm²

## Leave-One-Storm-Out

### Holdout: `HELENE_2024`
- Best model: `gbr_delta`
- Best MAE: `17.966` kJ/cm²
- Raw RTOFS MAE: `20.695` kJ/cm²
- MAE improvement: `2.729` kJ/cm²
- Best RMSE: `23.712` kJ/cm²
- Raw RTOFS RMSE: `26.638` kJ/cm²
- RMSE improvement: `2.926` kJ/cm²
### Holdout: `MILTON_2024`
- Best model: `raw_rtofs`
- Best MAE: `18.402` kJ/cm²
- Raw RTOFS MAE: `18.402` kJ/cm²
- MAE improvement: `0.000` kJ/cm²
- Best RMSE: `23.445` kJ/cm²
- Raw RTOFS RMSE: `23.445` kJ/cm²
- RMSE improvement: `0.000` kJ/cm²

## Frozen artifact

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
  "train_rows": 98,
  "train_platforms": 37,
  "train_events": [
    "HELENE_2024",
    "MILTON_2024"
  ],
  "train_date_min": "2024-09-23T06:14:35Z",
  "train_date_max": "2024-10-11T21:47:12Z"
}
```
