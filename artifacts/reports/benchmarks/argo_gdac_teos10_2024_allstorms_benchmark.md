# Multi-Storm HHP Benchmark

This report benchmarks simple correction baselines across the supplied storm
pair tables and then runs leave-one-storm-out tests to estimate event transfer.

## Combined grouped cross-validation

- Best model: `rf_delta`
- MAE: `12.470` kJ/cm²
- RMSE: `17.414` kJ/cm²
- R²: `0.842`
- Bias: `-0.113` kJ/cm²

## Combined raw RTOFS baseline

- MAE: `16.932` kJ/cm²
- RMSE: `23.826` kJ/cm²
- R²: `0.704`
- Bias: `-9.984` kJ/cm²

## Combined improvement over raw RTOFS

- MAE improvement: `4.462` kJ/cm²
- RMSE improvement: `6.411` kJ/cm²

## Leave-One-Storm-Out

### Holdout: `BERYL_2024`
- Best model: `gbr_delta`
- Best MAE: `15.641` kJ/cm²
- Raw RTOFS MAE: `16.498` kJ/cm²
- MAE improvement: `0.858` kJ/cm²
- Best RMSE: `20.254` kJ/cm²
- Raw RTOFS RMSE: `23.956` kJ/cm²
- RMSE improvement: `3.702` kJ/cm²
### Holdout: `HELENE_2024`
- Best model: `rf_delta`
- Best MAE: `16.470` kJ/cm²
- Raw RTOFS MAE: `19.443` kJ/cm²
- MAE improvement: `2.973` kJ/cm²
- Best RMSE: `22.527` kJ/cm²
- Raw RTOFS RMSE: `25.018` kJ/cm²
- RMSE improvement: `2.491` kJ/cm²
### Holdout: `MILTON_2024`
- Best model: `rf_delta`
- Best MAE: `14.675` kJ/cm²
- Raw RTOFS MAE: `17.870` kJ/cm²
- MAE improvement: `3.195` kJ/cm²
- Best RMSE: `18.593` kJ/cm²
- Raw RTOFS RMSE: `22.606` kJ/cm²
- RMSE improvement: `4.014` kJ/cm²

## Frozen artifact

```json
{
  "model_name": "rf_delta",
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
  "train_rows": 452,
  "train_platforms": 182,
  "train_events": [
    "BERYL_2024",
    "HELENE_2024",
    "MILTON_2024"
  ],
  "train_date_min": "2024-06-28T00:31:50Z",
  "train_date_max": "2024-10-11T21:47:12Z"
}
```
