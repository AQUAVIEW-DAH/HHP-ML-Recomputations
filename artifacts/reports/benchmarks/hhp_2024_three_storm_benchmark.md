# Multi-Storm HHP Benchmark

This report benchmarks simple correction baselines across the supplied storm
pair tables and then runs leave-one-storm-out tests to estimate event transfer.

## Combined grouped cross-validation

- Best model: `rf_delta`
- MAE: `12.899` kJ/cm²
- RMSE: `18.206` kJ/cm²
- R²: `0.845`
- Bias: `0.060` kJ/cm²

## Combined raw RTOFS baseline

- MAE: `17.918` kJ/cm²
- RMSE: `25.282` kJ/cm²
- R²: `0.701`
- Bias: `-11.046` kJ/cm²

## Combined improvement over raw RTOFS

- MAE improvement: `5.019` kJ/cm²
- RMSE improvement: `7.076` kJ/cm²

## Leave-One-Storm-Out

### Holdout: `BERYL_2024`
- Best model: `gbr_delta`
- Best MAE: `16.145` kJ/cm²
- Raw RTOFS MAE: `17.563` kJ/cm²
- MAE improvement: `1.418` kJ/cm²
- Best RMSE: `20.823` kJ/cm²
- Raw RTOFS RMSE: `25.481` kJ/cm²
- RMSE improvement: `4.657` kJ/cm²
### Holdout: `HELENE_2024`
- Best model: `ridge_delta`
- Best MAE: `18.594` kJ/cm²
- Raw RTOFS MAE: `20.695` kJ/cm²
- MAE improvement: `2.101` kJ/cm²
- Best RMSE: `23.309` kJ/cm²
- Raw RTOFS RMSE: `26.638` kJ/cm²
- RMSE improvement: `3.329` kJ/cm²
### Holdout: `MILTON_2024`
- Best model: `rf_delta`
- Best MAE: `15.567` kJ/cm²
- Raw RTOFS MAE: `18.402` kJ/cm²
- MAE improvement: `2.835` kJ/cm²
- Best RMSE: `19.800` kJ/cm²
- Raw RTOFS RMSE: `23.445` kJ/cm²
- RMSE improvement: `3.645` kJ/cm²

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
  "train_rows": 445,
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
