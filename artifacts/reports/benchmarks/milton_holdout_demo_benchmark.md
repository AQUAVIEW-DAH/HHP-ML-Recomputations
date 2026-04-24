# Multi-Storm HHP Benchmark

This report benchmarks simple correction baselines across the supplied storm
pair tables and then runs leave-one-storm-out tests to estimate event transfer.

## Combined grouped cross-validation

- Best model: `rf_delta`
- MAE: `12.734` kJ/cm²
- RMSE: `18.550` kJ/cm²
- R²: `0.823`
- Bias: `-0.456` kJ/cm²

## Combined raw RTOFS baseline

- MAE: `17.835` kJ/cm²
- RMSE: `25.583` kJ/cm²
- R²: `0.663`
- Bias: `-12.001` kJ/cm²

## Combined improvement over raw RTOFS

- MAE improvement: `5.101` kJ/cm²
- RMSE improvement: `7.033` kJ/cm²

## Leave-One-Storm-Out

### Holdout: `BERYL_2024`
- Best model: `raw_rtofs`
- Best MAE: `17.563` kJ/cm²
- Raw RTOFS MAE: `17.563` kJ/cm²
- MAE improvement: `0.000` kJ/cm²
- Best RMSE: `25.481` kJ/cm²
- Raw RTOFS RMSE: `25.481` kJ/cm²
- RMSE improvement: `0.000` kJ/cm²
### Holdout: `HELENE_2024`
- Best model: `gbr_delta`
- Best MAE: `19.445` kJ/cm²
- Raw RTOFS MAE: `20.695` kJ/cm²
- MAE improvement: `1.250` kJ/cm²
- Best RMSE: `24.879` kJ/cm²
- Raw RTOFS RMSE: `26.638` kJ/cm²
- RMSE improvement: `1.759` kJ/cm²

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
  "train_rows": 380,
  "train_platforms": 182,
  "train_events": [
    "BERYL_2024",
    "HELENE_2024"
  ],
  "train_date_min": "2024-06-28T00:31:50Z",
  "train_date_max": "2024-09-28T20:16:43Z"
}
```
