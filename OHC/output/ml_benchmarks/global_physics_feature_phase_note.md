# Global Physics Feature Phase

## What was added

We enriched the global `2024+2025` collocation table with additional **RTOFS-only**
features available from the cached daily global fields and 2D diagnostic files:

- `model_surface_temp_c`
- `model_ssh_m`
- `model_mixed_layer_thickness_m`
- `model_surface_boundary_layer_thickness_m`
- `model_temp_excess_26c`
- `d26_minus_mlt_m`
- `d26_minus_sblt_m`
- `d26_to_mlt_ratio`
- `d26_to_sblt_ratio`
- `warm_layer_thickness_positive_m`
- interaction terms with `abs_lat`

Source builder:

- `/home/suramya/HHP-Prediction/OHC/build_rtofs_global_physics_features_2024_2025.py`

Output table:

- `/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet`

## Why this phase was done first

These features are globally available now and remain fully model-only, so they
do not let the correction model see Argo-derived physics.

This gives us an immediate way to test whether additional upper-ocean structure
helps the correction task before attempting the heavier full-profile
stratification extension.

## Key modeling result

Year-holdout benchmark: **train on 2024, test on 2025**

Best of the tested global feature sets was the **pruned physics set**.

### TCHP

- raw RTOFS MAE: `16.169`
- base XGBoost MAE: `11.813`
- base + pruned global physics MAE: `11.559`

### D26

- raw RTOFS MAE: `14.823`
- base XGBoost MAE: `11.546`
- base + pruned global physics MAE: `11.363`

Summary file:

- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/year_holdout_global_physics_summary.csv`

## Most useful new features

From gain-based importance in the pruned model, the strongest added physics
signals were:

- `model_ssh_m`
- `model_ssh_x_abs_lat`
- `model_temp_excess_26c`
- `model_temp_excess_x_abs_lat`
- `model_mixed_layer_thickness_m`
- `d26_minus_mlt_m`

Importance file:

- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/year_holdout_global_physics_importance_all.csv`

## Redundant features identified

The redundancy screen showed that several derived warm-layer terms are mostly
reparameterizations of existing `D26` and surface-temperature information.

Especially redundant:

- `model_surface_temp_c` vs `model_temp_excess_26c`
- `d26_minus_mlt_m` vs `warm_layer_thickness_positive_m`
- `d26_minus_sblt_m` vs `d26_minus_mlt_m`

Correlation / redundancy artifacts:

- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/rtofs_global_physics_feature_feature_corr.csv`
- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/rtofs_global_physics_corr_tchp.csv`
- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/rtofs_global_physics_corr_d26.csv`
- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/rtofs_global_physics_vif_tchp.csv`
- `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/rtofs_global_physics_vif_d26.csv`

## Remaining blocker for full stratification / steric features

The next scientifically richer phase would compute **full-profile**
TEOS-10 features such as:

- buoyancy frequency (`N²`) summaries
- steric-height-style column summaries
- density-gradient and thermocline-strength summaries

However, this requires the raw **global 3D RTOFS archive files**
(`rtofs_glo.t00z.f06.archv.a`) for the collocated dates.

Those files were not retained locally after the global daily TCHP/D26 fields
were generated, and each compressed archive is on the order of **5.7 GB/date**.

So the full-profile phase is feasible, but it should be done only after we
either:

1. retain the global 3D archives during field generation, or
2. intentionally redownload the archive set for the dates we want to enrich.

## Recommended next step

Use the **pruned global physics feature set** as the new working benchmark, and
then plan a separate full-profile feature stage focused on retained global 3D
archives for:

- `N²` summaries
- steric-height proxy
- density-jump metrics around the thermocline / `D26`
