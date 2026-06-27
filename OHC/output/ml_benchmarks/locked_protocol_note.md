# Locked ML Evaluation Protocol

This note freezes the first official evaluation protocol for the `RTOFS -> Argo`
correction project.

## Goal

Evaluate correction models only at collocated Argo-supported points, using a
leakage-aware time split that approximates forward use.

## Data scope

- Training table:
  - `/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet`
- Years included:
  - `2024`
  - `2025`
- Total collocated rows:
  - `41,347`
- Total collocated dates:
  - `90`

Finite target rows:

- `delta_tchp_kj_per_cm2`
  - `12,543`
- `delta_d26_m`
  - `12,543`

## Eligibility rules

For each target, a row is eligible only if all of the following are finite:

- Argo target
- interpolated RTOFS target
- residual target (`Argo - RTOFS`)

So the actual train/eval pool is target-specific and based on finite residuals.

## Features frozen for this phase

- `year`
- `month_int`
- `lat`
- `lon`
- `abs_lat`
- `nearest_rtofs_grid_distance_km`
- `month_sin`
- `month_cos`
- `doy_sin`
- `doy_cos`
- `is_winter_jfm`
- `is_summer_jas`
- `is_other`
- `model_interp_tchp_kj_per_cm2`
- `model_interp_d26_m`

## Target definition

Two separate residual-learning targets:

1. `delta_tchp_kj_per_cm2`
2. `delta_d26_m`

Predicted corrected value:

`corrected = raw_rtofs + predicted_delta`

## Split protocol

- Split unit:
  - `date`
- Split type:
  - forward-chaining blocked validation
- Random row split:
  - explicitly forbidden
- Embargo:
  - `1` date between train and validation
- Number of folds:
  - `3`

Exact folds:

### Fold 1

- Train:
  - `20240131 -> 20240710`
- Embargo:
  - `1` date
- Validation:
  - `20240715 -> 20240909`

### Fold 2

- Train:
  - `20240131 -> 20240909`
- Embargo:
  - `1` date
- Validation:
  - `20240914 -> 20250226`

### Fold 3

- Train:
  - `20240131 -> 20250226`
- Embargo:
  - `1` date
- Validation:
  - `20250514 -> 20250831`

## Why this protocol was frozen

This was chosen to reduce the main leakage risks:

- same-day spatial clustering
- nearby profiles from the same ocean state appearing in both train and validation
- artificial optimism from random row splits

It also preserves the most realistic sequence:
- earlier dates predict later dates

## Important caveats

- We are still evaluating only where Argo exists.
- This is a collocated-point correction benchmark, not yet a gridded-field benchmark.
- `other` months are currently retained because they are part of the real 2025 overlap table and help preserve time diversity.
- This is not yet a final untouched holdout design; it is the locked benchmark protocol for current model-family comparison.

## Locked rerun scope

For the official rerun after freezing this note:

- baseline:
  - `raw_rtofs`
- main locked candidate:
  - `xgb_delta`

Why `xgb_delta`:

- It was the strongest standalone model in the exploratory run.
- The nonlinear ensemble slightly edged it for `TCHP`, but the ensemble is a second-order fusion product, not a single base model.
- Freezing the best standalone model first gives us a cleaner reference point.
