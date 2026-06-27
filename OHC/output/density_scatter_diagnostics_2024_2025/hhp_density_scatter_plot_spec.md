# HHP Density-Scatter Diagnostic Spec

This note documents the first-pass HHP diagnostics built in
`build_hhp_density_scatter_diagnostics.py`.

## Goal

Translate the current point-collocated HHP workflow into plots that are easy to
compare with the MLD-style 2-D PDF scatter figures requested by Dr. Jacobs,
while also adding region, patch, and feature-regime context.

## Inputs

- Locked semi-ablation OOF prediction table for TCHP:
  `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet`
- Locked semi-ablation OOF prediction table for D26:
  `/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet`
- Merged base/global/profile collocation features:
  `/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet`, `/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet`, `/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet`

## Corrected model used

- TCHP corrected model: `global_pruned`
- D26 corrected model: `drop_both_lat_interactions`

These are used because they are the current best locked-protocol candidates from
the semi-ablation pass.

## Plot families

1. Global density-scatter:
   observed vs raw RTOFS and observed vs corrected.
2. Presentation density-scatter:
   single-panel log10(PDF) figures in the MLD-style visual family, generated
   separately for raw and corrected predictions.
3. Macro-region density-scatter:
   same observed-vs-model diagnostic across 4 globe partitions.
4. 20-degree patch support + top-patch density-scatter:
   use 20° x 20° boxes and select the best-supported patch inside each region.
5. Error distributions:
   signed-error PDF and absolute-error CDF.
6. Feature relations:
   binned MAE curves vs selected physics features.

## Macro-region definition

- `high_latitude`: `|lat| > 45`
- `atlantic`: `-100 <= lon < 20` and not high latitude
- `indian`: `20 <= lon < 147` and not high latitude
- `pacific`: remaining longitudes and not high latitude

## 20-degree patch definition

- latitude start: `floor(lat / 20) * 20`
- longitude start: `floor(lon / 20) * 20`
- patch label example:
  `lat[0,20) lon[140,160)`

## Density-scatter style

- 2-D histogram on observed/model axes
- color shows `log10(PDF)`
- empty bins shown in white
- 1:1 line overlaid
- subplot title includes:
  `RMSE`, `MAE`, `Bias`, `Corr`, `Slope`, `n`

## Feature-relation style

For selected features, rows are divided into quantile bins. Within each bin the
script plots:

- raw mean absolute error
- corrected mean absolute error
- number of rows in the bin

This was chosen over raw scatter because it is easier to interpret when asking
whether error systematically changes with a physics regime.

## Output root

`/home/suramya/HHP-Prediction/OHC/output/density_scatter_diagnostics_2024_2025`
