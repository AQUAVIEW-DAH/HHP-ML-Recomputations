# HHP / OHC Context

This file is the durable context note for the current Hurricane Heat Potential (HHP) and Ocean Heat Content (OHC) work in `OHC/`. It is meant to save future sessions from reconstructing the pipeline from scratch.

## 1. Main objective

The working goal is to correct RTOFS upper-ocean heat-content diagnostics toward Argo-derived values, with the correction framed as:

- a point-collocated residual-learning problem for development and evaluation,
- and eventually a corrected RTOFS field that can be rendered and compared seasonally.

The two primary scalar targets currently in play are:

- `TCHP` in `kJ/cm²`
- `D26` in `m`

where `D26` is the depth of the 26 C isotherm and `TCHP` is the TEOS-10-backed heat content above that isotherm.

## 2. Core physical and computational assumptions

### 2.1 TEOS-10 calculation basis

Core thermodynamic calculations live in:

- `OHC/teos_ohc.py`
- `OHC/validate_teos_ohc.py`

The workflow uses TEOS-10 / `gsw` for seawater conversions and property calculations.

### 2.2 Collocation philosophy

This project is still a point-collocation pipeline, not a common-grid training pipeline.

Important consequence:

- Argo and RTOFS are **not** first regridded to a common global grid for training.
- Instead, native-grid RTOFS values are sampled to the exact Argo point and date.
- ML targets are built only on those collocated rows.

The main collocation script is:

- `OHC/build_rtofs_at_argo_points_multiyear.py`

The key spatial sampling rule is:

- 8-neighbor inverse-distance-squared interpolation from the native RTOFS grid to the exact Argo coordinate.

This design choice also explains why many later diagnostics look sparse or point-like.

### 2.3 100 km mask meaning

The 100 km mask used in seasonal rendering is a **support/display mask**, not the interpolation rule itself.

It is used to suppress display where the nearest collocated support is too far away. It is not the same thing as:

- Gaussian interpolation length scale,
- RBF epsilon,
- or the exact collocation interpolation weights.

## 3. Data assets currently used

### 3.1 Argo

Argo extraction logic relevant to the HHP workflow lives in:

- `ml/sources/argo_gdac_source.py`

Notable details already added there:

- `profile_index` and `profile_key` support for distinguishing `N_PROF` entries inside a single file.
- pressure-variable selection that rejects implausible `PRES_ADJUSTED` values.
- clipping of tiny negative near-surface pressures to `0 dbar`, with larger negatives rejected.

### 3.2 RTOFS

Regional / sampled workflows and newer global-archive work now coexist.

Relevant files:

- `OHC/build_rtofs_daily_tchp_fields.py`
- `OHC/build_rtofs_global_daily_tchp_fields.py`
- `ml/sources/rtofs_global_source.py`
- `ml/paths.py`

The global-cache path now used by the archive helpers is:

- `RTOFS_GLOBAL_CACHE_DIR = /data/suramya/rtofs_global_cache`

## 4. Training/evaluation tables

Current main table family:

- `OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet`
- `OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet`
- `OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet`

Summary note:

- total collocated rows across 2024-2025: `41,347`
- collocated dates: `90`
- finite residual rows used for the current TCHP / D26 learning pool: `12,543`

The finite residual subset is smaller because TCHP and D26 are only defined where the profile crosses the `26 C` isotherm.

## 5. Why the maps concentrate in warm-water regions

This was explicitly checked and documented.

Important scientific interpretation:

- raw Argo profile coverage is global,
- but valid `D26` and `TCHP` rows only exist where the upper ocean crosses `26 C`.

So the equatorial / subtropical concentration is caused by the target definition, not by Argo only existing there.

This is reflected in the schematic report and associated scripts:

- `OHC/output/reports/hhp_interpolation_ml_schematic_report.pdf`
- `OHC/plot_argo_2024_raw_global_distribution.py`
- `OHC/build_rtofs_at_argo_points_surface_diff_2024.py`

## 6. ML pipeline state

### 6.1 Baseline feature set

The base tabular feature set used before the added physics families contains:

- time features: `year`, `month_int`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos`
- seasonal flags: `is_winter_jfm`, `is_summer_jas`, `is_other`
- geography: `lat`, `lon`, `abs_lat`
- collocation quality: `nearest_rtofs_grid_distance_km`
- raw model state at the collocated point: `model_interp_tchp_kj_per_cm2`, `model_interp_d26_m`

### 6.2 Added global physics features

These are sampled from the RTOFS side only.

Main files:

- `OHC/build_rtofs_global_physics_features_2024_2025.py`
- `OHC/analyze_rtofs_global_physics_features.py`

Main added global diagnostics:

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

Important note:

- `SSH`, `MLT`, and `SBLT` are not Argo-derived variables in this pipeline.
- They are sampled from the RTOFS diagnostic products and are used as additional model-side context.

### 6.3 Added profile physics features

These are built from the collocated RTOFS vertical profile columns.

Main files:

- `OHC/build_rtofs_profile_physics_features_2024_2025.py`
- `OHC/analyze_rtofs_profile_physics_features.py`

Profile-family additions include:

- steric height proxies
  - `model_steric_0_1000_m`
  - `model_steric_0_2000_m`
  - `model_steric_1000_ref2000_m`
- Brunt-Vaisala summaries
  - `model_n2_mean_upper200_s2`
  - `model_n2_max_upper200_s2`
  - `model_n2_mean_to_d26_s2`
  - `model_n2_max_to_d26_s2`

These are RTOFS-side features only; Argo is not used to compute input features for the model.

### 6.4 Named feature-set recipes used in ablations

The most important named recipes are:

- `base`
  - original time + geography + collocation-distance + raw model targets only
- `base_plus_global_physics`
  - base plus the full global physics family
- `global_pruned`
  - base plus the pruned global subset: `SSH`, `MLT`, `SBLT`, `temp-excess`, `D26-minus-MLT`, `D26-to-SBLT`, plus three latitude interactions
- `global_pruned_plus_profile_core`
  - `global_pruned` plus `steric_1000_ref2000`, `N2_max_upper200`, and `N2_mean_to_D26`
- `drop_temp_lat_interaction`
  - `global_pruned`-style set without `temp-excess x |lat|`, with the profile-core terms included
- `drop_ssh_lat_interaction`
  - `global_pruned`-style set without `SSH x |lat|`, with the profile-core terms included
- `drop_both_lat_interactions`
  - D26-oriented set removing both `SSH x |lat|` and `temp-excess x |lat|`, keeping `MLT x |lat|`, plus the profile-core terms
- `surface_temp_swap`
  - same role as `drop_temp_lat_interaction`, but swaps `temp-excess` for raw surface temperature

## 7. Current best model status

Two evaluation lenses are relevant:

### 7.1 Year-holdout (train 2024, test 2025)

Summary file:

- `OHC/output/ml_benchmarks/year_holdout_physics_semi_ablation_summary.csv`

Headline winners from the year-holdout semi-ablation pass:

- `TCHP`: `drop_temp_lat_interaction` with `MAE ≈ 11.50`
- `D26`: `drop_both_lat_interactions` with `MAE ≈ 11.28`

### 7.2 Locked blocked-forward protocol

Summary files:

- `OHC/output/ml_benchmarks/locked_protocol_note.md`
- `OHC/output/ml_benchmarks/locked_physics_semi_ablation_oof_summary.csv`

This is the stricter protocol used as the more robust decision point.

Current recommended feature recipes under the locked protocol:

- `TCHP`: `global_pruned`
  - `MAE = 11.64`
  - `RMSE = 16.38`
  - `Bias = -0.13`
  - `Corr = 0.946`
- `D26`: `drop_both_lat_interactions`
  - `MAE = 10.76`
  - `RMSE = 14.50`
  - `Bias = -0.41`
  - `Corr = 0.930`

Raw RTOFS baseline under the same locked evaluation:

- `TCHP`: `MAE = 16.26`, `RMSE = 22.64`
- `D26`: `MAE = 14.61`, `RMSE = 19.86`

So the corrected models are materially better than raw at collocated points.

## 8. Correlation / redundancy findings

Main report:

- `OHC/output/reports/hhp_feature_correlation_ablation_report.pdf`

Script:

- `OHC/build_hhp_feature_correlation_ablation_report.py`

Key redundancy pairs already identified include:

- `model_surface_temp_c` vs `model_temp_excess_26c` with `|r| ≈ 1.00`
- `d26_minus_mlt_m` vs `warm_layer_thickness_positive_m` with `|r| ≈ 0.996`
- `model_steric_0_1000_m` vs `model_steric_0_2000_m` with `|r| ≈ 0.974`
- strong interaction redundancies with `abs_lat`

The practical takeaway is that the added physics features help, but only after pruning correlated variants and not assuming every extra feature is useful.

## 9. Steric-height work status

Main files:

- `OHC/build_argo_steric_height_2024.py`
- `OHC/render_argo_woa_steric_validation.py`

Current implementation notes:

- steric-height-style variables are derived from TEOS-10 dynamic height divided by `g = 9.81 m s^-2`
- the project currently uses that meter-based conversion in code
- separate paper-style rendering converts to `dyn cm` for visual comparison with the reference paper style

Important output families:

- `OHC/output/steric_height/`
- `OHC/output/reports/`

The paper-style validator renders:

- Argo steric height mean panel
- Argo minus WOA proxy difference panel

## 10. Diagnostics and reports already added

### 10.1 Schematic report

- `OHC/output/reports/hhp_interpolation_ml_schematic_report.pdf`

This report documents:

- TEOS-10 calculation flow
- exact-location collocation
- 100 km mask meaning
- interpolation methods tested
- why warm-water regions dominate the final maps
- the basic ML workflow and year-holdout strategy

### 10.2 Feature correlation / semi-ablation report

- `OHC/output/reports/hhp_feature_correlation_ablation_report.pdf`

This report documents:

- what features were added
- which are TEOS-10 vs non-TEOS-10
- Pearson correlation / redundancy diagnostics
- semi-ablation results
- target-specific recommendations

### 10.3 Plot gallery for feature verification

- script: `OHC/build_hhp_feature_diagnostic_gallery.py`
- output root: `OHC/output/feature_diagnostics/hhp_feature_gallery_2024_2025/`

Purpose:

- skim-friendly validation of data presence, ranges, and finite-row support for the main feature families

### 10.4 Density scatter / patch / error diagnostics

- script: `OHC/build_hhp_density_scatter_diagnostics.py`
- output root: `OHC/output/density_scatter_diagnostics_2024_2025/`
- spec note: `OHC/output/density_scatter_diagnostics_2024_2025/hhp_density_scatter_plot_spec.md`

This bundle now includes:

- global observed-vs-model density scatters
- presentation-style single-panel log10(PDF) density plots
- 4 macro-region density scatters
- 20 degree patch support maps
- top-patch density scatters
- signed-error PDF and absolute-error CDF
- feature-binned MAE diagnostics

These diagnostics operate on the locked semi-ablation prediction tables:

- `TCHP corrected = pred_obs__global_pruned`
- `D26 corrected = pred_obs__drop_both_lat_interactions`

## 11. Environment and rerun notes

Use the project virtualenv, not system Python:

```bash
/home/suramya/HHP-Prediction/hhp-env/bin/python <script.py>
```

Examples:

```bash
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/run_locked_xgb_physics_semi_ablation.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_feature_correlation_ablation_report.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_density_scatter_diagnostics.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_feature_diagnostic_gallery.py
```

## 12. Known open threads

These are still live and should not be rediscovered from scratch next time:

1. Dr. Jacobs wants more diagnostic plotting, especially:
   - named 20 degree regional box breakdowns,
   - MLD-style log-PDF scatter plots,
   - error/feature relation plots,
   - and broad plot coverage so odd behavior can be traced backward through the pipeline.
2. The current density diagnostic bundle already adds the MLD-style presentation plots, but not yet the exact mentor-defined named 20 degree regional box selection.
3. Sparse deep-profile features can produce thin support in some diagnostics; this is especially relevant for the deeper steric and Brunt-Vaisala summaries.
4. The current ML correction is still evaluated only at collocated Argo-supported rows. That is the correct current development regime, but it is not yet a field-level validation of behavior far from floats.
