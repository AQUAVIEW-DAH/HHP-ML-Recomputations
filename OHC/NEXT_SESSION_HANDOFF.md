# Next Session Handoff

Date: 2026-07-04 (previous: 2026-06-27)

## 0. Update 2026-07-04

### 0.1 Named 20 degree regional boxes now exist

`OHC/build_hhp_density_scatter_diagnostics.py` now renders the mentor-requested
fixed named 20 degree boxes in addition to the automatic top-supported patches:

- output family: `OHC/output/density_scatter_diagnostics_2024_2025/named_box_density/`
- per-target box location map + 8-box raw/corrected density scatters
- metrics rows with `scope = named_box` in `tables/density_scatter_metrics.csv`

The 8 fixed boxes (selected by pure box geometry, independent of the
macro-region longitude splits): Gulf of Mexico & NW Caribbean, Atlantic
hurricane MDR / E Caribbean, Arabian Sea, Bay of Bengal, Philippine Sea /
W Pacific warm pool, Coral Sea, Central equatorial N Pacific, SW Pacific
(Fiji sector).

The refreshed bundle was synced to Drive as `HHP-scatter-plots-2026-07-04`.

### 0.2 RTOFS reduced-field backfill restarted (was stalled)

The 2026-06-29 backfill died at `rtofs_tchp_20240409.nc` (113 of ~340 files).
It was relaunched on 2026-07-04 as a detached chained run (2024 remaining
dates, then all 336 missing 2025 dates):

- wrapper: `/data/suramya/rtofs_backfill_20260704/run_backfill_chain.sh`
- date lists: `/data/suramya/rtofs_backfill_20260704/remaining_{2024,2025}_dates.txt`
- logs: `/data/suramya/rtofs_global_ohc_fields_{2024,2025}/run_20260704_backfill.log`
- completion marker: `/data/suramya/rtofs_backfill_20260704/CHAIN_DONE`
- settings: `workers=3`, `--cleanup-archv-a`, ~10 min per date per worker

The wrapper is idempotent (already-built dates are skipped), so if it dies
again just rerun the wrapper. After it finishes: rebuild the collocation
table (`build_rtofs_at_argo_points_multiyear.py`), then rerun the global and
profile physics builders for the full calendar.

This note captures where the HHP / OHC work stood at the end of the current session.

## 1. What was completed recently

### 1.1 Major HHP code and reporting work now living in the repo

Large parts of the HHP workflow that had been living only as local work now exist as explicit scripts in `OHC/`, including:

- multiyear Argo-RTOFS collocation
- seasonal interpolation / rendering helpers
- year-holdout XGBoost training and sweep scripts
- locked blocked-forward evaluation scripts
- RTOFS-side global physics feature builders
- RTOFS profile physics feature builders
- steric-height computation and paper-style validation plots
- feature-diagnostic gallery generation
- feature correlation / semi-ablation PDF reporting
- density scatter / patch / error diagnostics

### 1.2 Current recommended correction models

Under the stricter locked blocked-forward protocol, the current working recommendation is:

- `TCHP`: `global_pruned`
- `D26`: `drop_both_lat_interactions`

Locked OOF summary file:

- `OHC/output/ml_benchmarks/locked_physics_semi_ablation_oof_summary.csv`

### 1.3 New density-scatter diagnostic bundle

The following script now exists and was run successfully from the project virtualenv:

- `OHC/build_hhp_density_scatter_diagnostics.py`

Output root:

- `OHC/output/density_scatter_diagnostics_2024_2025/`

Important generated families:

- `global_density/`
- `presentation_density/`
- `region_density/`
- `patch_density/`
- `error_distributions/`
- `feature_relations/`
- `tables/`

This includes the cleaner single-panel presentation-style plots that match the mentor-requested MLD visual family more closely.

## 2. Important implementation details not to forget

### 2.1 The workflow is still point-based

There is still no common-grid ML training table.

Everything uses:

- Argo truth at collocated rows
- raw RTOFS sampled to the same row
- residual target = `Argo - RTOFS`

That means sparse-looking diagnostics are expected.

### 2.2 RTOFS-to-Argo sampling rule

Current exact-location collocation uses:

- 8 nearest native RTOFS cells
- inverse-distance-squared weighting

### 2.3 Presentation density plots added this session

New presentation-style outputs:

- `OHC/output/density_scatter_diagnostics_2024_2025/presentation_density/tchp_raw_presentation_density_scatter.png`
- `OHC/output/density_scatter_diagnostics_2024_2025/presentation_density/tchp_corrected_presentation_density_scatter.png`
- `OHC/output/density_scatter_diagnostics_2024_2025/presentation_density/d26_raw_presentation_density_scatter.png`
- `OHC/output/density_scatter_diagnostics_2024_2025/presentation_density/d26_corrected_presentation_density_scatter.png`

These are observed-vs-model 2D histograms with color = `log10(PDF)`.

### 2.4 Base-calendar caveat now confirmed

The current 2024-2025 collocation table is not limited by Argo calendar coverage.
It is limited by the reduced daily global RTOFS TCHP/D26 field cache.

Confirmed inventory:

- Argo 2024 cache already spans the full year
- Argo 2025 cache already spans the full year
- reduced global RTOFS daily fields currently start at `20240131`

So for current backfill work, treat `2024-01-31` as the practical public-archive
start date in this workflow.

Consequence:

- white month blocks in the feature-gallery heatmaps mainly reflect missing
  base collocation rows from incomplete reduced RTOFS field coverage
- steric / Brunt-Vaisala profile-feature sparsity sits on top of that separate issue

## 3. Immediate next tasks

### 3.1 Highest-priority plotting follow-up

DONE 2026-07-04: the named 20 degree box family now exists (see section 0.1).
Remaining plotting follow-up is mentor review of the chosen box set — the 8
boxes are a TC-basin-oriented default, not yet mentor-confirmed coordinates.

### 3.2 Diagnostic extension likely worth doing

The current density bundle already has:

- global and macro-region density plots
- patch support maps
- top-patch density plots
- signed-error PDF / absolute-error CDF
- feature-binned MAE plots

Still worth adding next:

- mentor-defined named boxes
- more sparse-feature-friendly diagnostics for the deep steric and `N^2` families
- possibly a full-collocation version of the feature-relation plots so sparse features are less under-supported than in the locked OOF subset

### 3.3 Plot automation direction from mentor

Mentor guidance was to keep building a broad automated plot chain so anomalies can be traced backward through processing.

This means future sessions should prefer:

- one more script that adds plots systematically,
- not one-off manual figure generation.

## 4. Files that matter first when resuming

If a future session needs the minimum restart set, begin with these:

- `OHC/context.md`
- `OHC/NEXT_SESSION_HANDOFF.md`
- `OHC/build_hhp_density_scatter_diagnostics.py`
- `OHC/build_hhp_feature_diagnostic_gallery.py`
- `OHC/build_hhp_feature_correlation_ablation_report.py`
- `OHC/run_locked_xgb_physics_semi_ablation.py`
- `OHC/output/ml_benchmarks/locked_physics_semi_ablation_oof_summary.csv`
- `OHC/output/density_scatter_diagnostics_2024_2025/hhp_density_scatter_plot_spec.md`

## 5. Rerun commands

Use the project env:

```bash
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_density_scatter_diagnostics.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_feature_diagnostic_gallery.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/build_hhp_feature_correlation_ablation_report.py
/home/suramya/HHP-Prediction/hhp-env/bin/python /home/suramya/HHP-Prediction/OHC/run_locked_xgb_physics_semi_ablation.py
```

## 6. Repo hygiene choice made in this session

The repo now favors tracking:

- source code
- durable reports
- small benchmark summaries
- diagnostic manifests / curated plot bundles

and not tracking the heavyweight generated caches and large raw output tables by default.

That split is now encoded in `.gitignore`.
