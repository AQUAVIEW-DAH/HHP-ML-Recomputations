# Meeting follow-ups, due Thursday 2026-10-01

Meeting of 2026-09-24. Items as noted, then status and results.

1. nearest grid distance
2. year
3. why is the time block not correlating with the error
4. size of the withheld data: is it a full year?
5. when a tree splits, do we lose data?
6. why is `model_tchp_anom_from_1deg_mean` so high up in the tree splits?
7. why is time so unimportant when summer and winter looked starkly different earlier?
8. try keeping only `month_int` and removing the other time features
9. products, sums, means, modes of the features
10. can we find physical information the features miss, e.g. through functions of features (sin, cos, ...)?
11. GOFS reanalysis pipeline
12. lat/lon maps of the D26 and TCHP errors for the new model, maybe interpolated
13. results by next Thursday

All results use the rebuilt tables of 2026-09-24 (interpolation fix, one primary
profile per Argo cast). Drive folder: `HHP-analysis-2026-10-01/`.

## New baseline on clean data (supersedes every earlier number)

`OHC/exploration/run_moe_clean_rerun.py`, 53,742 out-of-fold profiles:

| | TCHP | D26 |
|---|---|---|
| raw RTOFS | 16.38 | 15.08 |
| single global model | 10.87 | 10.78 |
| **MoE blend (recommended)** | **10.68** | **10.57** |
| MoE, Gulf of Mexico | 12.33 | 11.92 |

The old 11.19 / 10.55 counted 12% of casts more than once and is retired.

## 4. Withheld data: not a full year

Three blocks of about five months each (1-date embargo):

| fold | trains on | validates on | validation rows |
|---|---|---|---|
| 1 | 2024-01-31 to 2024-09-17 | 2024-09-19 to 2025-02-22 | 16,737 |
| 2 | 2024-01-31 to 2025-02-22 | 2025-02-24 to 2025-07-30 | 19,505 |
| 3 | 2024-01-31 to 2025-07-30 | 2025-08-01 to 2025-12-31 | 17,500 |

Together they cover September 2024 to December 2025; the first seven months are
only ever trained on. Fold 1 has never seen October to January, nor 2025.

## 1, 2, 8. Grid distance, year, month_int only

`OHC/exploration/run_time_meta_ablation.py`, three seeds each (seed noise ~0.005).
Change in MAE against the full recipe:

| variant | TCHP | D26 |
|---|---|---|
| drop grid distance | +0.002 | +0.006 |
| drop year | +0.011 | -0.012 |
| month_int only (drop the other 8 time features) | +0.016 | +0.015 |
| month sin/cos only | +0.031 | -0.014 |
| day-of-year sin/cos only | +0.013 | -0.006 |
| no time features at all | +0.024 | +0.028 |

* Grid distance does nothing: remove it.
* Year is neutral on these scores and cannot generalise: no training fold can
  ever contain 2026. Remove it before the 2026 test.
* month_int alone is slightly worse than the full set and than a sin/cos pair:
  it is not cyclic, so December (12) and January (1) look maximally far apart
  and the trees need extra splits to rejoin them.
* Every time-feature change is within +/- 0.03.

## 3, 7. Why time barely matters

* Argo's monthly-mean TCHP swings 12.2 over the year and RTOFS's swings 12.3:
  RTOFS already reproduces the seasonal cycle, and the model receives it
  through the raw RTOFS value (correlation 0.93 with Argo).
* What the model corrects is the error, whose monthly means swing only 4.7
  while it scatters by 19 within any single month.
* Checked and rejected: that the two hemispheres cancel each other's seasons.
  Within each hemisphere month still barely correlates with TCHP (0.04 N,
  0.11 S) and the monthly peaks fall in the wrong months, because which regions
  the floats sample shifts from month to month and the warm-water filter drops
  higher-latitude profiles in winter.

## 5. Do tree splits lose data? No.

Every row lands in exactly one leaf of every tree (81,183 rows x 300 trees in
the locked model). A split partitions rows; nothing is discarded. What does
happen is that leaves can get thin: median 1,481 rows per leaf, but 5% of
leaves hold 13 or fewer and the smallest holds one, because the locked setting
allows a leaf of a single row. Raising that floor is being tested (item 10 run).

## 6. Why the 1-degree anomaly sits high in the trees

As a first split on the raw error it is weak (3.4% of the variance, against
10.8% for distance from the equator). But on the error left after position is
accounted for, it is the strongest input by a wide margin: 4.2%, against
0.3-0.5% for every other feature. The trees split on latitude first, then in
almost every branch the anomaly is the most useful next question. That is also
why it adds the most when grafted onto a small core.

## 12. Error maps for the new model

`moe_clean/{tchp,d26}_error_maps_before_after.png`: raw RTOFS versus MoE error,
as profile points and as a Gaussian-interpolated field (250 km scale, blank
beyond 200 km of a float). The MoE removes the bias, but the leftover error is
not random: it is too high across the central and eastern tropical Pacific and
the southern Indian Ocean, and still too low in the western warm pool.

## 9, 10. Missing physics and feature combinations

`OHC/exploration/run_missing_physics_search.py`, three seeds each. Change in MAE:

| addition | TCHP | D26 |
|---|---|---|
| **cyclic longitude (sin, cos)** | **-0.041** | **-0.041** |
| best feature combinations from the residual screen | -0.050 | -0.020 |
| El Nino index (Nino 3.4, from RTOFS SST) | +0.015 | +0.001 |
| family averages (anomaly, std, gradient) | +0.007 | +0.006 |
| leaf-size floor 20 / 50 / 100 | +0.005 / -0.002 / +0.002 | 0.000 / +0.006 / +0.020 |

* **Cyclic longitude** is the clean win, identical on both targets. Mechanism:
  the MoE's East-Pacific expert spans the dateline (160 E to 100 W), so in raw
  longitude its region is two disjoint ranges (160..180 and -180..-100) that the
  trees must stitch together. Sine and cosine make it one continuous region.
* **Combinations** help most for TCHP; the best were sums of standardised
  features such as z(D26) + z(SSH), plausibly two noisy measures of how deep the
  warm layer reaches, averaged. They were picked by screening residuals on these
  folds and disagree between targets, so they are kept out of the frozen recipe.
* **Only non-monotone functions and combinations can help a tree**: a one-to-one
  smooth transform (log, square) splits the same rows as the original feature.
* **El Nino: untested rather than refuted.** The east-Pacific over-correction is
  stable across all three folds (TCHP +3.6, +2.2, +5.2), and the mean Nino 3.4
  anomaly in every validation period was near neutral (-0.15, -0.29, -0.05). No
  validation period contained a real El Nino or La Nina, so this test could not
  have detected an ENSO effect. Only multi-year data (GOFS option C) can.

## Frozen recipe for the 2026 test

`OHC/exploration/run_frozen_recipe_dev.py`. Chosen on principle, not by score:
remove `year` (cannot generalise to 2026), the three deep-profile features
(cannot be built for 2026; 89% imputed anyway), and grid distance (no effect);
add cyclic longitude. TCHP 34 features, D26 32.

| development folds | current | frozen |
|---|---|---|
| TCHP single / MoE / MoE Gulf | 10.87 / 10.68 / 12.33 | **10.84 / 10.69 / 12.23** |
| D26 single / MoE / MoE Gulf | 10.78 / 10.57 / 11.92 | **10.73 / 10.54 / 11.68** |

## 2026 holdout tables

`OHC/build_holdout_2026_tables.py` -> `OHC/output/ml_collocation/holdout_2026/`.
138,640 profiles on 265 days (2026-01-01 to 09-22), 32,710 warm rows with one
profile per cast, observation time matched for 100%. Kept physically separate
from the development tables; to be evaluated exactly once, after the frozen
recipe is agreed.

## 11. GOFS reanalysis: needs a decision on purpose

The 2015 pilot works end to end. Against Argo, GOFS 3.1 reanalysis TCHP has MAE
6.6 and bias +0.6, against RTOFS's 16.3 and -10.7, but GOFS assimilates Argo and
is a hindcast, so the comparison is not like for like. The reanalysis ends in
2015 and RTOFS starts in 2024, so there are no overlapping dates. Options:

* A. reference field in float-free regions: not possible with the reanalysis
  (no overlap).
* B. multi-year SSH climatology for the anomaly features: ~2.8 TB for 2008-2015
  every 5 days. Cheapest and most defensible.
* C. more El Nino / La Nina variety for training: largest, and a different model
  than RTOFS, so any correction learned there has to transfer. Worth it only if
  the ENSO hypothesis holds.

## Figures (Drive: `HHP-analysis-2026-10-01/figures/`)

Made by `OHC/exploration/plot_meeting_followups_20261001.py` (01-10); 11-13 are copies of earlier outputs.

| figure | meeting item | what it shows |
|---|---|---|
| `01_items1-2-8_removing_grid_year_time.png` | 1, 2, 8 | change in error when grid distance, year or time features are removed |
| `02_items3-7_seasons_in_rtofs_vs_error.png` | 3, 7 | RTOFS follows the seasons; its error barely does |
| `03_item4_how_the_data_is_split.png` | 4 | the three ~5-month test blocks and the 2026 holdout on a timeline |
| `04_item5_tree_splits_keep_all_data.png` | 5 | profiles per leaf across all 300 trees; bigger-leaf test |
| `05_item6_why_the_anomaly_splits_early.png` | 6 | single-split strength on the raw error vs after location is known |
| `06_items9-10_new_features_and_dateline.png` | 9, 10 | candidate features; why sin/cos longitude removes the dateline seam |
| `07_item10_el_nino_untestable.png` | 10 | Nino 3.4 index over the test blocks; east-Pacific over-correction by month |
| `08_item11_data_coverage_rtofs_gofs_argo.png` | 11 | what RTOFS, GOFS and Argo data exist and what is downloaded |
| `09_item11_gofs_minus_rtofs_tchp_20240615.png` | 11 | GOFS minus RTOFS TCHP on one shared day |
| `10_frozen_recipe_vs_current.png` | 2026 test | frozen recipe vs current, development blocks |
| `11_item12_tchp_error_maps_before_after.png`, `12_item12_d26_error_maps_before_after.png` | 12 | raw RTOFS vs MoE error maps, points and interpolated |
| `13_item11_gofs2015_pilot_bias_by_region.png` | 11 | 2015 GOFS reanalysis pilot: bias by region |

## Update 2026-10-04: GOFS 3.1 analysis download

Three years, 2021-09-05 to 2024-09-04, daily 06Z (`OHC/exploration/build_gofs31_analysis_daily_fields.py`):
only temperature to 250 m over 42 S - 48 N and SSH are read, since the HYCOM server allows ~1-2.4 MB/s.
The 222 days overlapping RTOFS (2024-01-27 to 09-04) are complete; the rest finishes around 2026-10-05.
