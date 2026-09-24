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

## 9, 10. Missing physics and feature combinations: running

`OHC/exploration/run_missing_physics_search.py`. Hypothesis from item 12: the
east-west Pacific shape of the leftover error follows the El Nino / La Nina
state, which a correction learned on earlier months cannot anticipate. Tests an
RTOFS-only Nino 3.4 index, cyclic longitude (removes the artificial seam at the
dateline, in the middle of the Pacific), pairwise products, sums and ratios,
family averages, and a higher leaf-size floor.

Note on item 10: for tree models a one-to-one smooth transform of a feature
(log, square, ...) can never add information, because splitting on f and on
log f separates the same rows. Only new information, non-monotone transforms,
and combinations of features can help.

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
