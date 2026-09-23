# Diagnostics, smooth-model ladder, redundancy and emergence — 2026-09-10

Four batches answering the meeting-notes list. Scripts:
`run_latlon_diagnostics.py`, `run_smooth_model_ladder.py`,
`run_feature_redundancy.py`, `run_prune_graft_emergence.py`.
Outputs: `OHC/output/{latlon_diagnostics_20260908, smooth_ladder_20260909,
feature_redundancy_20260909, prune_graft_20260910}/`.

## 1. Is the difference between models just noise? — No

Date-block bootstrap of paired MAE differences, 1000 resamples, warm subset.
Every pairwise ranking is significant: TCHP RF−GP −0.25 (CI −0.31,−0.19);
D26 GP−RF −0.19 (CI −0.26,−0.10); both beat XGB/SVR by 0.3–1.3. Caveat: the
GP moves ±0.15 depending on which 3k points it is fitted on, so its D26
margin is fragile.

## 2. RMSE — no ranking flips

RF leads TCHP on both metrics (MAE 12.44 / RMSE 17.68); GP leads D26
(12.32 / 16.37). On the zero-filled all-rows scale raw RTOFS is MAE 4.9 but
RMSE 12.9: the budget is dominated by rare large misses (see §7).

## 3. RF overfitting — the leaf floor of 50 was too small

Train-vs-held-out sweep: held-out error keeps improving to leaf 100 (TCHP,
12.37) and is still falling at 200 (D26, 12.36); the train–holdout gap shrinks
from 1.7 to 0.4. **At leaf 200 the RF's D26 score (12.36) is statistically
indistinguishable from the GP's (12.32)** — the GP's D26 "win" was the RF's
under-regularisation, confirming the noise reading of its excess texture.

## 4. Why GoM discontinuities are large — not thin data

The Gulf is twice as densely sampled as the global median (128–165 profiles
per 1° vs 68–72) yet its seams are ~15% larger. Correlation of seam size with
sparsity ≈ 0 (−0.10); for D26 seams correlate with the GP's own gradient
(0.35). Conclusion: a genuinely sharp correction field (Loop Current), plus —
for TCHP, where neither correlation holds — box edges locking onto a front
whose position moves through the year.

## 5. The smooth models were mis-set, not bounded

Detail ladder (warm-subset OOF MAE; RF reference 12.44 / 12.50):

| variant | TCHP | D26 |
|---|---|---|
| **SVR, bumps 16× narrower** | **12.35** | **12.09** |
| SVR, bumps 4× narrower | 12.54 | 12.15 |
| GP Matérn 3/2 | 12.57 | 12.43 |
| GP learned scales, 6k pts | 12.67 | 12.26 |
| GP learned scales, 3k (as reported earlier) | 12.64 | 12.47 |
| GP two-scale kernel | 12.88 | 12.32 |
| GP forced 2°×8° | 13.13 | 12.73 |
| SVR default γ (as reported earlier) | 13.47 | 12.57 |

**The tuned SVR is the best position-only model of any family, on both
targets.** Its default bump width was ~16× too broad; the earlier conclusion
that "SVR is the weakest" was a tuning artifact and the email draft to
Dr. Jacobs needs that sentence corrected. Forcing the *GP* shorter hurt: its
likelihood-fitted scales were already right.

## 6. The noise floor (semivariogram)

Extrapolating the rising limb to zero separation: nugget 291 (kJ/cm²)² →
**floor RMSE 17.06**, and **71% of the total error variance lies below
~100 km**. A position-only predictor cannot beat that bound. RF sits 3.6%
above it, GP 5.1%. This explains the flat learning curve, the ≤0.6 spread
across families, and locates the remaining headroom in features that resolve
sub-100 km state.

## 7. Formal redundancy — the answer is "yes, but you cannot compress it"

* **Feature space is redundant**: 34 features carry a participation ratio of
  12.2 (D26: 35 → 13.6); only 10 (11) eigenvalues exceed 1; 16 (18)
  components hold 90% of the variance.
* **Skill is not**: XGBoost on the top-m principal components gives
  TCHP 12.08 at m=20 vs 11.39 at m=34 (D26 12.03 vs 10.75). The low-variance
  directions carry real predictive signal, so principal-component compression
  costs ~0.7 MAE.
* **Caution on the correlation score**: applied literally it ranks
  `model_interp_tchp`, `model_temp_excess_26c` and `d26_minus_mlt_m` as the
  *most* redundant, because the engineered features are derived from them —
  it finds hub variables, not useless ones. Pruning by this score alone would
  delete the indispensable base value.

## 8. Prune and graft, both directions — one feature does the work

From a preselected physical core (raw model value, SSH, temperature excess,
MLT, lat, lon, month sin/cos):

* `model_tchp_anom_from_1deg_mean` alone grafts **+0.407** onto the core —
  more than the entire 10-member neighborhood family (+0.353). D26:
  `model_d26_anom_from_1deg_mean` grafts +0.427 of the family's +0.435.
  **The neighborhood family reduces to one feature.**
* Calendar: ≈0 in both directions. Collocation geometry: negative both ways.
  Deep steric / stratification (D26): negative both ways — removing it
  *improves* the model, confirming Dr. Jacobs' suspicion that a 1000–2000 m
  steric height cannot inform a target living above ~160 m.
* **No family shows meaningful emergence relative to its best member**
  (largest +0.06). Sets behave like their strongest single member, which
  answers the "sets vs independent features" question.

## 9. The ~8,000 boundary cases ARE an emergence effect

Cumulative feature groups scored separately on boundary / warm / cold rows
(zero-filled targets, all 322k rows):

| D26, added cumulatively | boundary | warm | all |
|---|---|---|---|
| position | 40.93 | 12.16 | 4.93 |
| + calendar | 40.57 | 12.08 | 5.06 |
| + raw model value (0 where no 26 °C) | 38.42 | 11.69 | 4.91 |
| + SSH / MLT / SBLT | 38.20 | 11.22 | 4.79 |
| **+ temperature excess above 26 °C** | **31.20** | 11.25 | 4.39 |
| + SST neighbourhood | 30.88 | 11.30 | 4.40 |
| raw RTOFS | 42.29 | 14.67 | 5.17 |

Temperature excess cuts the boundary error by **7.0 m (−18%)** while the warm
rows do not improve at all (11.215 → 11.246). That is pure conditional value:
it is an edge detector, useless where 26 °C water certainly exists. TCHP shows
the same sign, smaller (11.06 → 10.82 at the boundary, warm slightly worse).

**The two results together are the important message.** `model_temp_excess_26c`
has a Spearman correlation with the D26 error of **0.015** — near the bottom of
the correlation-based ranking — yet it delivers the single largest boundary
improvement we have measured. Linear correlation screening would delete
precisely the feature whose value is an interaction. Correlation ranking is a
useful first pass, not a pruning rule.

Boundary error remains 31 m after everything, on 2.6% of rows carrying ~21% of
the D26 error budget. Open question for Dr. Jacobs: the D26 := 0 convention
creates that cliff at the isotherm edge; is that the behaviour he wants scored?

---

# Follow-up, 2026-09-16: the matrix he actually asked for, and what compresses

Script `run_redundancy_followup.py`; outputs `OHC/output/redundancy_followup_20260916/`.

## 10. The correlation matrix extended with the target errors

His ask was the N x N feature cross-correlation *plus* the correlation of each
feature to the TCHP error and the D26 error. We had the latter only as a CSV
column; it is now four extra rows and columns on the matrix itself (signed and
absolute error, both targets, in both recipes), pinned to the top-left edge in
red so relatedness-to-target and relatedness-to-each-other read off one figure.
`{t}_correlation_matrix_with_error.png`.

Reading it as a 2x2:

* **bright block, dark against the error strip = delete all but one.** The
  clearest case is the eight calendar encodings, which form a bright block
  among themselves and are essentially black against every error row. That
  matches the prune/graft result exactly (the calendar family contributes ~0
  in both directions).
* **isolated and dark against the error strip = delete outright.** Three
  features are black almost everywhere: `nearest_rtofs_grid_distance_km`,
  `model_steric_1000_ref2000_m`, `year`. All three were also negative in both
  prune and graft. These are the safest removals we have identified.
* **bright block with visible error correlation = keep one or two.** The
  raw-value cluster (`model_interp_d26_m`, `model_interp_tchp`,
  `d26_minus_mlt_m`, the TCHP local stds).
* **the counterexample, marked in blue.** `model_temp_excess_26c` is dark
  against the whole error strip and bright against the physics block, so the
  figure says delete it. It is the most valuable boundary-case feature we
  have. The annotation is on the figure deliberately.

## 11. How many directions of *useful* information? No low-dimensional answer

Three compressions compared at equal component counts
(`skill_vs_components_three_ways.png`):

| components kept | TCHP: variance | relevance | PLS | D26: variance | relevance | PLS |
|---|---|---|---|---|---|---|
| 5 | 13.11 | 13.33 | **12.10** | 13.01 | 12.37 | **11.55** |
| 20 | 12.08 | 12.00 | **11.89** | 12.03 | 11.60 | **11.34** |
| all raw inputs | 11.40 | | | 10.76 | | |

* Supervised ordering clearly helps: partial least squares with **5**
  components matches or beats variance-ordered principal components with
  **20**, a fourfold compression, and the gap is largest for D26.
* But **no linear compression reaches the raw inputs**, even choosing
  directions with the target in hand: PLS at 20 components is still 0.5
  (TCHP) and 0.6 m (D26) short.
* Meanwhile a *full-rank* rotation is harmless (34 principal components gave
  11.394 against 11.397 raw). So the loss comes from discarding directions,
  not from rotating the axes.
* `component_importance_vs_rank.png` shows why. The model's splitting gain per
  principal component does not decay with eigenvalue rank at all; the largest
  contributions are PC6 and PC2 (TCHP) and PC6 and PC19 (D26), with meaningful
  gain out to PC35.

**Conclusion.** The inputs are redundant in their linear structure (about 12
effective directions of variation) but their predictive content is not
compressible into a small linear subspace. The explanation is the same one the
boundary case gave us: value that lives in a threshold effect, such as
`model_temp_excess_26c` mattering only within about 0.6 degrees of zero, cannot
be isolated by any linear combination, because mixing it with other variables
destroys the threshold. Linear screening and linear compression fail for the
same reason.

## 12. What the trees reach for first

`first_splits.png`, gain in the first three levels. TCHP: `abs_lat` 43%,
`model_ssh_m` 15%, `model_interp_d26_m` 8%, `model_tchp_anom_from_1deg_mean`
7%. D26: `abs_lat` 41%, `model_d26_anom_from_1deg_mean` 30%, `lat` 10%,
`model_ssh_m` 7%. So the opening cuts are distance from the equator, how
energetic the local ocean is, and how much of a local bump the model claims.
`model_temp_excess_26c` is near the bottom here (0.2% / 2%), which is
consistent with it acting deep in the trees on a small subpopulation rather
than as a global split.

---

# CORRECTIONS, 2026-09-23 (post-audit)

An external review plus a line-by-line audit found six defects. What changed.

## C1. Interpolation bug (the only one that corrupts data)

`_interpolate_neighbor_values`, duplicated in
`build_rtofs_at_argo_points_multiyear.py` and
`build_rtofs_global_physics_features_2024_2025.py`, multiplied zero weights by
NaN values: `0.0 * nan = nan`, so one missing neighbour out of eight voided the
whole interpolation. Because TCHP/D26 grids are NaN both on land **and** below
26 C, the failure concentrated on the isotherm edge.

* **Verified:** 41% of sampled "Argo warm, model missing" rows are artifacts;
  1,426 such rows have model SST >= 26 C. The fix recovers exactly those rows
  and changes previously-valid rows by 0.000000.
* **Blast radius: 4 of 21 recent scripts** — those calling `.fillna(0)`:
  `run_latlon_only_models`, `run_latlon_diagnostics`, `run_smooth_model_ladder`,
  `run_prune_graft_emergence` Part 2. The other 17 use the warm filter, where a
  bug-NaN removes a row rather than corrupting it. **Zero warm rows carry a
  bug-affected physics feature** (structural: a row is warm only if all eight
  TCHP neighbours are finite, and SSH/MLT/SST exist wherever TCHP does), so the
  benchmark table and all MoE work are unaffected.
* **Still to do:** rebuild the collocation and physics tables, then re-run those
  four scripts. The boundary-emergence result is suspended until then, because
  `temp_excess >= 0` on exactly the artifact rows, so the model may have been
  detecting our bug rather than a physical edge.

## C2. The PCA conclusion was inverted

`if m < d:` skipped the rotation at the final point, so "all components" was
raw standardised features. Corrected (all three methods now always rotated):

| components | TCHP variance | relevance | PLS | D26 variance | relevance | PLS |
|---|---|---|---|---|---|---|
| 5 | 13.11 | 13.33 | **12.10** | 13.01 | 12.37 | **11.55** |
| 20 | 12.09 | 12.00 | **11.89** | 12.03 | 11.60 | **11.34** |
| all (34/35) | **12.01** | 12.03 | **11.85** | **11.51** | 11.51 | **11.32** |
| all raw, unrotated | | 11.40 | | | 10.76 | |

A full-rank rotation costs 0.61 (TCHP) and 0.76 (D26). PLS at full rank beats
PCA at full rank, and PLS saturates by about 20 components. So the predictive
information **is** compressible into roughly 20 linear directions; the residual
gap to raw features is the axis-alignment penalty trees pay for rotated inputs,
not information spread thinly across all directions. Section 11 said the
opposite and is withdrawn.

## C3. The noise floor is not well determined

The original semivariogram pooled pairs across dates, mixing spatial and
temporal variability. Same-day pairs give a much lower implied floor
(TCHP 17.4 -> 14.4; D26 14.2 -> 7.5), but the same-day curve is noisy and
non-monotonic at short range and its 0-10 km population is dominated by a few
dense clusters. **The honest statement is a range, not a number:** a
substantial share of the pooled nugget is temporal, so a model that knows the
date faces a materially lower floor than 17.1, but this data cannot pin it
down. The claim "within 3.6% of an unavoidable limit" is withdrawn, and with it
the conclusion that there is little headroom for richer models.

## C4. Correlation population mismatch

`temp_excess` correlation with the TCHP error is +0.136 on warm rows, **+0.342
on all rows**, -0.418 within boundary rows. The figure compared a warm-row
correlation against an all-rows gain; it now shows both populations. The
narrow point survives (correlation is a weak guide to value); "almost no
correlation at all" does not.

## C5. Methodological imprecision (conclusions survive)

* The RF stability map used `set(...)` on a bootstrap draw, collapsing it to a
  ~62% subsample. Now a true bootstrap. The old "74% of the tropics is
  significant" was a conservative lower bound. The MAE-difference bootstrap was
  always correct.
* Seam-figure density radius was the chord for **2 deg**, labelled 1 deg. Fixed.
* `linkage(1 - |R|)` passed a square matrix, which scipy reads as observation
  vectors, so the heatmap ordering was not 1 - |corr|. Now uses `squareform`.
* The SVR bump-width sweep stopped at x16, its own best value;
  `run_ladder_corrections.py` extends it to x128 to bracket the optimum.
* `rf_boot_std_median_smooth = 0.000` was reported as meaningful; it is zero
  only because "smooth regions" are mostly the empty extratropics.

## C6. Overreach in claims already sent to Dr. Jacobs

* "Stratification is physically unimportant" rests on features present on
  **11%** of warm rows. Supportable claim: no gain at current coverage.
* The SSH-versus-steric comparison used a steric feature median-imputed on
  **89%** of rows. Not valid as stated.
* Both need a short correction email.

## C7. New data-quality items (found in the audit, not in the review)

* **Stencil edge bias.** The neighbourhood builder accepts a window mean from
  as few as 25% valid cells and excludes sub-threshold cells rather than
  treating them as zero. Mean `tchp_anom_from_1deg_mean` is **-2.09** near the
  26 C edge against **+1.25** in the warm interior, where an unbiased anomaly
  would be near zero in both. This is our highest-value feature and the bias
  sits exactly where the boundary cases are. Also now inconsistent with the
  zero-fill convention. Decide the window rule with the rebuild.
* **11,536 warm rows share an exact (date, lat, lon) with another row**
  (11,530 positions, up to 4 each), within-position Argo spread 2.65 kJ/cm2.
  Not exact duplicates but distinct records at identical coordinates. Trace to
  the Argo build; it inflates effective sample size and breaks independence
  assumptions in the bootstrap.
