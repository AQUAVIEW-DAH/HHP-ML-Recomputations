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
