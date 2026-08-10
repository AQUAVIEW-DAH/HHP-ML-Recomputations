# Recommended model: mixture-of-experts blend — full specification

**Status: RECOMMENDED** as of 2026-08-09, superseding the single global XGBoost
recipes (`global_pruned` for TCHP, `drop_both_lat_interactions` for D26) that
were the recommendation since June 2026.

**Training/evaluation scripts:** `OHC/exploration/run_moe_regions.py` (v1
architecture), `OHC/exploration/run_moe_v2_tuning.py` (winning configuration).
**Result tables:** `OHC/output/moe_v2_tuning_20260811/`.
**Companion documents:** feature definitions in
[`gom_attribution.md`](gom_attribution.md) Part A; per-feature support in
[`../feature_inventory.md`](../feature_inventory.md); experiment history in
[`../notes/2026-08-11-meeting-followups.md`](../notes/2026-08-11-meeting-followups.md).

---

## 1. Architecture

For a collocated point with feature vector $\mathbf{x}$, raw RTOFS value $r$,
and observed target $y$, the corrected prediction is

$$\hat{y} \;=\; r \;+\; \alpha\, f_{\mathrm{geo}}(\mathbf{x}) \;+\; (1-\alpha)\, f_{\mathrm{regime}}(\mathbf{x})$$

a convex blend of two mixture-of-experts residual predictors that share the
same feature recipe and base learner but partition the ocean differently.

**Per-target winning configuration:**

| target | blend weight $\alpha$ (geographic) | regimes $K$ | prior weight $w$ |
|---|---|---|---|
| TCHP | 0.75 | 6 | 0.05 |
| D26 | 0.50 | 12 | 0.05 |

### 1.1 Geographic expert branch $f_{\mathrm{geo}}$

Five experts with a **hard positional gate** (every point belongs to exactly
one region; regions are fixed polygons, so the gate is defined everywhere,
observations or not):

| expert | definition |
|---|---|
| gulf_of_mexico | lat ∈ [18, 31], lon ∈ [−98, −80] |
| atlantic | lon ∈ [−100, 20), excluding the Gulf box |
| indian | lon ∈ [20, 100) |
| west_pacific | lon ∈ [100, 160) |
| epac_other | everything else |

Each expert is trained on **all** rows with sample weights

$$w_i = \begin{cases} 1.0 & \text{row } i \text{ in the expert's region} \\ 0.05 & \text{otherwise} \end{cases}$$

i.e. a local specialist with a weak global prior. The sweep over
$w \in \{0.05, 0.10, 0.15, 0.25\}$ chose 0.05 on every axis (both targets,
global and Gulf scores): the experts want strong specialisation, and larger
priors reproduce the v1 failure to capture the Gulf D26 behaviour.

### 1.2 Learned-regime expert branch $f_{\mathrm{regime}}$

$K$ experts over regimes discovered by k-means **in physics-state space with
no coordinates**, so the gate generalises to unsampled locations. Clustering
features (standardised, training-fold median imputation):

`model_ssh_m`, `model_temp_excess_26c`, `model_mixed_layer_thickness_m`,
`model_tchp_local_std_1deg`, `model_tchp_anom_from_1deg_mean`, `abs_lat`

Fitting (per fold, on training rows only): standardise, k-means with $K$
clusters (`n_init=8`, fixed seed). Expert $k$ is trained with the same
prior-weight scheme (cluster members 1.0, others 0.05).

**Soft gate.** For a prediction point, compute distances $d_k$ to the $K$
cluster centroids in the standardised space and weight the experts by

$$g_k(\mathbf{x}) \;=\; \frac{\exp\!\bigl(-d_k/T\bigr)}{\sum_{k'} \exp\!\bigl(-d_{k'}/T\bigr)},
\qquad T = \mathrm{median}(d)$$

so $f_{\mathrm{regime}}(\mathbf{x}) = \sum_k g_k(\mathbf{x})\, f_k(\mathbf{x})$.
The sweep over $K \in \{4, 6, 8, 12\}$ chose 6 for TCHP and 12 for D26.

### 1.3 Base learner (every expert, both branches)

XGBoost regression trees on the residual target $\delta = y - r$:

| hyperparameter | value |
|---|---|
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda (L2) | 1.0 |
| objective | squared error |
| missing values | training-fold median imputation |

These are the locked-protocol settings used across the whole project; the
capacity sweep (depth {4, 6, 8} × estimators {300, 800}) confirmed deeper or
larger settings do not help at current sample sizes.

### 1.4 Features

The per-target recipes are unchanged from the single-model era:
`global_pruned_plus_neighborhood` for TCHP (34 features) and
`drop_both_lat_interactions_plus_neighborhood` for D26 (35 features):
calendar encodings, location, raw model state, pruned RTOFS physics
diagnostics (SSH, MLT, SBLT, temperature excess and derived terms), deep
steric height and Brunt–Väisälä summaries, and the neighborhood-context
stencils. All formulas in [`gom_attribution.md`](gom_attribution.md) §A3;
support statistics in [`../feature_inventory.md`](../feature_inventory.md).

## 2. Evaluation protocol and headline numbers

Locked blocked-forward protocol: dates sorted chronologically, 3 consecutive
blocks, 1-date embargo, out-of-fold evaluation on the observed scale, full
2024–2025 calendar (60,666 evaluable rows).

| out-of-fold MAE | TCHP (kJ/cm²) | D26 (m) | TCHP Gulf | D26 Gulf |
|---|---|---|---|---|
| raw RTOFS | 16.34 | 14.67 | 15.21 | 15.35 |
| single global model (previous recommendation) | 11.397 | 10.755 | 13.05 | 11.81 |
| MoE v1 (w=0.15, K=6, no blend) | 11.281 | 10.634 | 12.42 | 11.92 |
| **MoE v2 blend (recommended)** | **11.189** | **10.553** | **12.41** | **11.52** |
| dedicated Gulf-local model (reference upper bound) | – | – | 12.37 | 11.23 |

## 3. Every ML method evaluated on the way here

| method | where tested | headline result | verdict |
|---|---|---|---|
| XGBoost, single global model, ~20 feature-recipe ablations | locked semi-ablation (`run_locked_xgb_physics_semi_ablation.py`) | best recipes 11.40 / 10.76 | was the recommendation; now the baseline |
| Coordinate-feature ablation (no lat/lon/interactions) | presentation ablation | 12.74 / 12.48 | coordinates worth ~1–1.4 MAE |
| GPBoost (trees + Gaussian-process spatial term, Vecchia) | spatial tiers eval | 13.89 / 13.22 under region-holdout | rejected: GP shrinks to zero in unseen regions |
| Regression kriging on residuals | considered in spatial directions survey | superseded by the GPBoost result | not pursued |
| WOA climatology as base model, ML-corrected | `run_woa_base_correction.py` | 17.21 / 13.85 vs 12.09 / 10.75 for corrected RTOFS | dynamical model earns its keep |
| Random forest (full and reduced recipes) | Gulf robustness batch | 12.15→11.82 / 11.02→10.87 | confirms SSH dominance and lean-beats-full; slightly behind boosting |
| XGBoost capacity sweep on reduced features | Gulf robustness batch | smallest settings win | sample-size-limited, not resolution-limited |
| Small neural network (MLP 64×64) | Gulf robustness batch | 12.87 / 12.42 | ~1 MAE behind trees at this data scale |
| Dedicated per-basin local model (Gulf) | `run_gom_local_vs_global.py` | 12.37 / 11.23 in-Gulf | proof of concept that motivated the MoE |
| MoE, geographic experts (hard gate) | `run_moe_regions.py`, v2 sweep | 11.23 / 10.61 at w=0.05 | strong; component of the blend |
| MoE, learned regimes (soft state-space gate) | same | 11.29 / 10.59 at best K | strong; component of the blend |
| **MoE blend of both gates** | `run_moe_v2_tuning.py` | **11.19 / 10.55** | **recommended** |

Feature-level negatives worth remembering alongside: surface-stratification
N² summaries (redundant with existing upper-200 m N², r = 0.93), barrier-layer
thickness under both ΔT criteria (no gain at current profile coverage), and
the steric-height reference swap (surface/1000 dbar adopted over
1000-ref-2000 on equal skill and better coverage).

## 4. Known caveats of the recommendation

1. **Blend hyperparameters (α, K, w) were selected on the same out-of-fold
   scores they are reported on** — a mild selection bias. The honest guard is
   that the margins over the untuned v1 are consistent across two targets and
   two scopes; a clean confirmation on the next data extension (e.g. gliders)
   is queued.
2. The Gulf D26 gap to the dedicated local model is narrowed (~60% recovered)
   but not closed.
3. Distribution narrowing: like every squared-error regressor, the correction
   under-disperses relative to observations; pointwise skill improves at every
   observed value except the lowest bin (see the conditional-skill analysis).
4. 2024↔2025 float-platform overlap means date-holdout is not a full
   future-generalisation guarantee (June 2026 audit).
5. Field deployment requires computing the neighborhood stencils and regime
   gate on the full grid; all inputs are RTOFS-internal, so no external data
   dependency is introduced.
