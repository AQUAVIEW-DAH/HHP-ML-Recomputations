# Meeting follow-ups — review date 2026-08-11

Source: meeting notes with Dr. Jacobs (early August 2026). Each item below has
the original shorthand, what it means, why it matters, what is needed to do it,
and current status. Items move from *queued* → *done* in place; results are
summarised inline so this file stays the single review point.

Naming pattern for these files: `docs/notes/YYYY-MM-DD-<topic>.md`, dated by
the review/checkpoint date.

---

## A. Diagnostics (cheap; run first)

### A1. Does the correction hurt at high TCHP? — *status: done, see results below*

**Original note:** "tchp oot of dist 50 kj rtofs better than the models vs argos"

**Meaning.** In the Gulf distribution plot, above roughly 50 kJ/cm² raw RTOFS
appeared to track Argo better than the corrected models did. This is the
expected signature of error-minimising regression: predictions get shrunk
toward the conditional mean, so extremes are under-predicted even when average
error improves.

**Why it matters.** The hurricane application lives in the upper tail. A
correction that improves mean error while degrading the high-heat-content tail
is the wrong trade for this use case, and we need to know exactly where the
crossover sits before presenting the correction as an improvement.

**What is needed.** Bin the locked out-of-fold rows by *observed* value
(12 quantile bins, ≥40 rows each); per bin compute MAE and bias for raw RTOFS
and for the corrected model; identify the observed value above which the
correction stops helping. Run globally and for the Gulf separately.
Script: `OHC/exploration/run_gom_diagnostics_followup.py` (item 1),
figures `*_conditional_skill.png`, tables `*_conditional_skill_{global,gom}.csv`.

**Results (2026-08-09 run).** **The hypothesis is not supported — the opposite
is true.** For TCHP the correction benefit *grows monotonically* with observed
value, globally and in the Gulf:

| observed TCHP | global: raw MAE → corrected | Gulf: raw MAE → corrected |
|---|---|---|
| ~5 (lowest bin) | 5.9 → 7.1 (**correction worse**) | 4.8 → 6.0 (**worse**) |
| ~45 | 12.1 → 10.1 | 11.5 → 9.8 |
| ~105 | 24.9 → 15.2 | 21.9 → 19.5 |
| ~165 (highest bin) | 41.3 → 19.4 (−22) | 41.9 → 24.2 (−9.7) |

No crossover exists at the high end for TCHP. The only regime where raw RTOFS
wins is the *lowest* bin (TCHP below ~15 kJ/cm², i.e. water barely above the
26 °C threshold), where the correction adds ~1 kJ/cm² of error.

For D26 the picture is similar but with one real exception inside the Gulf: a
crossover at **63 m**, where raw RTOFS is better across roughly 63–90 m
(−0.9 and −1.5 m of benefit in those two bins, ~360 profiles) before the
correction wins strongly again above 100 m. Globally, D26 shows the same
low-end penalty (below ~55 m) and a monotonically growing benefit above.

**Reconciliation with the distribution plot.** The meeting impression came from
the distribution figure, where the corrected curves over-concentrate near the
mode. Both statements are true and they are not in conflict: the correction is
**more accurate per profile** at high values while producing a **narrower
distribution** overall. Distribution shape and pointwise accuracy are different
properties; regression to the conditional mean narrows the former while
improving the latter.

**Follow-up to consider.** (a) The genuine weak spot is the *low* end, not the
high end — worth checking whether a floor/threshold rule (do not correct
downward near TCHP ≈ 0) removes that penalty, since negative TCHP is
unphysical. (b) The Gulf 63–90 m D26 dip deserves a look: it sits exactly where
the eastern/western branch split occurs (see A3), so it may be the same
Loop-Current-boundary confusion. (c) If distribution fidelity matters for the
downstream application, that is a separate objective (quantile loss or
variance inflation) and should be justified on its own terms, not as an error
fix.

### A2. Map the SSH contribution geographically — *status: done*

**Original note:** "ssh into a map as a function of lat and lon in the shap
plot 1 for the d26"

**Meaning.** Instead of only showing how strongly each model uses sea-surface
height on average, plot every Gulf profile at its own latitude/longitude and
colour it by how much SSH contributed to that profile's correction — global
model beside local model.

**Why it matters.** It converts an aggregate bar into a physical map: we expect
the Loop Current / eastern Gulf to light up strongly and the Bay of Campeche to
stay muted, which would confirm that the local model's advantage is
Loop-Current-driven rather than a generic fitting artefact.

**What is needed.** Reuse the out-of-fold SHAP matrices from the 2026-07-22
attribution run; scatter at profile positions with a diverging colour scale
shared between the two panels. Script item 2, figure `*_ssh_shap_map.png`.

**Results (2026-08-09 run).** The D26 map is the clearest figure we have
produced on this question. The Gulf-local panel traces the **Loop Current path
as a coherent red ribbon**: strongly positive SSH contributions (+10 to +22 m)
entering through the Yucatan Channel, curving through the eastern Gulf around
25–27 °N, and exiting through the Florida Straits, with the same signature on
the Campeche Bank and near the Yucatan shelf break. The western and northern
Gulf are uniformly negative (−5 to −12 m). The global panel over the identical
profiles is almost featureless pale blue: it applies a small negative SSH
adjustment nearly everywhere and never resolves the current.

This is the physical confirmation of the local model's advantage: it has
learned where the Loop Current is from the sea-surface-height field, and RTOFS's
D26 error is largest exactly along that path. Good candidate figure for the
seminar and for the paper.

### A3. Colour the dependence plots by a third feature — *status: done*

**Original note:** "color based on different parameters on the plot for the
contribution to see the different branches"

**Meaning.** The SSH dependence scatter for D26 splits into two distinct
branches at low SSH (one near −10 m, one near −5 m). Colour the points by other
features to reveal what separates the branches.

**Why it matters.** Branch structure in a contribution plot is interaction
structure by definition. Identifying the interacting feature tells us what
physical regime the model has learned to distinguish.

**What is needed.** Re-render the SSH dependence scatter coloured by longitude,
then by month. Script item 3, figure `*_ssh_dependence_coloured.png`.

**Already known (numerical check, 2026-08).** Splitting the low-SSH rows
(−0.3 ≤ SSH ≤ 0) at contribution −7.5 m and comparing feature means between the
two branches gives longitude as by far the strongest separator (standardised
difference 2.7), then latitude (1.9):

| | −10 m branch (n=556) | −5 m branch (n=422) |
|---|---|---|
| longitude | −88.4 (eastern Gulf) | −93.8 (SW Gulf / Campeche) |
| latitude | 26.6 | 22.7 |
| local D26 variability (1°) | 14.1 m | 8.9 m |
| mixed-layer thickness | 19 m | 30 m |

Interpretation: low sea level in the eddy-rich Loop Current sector implies a
much larger correction than the same sea level in the quiet southwestern Gulf.
The banding (rather than a smooth gradient) is a tree-model artefact: depth-4
trees respond in piecewise-constant steps.

### A4. Explain the global-vs-local contribution difference — *status: documented*

**Original note:** "why global vs local diff in shap plots"

**Meaning/answer.** Same features, same evaluation rows, same hyperparameters —
only the training pool differs. The global model must fit one response to SSH
that works everywhere on Earth, so its Gulf response is an average diluted by
regimes where SSH means less; the local model can specialise. Additionally the
attribution baseline differs: contributions are measured against each model's
own training distribution, so the global model's "expected output when SSH is
unknown" averages over global conditions and the local model's over Gulf
conditions. The comparison is therefore between *learned strategies*, not
between model-independent ocean properties.

**Where written down.** `docs/math/gom_attribution.md`, interpretation section.

---

## B. Robustness: does the conclusion survive a different model? (next)

### B1. Second model family — *status: queued*

**Original note:** "test another ml method to see if it comes up to similar
conclusion"

**Meaning.** Retrain on identical folds with a different algorithm (random
forest named explicitly; a linear or additive baseline is nearly free) and check
whether SSH still dominates in the Gulf and whether local still beats global.

**Why it matters.** A conclusion that survives a change of algorithm is a
statement about the ocean; one that does not is a statement about gradient
boosting. This is the cheapest credibility upgrade available before the
seminar.

**What is needed.** Reuse the blocked-forward folds and the Gulf subset;
train random forest (and optionally a linear model) on the same recipe; report
MAE side by side; compute permutation importance for the forest and compare the
feature ranking to the boosted model's contributions.

**Results (2026-08-09 run,** `OHC/output/gom_robustness_20260811/`**).** Both
conclusions survive the algorithm swap. Random forest Gulf MAE: full recipe
12.15 (TCHP) / 11.02 (D26); reduced set 11.82 / 10.87 — the lean-beats-full
ordering holds in the new model family too. And the forest's permutation
importance puts **SSH first for both targets** (then the local-anomaly
features, longitude), matching the boosted model's contribution ranking. The
SSH-dominance finding is therefore a property of the data, not of gradient
boosting. Random forest runs slightly behind boosted trees throughout
(~0.2–0.4 MAE), consistent with the literature on tabular data at this scale.

### B2. Resolution of the splits on the top features — *status: queued*

**Original note:** "look into how finely we can split into the random forest
only if we take the top parameters from the forest and see if we can ask the
model to look more strongly"

**Meaning.** Restrict to the top features (the survivors of the backward
elimination: month, SSH, local TCHP variability, TCHP anomaly for TCHP; month,
longitude, raw D26, SSH for D26), then allow the model more capacity — deeper
trees, more estimators, finer splits — so it can resolve the SSH response more
sharply instead of spending depth on weak features.

**Why it matters.** The elimination ladder already showed the lean model beats
the full recipe locally (TCHP 12.37 → 11.70; D26 11.23 → 10.63). Giving the lean
model more resolution is the natural next step, and the split thresholds
themselves become an interpretable description of the learned SSH response.

**What is needed.** Grid over depth and estimator count on the reduced feature
set; report the skill surface; extract and plot the distribution of split
thresholds on SSH.

**Results (2026-08-09 run).** More capacity does **not** help: the sweep over
depth {4, 6, 8} × estimators {300, 800} is won by the *smallest* settings
(TCHP: depth 4 / 300 trees, MAE 11.70; D26: depth 6 / 300 trees, 10.62 —
statistically indistinguishable from depth 4), and depth-8 or 800-tree variants
are uniformly worse. The split-threshold histograms show the winning models
already place hundreds of distinct splits along the SSH axis (408 for TCHP,
2,542 for D26), i.e. the SSH response is already finely resolved. Conclusion
for "can we ask the model to look more strongly": **the constraint is Gulf
sample size (~2.7k training rows), not model resolution** — more capacity just
overfits. The way to a sharper SSH response is more data (more years, gliders),
not deeper trees.

### B3. Small neural network on the top features — *status: queued*

**Original note:** "nn start with the most important parameters maybe
investigate it"

**Meaning.** Train a small dense network on the same reduced feature set.

**Why it matters.** Two reasons. First, it is the smoothness counterpoint to
trees: if the banded contribution structure becomes a continuous fan, that
confirms the bands are a tree artefact rather than physics. Second, it is the
natural stepping stone toward the architecture ideas in section C.

**What is needed.** Standardise inputs; small MLP (e.g. 2–3 hidden layers) on
the residual target; same folds; early stopping on a held-out slice of the
training block; compare skill and gradient-based sensitivities against the
boosted model's contributions.

**Results (2026-08-09 run).** A 64×64 network on the reduced set trails both
tree families clearly (TCHP 12.87, D26 12.42 — roughly a full MAE unit behind
boosted trees) and flips the bias positive. At ~2.7k training rows this is the
expected tabular-data outcome and it was an untuned first pass, but the
practical implication for section C stands: at current data volumes the
experts in any mixture-of-experts design should stay tree-based (or the gate
alone can be a small network); pure neural approaches become interesting only
once the training pool grows (more years, gliders, or the global MoE setting
where experts see far more rows).

---

## C. Architecture directions (research; roadmap, not this sprint)

### C1. Mixture of experts over regions — *status: roadmap, priority*

**Original note:** "moe in the different local regions"

**Meaning.** Instead of one global model or hand-drawn regional models, train
several expert models plus a learned gating network that decides, per profile,
which expert(s) to trust.

**Why it matters.** This is the principled generalisation of our strongest
empirical result: local beats global in the Gulf. A mixture of experts would
learn the regional partition from data rather than from our box definitions,
and could recover the local advantage everywhere without maintaining separate
regional pipelines. The 20° improvement maps
(`spatial-ablation-tile20-holdout`) are the motivating evidence.

**What is needed.** Expert models over the existing feature set; a gate driven
by location plus state; training on the same folds; evaluation both globally and
per named box; comparison against (a) global model, (b) per-basin local models.
Open questions: number of experts, gate inputs, and whether experts should be
region-specialised or regime-specialised.

**Results — v1 implemented (2026-08-09,**
`OHC/exploration/run_moe_regions.py`, outputs
`OHC/output/moe_regions_20260811/`**).** Design per the robustness findings:
tree-based experts, each trained on *all* rows with sample weights (own
region/regime 1.0, elsewhere 0.15 — a local specialist with a global prior).
Two gates compared: five geographic experts hard-gated by position (Gulf,
Atlantic, Indian, West Pacific, East/Central Pacific), and six k-means regimes
in physics-state space (no lat/lon) soft-gated by centroid distance, so the
regime gate is defined everywhere including float deserts.

Out-of-fold results, full global table:

| variant | TCHP MAE | D26 MAE | TCHP GoM | D26 GoM |
|---|---|---|---|---|
| global model | 11.397 | 10.755 | 13.05 | 11.81 |
| **MoE geographic** | **11.281** | **10.634** | **12.42** | 11.92 |
| MoE learned regimes | 11.327 | 10.683 | 13.04 | 11.84 |
| (dedicated Gulf-local, for reference) | – | – | 12.37 | 11.23 |

Readings: (1) both MoE variants beat the single global model everywhere-on-
average, and the geographic version sets **new overall bests** for both
targets. (2) In the Gulf, the geographic MoE recovers essentially all of the
dedicated local model's TCHP gain (12.42 vs 12.37, bias −5.7 → −3.9) from
within one unified system — the original motivation fulfilled. (3) It does
*not* yet recover the local D26 gain (11.92 vs 11.23), suggesting the 0.15
global-prior weight is too strong for D26's Gulf specialisation. (4) The
learned regimes recover latitude-band / stratification structure without being
given coordinates (`learned_regime_map.png`) — scientifically pleasing — but
no Gulf-specific regime emerges, so the regime gate does not help the Gulf.

**Next tuning steps.** Prior-weight sweep (0.05–0.3, possibly per-region);
combine gates (geographic experts + regime soft weights); regime count K sweep;
regime features that can isolate semi-enclosed basins without raw coordinates;
blend expert with global prediction instead of hard selection.

### C2. Gridded feature maps with attention — *status: roadmap*

**Original note:** "every feature having a 2d map even 10 deg grids and
attention layers"

**Meaning.** Present each feature as a 2-D map (e.g. 10° cells) and let an
attention mechanism learn which spatial context matters, rather than
hand-engineering neighbourhood summaries.

**Why it matters.** Our hand-built neighbourhood features (local variability,
gradients, anomaly) already gave a consistent gain; letting the model learn its
own spatial summaries is the systematic version of that idea.

**What is needed.** A gridded training representation (the pipeline is currently
point-collocated), which is a substantial data-engineering step. Related prior
art already surveyed in `OHC/SPATIAL_FEATURES_DIRECTIONS.md` (Tier 3).

### C3. Graph-based methods — *status: roadmap*

**Original note:** "maybe graph based methods"

**Meaning.** Represent profiles/regions as graph nodes with edges by proximity
or dynamical similarity, and use a graph neural network.

**Why it matters.** Graphs handle irregular sampling natively, which suits
scattered float positions better than a grid.

**What is needed.** Same gridded/graph data-engineering prerequisite as C2;
worth a literature pass before committing.

---

## Result summary

Output directory: `OHC/output/gom_diagnostics_20260811/`
(Drive: `HHP-gom-diagnostics-2026-08-11`). Script:
`OHC/exploration/run_gom_diagnostics_followup.py`.

- **A1 conditional skill — hypothesis overturned.** The correction gets
  *better*, not worse, as observed values rise (TCHP benefit reaches −22 kJ/cm²
  MAE in the top bin globally). The only weakness is the lowest bin for both
  targets, plus a narrow 63–90 m dip for D26 inside the Gulf. The distribution
  plot that prompted the question shows narrowing, which is a different
  property from pointwise accuracy.
- **A2 SSH maps — strongest confirmation so far.** The Gulf-local model's SSH
  contributions trace the Loop Current as a coherent path; the global model
  produces a nearly featureless field over the same profiles.
- **A3 coloured dependence — branch cause confirmed.** Longitude separates the
  two low-SSH branches (standardised difference 2.7): eastern Gulf gets ~−10 m,
  southwestern Gulf ~−5 m for identical sea level.
- **A4 explanation — written up** in `docs/math/gom_attribution.md`.

## Next actions

1. Run B1–B3 (random forest, capacity sweep on reduced features, small neural
   network) as one robustness batch on the existing folds.
2. Then C1 (mixture of experts over regions) as the main modelling direction.
3. Send Dr. Jacobs the A1 correction: his "raw better above 50" reading is not
   supported pointwise, and the low-end penalty is the actual issue worth
   discussing.
