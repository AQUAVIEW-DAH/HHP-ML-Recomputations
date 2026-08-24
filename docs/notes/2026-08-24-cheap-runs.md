# Cheap-runs batch — review date 2026-08-24

Follow-ups from the latest meeting notes (see
`2026-08-11-meeting-followups.md` for the earlier items). Outputs are under
`OHC/output/*_20260824/`; Drive folder `HHP-analysis-2026-08-24/` with one
subfolder per analysis.

## 1. Learning curve — data volume is NOT the current limit

`run_learning_curve.py` → `learning_curve_20260824/`. Retraining the best
single-model recipe on 25/50/75/100% of the training dates changes global MAE
by less than 0.1 in either target (TCHP 11.40 → 11.40; D26 10.85 → 10.76), and
the Gulf shows the same flatness. **Conclusion: at the current feature set,
quadrupling same-kind data buys almost nothing** — the constraint is the
information content of the inputs (representativeness of point collocation,
submesoscale noise), not sample count. Caveats: the curve subsamples within
the same two years, so genuinely new *years* or new *instruments* (gliders)
could still add diversity the subsample cannot; and the Gulf-local models sit
at ~3k rows where the earlier capacity result still applies. Practical
consequences: the multi-year HYCOM download is NOT justified for skill on the
current model (its value stays cross-model validation); gliders matter as
independent truth more than as skill fuel; progress comes from new
information (SSH anomaly, richer features) or architecture.

## 2. Expert cross-evaluation matrix — the MoE partition is validated

`run_expert_cross_eval.py` → `expert_cross_eval_20260824/*_expert_cross_matrix.png`.
Every diagonal cell is its column's best: each region is best served by its
own expert. The Gulf column shows the largest specialisation gap (own expert
11.76 m for D26 vs 12.75–14.02 for foreign experts); the Atlantic, Indian and
E-Pacific experts are nearly interchangeable with each other away from home.
Suggests a possible v3 simplification (merge near-interchangeable experts)
and confirms the Gulf needs its own.

## 3. Gulf expert vs dedicated Gulf-only model

Same script, `gulf_head_to_head.csv` and `*_gulf_expert_vs_dedicated_shap.png`.
On identical global folds: TCHP — the MoE expert now slightly *beats* the
dedicated model (12.41 vs 12.46); D26 — the dedicated model keeps a 0.34 m
edge (11.42 vs 11.76) and carries much less bias (−0.55 vs −1.98). The SHAP
comparison shows what the 0.05 global prior still suppresses for D26; the
remaining gap is the cost of the global prior, not of the architecture.

## 4. Gate-weight maps

`build_gate_weight_maps.py` → `gate_weight_maps_20260824/`. Dominant-regime
and gate-confidence maps for the winning K per target. The gate is soft:
median top weight 0.24 at K=6 (uniform would be 0.17) and 0.14 at K=12
(uniform 0.08) — the regime branch works by blending several experts rather
than picking one, consistent with its role as the smaller half of the blend.

## 5. Tree diagrams for Dr. Jacobs

`build_tree_diagrams.py` → `tree_diagrams_20260824/`: one depth-3 and one
depth-4 random-forest tree (sklearn rendering, feature names, thresholds,
leaf values, node sizes) and the first tree of the boosted model (manual
layout of the exported structure), all on the Gulf D26 reduced feature set
from a real locked-fold training block.

## 6. SSH anomaly feature (his suggestion) — complement, not replacement

`build_ssh_anom_feature.py` (+ climatology from all 701 days) and two new
locked-ablation recipes. Result: **adding** `ssh_anom_monthly_m` alongside raw
SSH gives the best D26 recipe yet (10.733 vs 10.755) and is neutral for TCHP;
**swapping** it in for raw SSH hurts both targets (D26 10.94, TCHP 11.48).
Reading: absolute SSH encodes basin structure the anomaly discards, and trees
can already synthesise anomaly-like behaviour from raw SSH plus location, so
the explicit anomaly is a small refinement. For the discussion with
Dr. Jacobs: his instinct is validated as an addition; the raw field should
stay.

## Not run (by design)

- Multi-year HYCOM pull: de-prioritised by the learning-curve result; awaiting
  discussion.
- Dr. Hill / AquaView slides: awaiting clarification of scope and audience.
