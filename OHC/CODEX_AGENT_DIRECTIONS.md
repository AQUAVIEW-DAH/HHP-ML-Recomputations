# Codex agent directions — ML correction of RTOFS (OHC / TCHP / D26)

> Source: research discussion session 2026-06-08 (research-hub `ocean-aquaview` track).
> These are directions + open decisions for the agent continuing the OHC ML-correction work.
> Run order is by decisiveness-per-effort. Everything in P0/P1 runs on the existing
> `argo_rtofs_collocated_2024_2025.parquet` — no new downloads, no 3D archives needed.

## Strategic framing (decided this session)

The corrected field is **mainly meant to correct RTOFS / the ocean model *everywhere*** (goal
**b**), including the float-sparse ocean and toward hurricane-intensity guidance — not only a
gridded product served near float coverage (goal **a**, secondary). Three consequences drive
the experiments below:

1. **lat/lon-dependent skill is suspect.** `abs_lat`/`lat`/`lon` are top SHAP features and the
   only train/test separation is date (2024 vs 2025). A model can score well on the
   year-holdout by *memorizing where RTOFS is wrong* — which is useless in the ~49% of the
   global grid with no Argo within 100 km. Features (lat/lon in/out) and the CV split
   (date vs spatial/platform grouping) are **orthogonal** knobs; the current design changed
   both vs the MLD project, confounding the comparison. NB: MLD's platform grouping is a
   leakage *guard*, not per-platform learning — MLD learns a pure state→bias map (no coords).
2. **The eval is blind to the deployment regime.** Skill is measured only at Argo points;
   the field is deployed where there are none. The year-holdout MAE says nothing about the
   float-desert, and trees *flatten* (don't extrapolate) out-of-support → risk of asserting
   a learned bias where RTOFS was fine = do-harm.
3. **The target is threshold-censored.** `teos_ohc.py:116`: when the column never crosses
   26 °C, compute returns **None for TCHP, OHC AND D26** (not 0), keeping only
   `surface_temp_c`; those rows are dropped by the finite-delta filter → ~70% of collocated
   Argo (41,347 → ~12,543) unused; model trained only on warm-pool states. **Not symmetric:**
   where SST<26 °C, TCHP=0 is physically real ("no hurricane fuel") and *should* be recoded 0;
   D26 is genuinely undefined there.

## P0 — diagnostics that decide direction (no retrain)

- **P0.1 — 4-cell threshold breakdown.** Partition collocated pairs by
  {Argo crosses 26 °C} × {RTOFS crosses 26 °C}. Output a 2×2 count table; in the
  `only-Argo` and `only-RTOFS` cells report RTOFS SST and TCHP bias and row counts.
  *Decision gate:* `neither` dominates the dropped rows ⇒ censoring is cosmetic, a hurdle
  model suffices (P2.7). Disagreement cells large ⇒ censoring eats the TC-genesis-band signal
  ⇒ go threshold-free (P2.6).
- **P0.2 — skill vs distance-to-nearest-training-float.** For each 2025 test row, compute km
  to the nearest 2024 training float; bin year-holdout MAE (TCHP & D26) by
  [0–50, 50–100, 100–250, 250–500, >500] km. Report the decay curve. This quantifies the
  goal-(b) deployment-regime problem.
- **P0.3 — float-clustering / leakage audit.** profiles-per-platform, per-platform spatial
  spread, and an estimate of how much the date-only split is inflated by same-float
  near-duplicates on adjacent dates.

## P1 — lat/lon causal ablation (re-run existing XGBoost, no new data)

- **P1.4 — 3-way generalization test** on the year-holdout. Report TCHP & D26 MAE/RMSE/bias for:
  - (a) lat/lon + date split  *(current baseline)*
  - (b) lat/lon + **leave-one-region-out** (tile the ocean into ~10–15° boxes; hold out whole
    tiles round-robin)
  - (c) **drop lat/lon/abs_lat**, state-only features (model_interp_tchp/d26 + RTOFS physics +
    season/cyclic-time) + date split
  *Interpretation:* gap (a)−(c) = skill attributable to geography lookup; (b) collapse =
  memorization that won't transfer; (c) ≈ (a) = genuinely state-conditioned (good — keep (c)).
- **P1.5 — honest split.** Re-run the headline eval with spatial/platform grouping instead of
  date-only; report the delta vs the current number.

## P2 — strategic fork (after P0.1)

- **P2.6 — threshold-free target pilot (preferred for goal b).** On the same collocated
  profiles, build OHC integrated to **fixed depths** (0–100 m and 0–700 m; defined everywhere,
  no 26 °C censoring) for both Argo and RTOFS. Train the delta correction on that, then
  **derive** TCHP & D26 from the corrected field. Checks: (i) recovers the dropped ~70%?
  (ii) derived TCHP↔D26 self-consistent (fixes the two-independent-targets problem)?
  (iii) skill vs direct-TCHP-correction on the warm-pool subset?
- **P2.7 — hurdle / two-part** (if P0.1 shows many clean zeros). Classifier for warm-pool
  presence (SST ≥ 26 °C) + magnitude regressor on the nonzero set; recode sub-threshold
  **TCHP = 0** (NOT D26). Compare to current.
- **P2.8 — independent desert validation.** Compare corrected-RTOFS TCHP/D26 against a gridded
  reanalysis (**GLORYS12 / ORAS5 / EN4** — already indexed in AQUAVIEW) and/or AOML/UAF gridded
  TCHP_D26, focused on float-sparse regions. Reanalysis ≠ truth; use to flag gross divergence
  (do-no-harm), not as a skill score.
- **P2.9 — do-no-harm guard.** Shrink predicted delta → 0 as distance-to-training-support grows
  (or via an OOD / leaf-coverage flag); verify corrected RTOFS never degrades vs raw in
  held-out regions.

## Pipeline hygiene

- Retain a thin global 3D RTOFS subset so the blocked profile-feature phase (N², steric proxy,
  density-jump) can proceed.
- Pick ONE canonical eval (year-holdout + spatial grouping); **retire the locked-protocol OOF**
  — its folds mix years (fold 3 trains through 2025-02-26 then validates on 2025-05-14…08-31),
  so it is not a clean future-holdout.
- Add a **field-level** validation. Current eval is collocated-point only; corrected global maps
  are rendered but never validated as fields.

## Open decisions — need human / mentor input (do not assume)

- Confirm deliverable = **correct-RTOFS-everywhere (b)** vs **near-float product (a)**. Sets the
  eval bar and the lat/lon feature decision.
- Is **TCHP-as-direct-target a fixed constraint** (AOML framing), or may we move to the
  threshold-free target + derive (P2.6)?
- What **independent reference** is acceptable for validating in the float-desert?
