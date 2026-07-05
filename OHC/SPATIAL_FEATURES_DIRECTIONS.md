# Spatial features for the HHP correction — brainstorm + literature directions

> Source: mentor request 2026-07-05 (Dr. Jacobs): the 20° patch diagnostics show clear
> regional / basin / hemispheric error tendencies; incorporate spatial structure into the
> ML pipeline, either as raw feature columns or as a spatially-aware model.
> This note reconciles that request with the earlier goal-b critique (lat/lon
> memorization does not transfer to the float-sparse ocean) and maps the option space
> found in the literature.

## 1. Framing: locality is signal, coordinates are a shortcut

The patch plots show real regional structure (e.g., raw RTOFS TCHP bias −24 kJ/cm² in the
Bay of Bengal vs −2 in the Gulf of Mexico). Two ways to let the model use it:

- **Location lookup** (raw lat/lon, region one-hots, distance-to-training-float):
  works at collocated points, cannot transfer to the ~49% of ocean without nearby Argo,
  risks asserting learned bias where RTOFS was fine (do-harm).
- **State/context lookup** (what the ocean *looks like* there: mesoscale context,
  climatological background, altimetry anomaly): available everywhere, generalizes,
  and *explains* the regional tendencies instead of memorizing them.

June 2026 methodology tests showed tile-holdout did **not** collapse, so the current model
is not pure geography memorization — but the goal-b deployment regime still demands
state/context features first, coordinates last.

Precedent for this framing: ECMWF's operational model-error ANN (Bonavita & Laloyaux 2020,
now in IFS) uses state predictors with location only as weak ancillary input; GFDL's
sea-ice increment CNN (Gregory et al. 2023/24) is fully coordinate-free via shared conv
filters. Nobody appears to have published ML correction of TCHP/D26 specifically — open niche.

## 2. Option ladder (decisiveness per effort)

### Tier 0 — raw feature columns, existing XGBoost, data already on disk (days)

- **S0.1 Neighborhood-context stencils** from our own reduced global daily fields
  (`/data/suramya/rtofs_global_ohc_fields_*`, 3298×4500 TCHP/D26/SST/OHC):
  for windows of ~0.5°, 1°, 2° around each collocation point compute local mean, std,
  gradient magnitude, Laplacian of TCHP/D26/SST. Encodes eddy/front/quiescent regime —
  the mesoscale context a point sample misses. RTOFS-side only, defined everywhere.
  This is the single cheapest way to answer "regional tendencies" with physics, and is
  the tabular analogue of what the CNN-correction papers get from convolutions.
- **S0.2 Better coordinate encodings** (only if coordinates are kept at all):
  3D unit vector (x, y, z) instead of raw lat/lon (kills dateline/pole artifacts);
  optionally a *low-frequency-only* spherical-harmonic basis (Rußwurm et al. 2023,
  `locationencoder`) so the model can express smooth basin-scale offsets but not
  pinpoint lookup. Keep out of the goal-b headline recipe unless tile-holdout stays flat.
- **S0.3 (flagged, deprioritized) RFSI-style neighbor features** — values/distances of
  k nearest training residuals (Sekulić et al. 2020). Strong for a near-float gridded
  product (goal a) but is location memorization by construction; only revisit if the
  deliverable decision flips to goal a.

### Tier 1 — new globally-complete predictor data (1–2 weeks)

- **S1.1 Climatology-anomaly features** (WOA23 and/or Roemmich-Gilson Argo climatology):
  climatological D26/D20/MLD/OHC at (point, month), plus `RTOFS − climatology` anomalies.
  The operational AOML/NESDIS TCHP products are literally built on
  climatology + altimetry (Goni & Trinanes reduced-gravity method), so
  climatology-relative features are the domain-standard, gap-free way to encode
  "where am I" physically. Small download, static per month.
- **S1.2 Satellite SSHA/ADT (CMEMS DUACS L4)** sampled at (point, date):
  every operational subsurface-from-surface method (AOML, NESDIS SOHCS, Navy
  MODAS/ISOP) leans on SSHA as the dominant predictor of D20/D26/OHC; DL
  reconstruction papers likewise rank SSH first in feature importance.
  Extra leverage for residual learning: `satellite SSHA − RTOFS SSH anomaly` directly
  measures where RTOFS's mesoscale field is displaced — plausibly the strongest
  single new feature for our residual target. Needs CMEMS account + daily L4 download.
- **S1.3 Static bathymetry features** (ETOPO/GEBCO): ocean depth, shelf flag,
  distance to coast. Cheap; matters where the reduced-gravity/two-layer picture breaks.

### Tier 2 — spatially-aware model, moderate change (2–4 weeks)

- **S2.1 GPBoost** (Sigrist, JMLR 2022; github.com/fabsig/GPBoost): boosted trees for the
  state→bias mean + a Gaussian-process spatial random effect (Vecchia approximation is
  fine at our ~12k rows). The GP absorbs smooth spatially-correlated residual structure
  and — key property — **shrinks toward the tree mean away from data**, which is an
  automatic do-no-harm guard in float deserts. Python API; mostly drop-in for our
  tabular pipeline.
- **S2.2 Regression kriging two-stage** (pykrige / verde): keep the current XGBoost,
  krige its out-of-fold residuals, add the kriged surface; kriging variance doubles as
  an uncertainty / applicability mask. Even less invasive than GPBoost; global
  stationarity of the residual variogram is the main caveat (fit per basin if needed).
- Benchmarks in the spatial-stats literature consistently find GP/kriging hybrids beat
  both plain ML and deep learning under sparse uneven sampling — our regime exactly.

### Tier 3 — architecture overhaul (only if Tiers 0–2 plateau)

- **S3.1 ConvNP / DeepSensor** (alan-turing-institute/deepsensor): gridded RTOFS context
  → off-grid Argo targets with calibrated uncertainty. This matches our geometry
  exactly (fields in, sparse points out) and is coordinate-free by construction, but it
  is a full model rewrite and needs the common-grid training story anyway.
- **S3.2 Gridded UNet increment learning** (Gregory-style DA-ML): requires building the
  common-grid target field first — the "significant overhaul" path; revisit after the
  threshold-free target question (P2.6) is settled.

## 3. Evaluation guardrails (mandatory for any tier)

1. **Spatial block CV** as a first-class eval next to the year-holdout: reuse the June
   tile-holdout harness (or verde `BlockKFold` / `spacv`) so every new spatial feature
   must win under leave-region-out, not just date-holdout. This is the memorization
   detector.
2. **Area-of-Applicability mask** (Meyer & Pebesma 2021, CAST): flag grid cells whose
   predictor vector is dissimilar to all training data; outside AOA fall back to raw
   RTOFS (implements P2.9 do-no-harm with an accepted method).
3. Report per-named-box metrics (now automated in the density diagnostics) for each
   feature ladder step, so "did the Bay of Bengal bias actually get explained by state
   instead of location" is directly visible.

## 4. Recommended sequence

1. **S0.1 stencil features now** (data on disk; new builder script
   `build_rtofs_neighborhood_features_2024_2025.py`), rerun the locked/tile-holdout
   semi-ablation with a `plus_neighborhood` recipe.
2. **S1.1 climatology features** next (WOA23 monthly D26/OHC prior + anomalies).
3. **S1.2 CMEMS SSHA** (needs credentials/download plan) — expected top performer.
4. **S2.1 GPBoost pilot vs S2.2 kriged-residual** on identical folds; adopt whichever
   wins under spatial CV.
5. Only then revisit Tier 3 overhaul.

Steps 1–2 are worth doing against the current table immediately; rebuild against the
post-backfill full-calendar table when it lands.

## 4.1 Status update 2026-07-05

- Mentor decision recorded: **main method stays RTOFS-only** (no new data
  dependencies). Everything beyond Tier 0 therefore runs as an explicit side
  exploration under `OHC/exploration/` (user decision: try all tiers anyway and
  compare, clearly marked as side work).
- **Tier 0 implemented and evaluated**: `build_rtofs_neighborhood_features_2024_2025.py`
  builds the stencil features (~73 s over 90 dates);
  `run_locked_xgb_physics_semi_ablation.py` gained `NEIGHBORHOOD_CORE` and two
  `*_plus_neighborhood` recipes. Locked OOF result: **new best for both targets**
  - TCHP `global_pruned_plus_neighborhood`: MAE 11.14 (was 11.64), bias -0.09
  - D26 `drop_both_lat_interactions_plus_neighborhood`: MAE 10.50 (was 10.76), bias -0.16
  - Still pending: tile-holdout (spatial CV) confirmation per the guardrail
    before promoting these to the recommended recipes.
- **HYCOM GOFS 3.1 reanalysis side track** started (`OHC/exploration/`):
  AWS Open Data bucket `hycom-gofs-3pt1-reanalysis`, 1994-2015 only (NOT a
  2024+ RTOFS replacement); value = adding ~a decade of Argo-collocatable
  history. Caveats logged in `exploration/README.md` (different model bias;
  NCODA assimilates Argo so residuals are not independent). Pilot = 72 dates
  of 2015 reduced to 2D OHC fields in `/data/suramya/gofs31_ohc_fields_2015`.
- GPBoost installed into `hhp-env` for the Tier-2 pilot (not yet run).

## 5. Open decisions for the mentor

- Confirm goal-b (correct-everywhere) still rules: it decides whether RFSI-type
  neighbor features (S0.3) are out of bounds and how heavily coordinates may be used.
- Is adding *observational* satellite SSHA as a predictor acceptable for the intended
  deployment (it makes the corrected product satellite-dependent, like AOML's), or must
  features stay purely RTOFS-internal?
- Which climatology is preferred as the background prior: WOA23, RG-Argo, or both?

## 6. Key references / repos

- Bonavita & Laloyaux 2020 JAMES; Farchi et al. 2024 (IFS operational ML error correction)
- Gregory et al. 2023 JAMES / 2024 GRL + github.com/William-gregory/DA-ML
- Liu, Bracco & Brajard 2023 JAMES (HYCOM mesoscale bias correction)
- Watt-Meyer et al. 2021 GRL (RF nudging correction, capped tendencies)
- Goni & Trinanes AOML TCHP method; NESDIS SOHCS ATBD (SSHA + climatology → D20/D26/OHC)
- Fox et al. 2002 (MODAS); Townsend et al. 2015 (ISOP) — Navy SSH+SST → synthetic profiles
- Sigrist 2022 JMLR GPBoost + github.com/fabsig/GPBoost
- Sekulić et al. 2020 (RFSI); Hengl et al. 2018 (RFsp)
- Meyer & Pebesma 2021 (AOA / CAST); verde BlockKFold; spacv
- Rußwurm et al. 2023 spherical-harmonic location encoding (locationencoder)
- DeepSensor (ConvNP) github.com/alan-turing-institute/deepsensor; OceanBench; 4DVarNet
