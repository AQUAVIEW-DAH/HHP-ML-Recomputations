# OHC / exploration — SIDE EXPLORATIONS (not the main method)

Everything in this directory is explicitly a **side exploration** relative to the
main HHP correction line. The mentor constraint for the main method is:
**no data dependencies beyond RTOFS** (plus Argo as truth). Scripts here violate
that constraint on purpose to measure what those extra dependencies would buy.

Current side tracks:

1. `build_gofs31_daily_ohc_fields.py` + `run_gofs31_pilot_2015.sh`
   - HYCOM GOFS 3.1 **reanalysis** (expt 53.X, GLBv0.08, AWS Open Data bucket
     `hycom-gofs-3pt1-reanalysis`) reduced to daily 2D TCHP/OHC/D26/SST/SSH
     fields, pilot year 2015.
   - Why: the public RTOFS archive only starts 2024-01-31; the reanalysis spans
     1994-2015 and could add a decade of Argo-collocatable history (2005+).
   - Caveats to keep in view (mentor: "every model has its own bias"):
     - GOFS 3.1 reanalysis is HYCOM+NCODA, a *different model realization* than
       operational RTOFS; error statistics learned on one do not automatically
       transfer to the other. Treat cross-model transfer itself as the research
       question, not an assumption.
     - NCODA assimilates Argo profiles (via Improved Synthetic Ocean Profiles),
       so Argo-minus-reanalysis residuals are **not independent** of the truth
       source and will be biased small vs the RTOFS-forecast regime.
     - Reanalysis (2015) vs operational nowcast (2024+) also differ in altimetry
       constellation and SST products assimilated.
   - Verdict criteria: does a correction model trained on GOFS-3.1-era
     collocations transfer to RTOFS 2024-2025 collocations better than chance?
     Do stencil-feature relationships look alike across the two models?

2. Tier-1 predictors from `../SPATIAL_FEATURES_DIRECTIONS.md` (WOA climatology
   priors, satellite SSHA) as they get implemented — each adds a non-RTOFS
   dependency and stays here until/unless the mentor promotes it.

Main-line work (Tier-0 neighborhood stencils, which are pure RTOFS) lives in the
parent `OHC/` directory, not here.
