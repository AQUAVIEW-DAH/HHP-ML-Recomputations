# HHP-Prediction — Working Notes

Scratchpad for in-progress decisions. Promote to docs/ when finalized.

## Physics / definitions

- **TCHP** = ρ · Cₚ · ∫[0 → D26] (T(z) - 26°C) dz
- **D26** = linearly interpolated depth where T = 26.0°C (from a profile with at least two bracketing samples)
- **ρ** — compute from T, S using TEOS-10 / GSW along the profile
- **Cₚ** — compute from TEOS-10 / GSW along the profile
- **Units** — output as kJ/cm² (divide J/m² result by 10⁷)

## Data decisions

- **Model:** RTOFS 3dz f006 6-hourly, `rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc` from `noaa-nws-rtofs-pds` S3
  - 40 depth levels 0-5000 m, 19 levels between 0-200 m — excellent TCHP coverage
  - Regional cut covers -105 to -40°W — reaches most of Atlantic MDR (full MDR extends to -20°W; eastern MDR would need different regional file or global RTOFS)
- **In-situ:** Argo GDAC only for the active training set, with depth QC requiring `max_depth_m ≥ 200`
  - Reject profiles that don't reach the 26°C isotherm (i.e., profiles where T at max_depth > 26°C — surface-only in very warm water)
  - Keep same-day RTOFS pairing pattern
- **Hurricane tracks:** IBTrACS via AQUAVIEW MCP — `get_item(collection="NOAA_AOML_HDB", item_id="IBTrACS_since1980_1")`
- **Validation:** AQUAVIEW `noaa_aoml_a121_e182_cf9a` (UAF collection — TCHP_D26 Fields 0.25°, 2021-present) for operational reference comparison

## Open questions

- Do we need SSH for the first feature set, or is D26 + integrated heat content + SST enough? Lean: no SSH for MVP; add later after baseline model works.
- Full ρ(z) integral or surface-constant-ρ approximation? Start with surface constant; verify error is <5% of TCHP value for MDR profiles.
- Holdout strategy: pick one full hurricane event (e.g., Helene Sep 24-28) and hold out the pre-track + track-passage window for that storm? Or hold out all of Oct 2024?

## Event candidates for replay sandbox

- **Beryl** 2024-06-28 to 2024-07-09 — earliest Cat 5 on record in Atlantic; passed through Caribbean and GoM
- **Helene** 2024-09-24 to 2024-09-28 — GoM landfall, catastrophic inland flooding
- **Milton** 2024-10-05 to 2024-10-10 — rapid intensification in central GoM, landfall Florida west coast

Milton is the cleanest RI case — explosive intensification over high-TCHP water.

## Build order

1. `hhp_core.py` — D26, TCHP integral, density, unit conversion
2. Quick local sanity check: pick a real RTOFS file, compute TCHP field over GoM, eyeball against known Loop Current heat pattern
3. `ml/sources/argo_gdac_source.py` — fork from MLD, relax depth QC, widen bbox
4. `ml/sources/wod_source.py` — same treatment
5. `ml/features.py` — D26, integrated heat content, kinetic energy, SST
6. `ml/processing/build_hhp_rtofs_2024.py` — training data builder over 2024 hurricane season
7. AQUAVIEW integration for IBTrACS track retrieval (`ml/sources/ibtracs_source.py`)
8. Validation comparator against AOML gridded TCHP_D26 product
9. `hhp_pipeline.py` + `api.py` + minimal dashboard

## Anti-patterns to avoid (carried forward from MLD experience)

- Don't trust raw observation counts — audit after every QC step
- Never use random train/test split; always grouped by platform/cruise
- Don't claim operational performance from random-split metrics
- Save every intermediate training CSV, not just the final — preserves the analysis trail
