# Full-year holdout: options (for Dr. Jacobs)

Question from the 2026-10-08 meeting: why not keep a full 12 months as the holdout?

What we have:
* Development data: RTOFS + Argo, 2024-01-31 to 2025-12-31 (23 months). Every choice so far (features,
  model settings, MoE) was made by testing on blocks inside this period.
* 2026 holdout: built and untouched, 2026-01-01 to 2026-09-22 (265 days, 32,710 warm profiles). RTOFS
  fields exist up to 2026-10-07.
* Full-year checks already done inside the development data: train on one calendar year, test on the
  other, and (2026-10-15) 12-month windows starting in Feb, May, Aug, Nov 2024 and Jan 2025. In every
  window the ranking is the same: adding RTOFS's recent history helps most (-0.37 to -0.53), Dr. Jacobs'
  16 features trail the full recipe (+0.14 to +0.40), and removing the calendar features changes the error
  by -0.16 to +0.10 with the sign depending on the window. So results do not depend on where the year
  starts, except for the calendar features, which are unstable.

Options:
1. Wait for all of 2026 (recommended). Freeze the recipe, train on 2024-2025, test once on Jan-Dec 2026.
   Untouched, forward in time, and a full seasonal cycle including the whole hurricane season. Ready about
   mid-January 2027 (Argo and RTOFS for December 2026 need a few days to arrive).
2. Test once now on Jan 1 - Sep 22, 2026 (9 months), then report Oct-Dec 2026 separately when it arrives.
   Fast, but the main number misses Oct-Dec, and the later full-year figure would partly reuse months
   already looked at.
3. Hold out 2025 and develop on 2024 only. Not recommended: 2025 is no longer untouched (all choices were
   tested on it), so every selection would have to be redone on 11 months of 2024 data.

Either way, the recipe should be frozen first. It is still moving: the RTOFS recent-history features
(-0.45) and the step-by-step build-up (kept steps beat the full recipe by 0.12 for TCHP and 0.21 for D26)
both change it.
