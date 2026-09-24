# Is the learned correction two physical operations? — 2026-09-24

Script `OHC/exploration/run_correction_decomposition.py`; outputs
`OHC/output/correction_decomposition_20260924/`. Warm rows only (the
interpolation bug cannot enter), locked folds, same benchmark as 11.40/10.76.

Hypothesis: correction ~ B(position, season) + beta * anomaly, where B is a
position-and-season bias map and the anomaly is RTOFS's own departure from its
1-degree neighbourhood mean.

## 1. Two operations recover most of the full model

| model | TCHP MAE | % of full model's gain | D26 MAE | % |
|---|---|---|---|---|
| raw RTOFS | 16.61 | 0 | 14.92 | 0 |
| B only (bias map) | 12.56 | 78 | 12.15 | 66 |
| **B + beta * anomaly (one extra number)** | **12.00** | **88** | **11.51** | **82** |
| B + beta by activity (five numbers) | 12.00 | 88 | 11.52 | 81 |
| trees on the same two ingredients | 11.94 | 90 | 11.40 | 84 |
| full recipe (34/35 inputs) | 11.40 | 100 | 10.74 | 100 |
| MoE blend | 11.19 | 104 | 10.55 | 104 |

* A bias map plus ONE damping coefficient recovers 88% (TCHP) and 82% (D26)
  of the full model's improvement over raw RTOFS.
* A straight line in the anomaly is nearly as good as trees on the same two
  ingredients (12.00 vs 11.94; 11.51 vs 11.40): the damping is close to linear.
* Having beta vary by eddy activity adds nothing (5 numbers = 1 number).
* Because the closed form has a single parameter beyond B, it is essentially
  immune to the selection-bias concern that affects every tuned configuration.

## 2. The damping coefficient

| | beta | 95% CI | fraction of RTOFS local structure kept (1+beta) |
|---|---|---|---|
| TCHP, all warm rows | -0.88 | [-0.93, -0.84] | 12% |
| TCHP, excluding near-edge (temp excess < 1 C) | -0.91 | [-0.96, -0.87] | 9% |
| D26, all warm rows | -0.81 | [-0.84, -0.78] | 19% |
| D26, excluding near-edge | -0.85 | [-0.88, -0.81] | 15% |

Reading: against point Argo profiles, only about 10-20% of the local mesoscale
structure RTOFS claims (departures from its own 1-degree mean) is borne out.
In signal-to-noise terms, the RTOFS local anomaly is roughly 85-90% error
variance. The best correction therefore replaces most of RTOFS's local
structure with its neighbourhood mean. Estimated from out-of-fold bias-map
residuals, so B cannot absorb the anomaly signal. Removing near-edge rows
(where the stencil anomaly is biased) makes the damping slightly stronger, so
the edge bias is not driving it.

Note: "borne out" is measured against point profiles. Argo representativeness
noise sits in the target, not the regressor, so it does not bias beta.

## 3. Amplitude or position? Mostly amplitude-like, with a position component

* **beta by eddy-activity quintile.** TCHP: flat at -0.76 to -0.78 across the
  four lower quintiles, then -0.96 [-1.02, -0.90] in the most active quintile,
  which does not overlap the others. D26: no monotone trend (-0.88, -0.88,
  -0.84, -0.94, -0.76). A flat beta is the amplitude signature; the TCHP drop
  in the most energetic fifth of the ocean is a position signature.
* **leftover error vs gradient**, holding |anomaly| and activity fixed. TCHP:
  +0.36 per standard deviation [+0.13, +0.58], significant. D26: +0.13
  [-0.03, +0.29], not significant.

Verdict: over most of the ocean the error behaves like a uniform
over-amplification of local structure. In the most energetic TCHP regimes an
additional displacement component appears: RTOFS features there are also in
the wrong place. D26 shows no clear position signature.

## 4. Asymmetry (flagged, not yet trusted)

Cold dips are damped more than warm bumps (TCHP -0.94 vs -0.84; D26 -0.90 vs
-0.75). Caution: near-edge rows carry a negatively biased anomaly and fall
mostly in the "cold dip" group, so this split has not been separated from the
stencil edge bias. Check before quoting.

## Caveats

* B is fitted by trees on position and season, so it is a smooth bias
  climatology but not itself a closed-form expression.
* The remaining 12% (TCHP) and 18% (D26) needs the other inputs.
* All scores are on the same development folds; the closed form has almost no
  free parameters, but the comparison models do.

---

# Post-fix reruns, 2026-09-24 (rebuilt tables, one primary profile per cast)

Tables rebuilt with the interpolation fix (`OHC/rebuild_tables_20260924.py`):
identical row counts, 3,978 rows recovered from NaN, 0 valid values lost, 0.0
maximum change on valid values, alignment guard passed on 322,616 rows. Every
row now carries observation time, platform, cycle, data mode and
`is_primary_profile`. Both reruns below use primary profiles only.

## The benchmark moved (the duplicates were adding TCHP noise)

On primary-only warm rows the full-recipe model scores **TCHP 10.87** (was
11.40) and **D26 10.78** (was 10.74); raw RTOFS 16.38 / 15.08. The duplicated
rows were secondary sampling schemes of the same cast at far lower vertical
resolution (~72 levels against ~500), and TCHP is an integral that is
sensitive to that resolution, so they injected noise into the TCHP target.
D26, a crossing depth, barely moved. **The previous benchmark numbers,
including the MoE's 11.19 / 10.55, are on a different population and must not
be compared with these; the MoE needs rerunning on the clean data.**

## Decomposition: holds

| | TCHP | D26 |
|---|---|---|
| bias map only, % of full gain | 79 | 64 |
| bias map + one damping number, % | **88** | **78** |
| beta, all warm primary rows | -0.82 [-0.85, -0.78] | -0.80 |
| beta, excluding near-edge | -0.85 | -0.81 |
| fraction of RTOFS local structure kept | 18% | 20% |
| gradient effect on leftover (per sd) | +0.29 [+0.11, +0.48] | +0.04 [-0.12, +0.23] |

Same story as before the fix; beta moved from -0.88 to -0.82 for TCHP. The
position (displacement) signature stays significant for TCHP only.

Asymmetry: D26's warm/cold difference nearly vanished (-0.78 vs -0.81; was
-0.75 vs -0.90), so it was largely artifact. TCHP's grew (warm bumps -0.71,
cold dips -0.95). Near-edge rows still fall mostly in the cold group, so the
TCHP asymmetry remains unverified.

## Emergent features: survived, halved

Boundary rows are now genuine disagreements about whether 26 C water exists.

| D26, cumulative | boundary | warm |
|---|---|---|
| + SSH / MLT / SBLT | 34.34 | 11.44 |
| **+ temperature excess** | **31.01** | 11.42 |
| raw RTOFS | 35.06 | 14.82 |

* Temperature excess now cuts D26 boundary error by **3.32 m (-9.7%)**, down
  from 7.00 m (-18.3%) before the fix. About half the original effect was the
  interpolation artifact.
* Warm rows still do not move (11.44 -> 11.42), so the conditional-value
  signature holds: the input helps only where the existence of 26 C water is
  in doubt.
* **TCHP: the effect is essentially gone** (7.78 -> 7.73). And on genuine TCHP
  boundary rows every model is **worse than raw RTOFS** (best 7.67 vs raw
  7.43). Near the threshold TCHP values are small, and a correction trained
  mostly on warm rows (typical correction +10) overshoots.
