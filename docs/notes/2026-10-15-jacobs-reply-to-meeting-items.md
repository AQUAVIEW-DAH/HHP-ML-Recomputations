# Dr. Jacobs' reply to the meeting-items email (2026-10-15): notes and plan

His point numbers follow the numbered email (1 noise level, 2 folding transforms, 3 build-up, 4 the 45 degree
note, 5 full-year testing, 6 recent history, 7 anomaly vs Argo, 8 misplaced fronts, 9 input relevance,
10 his 16 inputs). Status: recorded, not started.

## His notes (summary)

* His 16 inputs: "already beating the expert's intuition"; glad the expert set is still in the ballpark.
* 1 Noise test: very glad; a good point to bring to people: it shows how one knows a feature is noise.
* 2 Folding transforms (01a): great demonstration. Question: if y = x^2, does a tree split x at the same
  places as it would split x^2? Does using x^2 use leaves more efficiently? Example 2 (latitude): much of
  the difference is picked up by lat^2 or |lat|, but north and south still differ.
* 3 Build-up: reorder for someone new to the problem, and test each parameter within each group:
  1. Biases: |lat|, lat, lon, calendar sin/cos. Equivalent to binning/smoothing RTOFS - Argo on a
     lat/lon/calendar grid and treating it as RTOFS bias. Pictures that explain both the tree process and
     the biases. Then: is the residual related to any RTOFS parameter?
  2. RTOFS TCHP and D26 at the float: RTOFS skill beyond the bias. Is RTOFS variance too low, so the model
     raises the amplitude? Show how the model alters RTOFS TCHP and D26. Can the residuals be tied to
     specific RTOFS features?
  3. SSH, mixed layer, boundary layer, SST (eddies, surface and upper-ocean mixing, heat flux): which matters
     most? Then: how does feature-placement accuracy affect TCHP and D26?
  4. 1 degree averages, local spread and gradients (small-scale structure): evidence that feature placement
     hurts skill?
  5. Interactions: differences, products, products of differences. Any benefit?
* 4 (03a): reminds him of the PCA experiments; makes sense.
* "Recent history" is a new term to him: what is it?
* 5 Full-year testing: excellent; wants to discuss to be sure he understands.
* 6 Recent history: "very clever". What does it imply RTOFS is doing or not doing? Is RTOFS lagging the
  data? Are time differences increasing the rate RTOFS should be changing? May change the 5-step plan.
* 7 Anomaly vs Argo: are we looking at the right thing? Bin the x axis by squared error (RTOFS - Argo)^2;
  in each bin, the mean of (RTOFS at Argo - RTOFS 1 degree mean)^2. Remove the sign: the question is whether
  RTOFS small-scale structure is similar in size to the RTOFS - Argo error. Prefers regular x/y grid cells
  to hexagon bins in distribution plots.
* 8 Misplaced fronts: not fully understood; discuss. 6b: average over all profiles? A pattern needs all
  RTOFS features shifted the same way on average.
* 9 Input relevance: 7a, are the points jittered horizontally? 7c great; anything below the 95% confidence
  level should go. 7d: cos_semiannual near noise because of the semiannual phase; sin_semiannual is good.
  8a and 9a: what is shuffled, and how?

## Answers available now

* Recent history: 12 inputs per profile from RTOFS at the float's grid cell: 7 day mean (d-7..d-1), change
  over 3 and 7 days (d minus d-3, d minus d-7), spread over 8 days (d-7..d), for TCHP, D26 and SST. Only RTOFS
  up to the profile day. Definitions in 19_definitions.
* Lag question, first evidence: the 3 day change of RTOFS is negatively correlated with the error Argo -
  RTOFS (TCHP -0.28, D26 -0.32). If RTOFS lagged reality, a rising RTOFS would still be too low (positive
  correlation). Negative means RTOFS's recent changes overshoot: part of each day-to-day change is not in
  the ocean (e.g. assimilation jumps), which the 7 day mean smooths out. Needs a proper lag test.
* 6b: yes, averaged over all profiles (top) and over the 20% steepest-gradient profiles (bottom); a
  pattern would require a common average shift. Per-region version is the natural next step.
* 7a: yes, points are jittered horizontally (uniform +/-0.06) only so they do not overlap.
* 8a/9a shuffling: the model is fitted once; then one input's values are randomly permuted among the test
  profiles (each profile gets another profile's value, so the column keeps its distribution but loses its
  link to the profile), and the rise in test error is measured; repeated 10 times. 9a shuffles all columns of
  a group with the same permutation (keeps relations inside the group), and separately removes the group
  and refits.
* 7c threshold: current rule is "used less than the noise column in at least half of 60 fits"; switch to
  his 95% rule (feature's gain share above the 95th percentile of the noise column's).

## Plan (not started)

1. Explainer for y = x^2: splits on x vs x^2, leaves needed for the same error (synthetic, one figure).
2. Rebuild the step-by-step figure in his 5-step order, with each parameter tested individually inside
   each group (forward selection within group, noise-level keep rule), plus:
   - bias maps (lat/lon/calendar) and residual vs RTOFS parameters after step 1;
   - how the model changes RTOFS TCHP/D26 (amplitude: variance of corrected vs raw vs Argo, slope plots).
3. Lag test for recent history: Argo vs RTOFS on days d-7..d+3; correlation of RTOFS changes with error;
   whether recent history works through smoothing (mean) or trend (change).
4. Point 7 redo: bins of (RTOFS - Argo)^2 vs mean (RTOFS - 1 degree mean)^2, regular grid cells.
5. Regional shift test (6b) and 95% keep rule for 7c/7d.
6. Discussion items: points 5 and 8.
