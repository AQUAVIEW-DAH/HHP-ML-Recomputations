# Meeting of 2026-10-08 (Thursday): notes and plan

Two sets of notes from the same meeting: Suramya's own, and Dr. Jacobs' written notes on the plots
(Drive `HHP-analysis-2026-10-06/`). Status: recorded, not started.

## Suramya's notes (as written)

why does it degrade the sum exponent function of features, why is distance from median not one to one,
start with physical features and add sum mean exponent only on those for building simplicity and
intuition, to make it organized for a seminar and paper, sin and cosine for periodic for longitude is also
great because only lat might have trouble at the 45 degree angle, ask Dr. Jacobs regarding proper
structure too regarding the physical diff, why dont we keep full year as a holdout? 12 months, why is
rtofs anom diff closer to argo vals than rtofs ones, also how does it correlate to misplaced eddies and
fronts, create a feature, a nothing feature fill it with gaussian noise, and normalize it then whats the
effect of having that additional feature on having training and val errors, repeat it a bunch of times by
shuffling do this for the favourite features, should we do this in blocks of features as well? how can I
organize the information and start distilling the info in a more simpler and clearer way, writing down the
main points of our work, write short bullet points

## Dr. Jacobs' notes (summary of his email)

1. The signal has a large seasonal variation but the error does not: nice result. The season reaching the
   model through another variable's seasonal signal seems awkward.
2. Not about lost observations: what is the effect of very few observations in the lowest leaves
   ("diluting the data"; usual term: data fragmentation)?
   2b. Is the histogram over all leaves or only the final leaves? How were the error bars computed?
3. Front example: if RTOFS misplaces a front, RTOFS minus the area average can be large.
   3a. Why would Argo minus the area average be smaller in amplitude than RTOFS minus the area average?
   3b. Maybe the deviation from the local average is spatial information (where the point sits within a
   feature). Can a feature be used again in a later split? Do |lat|, lat, lon get used more than once?
4. How do training, test and holdout data interact; how are test and holdout used differently?
5. Sequential build-up: lat, lon -> physical features -> differences and products of physical features.
   Within each step, how do you show which features should be retained?
   5a. Top panel still has N features? Bottom panel adds sums/differences of the original N?
Suggested experiment: add a parameter of Gaussian random numbers; effect on training and validation error;
repeat with new realizations and shuffled training data; report the average error-variance reduction in
training and validation. Use it to judge features an oceanographer would expect to be physically related.

## Answers already given (2026-10-08)

* Squares / distance from the median replace each feature and fold the number line (+10 and -10 give the
  same value), so information is lost; distance from the median is not one-to-one for that reason.
* 2b: final leaves of all 300 trees of a model fitted on all clean rows (counts include the ~20% of rows
  each tree did not train on). Error bars: standard deviation over 3 random seeds; grey band +/- 2 sd.
* Small leaves are shrunk: leaf value ~ 0.03 x sum(errors) / (n + 1), and each prediction sums 300 trees.
* 3a: Argo's departure is NOT smaller in amplitude. TCHP: std of RTOFS minus its 1-degree mean 6.3,
  std of Argo minus that mean 15.4, correlation 0.10, slope 0.10 x 15.4 / 6.3 = 0.24. The 3a figure shows
  bin averages only and should show the full scatter again.
* 3b: features are reused: 299 of 300 trees ask some feature twice or more; lon 518 splits, TCHP anomaly
  431, |lat| 337, lat 309 over 300 trees.
* 4: training data builds the trees; test blocks are only scored, but their scores guide our choices;
  the holdout is never used for a choice and is scored once.

## Plan (not started)

1. One page of short bullet points of the main findings, for the seminar and paper.
2. Noise-feature test (training vs validation error, many repeats), then shuffle each favourite feature
   and each block of features against that noise baseline.
3. Sequential build-up: lat/lon -> physical features -> differences and products (-> calendar), keeping
   only what beats the noise baseline.
4. Misplaced fronts and eddies: redraw 3a with the full scatter; slope near steep gradients vs away from
   them; shift test of the RTOFS field; is the anomaly position information?
5. Full-year holdout: options for Dr. Jacobs (wait for all of 2026 vs hold out 2025).

Open question: what "only lat might have trouble at the 45 degree angle" refers to.
