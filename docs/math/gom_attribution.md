# Gulf of Mexico attribution analysis — exact math

**Script:** `OHC/exploration/run_gom_attribution_analysis.py`
**Outputs:** `OHC/output/gom_attribution_20260722/` → Drive `HHP-gom-attribution-2026-07-22`
**Figures covered:** `{tchp,d26}_gom_mean_shap.png`, `*_gom_shap_dependence.png`,
`*_gom_backward_elimination.png`, `*_gom_distributions.png` (+ CSVs)
**First written:** 2026-08-04 (documents the 2026-07-22 run)

## Setup common to all figures

**Data.** Gulf rows: collocated points with latitude $\in [18, 31]$ and
longitude $\in [-98, -80]$, keeping rows where the Argo observation, the raw
RTOFS value, and the residual are all valid: $n = 3{,}285$ rows over 623 dates
per target (TCHP in kJ/cm², D26 in m).

**Residual learning.** With $y$ the observed Argo value and $r$ the raw RTOFS
value at the same point and date, the learning target is the residual

$$\delta = y - r$$

An XGBoost regressor $f$ is fit to $\delta$; the corrected prediction is

$$\hat{y} = r + f(\mathbf{x})$$

where $\mathbf{x}$ is the 35-dimensional feature vector (best+neighborhood
recipe). $f$ is a sum of 300 regression trees of depth $\le 4$, trained by
gradient boosting on squared error with learning rate $0.03$, row subsample
$0.8$, per-tree feature subsample $0.8$, and L2 leaf penalty $\lambda = 1$.
Missing feature values are imputed with the **training-fold median** only.

**Blocked-forward folds.** The Gulf's 623 dates are sorted chronologically and
split into 3 consecutive blocks with a 1-date embargo at each boundary. Fold
$k$ trains on all dates before block $k$ and predicts block $k$, so every
prediction is out-of-fold (OOF): produced by a model that never saw that date.
Two model families are trained per fold on identical features: **global**
(training rows from all oceans) and **Gulf-local** (training rows from the
Gulf only), both evaluated on the same Gulf validation rows.

## 1. Mean-SHAP bars (`*_gom_mean_shap.png`, `*_gom_mean_shap.csv`)

The SHAP value $\phi_i(\mathbf{x})$ of feature $i$ for one prediction is the
Shapley value with features as players and the prediction as payout:

$$\phi_i(\mathbf{x}) \;=\; \sum_{S \subseteq F \setminus \{i\}}
\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\;
\Bigl[\, v\bigl(S \cup \{i\}\bigr) - v(S) \,\Bigr]$$

where $F$ is the set of all 35 features and $v(S)$ is the expected model
output when only the features in $S$ are known:

$$v(S) \;=\; \mathbb{E}\bigl[\, f(\mathbf{x}) \;\big|\; \mathbf{x}_S \text{ fixed at this row's values} \,\bigr]$$

Interpretation: $\phi_i$ is the average change in the prediction caused by
revealing feature $i$, averaged over every possible order of revealing the
features — which is what makes the attribution fair under feature correlation.

**Computation.** TreeSHAP (Lundberg et al. 2020) evaluates this exactly for
tree ensembles in polynomial time: at a split on an unknown feature the
algorithm follows both branches weighted by the fraction of training samples
that took each branch. XGBoost's `pred_contribs=True` implements this; no
sampling or approximation is involved.

**Additivity** (why the units are physical): for every row

$$f(\mathbf{x}) \;=\; \phi_0 + \sum_{i \in F} \phi_i(\mathbf{x})$$

with $\phi_0$ the training-mean prediction (base value, dropped from the
plots). The $\phi_i$ are therefore in residual units (m or kJ/cm²) and sum to
the actual correction.

**Aggregation.** SHAP matrices $\Phi \in \mathbb{R}^{3285 \times 35}$ are
assembled OOF (each row's values come from the fold model that held it out)
for both model families. The bar for feature $i$ is the mean absolute
contribution

$$\bar{I}_i \;=\; \frac{1}{n} \sum_{\text{rows } r} \bigl|\, \phi_i(\mathbf{x}_r) \,\bigr|$$

ranked by the local model's $\bar{I}_i$, top 12 shown. Absolute values are
used because a feature pushing $+5$ on half the rows and $-5$ on the rest is
doing 5 units of work despite a zero signed mean.

## 2. Dependence curves (`*_gom_shap_dependence.png`)

Scatter of the pairs $\bigl(x_{r,i},\; \phi_i(\mathbf{x}_r)\bigr)$ for the
top-4 features by local $\bar{I}_i$, global and local models overlaid. The
trend is the model's learned main effect of feature $i$; by additivity, the
**vertical spread at a fixed feature value is exactly the interaction
structure** (rows with equal $x_i$ but different context receive different
contributions). A flat cloud means the model ignores the feature.

## 3. Backward elimination (`*_gom_backward_elimination.png`, `.csv`)

**Score.** For a candidate active feature set $A$, retrain the Gulf-local
model per fold on $A$ only, assemble OOF corrected predictions
$\hat{y}_r(A) = r_r + f_A(\mathbf{x}_{r,A})$, and score on the observed scale:

$$\mathrm{MAE}(A) \;=\; \frac{1}{n} \sum_{\text{rows } r} \bigl|\, \hat{y}_r(A) - y_r \,\bigr|$$

**Greedy recursion.** Starting from the full recipe $A_0$ (35 features), at
each step $t$:

$$c^* \;=\; \arg\min_{c \,\in\, A_{t-1}} \; \mathrm{MAE}\bigl(A_{t-1} \setminus \{c\}\bigr),
\qquad A_t = A_{t-1} \setminus \{c^*\}$$

i.e., delete whichever feature hurts least (or helps most), repeat until 4
features remain. Cost: $\sum_{k=5}^{35} k \approx 580$ retrain-and-score
evaluations per target (each evaluation = 3 fold models).

**Reading the ladder.** Downward drift → removed features were dead weight
(the Gulf curve improves from 11.23 to 10.63 m: 35 features overfit 3k rows).
An upward jump at step $t$ → the feature removed there carried irreplaceable
skill. Caveats: the procedure is greedy and path-dependent, and correlated
features shield each other — which is why it is paired with SHAP: elimination
measures *replaceability*, SHAP measures *usage*.

## 4. Distributions (`*_gom_distributions.png`)

Normalized histograms with 45 equal-width bins spanning the observed values'
0.5th–99.5th percentile. For series $s$ (observed, raw RTOFS, corrected-global
OOF, corrected-local OOF), the height in bin $j$ is the probability density

$$d_j(s) \;=\; \frac{\mathrm{count}_j(s)}{n_s \,\Delta b}$$

with $\Delta b$ the bin width, so each curve integrates to 1 and series of
different lengths are comparable. Corrected series use the OOF predictions
defined above; no curve contains in-sample fitting.

## Interpretation caveat

TreeSHAP's conditional expectations use the training distribution's branch
weights, so the global model's "expected output when a feature is unknown"
averages over global conditions while the local model's averages over Gulf
conditions. The comparison therefore contrasts **learned strategies** ("what
does this model rely on"), not a model-independent property of the ocean.
