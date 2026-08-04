# Gulf of Mexico attribution plots (2026-07-22) — the entire math

**Script:** `OHC/exploration/run_gom_attribution_analysis.py`
**Outputs:** `OHC/output/gom_attribution_20260722/` → Drive folder `HHP-gom-attribution-2026-07-22`
**Plots covered:**
`{tchp,d26}_gom_mean_shap.png` · `{tchp,d26}_gom_shap_dependence.png` ·
`{tchp,d26}_gom_backward_elimination.png` · `{tchp,d26}_gom_distributions.png`
(+ the matching CSVs)

This document is self-contained: Part A derives every quantity that enters the
plots (targets, collocation, features, model, evaluation protocol); Part B
gives the math of each plot in the folder.

---

## Part A — Background: where every number in these plots comes from

### A1. The two targets

Both targets are computed identically for the Argo profile (truth) and, when
building the reduced model fields, for the RTOFS column, using TEOS-10 (`gsw`)
seawater properties.

**D26 — depth of the 26 °C isotherm (m).** With temperatures $T_k$ at depths
$z_k$ (increasing downward), find the first level $k$ where the profile
crosses 26 °C from above ($T_{k-1} > 26 \ge T_k$, with $T_1 \ge 26$ at the
surface) and interpolate linearly:

$$D26 \;=\; z_{k-1} \;+\; \frac{T_{k-1} - 26}{\,T_{k-1} - T_k\,}\,\bigl(z_k - z_{k-1}\bigr)$$

If the surface is colder than 26 °C or the profile never crosses it, D26 (and
TCHP) are undefined for that column — this is why only warm-water rows carry
targets.

**TCHP — tropical cyclone heat potential (kJ/cm²).** Heat content above the
26 °C isotherm:

$$\mathrm{TCHP} \;=\; \frac{1}{10^{7}} \int_{0}^{D26} \rho\bigl(S_A, T, p\bigr)\, c_p\bigl(S_A, T, p\bigr)\,\bigl(T(z) - 26\bigr)\, dz$$

where $\rho$ and $c_p$ are the TEOS-10 in-situ density and heat capacity,
$S_A$ is absolute salinity from practical salinity via `gsw_SA_from_SP`, and
$p$ is pressure from depth. The integral is evaluated layer-by-layer with the
last layer clipped at $D26$; the factor $10^{-7}$ converts J/m² to kJ/cm².

### A2. Collocation (how RTOFS is sampled to the Argo point)

For an Argo profile at position $\mathbf{p}$ on date $d$, the raw model value
$r$ is interpolated from the 8 nearest native RTOFS grid cells (found by
KD-tree on unit-sphere coordinates) with inverse-distance-squared weights:

$$r \;=\; \frac{\sum_{j=1}^{8} w_j\, v_j}{\sum_{j=1}^{8} w_j},
\qquad w_j = \frac{1}{\max(d_j,\ \varepsilon)^{2}}$$

where $v_j$ is the model field value at neighbor $j$, $d_j$ its great-circle
distance in km, and $\varepsilon = 10^{-6}$; an exact-position match
($d_j \le \varepsilon$) short-circuits to that cell's value. Non-finite
neighbors (land / no 26 °C crossing) are dropped from the sums.

**The learning target** is then the residual

$$\delta \;=\; y - r$$

with $y$ the Argo-derived target value; the corrected prediction is always

$$\hat{y} \;=\; r + f(\mathbf{x})$$

### A3. The features $\mathbf{x}$ used by these plots

Recipes: `global_pruned_plus_neighborhood` for TCHP (34 features) and
`drop_both_lat_interactions_plus_neighborhood` for D26 (35 features). The
groups, with formulas for the derived ones:

**Calendar** — `year`, `month_int`, and smooth encodings
$\text{month\_sin} = \sin(2\pi m/12)$, $\text{month\_cos} = \cos(2\pi m/12)$,
$\text{doy\_sin} = \sin(2\pi \cdot \text{doy}/365.25)$,
$\text{doy\_cos} = \cos(2\pi \cdot \text{doy}/365.25)$, plus season flags
(`is_winter_jfm`, `is_summer_jas`, `is_other`).

**Location** — `lat`, `lon`, $\text{abs\_lat} = |\text{lat}|$, and the
collocation-quality distance `nearest_rtofs_grid_distance_km` ($= d_1$ above).

**Raw model state** — the collocated raw values `model_interp_tchp_kj_per_cm2`
and `model_interp_d26_m` (i.e., $r$ for both targets).

**Global physics (RTOFS diagnostics sampled by the same IDW):** surface
temperature excess $T_s - 26$, sea-surface height (SSH), mixed-layer thickness
(MLT), surface boundary-layer thickness (SBLT), and derived combinations
$D26 - \mathrm{MLT}$, $D26/\mathrm{SBLT}$, plus interaction terms of the form
$\text{feature} \times |\text{lat}|$ (the TCHP recipe keeps
SSH·|lat|, MLT·|lat|, $T_s$-excess·|lat|; the D26 recipe keeps only
MLT·|lat|). The D26 recipe also carries three profile features (deep steric
height and two Brunt–Väisälä $N^2$ summaries).

**Neighborhood stencils** (computed from the reduced daily RTOFS fields around
the nearest grid cell $(y_0, x_0)$; window half-width $h$ cells with
$h \in \{6, 12, 25\}$ ≈ 0.5°, 1°, 2° at the 1/12° grid): for field
$F \in \{\mathrm{TCHP}, D26, \mathrm{SST}\}$ and window
$W_h = \{(y, x): |y - y_0| \le h,\ |x - x_0| \le h\}$ (longitude wraps),

$$\mu_h = \frac{1}{|V|} \sum_{(y,x) \in V} F_{y,x},
\qquad
\sigma_h = \sqrt{\frac{1}{|V|} \sum_{(y,x) \in V} \bigl(F_{y,x} - \mu_h\bigr)^2}$$

over the valid (finite) cells $V \subseteq W_h$, required to cover
$\ge 25\%$ of the window; the gradient magnitude from centered differences
scaled to per-100-km,

$$\|\nabla F\| \;=\; 100 \sqrt{ \left( \frac{F_{y,x+1} - F_{y,x-1}}{2\,\Delta x_{\mathrm{km}}} \right)^{\!2} + \left( \frac{F_{y+1,x} - F_{y-1,x}}{2\,\Delta y_{\mathrm{km}}} \right)^{\!2} }$$

with local grid spacings $\Delta x_{\mathrm{km}}, \Delta y_{\mathrm{km}}$ from
the curvilinear grid ($\Delta x$ shrinks with $\cos(\mathrm{lat})$); and the
mesoscale anomaly

$$a \;=\; F_{y_0, x_0} - \mu_{1^\circ}$$

The recipe uses $\sigma_{1^\circ}$, $\sigma_{2^\circ}$ (TCHP only),
$\|\nabla F\|$, and $a$ for each of the three fields.

### A4. The model $f$ (gradient-boosted trees)

$f$ is an additive ensemble of $M = 300$ regression trees:

$$f(\mathbf{x}) \;=\; F_0 + \eta \sum_{m=1}^{M} t_m(\mathbf{x})$$

with learning rate $\eta = 0.03$ and $F_0$ the constant base score. Trees are
grown sequentially: tree $m$ is fit to the pointwise negative gradient of the
squared-error loss at the current ensemble — for squared error this is simply
the current residual $\delta_i - F_{m-1}(\mathbf{x}_i)$. Each tree has depth
$\le 4$; each split is chosen to maximize the regularized gain

$$\mathrm{Gain} \;=\; \tfrac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]$$

and each leaf takes the closed-form optimal weight

$$w^{*} \;=\; -\,\frac{G}{H + \lambda}$$

where $G$ and $H$ are the sums of first and second loss derivatives over the
samples in the node (for squared error: $G = -\sum \text{residual}_i$,
$H = n_{\text{node}}$) and $\lambda = 1$. Randomization: each tree sees an
80 % row subsample and an 80 % feature subsample. Missing values are imputed
beforehand with the **training-fold median** of each feature.

### A5. Evaluation protocol (identical for every plot)

**Rows.** Gulf of Mexico: latitude $\in [18, 31]$, longitude $\in [-98, -80]$,
keeping rows where $y$, $r$, and $\delta$ are all valid → $n = 3{,}285$ rows
over 623 dates per target.

**Blocked-forward folds.** The Gulf dates are sorted chronologically and cut
into 3 consecutive blocks with a 1-date embargo at each boundary. Fold $k$
trains on all dates before block $k$ and predicts block $k$; hence every
plotted prediction is **out-of-fold (OOF)** — made by a model that never saw
that date or its embargo neighbor.

**The two model families.** Per fold, with identical features and
hyperparameters: **global** (trained on all-ocean rows of the training dates)
and **Gulf-local** (trained on Gulf rows only). Both are evaluated on the same
Gulf validation rows, so every difference between them is attributable to the
training pool alone.

---

## Part B — The math of each plot

### B1. `*_gom_mean_shap.png` / `*_gom_mean_shap.csv`

The SHAP value $\phi_i(\mathbf{x})$ of feature $i$ for one prediction is the
Shapley value with features as players and the prediction as the payout:

$$\phi_i(\mathbf{x}) \;=\; \sum_{S \subseteq F \setminus \{i\}}
\frac{|S|!\,\bigl(|F|-|S|-1\bigr)!}{|F|!}\;
\Bigl[\, v\bigl(S \cup \{i\}\bigr) - v(S) \,\Bigr]$$

where $v(S) = \mathbb{E}\bigl[f(\mathbf{x}) \mid \mathbf{x}_S \text{ fixed}\bigr]$
is the expected model output when only the features in $S$ are known. In
words: the average change in the prediction caused by revealing feature $i$,
averaged over all possible orders of revealing features — the averaging over
orders is what distributes credit fairly among correlated features.

**Exact computation.** TreeSHAP (Lundberg et al. 2020) evaluates the sum
exactly for tree ensembles: descending a tree, a split on a feature outside
$S$ follows both branches weighted by the fraction of training rows that took
each branch (the cover), which computes $v(S)$ without enumerating subsets.
XGBoost's `pred_contribs=True` implements this natively.

**Additivity** ties the values to physical units — for every row

$$f(\mathbf{x}) \;=\; \phi_0 + \sum_{i \in F} \phi_i(\mathbf{x})$$

with $\phi_0$ the training-mean prediction (dropped from the plots), so the
$\phi_i$ are in residual units (m, kJ/cm²) and sum to the actual correction
applied to that row.

**What the bars show.** OOF SHAP matrices $\Phi^{\mathrm{global}}$ and
$\Phi^{\mathrm{local}}$ (each $3285 \times |F|$, row-matched) are assembled by
taking each row's SHAP from the fold model that held it out. The bar for
feature $i$ is the mean absolute contribution

$$\bar{I}_i \;=\; \frac{1}{n} \sum_{r=1}^{n} \bigl|\, \Phi_{r i} \,\bigr|$$

ranked by the local model's $\bar{I}_i$, top 12 shown. Absolute values are
used because a feature contributing $+5$ on half the rows and $-5$ on the rest
does 5 units of work despite a signed mean of zero.

### B2. `*_gom_shap_dependence.png`

Scatter of $\bigl(x_{r,i},\ \Phi_{r i}\bigr)$ for the top-4 features by local
$\bar{I}_i$, both models overlaid on identical rows. The trend is the model's
learned response to feature $i$; by additivity, the **vertical spread at a
fixed feature value is exactly the interaction effect** — rows with equal
$x_i$ but different context get different credit. A flat cloud means the model
ignores the feature.

### B3. `*_gom_backward_elimination.png` / `.csv`

**Score.** For a candidate feature set $A$, retrain the Gulf-local model per
fold on $A$ only, assemble OOF predictions
$\hat{y}_r(A) = r_r + f_A(\mathbf{x}_{r,A})$, and score on the observed scale:

$$\mathrm{MAE}(A) \;=\; \frac{1}{n} \sum_{r=1}^{n} \bigl|\, \hat{y}_r(A) - y_r \,\bigr|$$

**Greedy recursion.** From the full recipe $A_0$:

$$c^{*}_t \;=\; \arg\min_{c \in A_{t-1}} \mathrm{MAE}\bigl(A_{t-1} \setminus \{c\}\bigr),
\qquad A_t \;=\; A_{t-1} \setminus \{c^{*}_t\}$$

— delete whichever feature hurts least (or helps most), repeat until 4
features remain. Total cost $\sum_{k=5}^{|A_0|} k \approx 580$
retrain-and-score evaluations per target (each = 3 fold models).

**Reading it.** Downward drift ⇒ removed features were dead weight (the D26
curve improves 11.23 → 10.63 m: the 35-feature global recipe overfits 3k local
rows). An upward jump at step $t$ ⇒ the feature removed there carried skill
the survivors cannot replace. Caveats: greedy and path-dependent, and
correlated features shield each other — hence the pairing with SHAP:
elimination measures *replaceability*, SHAP measures *usage*; a feature high
on both (SSH here) is unambiguously essential.

### B4. `*_gom_distributions.png`

Normalized histograms with 45 equal-width bins spanning the observed values'
0.5th–99.5th percentile (display clipping only). For each series
$s \in \{\text{observed},\ \text{raw RTOFS},\ \text{corrected-global OOF},\ \text{corrected-local OOF}\}$
the plotted height in bin $j$ is the probability density

$$d_j(s) \;=\; \frac{\mathrm{count}_j(s)}{n_s\, \Delta b}$$

with $\Delta b$ the bin width, so every curve integrates to 1 and series are
comparable regardless of row counts. The corrected curves use the OOF
predictions of §A5 — no curve contains in-sample fitting.

---

## Interpretation caveats

1. TreeSHAP's conditional expectations use the **training** distribution's
   branch weights: the global model's "output when a feature is unknown"
   averages over global conditions, the local model's over Gulf conditions.
   The comparison contrasts *learned strategies*, not model-independent ocean
   properties.
2. SHAP explains the model's *residual prediction*: "the model relies on SSH
   to predict RTOFS's Gulf error" is the defensible claim; a causal physical
   statement needs the oceanographic argument on top.
3. All skill numbers inherit the protocol caveats of the main pipeline
   (platform overlap between years; Gulf sample size).
