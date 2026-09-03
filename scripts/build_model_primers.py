"""Render the model-primer PDFs (one per model family) with matplotlib.

Audience: a reader with a strong mathematics background and no machine-learning
background. Each primer explains the algorithm from first principles and then
how it is applied to the RTOFS correction problem. Output: docs/math/model_primers/.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parents[1] / "docs" / "math" / "model_primers"
OUT.mkdir(parents=True, exist_ok=True)

PAGE = (8.27, 11.69)
X0, X1 = 0.09, 0.91
TOP, BOT = 0.94, 0.07
WRAP = 96

STYLE = {
    "h1": dict(size=16.5, weight="bold", dy_before=0.030, dy_line=0.030, dy_after=0.012),
    "h2": dict(size=12.5, weight="bold", dy_before=0.022, dy_line=0.022, dy_after=0.006),
    "p":  dict(size=10.3, weight="normal", dy_before=0.004, dy_line=0.0158, dy_after=0.004),
    "li": dict(size=10.3, weight="normal", dy_before=0.002, dy_line=0.0158, dy_after=0.002),
    "m":  dict(size=12.0, weight="normal", dy_before=0.010, dy_line=0.030, dy_after=0.010),
    "cap": dict(size=9.2, weight="normal", dy_before=0.002, dy_line=0.0145, dy_after=0.004),
}


def _sanitize(text: str) -> str:
    text = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", text)
    text = text.replace("\\bigl", "").replace("\\bigr", "").replace("\\!", "")
    text = text.replace("\\sigma_f, \\ell_1..\\ell_d, \\sigma_n", "\\sigma_f,\\ \\ell_j,\\ \\sigma_n")
    if "if\\ sample" in text:
        text = r"$w_i = 1\;\mathrm{inside\;the\;region},\qquad w_i = 0.05\;\mathrm{outside},$"
    return text


def _wrap_protect(text: str, width: int) -> list[str]:
    parts = re.split(r"(\$[^$]*\$)", text)
    prot = "".join(seg.replace(" ", "\x00") if seg.startswith("$") else seg for seg in parts)
    return [ln.replace("\x00", " ") for ln in textwrap.wrap(prot, width)] or [""]


def render(doc_title: str, subtitle: str, blocks: list[tuple[str, str]], fname: str) -> None:
    pdf = PdfPages(OUT / fname)
    fig = None
    y = TOP

    def new_page(first=False):
        nonlocal fig, y
        if fig is not None:
            pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=PAGE)
        fig.patch.set_facecolor("white")
        y = TOP
        if first:
            fig.text(X0, y, doc_title, size=20, weight="bold", va="top")
            y -= 0.035
            fig.text(X0, y, subtitle, size=11, style="italic", color="#444444", va="top")
            y -= 0.030
            fig.lines.append(plt.Line2D([X0, X1], [y, y], transform=fig.transFigure,
                                        color="#999999", linewidth=0.8))
            y -= 0.022
        fig.text(X1, 0.035, "HHP RTOFS-correction project — model primers, 2026-09-03",
                 size=7.5, color="#888888", ha="right")

    new_page(first=True)
    for kind, text in blocks:
        st = STYLE[kind]
        text = _sanitize(text)
        if kind in ("p", "li", "cap"):
            prefix = "   " if kind == "li" else ""
            lines = []
            for para_line in _wrap_protect(text, WRAP - (4 if kind == "li" else 0)):
                lines.append(prefix + para_line)
            if kind == "li" and lines:
                lines[0] = " • " + lines[0][3:]
        else:
            lines = [text]
        need = st["dy_before"] + len(lines) * st["dy_line"] + st["dy_after"]
        if y - need < BOT and kind != "h1":
            new_page()
        if kind in ("h1", "h2") and y - need - 0.10 < BOT:
            new_page()
        y -= st["dy_before"]
        for ln in lines:
            if kind == "m":
                fig.text((X0 + X1) / 2, y, ln, size=st["size"], ha="center", va="top")
            else:
                fig.text(X0, y, ln, size=st["size"], weight=st["weight"], va="top",
                         color="#333333" if kind == "cap" else "black")
            y -= st["dy_line"]
        y -= st["dy_after"]
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print("wrote", OUT / fname)


SETUP = [
    ("h2", "The problem these models solve"),
    ("p", "NOAA's ocean forecast model RTOFS reports, for any place and day, the Tropical Cyclone "
          "Heat Potential (TCHP, the heat stored above the 26 degree isotherm) and D26 (the depth "
          "of that isotherm). Autonomous Argo floats measure the same two quantities at scattered "
          "points. Comparing the two at the same place and day shows the forecast is biased. "
          "We therefore learn a correction: a function that predicts the forecast's error from "
          "information available at forecast time, so it can be subtracted everywhere."),
    ("p", "Formally, we have n observation pairs. For pair i, let $x_i$ be a vector of d known "
          "quantities (position, calendar, and fields taken from the forecast itself) and let"),
    ("m", r"$\delta_i \;=\; y_i^{\mathrm{Argo}} - y_i^{\mathrm{RTOFS}}$"),
    ("p", "be the observed error of the forecast. We seek a function f minimizing the expected "
          "size of $\\delta-f(x)$, and the corrected forecast is then"),
    ("m", r"$\hat{y}(x) \;=\; y^{\mathrm{RTOFS}}(x) + f(x).$"),
    ("p", "All models below are different ways of constructing f from the n samples. Their quality "
          "is measured by the mean absolute error of the corrected forecast on dates that come "
          "after every date used to fit f, so the score always reflects prediction of the future, "
          "never memory of the past."),
]

RF = SETUP + [
    ("h1", "1. A single decision tree"),
    ("p", "A decision tree represents f as a piecewise-constant function. It partitions the input "
          "space into axis-aligned boxes and assigns one constant prediction to each box. The "
          "partition is built greedily. Start with all n samples in one box. Consider every "
          "coordinate j of x and every threshold t: they define a candidate split of the box into "
          "the samples with $x_j\\le t$ and those with $x_j>t$. Choose the split that most reduces "
          "the squared-error cost"),
    ("m", r"$\mathrm{cost}(S) \;=\; \sum_{i \in S} (\delta_i - \bar{\delta}_S)^2,$"),
    ("p", "where $\\bar{\\delta}_S$ is the mean of the targets in box S. Repeat inside each new box "
          "until a stopping rule fires, for example a box holding fewer than a set number of "
          "samples. The final boxes are called leaves, and the prediction for a new x is simply "
          "the mean target of the leaf that x falls into."),
    ("p", "Two properties matter. First, the function is discontinuous: it jumps at box "
          "boundaries. Second, a deep tree is a low-bias, high-variance estimator: it can fit "
          "almost any shape, but the boxes it picks depend strongly on which samples it saw, so "
          "two trees fitted to two random halves of the data can differ a lot."),
    ("h1", "2. The random forest: averaging many trees"),
    ("p", "A random forest reduces that variance by averaging B trees that are deliberately "
          "decorrelated. Tree b is fitted on a bootstrap resample of the data (n samples drawn "
          "with replacement), and at every split only a random subset of the d coordinates is "
          "considered. The forest predicts the average"),
    ("m", r"$f(x) \;=\; \frac{1}{B}\sum_{b=1}^{B} T_b(x).$"),
    ("p", "The reason this works is the usual variance-of-a-mean argument. If each tree has "
          "variance $\\sigma^2$ and the average pairwise correlation between trees is $\\rho$, the "
          "variance of the average is"),
    ("m", r"$\rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2,$"),
    ("p", "so growing B kills the second term, and the injected randomness (bootstrap plus random "
          "coordinate subsets) keeps $\\rho$ small so the first term is small too. The averaged "
          "function is still piecewise constant, but with B overlapping partitions the jumps are "
          "much smaller and more numerous than for one tree."),
    ("h1", "3. How we apply it to the RTOFS correction"),
    ("p", "We use forests of B = 300 trees, grown until each leaf holds at least 50 samples. The "
          "leaf floor is our main protection against reading noise as signal: no prediction is "
          "ever the mean of fewer than 50 real profile errors."),
    ("li", "In the position-only experiment requested by Dr. Jacobs, the inputs are latitude and "
           "longitude alone, with TCHP and D26 set to 0 where the water never reaches 26 degrees. "
           "The forest's boxes then become literal rectangles on the map. A single 64-leaf tree, "
           "drawn as boxes, finds the tropical band edges, the West Pacific warm pool, and the "
           "Gulf of Mexico without guidance."),
    ("li", "On position alone the forest is the best of the tested models for TCHP "
           "(error 16.6 down to 12.4) because deep trees carve two dimensions finely."),
    ("li", "On the full 35 inputs the forest reaches 11.42 (TCHP) and 10.93 m (D26), a close "
           "second to gradient boosting. Its predictions jump by about 1.3 units at a typical "
           "box boundary, which would appear as faint rectangular seams if drawn as a map."),
    ("h2", "Where it is strong and weak here"),
    ("p", "Strong: very robust, almost no tuning, handles the sharp coastline-like transitions "
          "in the data, and the boxes are directly interpretable. Weak: predictions are "
          "discontinuous, it cannot extrapolate beyond the range of what it saw (every "
          "prediction is an average of training targets), and averaging makes it slightly "
          "blunter than boosting when many partially redundant inputs each carry a little "
          "signal."),
]

XGB = SETUP + [
    ("h1", "1. Fitting a sum of small corrections"),
    ("p", "Gradient boosting builds f as a sum of M small trees, fitted one after another, each "
          "correcting what the sum so far still gets wrong:"),
    ("m", r"$F_M(x) \;=\; \sum_{m=1}^{M} \nu\, f_m(x), \qquad 0<\nu\le 1 .$"),
    ("p", "With squared-error loss the recipe is simple. Let $r_i = \\delta_i - F_{m-1}(x_i)$ be "
          "the residuals of the current sum. Fit the next small tree $f_m$ to the residuals, add "
          "it in, and repeat. The factor $\\nu$ (the learning rate) deliberately takes only a "
          "small step in the direction of each new tree; many small steps generalize better than "
          "few large ones."),
    ("p", "The name comes from viewing this as gradient descent in function space: for a general "
          "loss $\\ell$, the residuals are replaced by the negative gradient "
          "$-\\partial\\ell(\\delta_i, F(x_i))/\\partial F(x_i)$, so each tree is a descent step."),
    ("h1", "2. What XGBoost adds"),
    ("p", "XGBoost is an implementation of this idea that chooses each tree using a second-order "
          "expansion of the loss plus an explicit penalty on tree complexity. For a fixed tree "
          "structure, let $G$ and $H$ be the sums of first and second derivatives of the loss "
          "over the samples in a leaf. The optimal leaf value and the value of a candidate split "
          "have closed forms:"),
    ("m", r"$w^{*} = -\frac{G}{H+\lambda}, \qquad \mathrm{gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right],$"),
    ("p", "where $\\lambda$ is a regularization constant that shrinks leaf values toward zero. "
          "Splits are only kept if their gain is positive after the penalty, which is how the "
          "method resists fitting noise. Two further randomizations mirror the forest: each tree "
          "sees a random 80 percent of the samples and of the input coordinates."),
    ("p", "The contrast with a random forest is the key point. A forest averages many deep, "
          "independent trees fitted in parallel to the same target. Boosting chains many shallow "
          "trees, each fitted to what remains unexplained. Shallow trees are weak alone but the "
          "chain is expressive, and because every tree works on the current residual, redundant "
          "inputs cost little: once one of two near-duplicate inputs has been used, the residual "
          "no longer rewards using the other."),
    ("h1", "3. How we apply it to the RTOFS correction"),
    ("p", "Our locked setting, fixed early and never tuned per experiment: M = 300 trees of depth "
          "4, learning rate 0.03, 80 percent row and column sampling, $\\lambda = 1$. Missing "
          "input values are replaced by the median of the fitting period."),
    ("li", "This is the engine of the recommended model. As a single global model it reaches "
           "11.40 (TCHP) and 10.76 m (D26) against 16.6 / 14.9 for the raw forecast."),
    ("li", "It is also the expert inside the mixture-of-experts blend (see the mixture primer), "
           "which reaches 11.19 / 10.55."),
    ("li", "A caution from the position-only experiment: with only latitude and longitude as "
           "inputs, depth-4 trees cannot carve fine spatial boxes, and boosting was the worst of "
           "the four families there. Its strength appears only when many informative inputs give "
           "the shallow trees good one-variable questions to ask."),
    ("h2", "Where it is strong and weak here"),
    ("p", "Strong: the best accuracy of every family on the full input set, graceful with "
          "redundant inputs, fast. Weak: needs its handful of settings chosen sensibly, is as "
          "discontinuous as any tree method, and like all tree methods it cannot predict values "
          "outside the range of the targets it was fitted on."),
]

MOE = SETUP + [
    ("h1", "1. Why one global function is not enough"),
    ("p", "A single f fitted to the whole ocean must average over regions whose error physics "
          "differ. The Gulf of Mexico is the clearest case: its errors follow the Loop Current, "
          "and a global fit dilutes that signal under two hundred times more data from "
          "elsewhere. A mixture of experts replaces the single f with several specialist "
          "functions plus a rule, called a gate, deciding which specialists speak for a given x "
          "and with what weight."),
    ("h1", "2. The architecture of our recommended model"),
    ("p", "Our corrected forecast blends two mixtures that share the same inputs and the same "
          "base learner (gradient-boosted trees, see the XGBoost primer) but partition the ocean "
          "differently:"),
    ("m", r"$\hat{y}(x) \;=\; y^{\mathrm{RTOFS}}(x) + \alpha\, f_{\mathrm{geo}}(x) + (1-\alpha)\, f_{\mathrm{regime}}(x).$"),
    ("h2", "2a. Geographic experts with a hard gate"),
    ("p", "Five experts own five fixed regions (Gulf of Mexico, Atlantic, Indian, West Pacific, "
          "East Pacific and the rest). The gate is hard: each location belongs to exactly one "
          "region, and only its expert is consulted there. Crucially, each expert is fitted on "
          "ALL samples, but with weights"),
    ("m", r"$w_i = 1\ \mathrm{inside\ the\ region}, \qquad w_i = 0.05\ \mathrm{outside},$"),
    ("p", "meaning every squared error in the fitting criterion is multiplied by $w_i$. The "
          "expert is a local specialist with a weak global prior: foreign data cannot dominate, "
          "but it fills gaps and stabilizes the fit where local data is thin. The weight 0.05 "
          "was chosen by trying 0.05, 0.10, 0.15 and 0.25; the smallest won everywhere, meaning "
          "the experts want strong specialization."),
    ("h2", "2b. Learned regimes with a soft gate"),
    ("p", "The second mixture lets the data define its own regions, in the space of physical "
          "state rather than position. Six standardized quantities (sea surface height, "
          "temperature excess above 26 degrees, mixed-layer thickness, two measures of local "
          "spatial variability, and distance from the equator) are clustered by k-means, which "
          "picks K centers minimizing"),
    ("m", r"$\sum_{i} \min_{k} \Vert z_i - c_k \Vert^2$"),
    ("p", "over the standardized state vectors $z_i$. One expert is fitted per cluster with the "
          "same 1 / 0.05 weighting. At prediction time the gate is soft: with $d_k$ the distance "
          "of the point's state to center k,"),
    ("m", r"$g_k(x) = \frac{\exp(-d_k/T)}{\sum_{k'} \exp(-d_{k'}/T)}, \qquad f_{\mathrm{regime}}(x)=\sum_k g_k(x)\, f_k(x),$"),
    ("p", "with T set to the median distance, so several experts blend smoothly. Because the "
          "state vector contains no coordinates, this gate generalizes to locations never "
          "sampled: a point is served by the experts for water that behaves like its water."),
    ("h2", "2c. The blend"),
    ("p", "The two mixtures err differently, so a convex combination beats both. The chosen "
          "settings are $\\alpha = 0.75$, K = 6 for TCHP and $\\alpha = 0.5$, K = 12 for D26."),
    ("h1", "3. Results and reading"),
    ("li", "Corrected error 11.19 (TCHP) and 10.55 m (D26) globally, against 11.40 / 10.76 for "
           "the best single global model and 16.6 / 14.9 for the raw forecast."),
    ("li", "In the Gulf, the mixture recovers most of the advantage of a dedicated Gulf-only "
           "model (12.41 vs 12.37 for TCHP; 11.52 vs 11.23 for D26) while remaining one global "
           "system."),
    ("li", "Every region prefers its own expert over any foreign expert, which validates the "
           "partition."),
    ("h2", "Where it is strong and weak here"),
    ("p", "Strong: captures regional error physics a global fit averages away, at no external "
          "data cost. Weak: more moving parts (weights, K, the blend factor) chosen on the same "
          "scores they are reported on, a mild selection bias we guard by checking consistency "
          "across two targets and two scopes; and the Gulf D26 gap to a dedicated local model "
          "is narrowed, not closed."),
]

SVR = SETUP + [
    ("h1", "1. Regression with a tolerance tube"),
    ("p", "Support vector regression fits a function while ignoring errors smaller than a chosen "
          "tolerance $\\varepsilon$. Only points falling outside the tube of half-width "
          "$\\varepsilon$ around the function influence the fit. For a linear function "
          "$f(x)=w^{T}x+b$, the fit solves"),
    ("m", r"$\min_{w,b} \;\ \frac{1}{2}\Vert w\Vert^2 \;+\; C\sum_{i=1}^{n} \max(0,\ |\delta_i - f(x_i)| - \varepsilon),$"),
    ("p", "a convex problem. The first term prefers flat, simple functions; the second charges "
          "for points outside the tube at a rate C. Small C means smoothness matters more than "
          "fitting every point; large C the reverse. The solution turns out to depend only on "
          "the points on or outside the tube, called the support vectors; all comfortably "
          "fitted points could be deleted without changing f."),
    ("h1", "2. The kernel trick"),
    ("p", "A linear function is too rigid for our problem. The kernel trick fixes this without "
          "ever building nonlinear coordinates explicitly. The solution of the problem above "
          "can be written using only inner products between input vectors, so we may replace "
          "every inner product by a kernel function k(x, x'). The fitted function becomes"),
    ("m", r"$f(x) = \sum_{i \in \mathrm{SV}} \beta_i\, k(x_i, x) + b,$"),
    ("p", "a weighted sum of bumps centered on the support vectors. We use the Gaussian "
          "(radial basis function) kernel"),
    ("m", r"$k(x,x') = \exp(-\gamma\Vert x-x'\Vert^2),$"),
    ("p", "whose one parameter $\\gamma$ sets the width of the bumps: large $\\gamma$ gives "
          "narrow bumps and a wiggly f, small $\\gamma$ gives broad bumps and a smooth f. "
          "This is mathematically equivalent to linear regression in an infinite-dimensional "
          "space of features, but computed entirely through the n-by-n kernel matrix."),
    ("h1", "3. How we apply it to the RTOFS correction"),
    ("p", "Inputs are standardized (each coordinate scaled to unit variance) because the kernel "
          "uses Euclidean distance. The cost of solving the problem grows roughly with the "
          "square of n, so we fit on a random subsample of 20,000 pairs. We use C = 50 and "
          "$\\varepsilon = 1$, i.e. errors below one unit are free."),
    ("li", "On position alone, the single isotropic width could not fit both the sharp "
           "north-south structure and the broad east-west structure: the fit came out too "
           "smooth and was the weakest smooth model (13.5 TCHP)."),
    ("li", "On the full 35 inputs it reaches 11.63 (TCHP) and 11.20 m (D26), behind both tree "
           "families but ahead of the Gaussian process."),
    ("h2", "Where it is strong and weak here"),
    ("p", "Strong: convex problem with a unique solution, produces a smooth function with no "
          "box seams, robust to outliers because of the tolerance tube. Weak: one global bump "
          "width for all directions and places, expensive beyond a few tens of thousands of "
          "samples, and its two constants (C, $\\varepsilon$) must be chosen by hand."),
]

GPR = SETUP + [
    ("h1", "1. A probability distribution over functions"),
    ("p", "Gaussian process regression treats the unknown f itself as random. A Gaussian "
          "process is a distribution over functions with the property that the values of f at "
          "any finite set of points are jointly Gaussian. It is fully specified by a mean "
          "(taken as zero after centering) and a covariance kernel"),
    ("m", r"$\mathrm{Cov}(f(x), f(x')) = k(x, x'),$"),
    ("p", "which encodes one belief: points close in input space have similar function values. "
          "We use the anisotropic Gaussian kernel"),
    ("m", r"$k(x,x') = \sigma_f^2\, \exp\left(-\sum_{j=1}^{d} \frac{(x_j-x_j')^2}{2\,\ell_j^{\,2}}\right),$"),
    ("p", "with one length scale $\\ell_j$ per input direction: the function is allowed to vary "
          "quickly along directions with small $\\ell_j$ and slowly along directions with "
          "large $\\ell_j$."),
    ("h1", "2. From prior to prediction"),
    ("p", "Observed targets are modeled as $\\delta_i = f(x_i) + \\epsilon_i$ with Gaussian "
          "noise of variance $\\sigma_n^2$. Because everything is jointly Gaussian, "
          "conditioning on the n observations is exact linear algebra. With K the n-by-n "
          "matrix $K_{ij}=k(x_i,x_j)$ and $k_*$ the vector $k(x_i, x_*)$, the predictive mean "
          "and variance at a new point $x_*$ are"),
    ("m", r"$\mu(x_*) = k_*^{T} (K+\sigma_n^2 I)^{-1} \delta, \qquad v(x_*) = k(x_*,x_*) - k_*^{T}(K+\sigma_n^2 I)^{-1} k_* .$"),
    ("p", "The mean is a weighted average of all observed errors, with weights determined by "
          "the kernel, and the variance says how uncertain the estimate is, growing far from "
          "data. Oceanographers will recognize this as the same mathematics as optimal "
          "interpolation / objective mapping; the machine-learning form simply also learns the "
          "kernel constants ($\\sigma_f, \\ell_1..\\ell_d, \\sigma_n$) from the data, by "
          "maximizing the likelihood of the observations."),
    ("p", "The price is the inverse of an n-by-n matrix, whose cost grows with the cube of n, "
          "so exact fits are limited to a few thousand samples; we fit on a random subsample "
          "of 3,000 pairs."),
    ("h1", "3. How we apply it to the RTOFS correction"),
    ("li", "In the position-only experiment the two length scales were learned as roughly 7.5 "
           "degrees in latitude and 33 (TCHP) to 62 (D26) degrees in longitude. The error "
           "field really is stretched east to west, exactly the anisotropy Dr. Jacobs "
           "anticipated, and the resulting map is smooth with no box seams."),
    ("li", "On position alone it was the best model for D26 (12.17 m) and close to the forest "
           "for TCHP."),
    ("li", "On the full 35 inputs it was the weakest family (12.05 / 11.39): one length scale "
           "per direction over 35 standardized inputs is a blunt instrument compared to two "
           "interpretable spatial directions, and 3,000 samples cannot pin down 37 kernel "
           "constants well."),
    ("h2", "Where it is strong and weak here"),
    ("p", "Strong: principled uncertainty at every point, smooth fields suitable for published "
          "maps, and interpretable length scales that answered a physical question directly. "
          "Weak: cubic cost in n forces subsampling, and its advantage fades in high "
          "dimensions."),
]

DOCS = [
    ("Decision Trees and Random Forests", "How averaging many randomized trees fits the RTOFS error, and what its boxes mean", RF, "random_forest.pdf"),
    ("Gradient Boosting and XGBoost", "How a chain of small corrections becomes our best single model", XGB, "gradient_boosting_xgboost.pdf"),
    ("The Mixture-of-Experts Blend", "The recommended model: regional and regime specialists behind two gates", MOE, "mixture_of_experts.pdf"),
    ("Support Vector Regression", "Fitting a smooth function with a tolerance tube and a Gaussian kernel", SVR, "support_vector_regression.pdf"),
    ("Gaussian Process Regression", "Optimal interpolation that learns its own length scales", GPR, "gaussian_process_regression.pdf"),
]

if __name__ == "__main__":
    for title, sub, blocks, fname in DOCS:
        render(title, sub, blocks, fname)
