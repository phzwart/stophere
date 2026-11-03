# Bayesian Optimization: a Practical Primer

## 1. The problem

Bayesian optimization (BO) searches for an input

$$
x^*\in\arg\max_{x\in\mathcal X} f(x)
$$

when evaluating the unknown objective `f` is expensive. An evaluation may be a
physical experiment, simulation, or model-training run. BO is useful when we
can afford tens or hundreds of evaluations, not millions.

An observation is

$$
y_i=f(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma_n^2).
$$

The objective is the real input-output relationship. The surrogate is a model
of that relationship; it is not the objective itself.

## 2. The BO loop

1. Evaluate a small initial design.
2. Fit a probabilistic surrogate to `(x, y)`.
3. Use an acquisition function to score candidate inputs.
4. Evaluate the best-scoring candidate on the real objective.
5. Refit and repeat until a stopping rule fires or the budget is exhausted.
6. Return an input using a separately defined recommendation rule.

These roles must remain separate:

| Object | Question answered |
|---|---|
| Objective `f(x)` | What result does the real experiment produce? |
| GP surrogate | What functions remain plausible after the observations? |
| Acquisition `a(x)` | Which `x` should be evaluated next? |
| Stopping rule | Is another evaluation worth performing? |
| Recommendation | Which `x` should be returned at the end? |

## 3. Gaussian-process surrogate

A Gaussian process (GP) places a probability distribution over functions:

$$
f\sim\mathcal{GP}(m(x),k(x,x')).
$$

It does **not** search for one function and declare that function to be true.
Before data, many functions are plausible. After observing data, functions that
do not agree with the observations become less plausible. The result is a
posterior distribution over functions.

For training inputs `X`, observations `y`, and a candidate `x`, conditioning a
zero-mean GP gives

$$
\mu(x)=k_{xX}(K+\sigma_n^2I)^{-1}y,
$$

$$
\sigma^2(x)=k(x,x)-k_{xX}(K+\sigma_n^2I)^{-1}k_{Xx}.
$$

The posterior mean is the model's current prediction. The posterior variance
is uncertainty **under the model assumptions**. The covariance vector `k_xX`
determines how strongly each observation influences the new point.

### Three kinds of quantities that are often confused

| Kind | Examples | What it controls |
|---|---|---|
| Latent/model values | `f(x_i)` | Unknown function values inferred by the GP |
| GP hyperparameters | length scale, output scale, noise, mean | Which functions the model considers plausible |
| Decision settings | UCB `β`, EI threshold `ξ`, stopping `ε,δ` | How the posterior is used to sample or stop |

A length scale changes the posterior itself. UCB's `β` does not change the
posterior; it changes the decision made from that posterior. This distinction is
important when diagnosing a bad BO trajectory.

### 3.1 Kernel family: what shapes are plausible?

The kernel is a rule for covariance: it says how similar the function values at
two inputs should be before seeing the observations. Two common stationary
kernels are:

$$
k_{RBF}(d)=\sigma_f^2\exp\left(-\frac{d^2}{2\ell^2}\right),
$$

$$
k_{Mat\acute ern\,5/2}(d)=\sigma_f^2
\left(1+\frac{\sqrt5d}{\ell}+\frac{5d^2}{3\ell^2}\right)
\exp\left(-\frac{\sqrt5d}{\ell}\right),
$$

where `d=|x-x'|` in one dimension.

- **RBF effect:** sampled functions are extremely smooth. It is appropriate
  when the process changes gradually and sharp local features are implausible.
- **Matérn effect:** sampled functions can be rougher. Matérn-5/2 is twice
  differentiable and is often a more forgiving BO default.
- **Life analogy:** the RBF kernel describes a freshly paved rolling road;
  Matérn describes a natural hiking trail. Both have nearby points that resemble
  one another, but the hiking trail permits more irregular local changes.
- **Failure mode:** if the real function has a cliff or narrow spike, an overly
  smooth kernel can interpolate confidently across it and make BO stop early.

The kernel family is usually chosen by scientific knowledge and validation,
not reliably “discovered” from a tiny BO dataset.

### 3.2 Length scale `ℓ`: how far does information travel?

For the RBF kernel, correlation at distance `d` is

$$
\rho(d)=\exp\left(-\frac{d^2}{2\ell^2}\right).
$$

Consider two inputs separated by `d=0.2`:

| Length scale | Correlation | Interpretation |
|---:|---:|---|
| `ℓ=0.1` | `exp(-2)=0.135` | Weak relationship; information stays local |
| `ℓ=0.2` | `exp(-0.5)=0.607` | Moderate transfer |
| `ℓ=0.5` | `exp(-0.08)=0.923` | Strong relationship; information travels far |

- **Small `ℓ`:** permits rapid wiggles. The mean follows observations locally,
  but uncertainty rises quickly between points. BO tends to explore more.
- **Large `ℓ`:** assumes slow variation. A few observations influence most of
  the domain, producing smoother means and lower posterior uncertainty.
- **Life analogy:** a temperature reading in a small, uniformly heated room may
  tell us about locations several meters away, corresponding to a long length
  scale. On a patchy grill containing hot coals and cold edges, a reading tells
  us only about a small neighborhood, corresponding to a short length scale.
- **Stopping consequence:** an overestimated `ℓ` is especially dangerous. It
  can spread confidence too far, hide an unsampled narrow peak, shrink EI, and
  trigger a false stop. An underestimated `ℓ` is conservative but may waste
  evaluations by keeping uncertainty high everywhere.

Input scaling matters. A length scale of `0.2` means something different when
`x∈[0,1]` than when `x∈[0,1000]`. Normalize bounded inputs before fitting or use
priors expressed on the original scientific scale.

### 3.3 Output scale `σ_f²`: expected vertical variation

In `k(x,x')=σ_f²ρ(x,x')`, the output scale multiplies every prior covariance.
At one point, the prior variance is `k(x,x)=σ_f²`, so the prior standard
deviation is `σ_f`.

- **Small output scale:** the model expects the latent function to remain close
  to its mean.
- **Large output scale:** the model considers tall peaks and deep valleys
  plausible and usually produces wider posterior intervals.
- **Life analogy:** if the mean is the center position of a speaker cone, the
  output scale is its volume knob: it controls the expected amplitude, not the
  horizontal frequency of changes.
- **Simple example:** with correlation `ρ=0.6`, `σ_f²=1` gives covariance `0.6`;
  `σ_f²=4` gives covariance `2.4` and prior standard deviation `2` instead of
  `1`. The correlation pattern is unchanged, but its vertical scale is larger.
- **Failure mode:** too small an output scale makes genuinely high values seem
  implausible and can suppress exploration. Too large a value produces broad
  intervals and may cause unnecessary exploration.

Standardizing `y` makes a prior centered near `σ_f²≈1` easier to interpret.

### 3.4 Observation noise `σ_n²`: how much should data be trusted?

The observation model is

$$
y_i=f(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma_n^2).
$$

Adding `σ_n²I` to the training covariance lets observations deviate from the
latent function. For one zero-mean observation at `x_1` with prior variance
`σ_f²`, the posterior mean at the same point is

$$
\mu(x_1)=\frac{\sigma_f^2}{\sigma_f^2+\sigma_n^2}y_1.
$$

If `σ_f²=1` and `y_1=2`, then:

| Noise variance | Posterior mean at `x_1` | Meaning |
|---:|---:|---|
| `0.01` | `1/1.01×2≈1.98` | Almost fully trust the reading |
| `1` | `1/2×2=1` | Shrink strongly toward the prior mean |
| `9` | `1/10×2=0.2` | Treat the reading as mostly noise |

- **Life analogy:** repeated bathroom-scale measurements fluctuate because the
  scale is imperfect. Noise describes that measurement error; it does not mean
  a person's underlying mass jumps by the same amount each second.
- **Too little noise:** the GP chases random fluctuations, learns an artificially
  short length scale, and becomes overconfident at observed points.
- **Too much noise:** real structure is dismissed; the posterior remains broad,
  and BO may repeatedly sample regions it has already measured.
- **Condition:** if noise changes with `x` (heteroscedasticity), one global
  `σ_n²` is misspecified. Replicates or a heteroscedastic model may be needed.

### 3.5 Mean function: where does the model return far from data?

The mean function `m(x)` is the baseline before covariance-based corrections.
A constant mean `c` is common:

$$
m(x)=c.
$$

- **Life analogy:** when weather stations provide no local evidence, the model
  falls back to a reference climate temperature.
- **Impact:** in data-rich regions the observations dominate; in sparse or
  distant regions predictions move back toward the prior mean. An implausibly
  high mean can make unexplored regions look attractive, while a low mean can
  discourage exploration.
- **Simple example:** with standardized outputs, a zero mean says that far from
  data the response is expected near the training average. With raw outputs
  around `100`, a fixed zero mean creates an unreasonable extrapolation target.

### 3.6 ARD length scales: one sensitivity scale per input

Automatic relevance determination (ARD) uses

$$
k(x,x')=\sigma_f^2\exp\left[-\frac12
\sum_{j=1}^{p}\frac{(x_j-x'_j)^2}{\ell_j^2}\right].
$$

Each `ℓ_j` controls how quickly the response changes along coordinate `j`.
After comparable input scaling, a large fitted `ℓ_j` suggests slow variation
along that dimension; a small value suggests sensitivity.

Suppose temperature ranges from `0–1000 °C` and concentration from `0–1`.
Without normalization, a numerical length scale cannot be compared across the
two units. After mapping both inputs to `[0,1]`, fitted values
`ℓ_temperature=0.1` and `ℓ_concentration=1.5` suggest faster variation with
temperature. This is evidence under the model, not causal feature importance.

With few observations and many dimensions, ARD is weakly identified: many
combinations of length scales explain the same data. Priors, bounds, dimension
reduction, or more observations may be necessary.

### 3.7 Jitter is not observation noise

Jitter is a tiny diagonal term added for numerical stability before a Cholesky
factorization:

$$
K_{stable}=K+\sigma_n^2I+\eta I,
$$

where `η` may be around `10^-6` in the chosen numerical precision.

- **Noise** is part of the statistical model and should represent measurement
  variation.
- **Jitter** is an implementation safeguard against nearly singular matrices
  and floating-point error.

Increasing jitter until the program runs can silently change predictions. It
should be logged, and a large required jitter should trigger a diagnosis of
duplicate points, scaling, kernel parameters, or precision.

### 3.8 Priors and constraints

Positive hyperparameters are often represented by transformed unconstrained
variables, for example `ℓ=softplus(r)` or `ℓ=exp(r)`. Priors and bounds prevent
pathological solutions such as an almost-zero length scale or enormous noise.

- A **prior** says which values were plausible before these observations and
  contributes to maximum a posteriori fitting.
- A **constraint** defines allowed numerical values.
- Neither replaces a scientific justification or predictive checks.

With only six data points, a weak likelihood may not distinguish `ℓ=0.2` from
`ℓ=2`. A sensible prior can stabilize fitting, but a strong incorrect prior can
also create confident error. Prior sensitivity should therefore be tested.

### 3.9 Hyperparameter interactions and identifiability

Hyperparameters are not learned independently.

- Short length scale plus low noise can explain jagged observations as real
  signal.
- Long length scale plus high noise can explain the same observations as a
  smooth function measured imprecisely.
- Mean and output scale can trade off when data cover only a small region.
- Output scale and noise can both increase observed marginal variance.

This is an identifiability problem: multiple explanations fit the same small
dataset. One fitted optimum should not be mistaken for certainty. Useful checks
include multiple optimizer restarts, parameter trajectories, profile plots,
prior sensitivity, and predictive performance on held-out or repeated data.

### 3.10 How the model is fitted

For mean vector `m` and covariance
`C_θ=K_θ+σ_n²I`, the exact log marginal likelihood is

$$
\log p(y\mid X,\theta)=
-\frac12(y-m)^T C_\theta^{-1}(y-m)
-\frac12\log|C_\theta|
-\frac n2\log(2\pi).
$$

The terms have different jobs:

1. **Data fit:** `-(1/2)(y-m)^T C^-1(y-m)` rewards explaining the observed
   pattern, measured in covariance-adjusted distance.
2. **Complexity/volume penalty:** `-(1/2)log|C|` discourages explaining every
   possible dataset equally well. It implements an automatic Occam tradeoff.
3. **Normalization:** `-(n/2)log(2π)` makes this a valid log density and is
   constant with respect to the hyperparameters.

A practical fitting sequence is:

1. Normalize bounded inputs and usually standardize outputs.
2. Initialize raw length scale, output scale, noise, and mean parameters.
3. Transform parameters so required quantities stay positive.
4. Build `C_θ` from the current parameters.
5. Compute a Cholesky factor `L` such that `C_θ=LL^T`.
6. Use triangular solves to evaluate the likelihood; do **not** explicitly form
   `C_θ^-1` in implementation.
7. Use automatic differentiation to obtain likelihood gradients.
8. Optimize the negative log likelihood or negative log posterior, preferably
   from multiple starts when the dataset is small.
9. Record the chosen parameters, objective value, warnings, jitter, and boundary
   hits.
10. Refit after each new BO observation, normally warm-starting from the prior
    iteration while retaining a recovery path if optimization fails.

#### A two-point numerical intuition

For two noiseless inputs separated by `d`, an RBF kernel with unit output scale
has

$$
K=\begin{bmatrix}1&\rho\\\rho&1\end{bmatrix},
\qquad \rho=\exp\left(-\frac{d^2}{2\ell^2}\right).
$$

If the observed values are similar, such as `y=[1,1]`, a larger correlation
can explain them efficiently. If they are opposite, such as `y=[1,-1]`, a very
large correlation makes that observation pattern surprising, pushing the fit
toward a shorter length scale or more noise. The log determinant prevents the
optimizer from judging fit only by pointwise residual size.

Maximum marginal likelihood or MAP fitting is empirical Bayes: downstream
predictions usually plug in one hyperparameter estimate and ignore uncertainty
about the hyperparameters themselves. With very little data, fully Bayesian
integration or at least sensitivity analysis may give more honest uncertainty.

### 3.11 When posterior uncertainty is meaningful

The plotted GP interval is meaningful only conditional on several conditions:

- the objective is adequately represented by the kernel and mean;
- the likelihood represents the noise distribution and its dependence on `x`;
- inputs, outputs, and bounds are correctly scaled;
- fitting found a reasonable solution rather than a boundary or local optimum;
- numerical linear algebra and Monte Carlo errors are controlled;
- the candidate domain has not omitted important regions.

Diagnostics should include fitted-parameter trajectories, standardized
residuals, leave-one-out predictions, replicate behavior, interval coverage on
synthetic benchmarks, prior sensitivity, and warnings or boundary hits.

Common deviations and possible responses are:

| Deviation | Symptom | Possible response |
|---|---|---|
| Nonstationarity | Smooth region and sharp region share one bad `ℓ` | Input warping, local/nonstationary GP, partitioning |
| Heteroscedastic noise | Residual spread changes with `x` | Replicates or heteroscedastic likelihood |
| Discontinuity | GP smooths across a jump | Change kernel/model or divide known regimes |
| Categorical inputs | Euclidean distance is meaningless | Categorical/mixed-space kernel |
| High dimension | ARD unstable, acquisition hard to optimize | Structure, embeddings, active subspaces |
| Outliers/heavy tails | Noise estimate or mean is distorted | Robust likelihood, data-quality investigation |

### 3.12 Important knobs that are not GP hyperparameters

These settings affect BO decisions but are not fitted properties of the GP:

- **UCB `β`:** exploration weight. With scores
  `μ+sqrt(β)σ`, increasing `β` can switch the choice from a high-mean point to a
  high-uncertainty point.
- **EI/PI `ξ`:** improvement margin. Replacing `f_best` with `f_best+ξ` asks for
  a more substantial improvement; conventions differ, so the sign and formula
  must be documented.
- **Acquisition optimizer restarts/raw samples:** numerical search effort. Too
  little effort can return a bad local maximum even when the acquisition is
  correct.
- **Batch size `q`:** number of points chosen before observing outcomes. Batch
  acquisition must account for correlation and lost adaptivity.
- **Stopping tolerance `ε`:** how much final value loss is practically
  acceptable. It should come from scientific or operational utility.
- **Stopping risk `δ`:** tolerated posterior probability of exceeding `ε`.
  Smaller `δ` demands stronger evidence and usually more evaluations.

For example, `ε=0.01, δ=0.05` and `ε=0.10, δ=0.20` describe fundamentally
different decisions even with the identical GP posterior. They should be
reported as user-defined decision conditions, not learned model facts.

## 4. Exploration and exploitation

An acquisition function is cheap to evaluate because it uses the surrogate.
It balances high predicted value (exploitation) and high uncertainty
(exploration).

| Strategy | Intuition | Important parameter or risk |
|---|---|---|
| EI / LogEI | Expected amount by which `x` beats the incumbent | Can become tiny numerically; use LogEI |
| PI / LogPI | Probability of beating a threshold | May prefer safe, tiny improvements |
| UCB | `μ(x) + sqrt(β)σ(x)` | `β` controls exploration |
| Thompson sampling | Sample one plausible function and maximize it | Must preserve joint posterior correlation |
| MES | Learn the unknown maximum value | More computation and approximation |

For noiseless maximization, expected improvement over best value `f_best` is

$$
EI(x)=(\mu-f_{best})\Phi(z)+\sigma\phi(z),\quad
z=\frac{\mu-f_{best}}{\sigma}.
$$

The first term rewards predicted improvement; the second rewards uncertainty.

## 5. Recommendation and regret

In the noiseless pilot, a transparent recommendation is the best evaluated
point:

$$
\hat x_t\in\arg\max_{x_i\in D_t} f(x_i).
$$

Simple regret is

$$
r_t=f^*-f(\hat x_t).
$$

The recommendation is acceptable for tolerance `ε` when `r_t≤ε`. In a real
experiment `f*` is unknown; benchmark truth is used only by the evaluator, not
by the optimizer or stopping rule.

## 6. Stopping rules

- **Fixed budget:** reproducible cost control, but no quality claim.
- **Stagnation:** stop after little recent improvement; easily fooled by a
  missed region.
- **Negligible EI:** stop when maximum diagnostic EI is small; inherits GP
  assumptions and numerical sensitivity.
- **Confidence gap:** compare the largest plausible value with a lower bound at
  the recommended point. Pointwise intervals do not automatically give global
  or repeated-check coverage.
- **Posterior acceptable-regret probability:** stop when the joint posterior
  assigns high probability to regret at most `ε`; Monte Carlo error and model
  misspecification must be reported.

No recent improvement is evidence about the sampled trajectory, not proof that
the global optimum has been found. A narrow, remote peak is the canonical
counterexample.

## 7. Tool roles

- **BoTorch:** acquisition functions and Bayesian-optimization building blocks.
- **GPyTorch:** scalable GP models, kernels, likelihoods, and fitting machinery.
- **Ax:** higher-level experiment management built around BoTorch.
- **Optuna:** general hyperparameter optimization with multiple samplers.
- **SciPy optimizers:** deterministic numerical optimization; useful for
  maximizing a fitted acquisition function, not a replacement for uncertainty.

## 8. Project-specific validity checks

1. Keep benchmark truth hidden from the optimizer and stopper.
2. Distinguish field seed, optimizer seed, fitting seed, sampling seed, and
   noise seed.
3. Record fitted length scales, noise, warnings, and failures each iteration.
4. Score early stopping by regret and false-stop rate, not stopping time alone.
5. Pair methods on the same initial designs and candidate catalogues.
6. Treat fitted-GP probability statements as conditional on the model.

## Glossary

**Posterior:** updated distribution after observing data. **Kernel:** covariance
rule connecting inputs. **Hyperparameter:** a model parameter fitted outside the
latent function values. **Incumbent:** current best reference value.
**Acquisition value:** utility of evaluating a candidate next. **Simple regret:**
value lost by the final recommendation. **Calibration:** whether stated
uncertainties match observed frequencies.
