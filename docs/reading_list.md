# Phase 1 Annotated Reading List

Recommended order: 1 → 6 → 3 → 7, then acquisition papers 4–5, then stopping
papers 8–10. The objective is implementation-level understanding: for every
paper, identify the decision being made, the probability statement being used,
and the assumptions needed for that statement to be trustworthy.

## 1. Frazier (2018), *A Tutorial on Bayesian Optimization*

- **Question addressed:** How can we optimize an expensive black-box function
  using a small number of sequential evaluations?
- **Main contribution:** This tutorial connects GP regression, posterior
  uncertainty, acquisition functions, and the sequential BO loop in one
  treatment. It is the best first paper for understanding the vocabulary and
  the distinction between learning the objective and deciding where to sample.
- **Mechanics to understand:** At iteration `t`, condition the GP on
  `D_t={(x_i,y_i)}`, compute a posterior mean and covariance, maximize an
  acquisition function, observe one new value, and repeat. BO does not select
  one “true function” from the GP; it maintains a distribution over plausible
  functions and uses that distribution to make a decision.
- **Concrete example:** If two candidates have posterior summaries
  `(μ,σ)=(0.90,0.05)` and `(0.75,0.30)`, the first is attractive because of its
  predicted value while the second may be attractive because it could reveal a
  much higher value. Expected improvement combines both effects.
- **Use in this project:** Provides the conceptual base for the primer, the toy
  BO loop, and the separation among acquisition, stopping, and recommendation.
- **Assumption/failure mode:** Posterior uncertainty is conditional on the
  chosen kernel, likelihood, priors, and fitted hyperparameters. A narrow
  posterior is not automatically a calibrated statement about the real world.
- **Reading check:** Why can choosing the largest posterior mean fail? Why can
  choosing the largest posterior variance also fail?
- [Paper](https://arxiv.org/abs/1807.02811)

## 2. Shahriari et al. (2016), *Taking the Human Out of the Loop*

- **Question addressed:** How do the many BO models, acquisition functions, and
  application settings fit into a common framework?
- **Main contribution:** A broad taxonomy of BO, including GP surrogates,
  classical acquisition functions, information-based methods, constraints,
  parallel evaluations, and high-dimensional challenges.
- **Mechanics to understand:** The surrogate is the probabilistic belief model;
  the acquisition function is a decision policy computed from that model.
  Changing the acquisition does not refit the data model, and changing the
  kernel does not by itself define which experiment comes next.
- **Concrete example:** EI and UCB can use the same fitted GP but rank candidates
  differently: EI asks about improvement over an incumbent, while UCB adds an
  explicit uncertainty bonus controlled by `β`.
- **Use in this project:** Supplies a terminology map and a checklist of
  extensions and failure modes against which our benchmark can be described.
- **Assumption/failure mode:** It is a survey rather than a single algorithm;
  implementations and numerical best practices have evolved since publication.
- **Reading check:** For each method in the survey, can we say separately what
  is modeled, what is optimized, and what computational approximation is made?
- [Paper](https://ieeexplore.ieee.org/document/7352306)

## 3. Jones, Schonlau & Welch (1998), *Efficient Global Optimization*

- **Question addressed:** How should the next expensive deterministic function
  evaluation be selected during global optimization?
- **Main contribution:** Efficient Global Optimization (EGO), which combines a
  kriging/GP model with expected improvement (EI).
- **Core calculation:** For maximization,

  $$
  EI(x)=(\mu(x)-f_{best})\Phi(z)+\sigma(x)\phi(z),\qquad
  z=\frac{\mu(x)-f_{best}}{\sigma(x)}.
  $$

  The first term is predicted improvement weighted by its probability; the
  second is the value of uncertainty. EI has the same units as the objective.
- **Concrete example:** If `f_best=1`, `μ=1`, and `σ=0.2`, then `z=0` and
  `EI≈0.2×0.399=0.0798`, even though the predicted improvement is zero. The
  point is valuable because the model is uncertain there.
- **Use in this project:** Defines the baseline acquisition and makes the
  incumbent convention explicit. The implementation uses LogEI for numerical
  stability while preserving the ideal EI maximizer.
- **Assumption/failure mode:** Classical EGO is oriented toward deterministic
  observations. With noise, the observed maximum can be a noise spike and the
  definition of the incumbent needs more care.
- **Reading check:** What happens to EI when `σ→0`? Under what condition does it
  reduce to positive predicted improvement?
- [Paper](https://doi.org/10.1023/A:1008306431147)

## 4. Srinivas et al. (2010), *GP Optimization in the Bandit Setting*

- **Question addressed:** Can exploration and exploitation be balanced while
  obtaining theoretical regret guarantees?
- **Main contribution:** GP-UCB and bounds involving the maximum information
  gain of the kernel. The method selects

  $$
  x_t\in\arg\max_x \mu_{t-1}(x)+\sqrt{\beta_t}\sigma_{t-1}(x).
  $$

- **Mechanics to understand:** `μ` supplies exploitation, `σ` supplies
  exploration, and `β_t` controls the size of the uncertainty bonus. The paper
  primarily analyzes cumulative regret, whereas this project evaluates the
  final recommendation using simple regret.
- **Concrete example:** Candidate A has `(μ,σ)=(0.9,0.05)` and B has
  `(0.7,0.30)`. With `β=1`, their UCB scores are `0.95` and `1.00`, so B wins.
  With `β=0.04`, the scores are `0.91` and `0.76`, so A wins.
- **Use in this project:** Gives a controlled way to vary exploration and shows
  why the acquisition parameter must be logged rather than treated as a minor
  implementation detail.
- **Assumption/failure mode:** The guarantees require a valid confidence
  construction and assumptions on the objective/kernel relationship. A fitted
  real-world GP does not automatically inherit the theorem. Pointwise credible
  intervals also do not automatically become simultaneous confidence bands.
- **Reading check:** Why can low cumulative regret and low final simple regret
  favor different sampling behavior?
- [Paper](https://arxiv.org/abs/0912.3995)

## 5. Wang & Jegelka (2017), *Max-value Entropy Search*

- **Question addressed:** Can an evaluation be chosen according to how much it
  teaches us about the unknown optimal value rather than its immediate gain?
- **Main contribution:** Max-value Entropy Search (MES), which maximizes mutual
  information between a prospective observation and `f*`, the unknown maximum
  value. It is an information-theoretic alternative to improvement-based rules.
- **Mechanics to understand:** MES samples or approximates a distribution over
  `f*` and scores how much observing `y(x)` is expected to reduce its entropy.
  This is different from choosing the point with maximum uncertainty: useful
  uncertainty must be relevant to the optimum.
- **Concrete example:** A highly uncertain low-value region may have large
  variance but almost no chance of changing our belief about `f*`; MES can
  prefer a moderately uncertain region near a plausible competing peak.
- **Use in this project:** Provides a fifth acquisition family and a useful
  contrast with EI, UCB, and Thompson sampling when analyzing stopping behavior.
- **Assumption/failure mode:** Sampling `f*`, approximating entropy, and
  optimizing the acquisition add computational and Monte Carlo error. The
  result still depends on the fidelity of the GP posterior.
- **Reading check:** What random variables appear on the two sides of the mutual
  information, and why is learning `f*` not identical to learning `x*`?
- [Paper](https://proceedings.mlr.press/v70/wang17e.html)

## 6. Balandat et al. (2020), *BoTorch*

- **Question addressed:** How can modern, composable, Monte Carlo BO be built
  efficiently on automatic differentiation and tensor computation?
- **Main contribution:** BoTorch represents acquisition computations directly
  in PyTorch, enabling flexible models, batch acquisitions, constraints, and
  gradient-based acquisition optimization.
- **Mechanics to understand:** Reparameterization writes posterior samples as a
  differentiable transformation of base random samples. Fixed base samples make
  a Monte Carlo acquisition surface smoother during numerical optimization.
- **Concrete example:** Instead of deriving a closed form for a batch of three
  candidates, draw joint posterior samples, compute improvement for each draw,
  average it, and backpropagate through that average to the candidate locations.
- **Use in this project:** It is the implementation framework for the toy loop
  and future benchmark. Model fitting, acquisition construction, acquisition
  optimization, observation, and recommendation should remain separate steps.
- **Assumption/failure mode:** A flexible library cannot choose a scientifically
  suitable prior, stopping tolerance, or evaluation metric. Default settings
  are software choices, not evidence that the resulting uncertainty is valid.
- **Reading check:** Why must posterior samples for a batch preserve joint
  correlation? What problem do restarts and raw samples solve?
- [Paper](https://arxiv.org/abs/1910.06403)

## 7. Ament et al. (2023), *Unexpected Improvements to Expected Improvement*

- **Question addressed:** Why can EI optimization fail numerically even when
  the mathematical definition is sensible?
- **Main contribution:** LogEI and related numerically stable formulations that
  avoid underflow and vanishing gradients in low-improvement regions.
- **Mechanics to understand:** Because logarithm is strictly increasing,
  `argmax EI(x)=argmax log(EI(x))` when EI is positive. Stable computations use
  identities tailored to extreme normal-tail probabilities instead of first
  rounding a tiny EI value to zero.
- **Concrete example:** EI values `10^-40` and `10^-60` may both become
  numerically unusable in a computation, while their logs `-92.1` and `-138.2`
  remain distinguishable and optimizable.
- **Use in this project:** Justifies using `LogExpectedImprovement` in the
  example and pilot, and warns against interpreting a computed zero acquisition
  value as scientific evidence that no useful point remains.
- **Assumption/failure mode:** Numerical stability repairs the calculation, not
  a misspecified kernel, bad length scale, or poorly calibrated posterior.
- **Reading check:** Why can LogEI change numerical optimization behavior while
  leaving the ideal mathematical maximizer unchanged?
- [Paper](https://arxiv.org/abs/2310.20708)

## 8. Makarova et al. (2022), *Automatic Termination for HPO*

- **Question addressed:** When should hyperparameter optimization terminate
  because the remaining potential gain is unlikely to justify another trial?
- **Main contribution:** A probability-aware termination framework that bounds
  remaining optimization potential while explicitly considering statistical
  error. Stopping is treated as a risk decision, not merely lack of recent
  improvement.
- **Mechanics to understand:** A practical rule must specify a tolerance, a
  permissible error probability, and how repeated decisions across iterations
  affect that error. The output is a statement about remaining opportunity,
  conditional on the model and approximations.
- **Concrete example:** “Stop if the probability of gaining more than 0.01 is
  below 0.05” communicates both practical value and risk; “stop after five flat
  iterations” does neither.
- **Use in this project:** Motivates explicit error budgets and measurement of
  false stops, rather than judging a stopping method only by saved evaluations.
- **Assumption/failure mode:** Hyperparameter-tuning tasks may differ from the
  controlled synthetic functions used here. Reusing the criterion requires an
  explicit mapping of assumptions, not only a code translation. Repeated
  checking can inflate the total chance of an erroneous stop.
- **Reading check:** What probability is controlled, over what source of
  randomness, and is the guarantee per check or over the whole trajectory?
- [Paper](https://proceedings.mlr.press/v188/makarova22a.html)

## 9. Ishibashi et al. (2023), *A Stopping Criterion for BO*

- **Question addressed:** Can BO stop based on how much expected decision
  quality is still changing rather than on raw incumbent stagnation?
- **Main contribution:** A stopping criterion based on the change in expected
  minimum simple regret, aligning termination with the final recommendation.
- **Mechanics to understand:** Simple regret concerns the value lost by the
  returned point. The criterion estimates a posterior decision quantity across
  iterations; it therefore depends on posterior integration and numerical
  approximations, not just the last observed value.
- **Concrete example:** Two iterations may produce no new incumbent but sharply
  reduce uncertainty around a competing peak. A stagnation rule sees no
  progress, while a regret-oriented rule can recognize the improved decision.
- **Use in this project:** Clarifies why stopping should target recommendation
  quality and supplies a comparison between heuristic and posterior rules.
- **Assumption/failure mode:** Posterior regret inherits model misspecification,
  and Monte Carlo or finite-catalog approximations can themselves create an
  apparently small change. These errors should be measured.
- **Reading check:** Could the estimated expected regret stabilize at the wrong
  value? Construct a narrow-peak example in which this happens.
- [Paper](https://proceedings.mlr.press/v206/ishibashi23a.html)

## 10. Wilson (2024), *Stopping BO with Probabilistic Regret Bounds*

- **Question addressed:** Can we stop when the recommended solution is probably
  within a user-chosen tolerance of the unknown optimum?
- **Main contribution:** Frames stopping through probabilistic regret bounds,
  connecting a practical tolerance `ε` and risk level `δ` to the recommendation.
- **Target condition:** A rule may seek a statement of the form

  $$
  P\left(r_t\leq\epsilon\mid D_t\right)\geq 1-\delta,
  \qquad r_t=f^*-f(\hat x_t).
  $$

  This is more interpretable than “the acquisition became small,” because it
  directly refers to the loss incurred by the returned point.
- **Concrete example:** With `ε=0.02` and `δ=0.05`, the intended interpretation
  is that, under the fitted posterior, the recommendation has at least 95%
  probability of being within 0.02 of the optimum. It is not automatically a
  95% real-world repeated-sampling guarantee.
- **Use in this project:** Direct conceptual basis for S5, for reporting false
  stops, and for separating the stopping rule from the recommendation rule.
- **Assumption/failure mode:** The probability is conditional on the GP model,
  fitted hyperparameters, domain approximation, and Monte Carlo accuracy.
  Repeated checks and optional stopping also require explicit care.
- **Reading check:** Which part of the claimed confidence is posterior/model
  probability, which part is numerical error, and how will benchmark truth be
  used without leaking into the stopping decision?
- [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b204de7078301292a8876a762eed3dcb-Abstract-Conference.html)

## Two papers to study first

1. **Frazier (2018):** read Sections 1–4 first and be able to narrate the entire
   model → acquisition → observation → refit loop without equations.
2. **Wilson (2024):** identify the exact regret random variable, probability
   level, computational approximation, and assumptions behind the stop.

Read **BoTorch (2020)** alongside implementation and **LogEI (2023)** before
trusting very small acquisition values as evidence for stopping. For every
paper, write one sentence answering: “What would make this method confidently
wrong?” That question is central to the benchmark design.
