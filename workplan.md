# Workplan: When is Bayesian optimization done?

## Project goal

Determine when Bayesian optimization (BO) can stop with an acceptably good solution, how many evaluations different stopping rules save, and which conditions cause them to stop too early.

Use **BoTorch** to compare acquisition strategies and stopping criteria on controlled target functions. Produce a reproducible GitHub repository containing explanations, code, examples, configurations, and results.

The central question is:

> For a given landscape and acquisition strategy, which stopping rule saves evaluations while maintaining an acceptable probability of returning a good solution?

The first task is to build shared understanding. Start with a basic BO primer and one worked example, supported by an annotated reading list. The student and other collaborators should be able to follow the main ideas, assumptions, and limitations.

## Scope

- Scalar maximization over bounded domains.
- One-dimensional examples for learning; two- and three-dimensional functions for the main study.
- A function interface that can later support N dimensions.
- Sequential evaluations, with one point selected per iteration.
- Gaussian-process surrogates implemented with BoTorch.
- Five acquisition strategies and five stopping approaches.
- A noiseless pilot, followed by a focused noisy extension.
- At least five optimizer seeds per function-policy combination in the pilot.

Start with existing methods and a small, transparent implementation. New acquisition functions, neural surrogates, large-scale GP methods, multi-objective optimization, constraints, asynchronous batches, and dashboards are outside the initial scope.

## Phase 1: Build a basic BO primer and worked example

### Task 1.1: Assemble and annotate a ten-paper reading list

Use the starting list at the end of this workplan. It is a selection relevant to this project, not a ranking by citation count or popularity.

For each paper, record:

- The problem it addresses.
- Its main contribution in a few sentences.
- The assumptions relevant to this project.
- One important equation, algorithm, or figure, explained in plain language.
- Its relevance to implementation or evaluation.
- Any unclear points or limitations.

Begin with an introductory source and the BoTorch tutorial. Read the specialized acquisition and stopping papers as their concepts become relevant. Understanding every proof is not required to begin the worked example.

**Deliverable:** `docs/reading_list.md`.

### Task 1.2: Create a short introductory guide

Write a primer of approximately 4–6 pages, excluding references. Use AI to draft and explain material, then check the claims, equations, and citations against the original sources. Refine unclear explanations so the guide is useful to the student and other collaborators.

Cover:

1. What BO is useful for, and why expensive function evaluations motivate it.
2. The objective function, observations, and measurement noise.
3. Gaussian-process predictions: mean, uncertainty, kernels, and length scales.
4. Exploration and exploitation.
5. The difference between an objective function, an acquisition strategy, a recommendation rule, and a stopping criterion.
6. An overview of the five acquisition strategies in this project.
7. Simple regret and the meaning of an acceptably good solution.
8. Why no recent improvement does not necessarily imply completion.
9. How an incorrect surrogate can become confidently wrong.
10. The roles of BoTorch, GPyTorch, Ax, Optuna, and conventional optimizers such as those in SciPy. This is an overview, not a comparison requiring separate implementations.

Include a short glossary and a diagram or worked figure of the BO loop.

**Deliverable:** `docs/bo_primer.md`.

### Task 1.3: Build one visual BoTorch example

Implement a separate one-dimensional toy example, such as

$$
f(x)=\sin(6\pi x)+0.4x,\qquad x\in[0,1].
$$

Use a GP and expected improvement, with a fixed random seed. Show several iterations of:

- The true function and observed points.
- The posterior mean and latent uncertainty.
- The acquisition function.
- The next selected point.
- The best evaluated value so far.

Explain what is fitted, what is optimized, and what counts as an objective evaluation. Plotting the true function is for illustration; the optimizer only receives the observations it requests.

Use the numerically stable log-EI implementation where appropriate. Record the package versions that successfully run the example.

**Deliverable:** `examples/01_bo_intro.ipynb`, with a clear way to run it from a clean environment.

### Phase 1 completion

- [x] Annotated reading list is available.
- [x] The primer explains the main concepts with checked sources.
- [x] The example runs and its figures make the BO loop understandable.
- [x] Questions and unclear points are recorded for discussion.

Completion evidence is summarized in [`docs/phase1_results.md`](docs/phase1_results.md).

The aim is shared understanding. Use discussion to improve the explanations and identify where another example would help.

## Phase 2: Build and document the target-function bank

Organize functions by the properties they test. Implement or reuse the following bank.

| Family | Property being tested | Dimensions |
| --- | --- | --- |
| Quadratic peak | One smooth global maximum | 2, 3; scalable to N |
| Rotated ellipsoid | Strong anisotropy and coupled coordinates | 2, 3; scalable to N |
| Rippled dominant peak | A broad trend with short-scale local variation | 2, 3; scalable to N |
| Competing Gaussian peaks | Several separated, similarly good peaks | 2, 3; scalable to N |
| Rosenbrock | A narrow, curved valley before conversion to maximization | 2, 3; scalable to N |
| Ackley | Local structure around a dominant optimum and a relatively flat outer region | 2, 3; scalable to N |
| Rastrigin | Repeated competing local structure | 2, 3; scalable to N |
| Hidden narrow peak | A broad attractive region plus a narrow, better remote peak | 2, 3; scalable to N |
| Gaussian random field: long scale | Broad stochastic landscape features | 2, 3; scalable to N |
| Gaussian random field: short scale | Many small-scale features | 2, 3; scalable to N |
| Gaussian random field: rougher covariance | Sensitivity to covariance smoothness assumptions | 2, 3; scalable to N |
| Branin | Standard benchmark with equivalent global minima before negation | 2 only |
| Hartmann-3 | Standard multimodal benchmark | 3 only |

The first eleven families in both dimensions, plus the two anchors, give **24 cases**.

### Implementation requirements

- [ ] Use a common input domain of `[0, 1]^d`, with documented transformations to native domains.
- [ ] Express all objectives as maximization problems.
- [ ] Document output scaling, parameters, and known or numerical reference optima.
- [ ] Keep the latent function separate from the observation-noise wrapper.
- [ ] Support batched evaluation and an explicit dimension argument.
- [ ] Keep Branin and Hartmann-3 as fixed-dimensional anchors.
- [ ] Generate 2D plots and several informative slices for 3D functions.
- [ ] Check known values, dimensional consistency, and repeatability.

For Gaussian random fields, generate a field once and hold it fixed throughout a run. A **field seed** defines the landscape; an **optimizer seed** defines an optimization attempt. Do not resample the latent field at each query.

Document the field generator and its approximation. A finite spectral approximation to a Matérn covariance does not reproduce all fine-scale properties of an exact Matérn sample path. Distinguish rapid spatial variation from observation noise and mathematical nonsmoothness.

**Deliverables:** `src/benchmarks.py`, `tests/test_benchmarks.py`, `docs/benchmarks.md`, and a reproducible gallery example.

## Phase 3: Implement five acquisition strategies

Use the same surrogate specification, initial design, domain, evaluation budget, and recommendation rule across methods.

| ID | Strategy | Target of the next evaluation | BoTorch route for the initial study |
| --- | --- | --- | --- |
| A1 | Expected improvement (EI) | Expected positive gain over the incumbent | `LogExpectedImprovement` |
| A2 | Probability of improvement (PI) | Probability of exceeding the incumbent by a specified margin | `LogProbabilityOfImprovement` |
| A3 | Upper confidence bound (UCB) | Predicted value plus an uncertainty allowance | `UpperConfidenceBound` |
| A4 | Thompson sampling (TS) | Maximum of one jointly sampled plausible function | `MaxPosteriorSampling` |
| A5 | Max-value entropy search (MES) | Information about the unknown maximum value | `qMaxValueEntropy`, with `q=1` |

These are established representative strategies. Explain their assumptions and expected failure modes rather than presenting them as a verified popularity ranking.

### Common protocol

For the first controlled screen, evaluate the continuous functions through a **shared finite catalogue of candidate points**. This makes the best allowed value exactly computable for scoring, including for random fields. Later repeat selected comparisons with continuous acquisition optimization.

| Setting | Initial choice |
| --- | --- |
| Candidate catalogue | 2,048 fixed scrambled Sobol points for each dimension setting |
| Initial observations | `5 × d` distinct catalogue points, paired across policies for each seed |
| Surrogate | `SingleTaskGP` with an explicitly supplied scaled ARD Matérn-5/2 kernel |
| Numerical precision | Double precision |
| Output handling | Standardize for fitting; evaluate stopping thresholds in the declared objective units |
| Batch size | One point per iteration |
| Total budget | 100 observations in 2D; 150 in 3D, including initialization |
| Pilot observations | No added measurement noise; record the numerical noise floor used by the GP |
| Recommendation | Best evaluated point in the noiseless study |
| Reference optimizer | Uniform random search with the same evaluation budget |

Fit the GP after each observation and record fitted hyperparameters and failures. Use separate random streams for initialization, model fitting, acquisition sampling, measurement noise, and stopping diagnostics.

In the noiseless study, avoid repeat queries. Stopping diagnostics still consider the full catalogue, including evaluated points. Thompson sampling requires joint posterior samples that preserve correlation across locations.

The numerical settings above are proposed starting choices. Profile a small example, resolve implementation issues, and freeze the configuration before the full screen. Record any changes and their reasons.

**Deliverables:** `src/policies.py`, a transparent run loop, a saved configuration, and one example comparing policies.

## Phase 4: Implement five stopping approaches

Keep stopping separate from the acquisition strategy and recommendation rule.

For maximization, define simple regret as

$$
r_t=f^\star-f(\hat{x}_t),
$$

where $f^\star$ is the best allowed value and $\hat{x}_t$ is the point returned at iteration $t$. A result is acceptable when $r_t\leq\epsilon$.

| ID | Stopping approach | Operational meaning | Main limitation to investigate |
| --- | --- | --- | --- |
| S1 | Fixed budget | Stop after a specified number of evaluations | Controls cost without establishing solution quality |
| S2 | Incumbent stagnation | Improvement over a recent window is below a threshold | Can stop while trapped or underexploring |
| S3 | Negligible expected improvement | Maximum EI over the allowed domain is below a threshold | Depends on the surrogate; sensitive to numerical issues |
| S4 | Global confidence gap | Best plausible value is within tolerance of a lower bound at the returned point | Requires reliable simultaneous uncertainty bounds |
| S5 | Posterior probability of acceptable regret | Posterior probability that regret is below tolerance is sufficiently high | Depends on model correctness and probability-estimation accuracy |

### Implementation notes

- Evaluate S3 using a common EI diagnostic on every trajectory, regardless of the acquisition strategy used to choose points. Acquisition values from EI, UCB, and MES have different meanings and cannot share one raw threshold.
- For S4, evaluate `max U(x) - L(x_hat)`. The lower bound must refer to the point actually returned. Pointwise intervals do not automatically provide simultaneous coverage across locations and repeated checks.
- For S5, sample the maximum and the recommended-point value from the **same joint latent posterior draw**. Account for Monte Carlo error and repeated stopping checks.
- Label fitted-GP confidence claims as conditional on model assumptions; measure their actual reliability on the benchmarks.
- Keep the same hard budget as a fallback for every rule. Record a criterion firing separately from budget exhaustion.
- A stopping rule sees only the data available at the current iteration. It cannot access benchmark truth or future observations.

### Initial stopping settings

Use `epsilon = 0.02` in the benchmark's documented output units, with scoring sensitivity at `0.01` and `0.05`. Check that these tolerances are meaningful relative to each landscape's competing peaks.

Begin stopping checks after `n_initial + 10` observations, then check every five observations and at the final budget. Suggested heuristic settings are a 10-observation stagnation window with gain threshold `0.002`, and maximum-EI threshold `0.001`.

For the probability-based rules, specify the confidence construction and numerical error allowance in `docs/stopping_rules.md` before running the screen. For example, separate a 5% total model-based allowance into posterior regret and Monte Carlo decision-error components. Do not label an uncorrected Monte Carlo proportion a certificate.

Validate implementations on an easy function, a deliberately difficult hidden-peak case, and a small GP-generated case with known, correctly matched covariance assumptions.

**Deliverables:** `src/stopping.py`, `docs/stopping_rules.md`, and examples showing both sensible stopping and premature stopping.

## Phase 5: Run the full pilot screen

Run at least five optimizer seeds for every function-policy combination.

| Quantity | Noiseless pilot |
| --- | --- |
| Function cases | 24 |
| Acquisition strategies | 5 |
| Optimizer seeds per case-policy pair | 5 |
| Full BO trajectories | `24 × 5 × 5 = 600` |
| Stopping approaches applied to each trajectory | 5 |
| Scored stopping outcomes | `24 × 5 × 5 × 5 = 3,000` |

The random-search reference adds 120 trajectories. With the proposed budgets, the 600 BO trajectories require at most 75,000 objective evaluations, including initialization; the reference adds 15,000.

### Evaluate stopping rules on shared trajectories

Run each acquisition trajectory to its full budget. At each scheduled check, record whether each stopping rule would have fired, and save its first trigger time.

This gives the complete comparison without rerunning the optimizer separately for every stopping rule. It is valid when stopping rules only observe the trajectory. A rule that requests extra measurements or changes subsequent sampling requires its own experiment.

Stopping diagnostics must not consume the acquisition policy's random stream or otherwise change the trajectory. Never use later observations to recompute an earlier stopping decision.

### Launch sequence

- [ ] Complete one end-to-end run.
- [ ] Profile three contrasting functions, two policies, and two seeds.
- [ ] Freeze the pilot configuration and record software versions.
- [ ] Run the complete 600-trajectory BO matrix and random reference.
- [ ] Account for every planned case, including numerical failures and rules that never trigger.
- [ ] Save raw traces and generate results from scripts rather than manual spreadsheet edits.

Use one fixed random-field realization per covariance regime and dimension in the pilot. Additional field realizations are a separate factor in confirmation.

**Deliverables:** run configuration, execution script, trace format, results table, and a short pilot report.

## Phase 6: Evaluate reliability, cost, and failure conditions

Compare methods at comparable solution quality and reliability. A low average stopping time alone is insufficient.

| Metric | What to report |
| --- | --- |
| Simple regret at termination | Loss in true latent value relative to the best allowed point |
| Success rate | Fraction of runs with regret below tolerance, including budget terminations |
| False-stop rate among early triggers | Fraction of early triggers returning an unacceptable solution; include numerator and denominator |
| Unconditional premature-stop frequency | Fraction of all runs that stop early and return an unacceptable solution |
| Objective evaluations | Total observations consumed, including initialization and repeats |
| Computation time | GP fitting, acquisition selection, and each stopping diagnostic separately |
| Non-trigger fraction | Fraction reaching the hard budget without the criterion firing |
| Model diagnostics | Uncertainty coverage, fitted length scales, and behavior in missed regions |

Produce:

- [ ] Landscape plots with evaluation locations and stopping points.
- [ ] Regret-versus-evaluation curves.
- [ ] Stopping-time distributions with individual seeds visible.
- [ ] False-stop frequency versus evaluation cost.
- [ ] A summary table by landscape, acquisition strategy, and stopping rule.
- [ ] Short explanations of representative successes and failures.

Keep benchmark truth in the evaluator, separate from the optimizer and stopper. Use separate development and confirmation seeds or field realizations when tuning thresholds.

Five seeds provide a pilot view of variability. They do not establish a small failure probability: zero failures in five independent runs still gives an approximately 45% one-sided 95% binomial upper bound. Do not treat the 3,000 stopping outcomes as independent observations; several come from the same trajectory.

**Deliverables:** `src/evaluation.py`, reproducible figure scripts, and `docs/pilot_results.md`.

## Phase 7: Confirm selected findings and extend the study

Choose follow-up experiments to answer specific questions arising from the pilot.

1. Increase optimizer seeds, initially to 20–30 for selected comparisons. Use a sample-size calculation for a specific reliability claim.
2. Add at least five field realizations for the random-field regimes being studied. Preserve the distinction between field variability and optimizer variability.
3. Check sensitivity to catalogue size, stopping thresholds, and field approximation resolution.
4. Repeat selected policy-stopper combinations using continuous acquisition optimization.
5. Add a controlled noisy condition, with known Gaussian observation noise. Use noise-aware EI, a documented noisy PI convention, and a posterior-mean recommendation rather than selecting the largest noisy observation.

Finite-catalogue performance is a statement about the allowed catalogue. It is not a guarantee over the continuous domain: a catalogue can miss a narrow peak entirely. For continuous tests, distinguish known optima from numerical reference values. A best-found reference value can underestimate the true maximum and therefore understate regret.

Keep these extensions focused. Increasing dimension, introducing unknown or heavy-tailed noise, or changing surrogate families can become later projects.

**Deliverable:** `docs/confirmation_results.md`, with frozen configurations, uncertainty estimates, and explicit limitations.

## Repository organization

The following are planned locations; create them as each phase is completed.

| Path | Purpose |
| --- | --- |
| `README.md` | Project question, installation, and one example command |
| `WORKPLAN.md` | This workplan and progress tracking |
| `docs/bo_primer.md` | Introductory explanation for the student and collaborators |
| `docs/reading_list.md` | Annotated papers and reading notes |
| `docs/benchmarks.md` | Landscape definitions, scaling, and reference optima |
| `docs/stopping_rules.md` | Equations, assumptions, thresholds, and numerical details |
| `docs/pilot_results.md` | Complete pilot summary |
| `docs/confirmation_results.md` | Focused follow-up and limitations |
| `examples/` | Introductory notebook and small reproducible demonstrations |
| `src/` | Benchmarks, policies, stopping rules, runner, and evaluation |
| `configs/` | Versioned experiment settings |
| `tests/` | Checks for scientific correctness and information separation |
| `results/` | Small result tables, example traces, and figure-generation instructions |

Use small commits and issues corresponding to the phases. Keep numerical results traceable to the configuration, random seeds, software versions, and code revision. Store large generated outputs separately when needed and document how to reproduce them.

## Milestones and working rhythm

| Milestone | Tangible result |
| --- | --- |
| 1. Shared understanding | Primer, annotated reading list, and visual BO example |
| 2. Validated functions | Documented bank and landscape gallery |
| 3. Working methods | Five acquisition strategies and five stopping observers |
| 4. Complete pilot | Full five-seed matrix, including failures and non-triggers |
| 5. Focused confirmation | Additional evidence for specific findings |
| 6. Reproducible repository | Clear conclusions, examples, and commands to reproduce figures |

Aim to complete the first milestone in roughly one week, adjusting for the student's background. Review progress through a short weekly discussion: what is now understood, what remains unclear, and what the next experiment will resolve. Keep the primer and notes useful to everyone joining the project.

## Starting reading list: ten papers

Read the introductory material first, then use the methodological papers as references while implementing the relevant phase.

| # | Paper | Why it is included |
| --- | --- | --- |
| 1 | Frazier (2018), [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811) | First introduction to the GP, acquisition, and sequential decision loop |
| 2 | Shahriari et al. (2016), [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://ieeexplore.ieee.org/document/7352306) | Broader overview and terminology; read selected sections |
| 3 | Jones, Schonlau and Welch (1998), [Efficient Global Optimization of Expensive Black-Box Functions](https://doi.org/10.1023/A:1008306431147) | Classical expected-improvement framework |
| 4 | Srinivas et al. (2010), [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design](https://arxiv.org/abs/0912.3995) | UCB and the assumptions underlying regret guarantees |
| 5 | Wang and Jegelka (2017), [Max-value Entropy Search for Efficient Bayesian Optimization](https://proceedings.mlr.press/v70/wang17e.html) | Information-based acquisition strategy |
| 6 | Balandat et al. (2020), [BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403) | Implementation framework and Monte Carlo acquisition calculations |
| 7 | Ament et al. (2023), [Unexpected Improvements to Expected Improvement for Bayesian Optimization](https://arxiv.org/abs/2310.20708) | Why numerical treatment of EI matters |
| 8 | Makarova et al. (2022), [Automatic Termination for Hyperparameter Optimization](https://proceedings.mlr.press/v188/makarova22a.html) | Remaining optimization potential and statistical error |
| 9 | Ishibashi et al. (2023), [A stopping criterion for Bayesian optimization by the gap of expected minimum simple regrets](https://proceedings.mlr.press/v206/ishibashi23a.html) | A stopping criterion based on changes in expected regret |
| 10 | Wilson (2024), [Stopping Bayesian Optimization with Probabilistic Regret Bounds](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b204de7078301292a8876a762eed3dcb-Abstract-Conference.html) | Stopping when a solution is probably within a specified tolerance |

### Supporting resources

- [Garnett, Bayesian Optimization](https://bayesoptbook.com/): textbook background and selected introductory sections.
- [BoTorch closed-loop tutorial](https://botorch.org/docs/tutorials/closed_loop_botorch_only): use its loop structure while simplifying to the initial unconstrained, sequential example.
- [BoTorch acquisition API](https://botorch.readthedocs.io/en/stable/acquisition.html): verify API signatures against the installed version.
- [BoTorch Thompson-sampling tutorial](https://botorch.org/docs/tutorials/thompson_sampling): joint posterior sampling and candidate selection.
- [BoTorch MES tutorial](https://botorch.org/docs/tutorials/max_value_entropy): implementation reference.
- [SFU optimization test-function collection](https://www.sfu.ca/~ssurjano/optimization.html): standard benchmark definitions.
- [Xie et al., Cost-aware Stopping for Bayesian Optimization](https://arxiv.org/abs/2507.12453): optional follow-up reading on explicit evaluation costs.

## Definition of project completion

The repository should allow another researcher to reproduce a small example and understand:

1. Which stopping rules save evaluations at an acceptable error rate.
2. How their performance depends on the landscape and acquisition strategy.
3. Which failures arise from model assumptions, numerical approximations, or insufficient exploration.
4. What the results establish, and what remains uncertain.

The final product is a clear, reproducible comparison with useful examples and documented limits.

