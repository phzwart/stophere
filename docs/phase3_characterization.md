# Phase 3 prototype — BO characterization

> **Plan alignment:** This is an exploratory NumPy Phase 3 screening prototype. It studies acquisition-policy and GP-setting sensitivity but does not complete the planned Phase 3: it does not use BoTorch and includes only UCB, EI, and PI (not Thompson sampling or MES). Simplified stopping diagnostics are retained as exploratory measurements; they are not formal Phase 4 deliverables. See [roadmap and status](roadmap_and_status.md) for the phase-by-phase mapping and next steps.

## Research goal

The project's success criterion is **reliable characterization**, not algorithmic novelty or maximum difficulty. The practical question is: under which objective shapes, GP assumptions, acquisition settings, and stopping rules does BO reach a useful solution—and when does it miss or stop too early?

This experiment is an initial, controlled screening study. It is not evidence that one optimizer is universally best, and it is not a replacement for the planned BoTorch implementation.

## Experimental matrix

Run from the repository root:

```bash
MPLBACKEND=Agg python -m experiments.phase3_characterization
```

The runner evaluates 9 systems: smooth 2D, anisotropic 2D, rippled 2D, competing peaks 2D, hidden narrow peak 2D, curved-valley Rosenbrock 3D, flat/multimodal Ackley 3D, rough random field 2D, and the auxiliary 3×3 Ising MRF. Continuous cases use fixed grids (21 points per axis in 2D; 9 per axis in 3D); the MRF search catalog exhausts all 512 binary states. Grids are the declared finite optimization domains for these pilot measurements—not proofs about a continuous domain.

Each system has 5 paired seeds, 5 common initial observations, and a maximum of 24 evaluations. The matrix has 12 distinct configurations: one nominal RBF/GP-UCB configuration plus one-factor contrasts in kernel family (RBF, Matérn 3/2, Matérn 5/2), length scale (0.5× and 2× nominal), UCB exploration weight β (0.5 and 8 versus nominal 2), acquisition (EI and PI versus UCB), a 3%-of-catalogue-range Gaussian observation-noise condition, one kernel/length-scale interaction, and random search. The nominal setting is represented once; duplicate baseline runs are deliberately avoided. This gives 9 × 12 × 5 = 540 trajectories.

All GPs are exact, fixed-hyperparameter NumPy implementations using the same candidate catalogues. We do **not** fit kernel hyperparameters in this screening pass. The Matérn families encode different smoothness assumptions; shorter length scales permit faster changes, longer length scales impose smoother changes; larger β emphasizes uncertainty in UCB. EI and PI trade exploration differently from UCB. Observation noise is added to measurements—not to the latent objective—and the GP is given the matching noise variance in that condition. It is a sensitivity case, not a noise-misspecification study.

### Stopping rules

Each full trajectory is replayed through four online stopping rules. This includes random search as a search policy. To make stopping diagnostics comparable across search configurations, they use a **shared diagnostic GP**: fixed RBF kernel and each system's declared nominal length scale, refit to the observations available at that step. The acquisition policy itself still uses its configured kernel and length scale. The noise likelihood is matched to the observation-noise condition. Since the prefix up to a stopping point is the same as the corresponding full run, this estimates what each rule would have saved without changing the earlier choices. This is a prototype convention, not Peter's final Phase 4 common-EI / confidence-gap / posterior-regret specification.

1. `fixed_budget`: always use all 24 evaluations.
2. `patience_5`: stop after at least five post-initial observations if the incumbent has not improved by more than 1% of the recent observed response standard deviation over the last five evaluations.
3. `global_ucb_gap_0.05`: from at least two post-initial evaluations onward, stop if the largest posterior 95% UCB over the catalog is within 0.05 observed response standard deviations of the incumbent.
4. `max_p_improve_0.05`: stop if, for every catalog point, the largest marginal posterior probability of beating the incumbent by 0.01 observed response standard deviations is at most 0.05.

The latter two are diagnostics, not formal guarantees: the UCB gap depends on the GP being reasonably calibrated, and maximum marginal probability is not the joint probability that any candidate beats the incumbent. All decision thresholds use the observations/model, not the evaluator-only optimum. Evaluation success is reported separately as final normalized simple regret ≤0.05, with the finite-catalogue optimum/range used only after the run. A false stop means a stopping rule terminates while this success threshold is not met.

## Reading the results

`summary.csv` reports final normalized regret by system/configuration over five seeds. `comparison_deltas.csv` reports paired seed-wise changes against the nominal configuration within each system (negative is better). `stopping_rules.csv` records each counterfactual stop prefix, false stops, and evaluations saved. `trajectories.csv` contains the step-by-step raw data, including the diagnostic GP's kernel and length scale for auditability. `success_profiles.csv` adds pooled success plus the median and 25th/75th percentiles of system-level success. The quantiles describe spread across benchmark systems, not sampling uncertainty or confidence intervals. `success_quantiles_final_budget.png` and `success_by_budget.png` visualize those summaries.

## Pilot results

The current matrix contains 540 trajectories (9 systems × 12 configurations × 5 seeds). Across all system/configuration cells, the fixed-budget mean normalized regret is **0.129** with **69.3%** success at ε=0.05. This pooled mean is descriptive only; easy systems such as the grid-aligned Ackley case and more difficult systems such as 3D Rosenbrock differ greatly, so per-system outcomes matter more than the aggregate.

### GP and hyperparameter contrasts

Each number below is the mean of the per-system five-seed mean; it is a compact descriptive comparison, not a statistical test. Success is the fraction of runs with final normalized regret ≤0.05.

| Configuration | Mean normalized regret | Success rate |
| --- | ---: | ---: |
| Nominal RBF + UCB (β=2) | 0.1314 | 71.1% |
| Matérn 3/2 kernel | 0.1171 | 71.1% |
| Matérn 5/2 kernel | 0.1015 | 75.6% |
| RBF length scale ×0.5 | 0.1104 | 73.3% |
| RBF length scale ×2 | 0.1536 | 64.4% |
| UCB β=0.5 | 0.1106 | 73.3% |
| UCB β=8 | 0.1209 | 68.9% |
| Expected improvement | 0.1179 | 73.3% |
| Probability of improvement | 0.1298 | 73.3% |
| Matérn 3/2 with length scale ×2 | 0.0958 | 77.8% |
| 3%-range observation noise, matched GP noise model | 0.1507 | 64.4% |
| Random search | 0.2105 | 44.4% |

The strongest averaged row is the **Matérn-3/2 ×2 interaction** (0.0958), but this does not establish a generally best setting: improvements are concentrated in some systems. Relative to nominal, the paired mean regret change was −0.0356 for that interaction, −0.0299 for Matérn-5/2, −0.0210 for shorter RBF length scale, and +0.0791 for random search. A shorter length scale helped the rough/rippled cases, while the 2× RBF length scale was worse on average and especially poor on the narrow/curved structure. Noise increased average regret from 0.1314 to 0.1507 despite matching the GP's observation-noise variance.

Per-system nominal regret demonstrates why characterization matters: smooth 2D = 0.000, anisotropic 2D = 0.000, competing modes 2D = 0.000, rippled 2D = 0.038, hidden narrow 2D = 0.158, rough random field 2D = 0.089, curved-valley Rosenbrock 3D = 0.859, flat/multimodal Ackley 3D = 0.000, and Ising MRF = 0.039. The 3D Rosenbrock result is a strong failure case at this small budget/grid; it deserves separate dimension/budget analysis. Some systems are so easy on these grids that all settings can tie at zero regret.

| System | Nominal RBF/UCB | Random | Best observed GP setting* |
| --- | ---: | ---: | --- |
| Smooth 2D | 0.000 | 0.005 | 0.000 (EI) |
| Anisotropic 2D | 0.000 | 0.001 | 0.000 (PI) |
| Competing modes 2D | 0.000 | 0.145 | 0.000 (β=0.5) |
| Rippled 2D | 0.038 | 0.038 | 0.003 (RBF length scale ×0.5) |
| Hidden narrow 2D | 0.158 | 0.243 | 0.158 (PI) |
| Rough random field 2D | 0.089 | 0.175 | 0.042 (β=8) |
| Curved-valley Rosenbrock 3D | 0.859 | 0.747 | 0.586 (Matérn 3/2, length scale ×2) |
| Flat/multimodal Ackley 3D | 0.000 | 0.426 | 0.000 (several settings tie) |
| Ising MRF 3×3 | 0.039 | 0.113 | 0.007 (EI) |

*“Best” is the lowest observed mean among these same pilot runs, so it is selection-biased and shown only to locate follow-up cases. It needs confirmation on fresh seeds before being treated as a reliable setting.

### Stopping-rule trade-off

| Rule | Mean stopped regret | Success rate | False-stop rate | Mean evaluations saved (of 24) | Mean stop evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed budget | 0.129 | 69.3% | 0.0% | 0.00 | 24.00 |
| Patience-5 | 0.165 | 56.1% | 43.9% | 10.57 | 13.43 |
| Global UCB gap ≤0.05 SD | 0.134 | 67.8% | 32.2% | 2.42 | 21.58 |
| Max marginal P(improve) ≤0.05 | 0.137 | 66.5% | 33.5% | 3.27 | 20.73 |

In this pilot, patience saves the most calls but often stops too early. The posterior-based rules save fewer calls and still false-stop about one third of runs. Across all 60 MRF matrix trajectories (12 configurations × 5 seeds), patience saves 9.9 calls on average with a 73.3% false-stop rate; global UCB gap saves 2.55 calls with a 23.3% false-stop rate; maximum marginal P(improve) saves 3.75 calls with a 30.0% false-stop rate. This is not yet a calibrated stopping-rule recommendation: the threshold and rule definitions need a sensitivity sweep, and the GP's uncertainty is not guaranteed to be calibrated.

The shared diagnostic-GP convention was made explicit and enforced in the current run. Search configurations retain their own model for acquisition decisions; stopping summaries use the same nominal RBF length scale for a given system. This changes the stopping metrics relative to the earlier run, while the search trajectories and fixed-budget regret results remain unchanged. The change makes cross-configuration stopping comparisons cleaner, but does not make the simplified diagnostics equivalent to the planned Phase 4 rules.

## Limitations and next steps

- Five seeds and finite grids are exploratory; there are no confidence claims or broad rankings.
- The 3D and MRF cases have different catalogue sizes/geometry; normalized regret helps compare scale but not intrinsic difficulty.
- Fixed GP parameters do not represent empirical-Bayes/MLE fitting. Add a separately labeled fitted-hyperparameter condition, and log fitted parameters and convergence warnings.
- The synthetic noise case assumes a known, matched likelihood variance. Add repeated noisy measurements and a deliberately misspecified-noise contrast before drawing robustness conclusions.
- Current code uses generic Euclidean kernels on an encoded binary MRF and does not exploit its graph. A graph/Hamming-aware covariance is a future comparison, not part of this result.
- Stopping thresholds are plausible pilot values, not calibrated universal defaults. Report the regret/savings/false-stop frontier over multiple thresholds, and include per-system and paired-seed outcomes.
- Treat this as a Phase 3 characterization prototype, then implement and validate the five planned acquisition strategies in BoTorch before increasing budgets or making project-level claims.
- Use `phase3_budget_profiles.py` for the 50/150-evaluation BO-policy screen. The existing 24-evaluation CSV cannot support those checkpoints; they require new trajectories.
- See `phase3_budget_profiles.md` for the completed 180-trajectory screen and measured results.
- Build the full cross-algorithm matrix separately, following `phase3_full_matrix_plan.md`, with BAM, SNOBFIT, MCS, CMA-ES, and NOMAD on continuous box-constrained systems.
- Implement the exact planned Phase 4 stopping definitions (including a common EI diagnostic, `max U(x) - L(x_hat)`, and joint-posterior probability of acceptable regret), keeping them separate from acquisition decisions.

The desired deliverable is a map of **conditions and failure modes**: objective × model assumption × acquisition/hyperparameter × stopping rule → quality, reliability, and evaluation cost.
