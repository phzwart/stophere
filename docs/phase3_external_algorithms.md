# Phase 3 external derivative-free algorithm screen

## Purpose and scope

This screen responds to the request to check algorithms beyond Bayesian optimization. It validates official Python implementations of CMA-ES, SNOBFIT, NOMAD, and BAM against a continuous uniform-random baseline.

The common scope is the six continuous two-dimensional benchmark systems. Every problem uses the unit box, maximization, five starting-point seeds, a maximum budget of 150 objective evaluations, and success defined as normalized simple regret at most 0.05. The screen contains 150 algorithm-system-seed runs.

This is not yet the final BO-versus-DFO matrix. The existing NumPy BO prototype selects from finite catalogues, while these external solvers search the continuous box. Pooling their results would confound the algorithm with the domain representation. A final comparison requires the planned continuous BoTorch protocol and a frozen shared initialization rule.

## Implementations and budget controls

| Method | Package | Version | Budget handling |
| --- | --- | ---: | --- |
| CMA-ES | `cma` | 4.5.0 | Ask/tell with population size 5, so 50 and 150 are exact generation boundaries |
| SNOBFIT | `SQSnobFit` | 0.4.5 | Direct official batch interface; the final batch is truncated before evaluation to prevent the high-level driver from overshooting |
| NOMAD | `PyNomadBBO` | 4.6.0 | `MAX_BB_EVAL=150`; natural early termination retains its final incumbent at later budget checkpoints |
| BAM | `bam-solver` | 5.0.0 | `max_evals=150`, official free mode for at most two variables |
| Uniform random | NumPy | 1.26.4 | One continuous unit-box point per evaluation |

SNOBFIT 0.4.5 requires NumPy 1.26; it fails under NumPy 2 in the package's internal array assignments. Two NOMAD runs terminated naturally before 150 evaluations, at 131 and 143 calls. Their measured incumbents are carried forward for the 150-evaluation maximum-budget profile, while the audit table retains the actual evaluation counts.

## Results

| Method | Success at 20 | Success at 50 | Success at 100 | Success at 150 |
| --- | ---: | ---: | ---: | ---: |
| CMA-ES | 50.0% | 63.3% | 63.3% | 66.7% |
| SNOBFIT | 63.3% | 83.3% | 93.3% | **93.3%** |
| NOMAD | 56.7% | 60.0% | 66.7% | 70.0% |
| BAM | **83.3%** | 83.3% | 83.3% | 83.3% |
| Uniform random | 46.7% | 66.7% | 76.7% | 86.7% |

BAM reaches acceptable solutions quickly on five systems but fails on every `hidden_narrow_2d` run, so its pooled rate plateaus. SNOBFIT improves through 100 evaluations and has the highest measured 150-evaluation success. Uniform random eventually exceeds BAM, NOMAD, and CMA-ES in pooled success because the low-dimensional domain makes broad coverage effective and because failures are concentrated in the hidden-narrow, rough-field, and competing-modes systems.

These results do not establish a universal ranking. There are only six systems and five seeds, the algorithms use their natural initialization semantics, and hyperparameters were not tuned per method. The main result is methodological: per-system failure modes and success-versus-budget profiles are necessary, because a single final average hides qualitatively different behavior.

## Mean adapter wall time per run

The objectives are computationally cheap, so these values mostly measure optimizer overhead on this machine and should not be interpreted as beamline cost.

| Method | Mean wall time |
| --- | ---: |
| CMA-ES | 0.41 s |
| SNOBFIT | 1.14 s |
| NOMAD | 1.67 s |
| BAM | 7.77 s |
| Uniform random | 0.01 s |

## Remaining external algorithms

- MCS: the authors' reference implementation is MATLAB-based. No unverified Python translation is labeled as MCS in this project.
- BAM in three dimensions: the official free mode is limited to two variables. Academic users can request a free license before the 3D matrix is run.
- Ising MRF: it remains a separate discrete protocol; continuous box solvers are not applied through an arbitrary binary repair rule.

## Outputs

- `results/phase3/external_algorithms/trajectories.csv`: every completed objective evaluation.
- `results/phase3/external_algorithms/run_status.csv`: versions, evaluation counts, natural termination, errors, and wall time.
- `results/phase3/external_algorithms/success_profiles.csv`: pooled success and cross-system quantiles.
- `results/phase3/external_algorithms/success_by_budget.png`: 5/10/20/50/100/150 maximum-budget profiles.
- `results/phase3/external_algorithms/success_quantiles_final_budget.png`: final medians, interquartile ranges, and pooled rates.

