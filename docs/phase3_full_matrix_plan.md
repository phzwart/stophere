# Phase 3 full comparison matrix plan

## Purpose

The next comparison should separate two questions:

1. How quickly does each method find an acceptable solution as the evaluation budget grows?
2. How variable is that result across random seeds and objective landscapes?

Success remains normalized simple regret at most 0.05. At each checkpoint, report the pooled success rate and the median, 25th percentile, and 75th percentile of the system-level success rates. These quantiles describe heterogeneity across systems; they are not confidence intervals.

## Budget checkpoints and repeats

- Checkpoints: 5, 10, 20, 50, 100, and 150 evaluations.
- Full matrix: 10 paired seeds per system and algorithm.
- Use the same initial observations for every compatible algorithm within a system and seed.
- Preserve complete trajectories so any additional checkpoint can be computed without rerunning the experiment.

## Algorithm families

The attached Branch-and-Model paper compares BAM with SNOBFIT, MCS, CMA-ES, and NOMAD. Those five derivative-free optimizers should be evaluated alongside the Bayesian-optimization policies and random search on the continuous box-constrained systems.

The 3 × 3 Ising MRF is not directly compatible with continuous box-constrained solvers. Keep it in a separate discrete comparison until an explicit binary encoding, repair rule, and distance-aware protocol are validated. Do not mix its results into the continuous-solver ranking.

| Family | Methods | Current status |
| --- | --- | --- |
| Bayesian optimization | nominal GP-UCB, EI, PI; later Thompson sampling and MES | UCB, EI, and PI prototype available |
| Sampling baseline | random search | available |
| Global derivative-free | BAM, SNOBFIT, MCS, CMA-ES | adapter and dependency validation required |
| Local/direct search | NOMAD | adapter and dependency validation required |

## Fair-comparison requirements

- Count objective-function calls only; log optimizer overhead separately.
- Use identical box bounds, objective orientation, success threshold, budgets, and paired seeds.
- Record failures, invalid points, duplicate evaluations, and early termination explicitly.
- For stochastic solvers, compare the same number of runs and report seed-level results.
- Validate each adapter on analytic functions with known optima before entering the full matrix.
- Treat a reimplementation of BAM from the paper as a separate method unless it is verified against the authors' implementation.

## Two-week execution sequence

1. Days 1-3: finalize the success-profile figures and validate the 50/150-step runner.
2. Days 3-6: build and unit-test adapters for the available derivative-free solvers.
3. Days 6-9: run a small smoke matrix, resolve domain and termination mismatches, then freeze the protocol.
4. Days 9-12: run the paired-seed full matrix and generate quantile, budget-profile, regret, and runtime tables.
5. Days 12-14: audit the results, package the figures and methods note, and prepare the group-meeting discussion for beamline deployment.

## Beamline meeting package

The package should contain the frozen protocol, algorithm/version table, per-system results, success-versus-budget profiles, quantile spreads, failure cases, computational overhead, and a short proposal for safe beamline testing. The beamline proposal should specify the initial design, allowed control domain, failure/abort conditions, evaluation latency, data logging, and a conservative stopping policy.
