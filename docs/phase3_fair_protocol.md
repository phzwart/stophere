# Phase 3 fair-comparison protocol

## Status and purpose

`configs/phase3_fair_protocol.yaml` is protocol version 1. It is frozen for
implementation and smoke testing. Smoke tests may reveal a numerical or runtime
problem, but no setting may be changed because one policy appears to perform
better or worse. Any scientifically material change creates protocol version 2
and requires all affected comparisons to be rerun.

The comparison changes one factor: the acquisition policy. The objective,
candidate catalogue, initialization, surrogate, fitting procedure, budget,
recommendation, and scoring rule remain common.

## Primary experiment

- Optimize each continuous benchmark over a fixed catalogue of 2,048 scrambled
  Sobol points. A catalogue is fixed by dimension and shared by every policy.
- Use `5 * d` distinct initial points. Within each benchmark and optimizer seed,
  all policies receive exactly the same initial points.
- Run 100 total observations in 2D and 150 in 3D, including initialization.
- Fit a double-precision `SingleTaskGP` after each observation. Use an explicitly
  supplied scaled ARD Matérn-5/2 covariance, standardized outcomes, and the same
  priors and numerical noise variance for every policy.
- Select one unevaluated point per iteration. Repeat queries are forbidden in
  the noiseless finite-catalogue experiment.
- Compare LogEI, LogPI, UCB, joint-posterior Thompson sampling, and MES. The
  implementation changes only the policy-specific selection step.
- Return the best evaluated point. The exact reference is the best objective
  value among the allowed catalogue points.
- Define primary success as normalized simple regret at most 0.05 and also
  report sensitivity at 0.01 and 0.02.
- Run every acquisition trajectory to the hard budget. Phase 4 stopping rules
  are observers: they record their first trigger without changing Phase 3.

## Pairing and random streams

The catalogue seed depends only on dimension. Initial-design seeds are paired
across policies. Thompson sampling has its own policy stream. Later noise and
stopping diagnostics must also receive separate streams so they cannot alter
the acquisition trajectory.

## What the primary experiment does not claim

The finite catalogue makes fairness and exact catalogue scoring transparent,
but it is not identical to continuous acquisition optimization. Selected cases
will therefore be repeated later in a continuous domain. External continuous
optimizers such as CMA-ES, SNOBFIT, NOMAD, and BAM remain a separate screen
until the BoTorch policies receive the same continuous-domain treatment.

The first pilot is noiseless. A noisy experiment is a separate extension with a
posterior-mean recommendation and its own declared observation model; noisy and
noiseless success rates are not pooled.

## Freeze checklist

Before the full matrix begins:

1. Peter confirms the domain mode, budget, primary success threshold, and the
   choice to treat noise as a separate extension.
2. All five policies pass the four-system smoke test.
3. Paired initialization, absence of duplicate queries, deterministic replay,
   exact evaluation counts, and absence of truth leakage are tested.
4. Package versions, the protocol file, raw traces, warnings, and failures are
   retained with the results.
5. The repository records the exact commit used for the full matrix.

