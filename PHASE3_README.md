# Phase 3: acquisition and model characterization

## Formal BoTorch implementation status

The current real-data priority is optimization. The ALS three-target offline
optimization pre-pilot is part of Phase 3 because it compares acquisition
policies on measured objective values. Its passive stopping scores are a Phase
4 diagnostic on those same trajectories. It is not the Phase 5 full synthetic
benchmark matrix. See `docs/als_three_target_replay.md`.

The fair-comparison rules are now explicit in
`configs/phase3_fair_protocol.yaml` and explained in
`docs/phase3_fair_protocol.md`. The shared run loop is in
`src/botorch_runner.py`, while `src/policies.py` contains the only code that
changes among LogEI, LogPI, UCB, Thompson sampling, and MES. The four-system
smoke runner is `experiments/phase3_botorch_smoke.py`.
The resumable four-system full-budget decision gate is
`experiments/phase3_full_budget_prepilot.py`; see
`docs/phase3_full_budget_prepilot.md`.

Smoke-test output verifies software wiring only. The complete 100-trajectory
four-system pre-pilot is now available under
`results/phase3/full_budget_prepilot/`. It is the decision gate for protocol
revision, not the final 24-system scientific comparison. The complete matrix
remains pending until the protocol receives collaborator sign-off and all
acceptance checks pass.

Phase 3 studies how acquisition choices and Gaussian-process assumptions affect Bayesian-optimization behavior. This branch builds on the validated Phase 2 benchmark bank; Phase 2 files remain available as dependencies, while Phase 3-specific code, tests, documentation, and results use explicit `phase3` names and paths.

## Phase 3 files

- `experiments/phase3_characterization.py`: reproducible NumPy exact-GP characterization runner.
- `experiments/phase3_success_profiles.py`: pooled success and cross-system quantile summaries.
- `experiments/phase3_budget_profiles.py`: extended 5/10/20/50/100/150-evaluation BO-policy screen.
- `experiments/phase3_external_algorithms.py`: official CMA-ES, SNOBFIT, NOMAD, and BAM adapter screen on the common continuous 2D scope.
- `experiments/phase3_random_baseline_audit.py`: exact without-replacement audit of the random-search success profile.
- `experiments/phase3_trajectory_visuals.py`: objective heat maps with initial points, later queries, recommendations, optima, and first stop signals.
- `experiments/phase3_run_prepilot_batch.py`: resumable bounded-parallel executor for missing pre-pilot tasks.
- `experiments/phase3_summarize_prepilot.py`: policy, warning, and truth-aware stopping aggregation for the full-budget pre-pilot.
- `tests/test_phase3_characterization.py`: scientific and software checks for the Phase 3 runner.
- `tests/test_phase3_external_algorithms.py`: optional-dependency adapter and strict-budget smoke tests.
- `docs/phase3_characterization.md`: methods, results, limitations, and next steps.
- `docs/phase3_budget_profiles.md`: measured 50/100/150-evaluation success results and interpretation.
- `docs/phase3_external_algorithms.md`: external-algorithm protocol, results, limitations, and dependency audit.
- `docs/phase3_full_matrix_plan.md`: two-week algorithm-comparison and beamline-meeting plan.
- `results/phase3/characterization/`: raw trajectories, summaries, paired deltas, exploratory stopping diagnostics, and summary plot.
- `results/phase3/mrf_bo_pilot/`: earlier MRF pilot retained as a separate historical protocol.
- `results/phase3/external_algorithms/`: continuous 2D external-algorithm trajectories, audit table, success profiles, and figures.
- `results/phase3/random_baseline_audit/`: exact and observed random-search success probabilities.

## Run the prototype

```bash
MPLBACKEND=Agg python -m experiments.phase3_characterization
python -m pytest -q tests/test_phase3_characterization.py
```

To generate the extended-budget BO-policy profile without the exploratory stopping diagnostics:

```bash
python -m experiments.phase3_budget_profiles --budget 150 --workers 4
```

Install the optional official solver packages and run the external 2D screen:

```bash
python -m pip install -r requirements-phase3-algorithms.txt
python -m experiments.phase3_external_algorithms --budget 150 --seeds 5 --workers 4
```

## Scope boundary

The current runner compares UCB, EI, PI, kernel families, length scales, exploration weights, matched observation noise, and random search across nine systems. It uses a transparent fixed-hyperparameter NumPy GP.

The NumPy characterization prototype does not complete the formal Phase 3
comparison. The shared BoTorch protocol and five acquisition policies are
implemented and have completed a four-system pre-pilot and the ALS optimization
pre-pilot. The full 24-case formal matrix is still planned as Phase 5. The
NumPy prototype's stopping diagnostics remain exploratory; the separately
implemented observers and their ALS replay scores belong to Phase 4.

The processed ALS optimization replay is an offline measured-data pre-pilot,
not a new beamline run. It contains 90 trajectories (3 objectives × 6 policies
× 5 paired seeds). `experiments/plot_als_three_target_diagnostics.py` recreates
local high-contrast target maps, a log-scale peak-height distribution, success
bars with Wilson intervals, and a small held-out policy-selection audit. The
catalogue and all resulting figures/CSVs remain private and are intentionally
excluded from Git; see `docs/als_three_target_replay.md` for the local command
and current interpretation limits.

The external screen validates official adapters for BAM, SNOBFIT, CMA-ES, and NOMAD on the common six-system continuous 2D scope. It is still not the full cross-algorithm matrix: the current NumPy BO prototype uses finite catalogues, MCS has no validated Python adapter, and BAM requires an academic license above two variables. The Ising MRF remains a separate discrete protocol.
