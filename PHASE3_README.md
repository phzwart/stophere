# Phase 3: acquisition and model characterization

## Formal BoTorch implementation status

The fair-comparison rules are now explicit in
`configs/phase3_fair_protocol.yaml` and explained in
`docs/phase3_fair_protocol.md`. The shared run loop is in
`src/botorch_runner.py`, while `src/policies.py` contains the only code that
changes among LogEI, LogPI, UCB, Thompson sampling, and MES. The four-system
smoke runner is `experiments/phase3_botorch_smoke.py`.

Smoke-test output verifies software wiring only. It must not be reported as the
formal Phase 3 scientific comparison. The complete matrix remains pending until
the protocol receives collaborator sign-off and all acceptance checks pass.

Phase 3 studies how acquisition choices and Gaussian-process assumptions affect Bayesian-optimization behavior. This branch builds on the validated Phase 2 benchmark bank; Phase 2 files remain available as dependencies, while Phase 3-specific code, tests, documentation, and results use explicit `phase3` names and paths.

## Phase 3 files

- `experiments/phase3_characterization.py`: reproducible NumPy exact-GP characterization runner.
- `experiments/phase3_success_profiles.py`: pooled success and cross-system quantile summaries.
- `experiments/phase3_budget_profiles.py`: extended 5/10/20/50/100/150-evaluation BO-policy screen.
- `experiments/phase3_external_algorithms.py`: official CMA-ES, SNOBFIT, NOMAD, and BAM adapter screen on the common continuous 2D scope.
- `tests/test_phase3_characterization.py`: scientific and software checks for the Phase 3 runner.
- `tests/test_phase3_external_algorithms.py`: optional-dependency adapter and strict-budget smoke tests.
- `docs/phase3_characterization.md`: methods, results, limitations, and next steps.
- `docs/phase3_budget_profiles.md`: measured 50/100/150-evaluation success results and interpretation.
- `docs/phase3_external_algorithms.md`: external-algorithm protocol, results, limitations, and dependency audit.
- `docs/phase3_full_matrix_plan.md`: two-week algorithm-comparison and beamline-meeting plan.
- `results/phase3/characterization/`: raw trajectories, summaries, paired deltas, exploratory stopping diagnostics, and summary plot.
- `results/phase3/mrf_bo_pilot/`: earlier MRF pilot retained as a separate historical protocol.
- `results/phase3/external_algorithms/`: continuous 2D external-algorithm trajectories, audit table, success profiles, and figures.

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

This prototype does not complete Peter's planned Phase 3. The formal Phase 3 deliverable still requires a shared BoTorch protocol for LogEI, LogPI, UCB, Thompson sampling, and max-value entropy search. The included stopping diagnostics are exploratory measurements, not formal Phase 4 stopping rules.

The external screen validates official adapters for BAM, SNOBFIT, CMA-ES, and NOMAD on the common six-system continuous 2D scope. It is still not the full cross-algorithm matrix: the current NumPy BO prototype uses finite catalogues, MCS has no validated Python adapter, and BAM requires an academic license above two variables. The Ising MRF remains a separate discrete protocol.
