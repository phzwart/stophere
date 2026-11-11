# Phase 4 — Stopping observers

Phase 4 asks when an already-running Bayesian optimization trajectory could
have stopped. It does not change where Phase 3 samples.

Implemented components:

- `src/stopping.py`: five trajectory-preserving stopping observers;
- `configs/phase4_stopping_observers.yaml`: provisional thresholds and risk
  settings awaiting collaborator confirmation;
- `docs/stopping_rules.md`: operational definitions, probability statements,
  numerical approximations, and limitations;
- `experiments/phase4_stopping_smoke.py`: reduced three-landscape validation;
- `tests/test_phase4_stopping.py`: checks information separation, hard-budget
  fallback, first-trigger logging, and acquisition-path invariance;
- `results/phase4/stopping_smoke/`: implementation smoke output, not formal
  scientific performance evidence.

The full Phase 4 comparison must wait for the objective-unit tolerance and risk
level to be confirmed. The default 0.02 tolerance is explicitly provisional.
