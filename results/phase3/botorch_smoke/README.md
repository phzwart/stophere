# Formal Phase 3 implementation smoke test

This directory verifies that the shared BoTorch runner and all five policy
selectors execute on four contrasting systems:

- a smooth 2D quadratic peak;
- a 2D hidden narrow peak;
- a 2D rough random field;
- a 3D Rosenbrock curved valley.

The run used one paired seed, a reduced 128-point catalogue, and one acquisition
step after the common `5 * d` initialization. All 20 system-policy combinations
completed without GP fitting warnings. The trace contains 40 rows because it
records the initialized state and the state after the one acquisition step.

This is software evidence, not a policy-performance experiment. The catalogue,
budget, seeds, and number of acquisition steps are intentionally too small for
scientific ranking. The formal protocol remains the 2,048-point catalogue with
100 evaluations in 2D and 150 in 3D.

Files:

- `metadata.json`: scope and exact software versions;
- `trajectories.csv`: per-state trace used to verify interfaces and logging.

