# Expanded formal BoTorch smoke test

Run date: 2026-11-13

This directory contains implementation evidence, not a scientific policy
comparison. The run used four contrasting systems, all five formal acquisition
policies, two paired optimizer seeds, a 128-point smoke catalogue, and five
acquisition steps after the `5*d` initialization.

## Outcome

- 40 planned trajectories;
- 40 completed trajectories;
- zero failed trajectories;
- zero GP fit warnings;
- 240 saved posterior states.

All policies had a 50% pooled final success rate in this deliberately short
run. This equality is not evidence that the policies are equivalent. System
difficulty dominated the short horizon: quadratic peak reached 100% success,
rough random field reached 0%, and hidden narrow peak and 3D Rosenbrock each
reached 50%.

Files:

- `trajectories.csv`: saved state after initialization and each acquisition;
- `run_audit.csv`: one status, runtime, warning, and final-metric row per plan;
- `metadata.json`: exact scope and software versions.
