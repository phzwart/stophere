# Four-system full-budget pre-pilot

Run date: 2026-11-13

`manifest.csv` contains the complete 100-task plan: four systems, five
acquisition policies, and five paired seeds. The current result set is only a
one-policy/one-seed slice and must not be used for a policy ranking.

## Completed full-budget slice

MES with optimizer seed 0 completed on all four pre-pilot systems:

| System | Budget | Final normalized regret | Success <= 0.05 | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| quadratic peak 2D | 100 | 0.000 | yes | 22.6 |
| hidden narrow peak 2D | 100 | 0.414 | no | 59.9 |
| rough random field 2D | 100 | 0.023 | yes | 58.3 |
| Rosenbrock 3D | 150 | 0.000 | yes | 74.6 |

## Descriptive stopping outcomes

The four trajectories are too few and too unbalanced to rank stopping rules.
They are useful for checking failure modes:

- incumbent stagnation triggered early on all four and false-stopped on two;
- negligible expected improvement triggered early on all four and
  false-stopped on one;
- global confidence gap and posterior acceptable regret each triggered early
  on three, false-stopped on one, and used the hard-budget fallback on the rough
  field;
- the fixed-budget rule provides the no-savings reference.

The hidden-narrow trajectory remained unsuccessful at budget, showing why a
stopping trigger cannot be interpreted as optimization success without the
truth-aware evaluator.

## Runtime diagnostic

The same 25-evaluation quadratic MES task took 7.58 seconds with Phase 4
observers and 7.08 seconds without them, approximately 7% overhead in this
single diagnostic case. More policies and difficult systems are required
before making a general runtime claim.

Files:

- `raw/`: completed full-budget trajectory and metadata files;
- `summary/trajectory_summary.csv`: one row per completed trajectory;
- `summary/stopping_outcomes.csv`: truth-aware outcome per rule and trajectory;
- `summary/stopping_rule_summary.csv`: partial descriptive aggregate;
- `profile_with_stopping/` and `profile_without_stopping/`: matched short
  runtime diagnostic.
