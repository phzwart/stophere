# Phase 3 and Phase 4 full-budget pre-pilot

## Purpose

The pre-pilot is the decision gate between implementation smoke testing and the
24-system Phase 5 matrix. It uses the same four contrasting systems as the
formal BoTorch smoke test:

- a smooth 2D quadratic peak;
- a 2D hidden narrow peak;
- a rough 2D random field;
- a 3D curved Rosenbrock valley.

Each of the five acquisition policies runs with five paired seeds, producing a
100-task plan. Unlike the smoke test, the formal pre-pilot uses the complete
100-evaluation 2D and 150-evaluation 3D budgets. Phase 4 observers remain
passive and every trajectory continues to the hard budget.

## Safety and execution

Running the module without a task index writes only the deterministic manifest:

```bash
python -m experiments.phase3_full_budget_prepilot
```

Run one resumable task:

```bash
python -m experiments.phase3_full_budget_prepilot --task-index 0
```

Resume all missing tasks with bounded parallelism:

```bash
python -m experiments.phase3_run_prepilot_batch --workers 4
```

Summarize the completed matrix:

```bash
python -m experiments.phase3_summarize_prepilot
```

The summary command writes one row per trajectory, one truth-aware outcome per
trajectory and stopping rule, a policy aggregate, and a two-panel summary
figure.

`--budget-override` is available for implementation diagnostics and runtime
profiling. Any output created with that flag is labelled diagnostic and is not
part of the formal pre-pilot. `--without-stopping` isolates acquisition runtime
when profiling the additional cost of the Phase 4 path sampler.

## Acceptance gate

Do not expand to the 600-task Phase 5 matrix until the pre-pilot confirms:

1. every acquisition completes under the frozen budget and catalogue;
2. paired initial designs and trajectory replay remain deterministic;
3. Phase 4 diagnostics do not change the selected acquisition path;
4. runtime and memory are acceptable, especially for MES and Matheron paths;
5. success and stopping metrics have enough variation to distinguish methods;
6. the objective-unit tolerance and risk interpretation have collaborator
   approval.

## Completed pre-pilot results

All 100 planned full-budget trajectories completed. The pooled success rate was
70%. LogEI, LogPI, UCB, and MES each reached 75%; Thompson sampling reached
50%. All policies solved the smooth quadratic and 3D Rosenbrock cases in all
five seeds. The hidden narrow peak remained the principal failure: LogEI and
LogPI each succeeded once in five seeds, while UCB, Thompson sampling, and MES
did not succeed. On the rough field, MES and UCB succeeded in all five seeds,
LogEI and LogPI in four, and Thompson sampling in none.

One GP fit warning occurred at evaluation 83 of the rough-field LogEI seed-3
trajectory. The optimizer still completed and the trajectory succeeded. This
warning is retained in the raw CSV and task summary rather than hidden.

The Phase 4 thresholds remain provisional. Under those values, the global
confidence-gap observer triggered early on 70% of trajectories, saved 53.3
evaluations on average, and false-stopped 24% of all trajectories. Stagnation
triggered early on every trajectory, saved 81.2, and false-stopped 46%.
Negligible expected improvement triggered early on 96%, saved 84.3, and
false-stopped 50%. Posterior acceptable regret triggered early on 75%, saved
70.1, and false-stopped 41%. These values show why the thresholds must be
calibrated before live use.

Outputs are under `results/phase3/full_budget_prepilot/`, including
`task_summary.csv`, `policy_summary.csv`, `stopping_observer_summary.csv`,
`prepilot_summary.png`, and representative 2D trajectory overlays.
