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

Run one resumable task only after the protocol decision gate:

```bash
python -m experiments.phase3_full_budget_prepilot --task-index 0
```

Summarize every completed full-budget task currently present:

```bash
python -m experiments.phase3_prepilot_summary
```

The summary command writes one row per trajectory, one truth-aware outcome per
trajectory and stopping rule, and a partial aggregate. A partial aggregate is
descriptive only until the complete paired acquisition/seed plan is present.

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
