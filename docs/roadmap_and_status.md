# Project roadmap and status

This page separates Peter's original phase plan from the exploratory work added while learning the methods. The workplan is the source of truth for the intended study; a prototype touching a phase does not mean that phase's deliverables are complete.

## Where the work currently stands

| Peter's phase | Intended work | Current state |
| --- | --- | --- |
| 1 — Foundations | Reading list, primer, and visual BO example | Learning materials, example, results, and presentation are in the repository. |
| 2 — Test functions | Validate the 24 continuous 2D/3D benchmarks; add the separately reported MRF extension | Benchmark bank, tests, documentation, and generated 2D/3D gallery figures are present. The Phase 2 benchmark tests pass (33 tests). The 3×3 Ising MRF is an auxiliary discrete extension—not part of the 24 continuous cases. |
| 3 — Acquisition methods | Implement EI, PI, UCB, Thompson sampling, and MES in BoTorch under a common protocol | **Expanded smoke complete; pre-pilot started.** All five policies completed a 40-trajectory, two-seed expanded smoke with no failed runs or fit warnings. A one-policy/one-seed full-budget slice now covers all four pre-pilot systems. The complete paired policy/seed pre-pilot remains pending, so no policy ranking is claimed. |
| 4 — Stopping rules | Implement the five specified stopping approaches with shared diagnostics and model-aware uncertainty | **Implementation smoke complete; thresholds still provisional.** All five formal observers, first-trigger logging, repeated-look corrections, hard-budget fallback, documentation, and trajectory-invariance tests are present. Four full-budget MES trajectories now exercise the observers, but scientific thresholds still require collaborator confirmation. |
| 5 — Pilot screen | Full 24-case × 5-acquisition × 5-seed BoTorch pilot, plus random reference | **Execution scaffold present; matrix not launched.** A deterministic 600-task manifest and resumable per-task runner are available. The current 540-trajectory NumPy matrix and separate continuous external-optimizer screen remain diagnostic context, not substitutes for the formal pilot. |
| 6 — Reliability analysis | Regret, success, false-stop frequency, cost, model diagnostics, and failure analysis | **Scoring layer started.** Truth-aware first-trigger scoring is separated from online observers. Aggregation, uncertainty intervals, figures, and full failure analysis still require Phase 5 data. |
| 7 — Confirmation | More seeds/field realizations and focused robustness checks | Not started. |

## What the current Phase 3 prototype does

`experiments/phase3_characterization.py` is a transparent, fixed-hyperparameter NumPy exact-GP implementation. It runs 9 systems (including the MRF), 12 configurations, and 5 paired seeds—540 trajectories total—with 5 initial observations and a budget of 24. Its search configurations vary RBF/Matérn kernels, length scale, UCB exploration weight, EI/PI/UCB, a matched-noise condition, one interaction, and random search.

Stopping diagnostics now use a common RBF GP with each system's nominal length scale, independent of the search configuration's kernel/length-scale setting. This improves the comparison of stopping diagnostics across search policies. It remains a pilot convention, not the final stopping-rule design in Peter's plan. The diagnostics are applied to completed trajectories to estimate counterfactual stop times; this is valid for observers that do not change the search policy. In the regenerated run, the UCB-gap proxy saved 2.42 evaluations on average and false-stopped 32.2% of runs; maximum marginal P(improve) saved 3.27 and false-stopped 33.5%. Those are descriptive results under this prototype's definitions, not performance guarantees.

The MRF's first standalone pilot is retained under `results/phase3/mrf_bo_pilot/` as a historical experiment. It used a different protocol from the later matrix, so its results should not be pooled with the matrix results. The matrix itself includes a fresh MRF condition.

The external-optimizer screen is retained under `results/phase3/external_algorithms/`. It uses six continuous 2D systems, five methods, five seeds, and a maximum budget of 150. It validates official solver adapters and budget accounting, but it is not pooled with the finite-catalogue BO prototype because the search domains and initialization semantics differ.

## Repository map

| Location | Contents |
| --- | --- |
| `workplan.md` | Original research question, planned phases, protocols, and deliverables |
| `PHASE1_README.md`, `PHASE2_README.md` | Short, phase-specific entry points |
| `docs/` | Concepts, reading notes, benchmark definitions, status, and experiment reports |
| `examples/` | Small runnable demonstrations and landscape gallery |
| `src/` | Benchmark definitions and evaluation inputs |
| `experiments/` | Reproducible experimental runners |
| `tests/` | Scientific and software correctness checks |
| `results/phase1/` | Worked-example trace and summary |
| `results/phase3/` | Phase 3 characterization outputs and the earlier MRF pilot |
| `presentations/` | English project presentation |
| `configs/phase3_fair_protocol.yaml` | Frozen common settings for the formal finite-catalogue comparison |
| `results/phase3/botorch_smoke/` | Reduced implementation smoke trace; not scientific performance evidence |

Generated Python caches and local test caches are ignored by Git and should not be committed. Keep the two MRF output sets: they document different protocols rather than duplicated runs.

## Immediate next steps

1. Ask Peter to sign off on the frozen domain, budget, primary success threshold, and decision to keep noise as a separate extension.
2. **Completed:** run two paired smoke seeds with five acquisition steps, profile all policies, and retain an audit row for every planned run.
3. **Started:** run the four-system full-budget pre-pilot. The MES/seed-0 slice is complete; the other 96 acquisition/seed tasks remain.
4. Obtain collaborator sign-off on the Phase 3 finite-catalogue protocol and the Phase 4 objective-unit tolerance, false-stop risk, and repeated-look interpretation.
5. Complete the paired pre-pilot, profile Phase 4 cost across policies and difficult landscapes, and freeze protocol v2 before the 600-task matrix. The NumPy screen remains diagnostic context, not a substitute for the formal comparison.

The expanded smoke runner now records one status/runtime row per planned
trajectory. The resumable 100-task full-budget pre-pilot entry point and
deterministic manifest are available in
`experiments/phase3_full_budget_prepilot.py`. These additions prepare the next
gate but do not change the requirement for collaborator approval before the
full scientific matrix.

## 2026-11-13 execution update

- Expanded smoke: 40/40 trajectories completed, zero failures, and zero GP fit
  warnings using four systems, five policies, two paired seeds, and five steps
  after initialization. The short smoke produced a 50% pooled success rate for
  every policy and therefore is implementation evidence, not a ranking.
- Full-budget slice: MES with optimizer seed 0 completed on the four pre-pilot
  systems. Final normalized regrets were 0.000 (quadratic), 0.414 (hidden
  narrow), 0.023 (rough field), and 0.000 (3D Rosenbrock). Three of four met
  the 0.05 success definition.
- Stopping diagnostics on this four-trajectory slice remain descriptive. The
  stagnation rule false-stopped on two trajectories; max-EI false-stopped on
  one; global-gap and posterior-regret each false-stopped on one and did not
  trigger early on the rough field. The sample is too small and unbalanced for
  a rule ranking.
- A 25-evaluation MES runtime profile took 7.58 seconds with Phase 4 observers
  and 7.08 seconds without them on the same machine, about 7% overhead in this
  diagnostic case. Full-budget runtimes ranged from 22.6 to 74.6 seconds for
  the four completed trajectories.

