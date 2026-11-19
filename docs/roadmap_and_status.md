# Project roadmap and status

This page separates Peter's original phase plan from the exploratory work added while learning the methods. The workplan is the source of truth for the intended study; a prototype touching a phase does not mean that phase's deliverables are complete.

## Where the work currently stands

| Peter's phase | Intended work | Current state |
| --- | --- | --- |
| 1 — Foundations | Reading list, primer, and visual BO example | Learning materials, example, results, and presentation are in the repository. |
| 2 — Test functions | Validate the 24 continuous 2D/3D benchmarks; add the separately reported MRF extension | Benchmark bank, tests, documentation, and generated 2D/3D gallery figures are present. The Phase 2 benchmark tests pass (33 tests). The 3×3 Ising MRF is an auxiliary discrete extension—not part of the 24 continuous cases. |
| 3 — Acquisition methods | Implement and compare EI, PI, UCB, Thompson sampling, and MES in BoTorch under a common protocol | **Implementation, synthetic pre-pilot, and real ALS optimization pre-pilot complete.** The 100-trajectory four-system pre-pilot and 90-trajectory three-target ALS replay are decision evidence, not a final ranking. The 24-case formal comparison remains unfinished. |
| 4 — Stopping rules | Implement and evaluate the five specified stopping approaches with shared diagnostics and model-aware uncertainty | **Observer implementation and provisional scoring complete.** The five observers were evaluated on the same ALS optimization trajectories; thresholds remain provisional and false-stop rates on sparse targets are too high for live use. |
| 5 — Pilot screen | Run the full 24-case × 5-acquisition × 5-seed formal matrix, plus random reference | **Scaffold present; execution not started.** A deterministic 600-task manifest and resumable runner are available. The ALS three-target replay is categorized under Phase 3 acquisition comparison and Phase 4 stopping diagnostics; it does not replace this synthetic benchmark matrix. |
| 6 — Reliability analysis | Regret, success, false-stop frequency, cost, model diagnostics, and failure analysis | **Pre-pilot scoring layer complete.** Truth-aware first-trigger scoring is separated from online observers; policy summaries, stopping summaries, quantile figures, random-search probability audit, and trajectory overlays are present. Full reliability claims still require Phase 5 and confirmation data. |
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
3. **Completed:** run and aggregate the 100-trajectory four-system full-budget pre-pilot.
4. Obtain collaborator sign-off on the Phase 3 finite-catalogue protocol and the Phase 4 objective-unit tolerance, false-stop risk, and repeated-look interpretation.
5. Finish the real-data optimization pre-pilot protocol review and use Peter's decisions to freeze Phase 3/4 protocol v2 before the Phase 5 600-task synthetic matrix. Mapping is outside the current project priority.

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

## 2026-11-17 execution update

- Full-budget pre-pilot: 100/100 trajectories complete across four systems,
  five policies, and five paired seeds. Pooled success is 70%. LogEI, LogPI,
  UCB, and MES each achieved 75%; Thompson sampling achieved 50%.
- The hidden narrow peak is the dominant failure. It was solved in only two of
  25 policy-seed runs. Thompson sampling also failed every rough-field run.
- The provisional stopping rules saved 53-84 evaluations on average but
  false-stopped 24-50% of all trajectories. The thresholds are not ready for
  live deployment.
- One GP optimization warning occurred in one of 100 trajectories; the run
  completed and succeeded, and the warning is retained for auditability.
- The random-search plateau was checked against its exact
  without-replacement success probability. The exact expectation increases
  from 59.5% at 100 evaluations to 66.8% at 150, while the five-seed observed
  value remains 62.5% at both checkpoints.
- Private processed ALS and PETRA III data were downloaded and audited. The
  processed ALS archive contains a complete 57-by-57 grid of reduced spectra,
  which supports offline replay without committing private data.
- Historical side experiment, outside the current optimization priority: a
  five-seed ALS mapping replay compared maximum posterior variance with
  random sampling from the same 25 starting measurements. At 99 measurements,
  maximum variance had lower mean normalized map RMSE (0.2860 versus 0.2995)
  and lower maximum posterior uncertainty (0.2294 versus 0.3211). This uses a
  provisional q-at-raw-maximum response and fixed published length scales.
- A one-seed ALS optimization replay ran all five formal BoTorch policies to
  99 measurements. MES found the catalogue best point; LogEI and UCB reached
  normalized regret 0.0060; LogPI and Thompson sampling remained near 0.18.
  This validates real-data ingestion and replay, but is not a scientific policy
  ranking because the target is provisional and variability is not measured.

## 2026-11-18 execution update

- The processed ALS spectra were converted into three provisional Phase 3 optimization
  targets: total integrated intensity, maximum detected peak height, and
  minimum peak FWHM.
- A resumable paired Phase 3 replay now compares LogEI, LogPI, UCB, Thompson sampling,
  MES, and a random-search reference with the same measured catalogue, 25-point
  Sobol initialization, 99-measurement budget, GP surrogate, recommendation
  rule, and retrospective scoring.
- All five passive stopping observers are evaluated without changing any
  acquisition trajectory or using future observations or the true optimum.
- The reusable Phase 3 code, tests, and declared provisional protocol are committed.
  The private source data and all data-derived trajectories, figures, and
  numerical summaries remain outside the repository pending approval to share.
- The main scientific gate is now target definition and risk calibration, not
  software availability: the q window, background treatment, peak/FWHM
  definition, success tolerance, acceptable false-stop risk, and live cost
  model still require collaborator confirmation.

