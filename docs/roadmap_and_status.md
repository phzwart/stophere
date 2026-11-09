# Project roadmap and status

This page separates Peter's original phase plan from the exploratory work added while learning the methods. The workplan is the source of truth for the intended study; a prototype touching a phase does not mean that phase's deliverables are complete.

## Where the work currently stands

| Peter's phase | Intended work | Current state |
| --- | --- | --- |
| 1 — Foundations | Reading list, primer, and visual BO example | Learning materials, example, results, and presentation are in the repository. |
| 2 — Test functions | Validate the 24 continuous 2D/3D benchmarks; add the separately reported MRF extension | Benchmark bank, tests, documentation, and generated 2D/3D gallery figures are present. The Phase 2 benchmark tests pass (33 tests). The 3×3 Ising MRF is an auxiliary discrete extension—not part of the 24 continuous cases. |
| 3 — Acquisition methods | Implement EI, PI, UCB, Thompson sampling, and MES in BoTorch under a common protocol | **Not complete.** An exploratory NumPy prototype currently screens EI, PI, UCB, kernels, and hyperparameters. It has no BoTorch reproduction, Thompson sampling, or MES. This is the next planned phase to start. |
| 4 — Stopping rules | Implement the five specified stopping approaches with shared diagnostics and model-aware uncertainty | **Not complete.** The prototype has fixed budget, a patience heuristic, a global-UCB-gap proxy, and a maximum marginal-improvement-probability proxy. These are not all the planned definitions. |
| 5 — Pilot screen | Full 24-case × 5-acquisition × 5-seed BoTorch pilot, plus random reference | **Not complete.** The current 540-trajectory NumPy matrix covers 9 systems, 12 search configurations, and 5 seeds at budget 24. A separate 150-run continuous 2D screen validates CMA-ES, SNOBFIT, NOMAD, and BAM adapters. Neither is the planned BoTorch pilot. |
| 6 — Reliability analysis | Regret, success, false-stop frequency, cost, model diagnostics, and failure analysis | Preliminary descriptive metrics exist for the screen; the planned complete analysis and uncertainty treatment remain. |
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

Generated Python caches and local test caches are ignored by Git and should not be committed. Keep the two MRF output sets: they document different protocols rather than duplicated runs.

## Immediate next steps

1. Start the actual planned Phase 3 with a small BoTorch end-to-end pilot: one 2D landscape, one seed, one acquisition first; verify the GP fit, acquisition optimization, trace, and reproducibility.
2. Expand to all five planned acquisition strategies only after that small path is stable. Record fitted hyperparameters, package versions, numerical warnings, and the exact recommendation rule.
3. Implement Phase 4 as separate stopping observers using the workplan's operational definitions; keep each rule's model inputs and threshold units explicit.
4. Freeze the protocol before launching the full pilot. The current NumPy screen is useful for choosing diagnostic cases, but its pooled rankings are not a substitute for the planned BoTorch comparison.

