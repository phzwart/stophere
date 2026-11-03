# Phase 1 structured results

- `bo_intro_trace.csv` records every initial and BO-selected evaluation.
- `bo_intro_summary.json` records the seed, evaluation budget, recommendation, retrospective reference optimum, regret, and package versions.

The dense reference optimum is used only for retrospective teaching and evaluation. It is never exposed to the optimizer or used to select the next experiment.
