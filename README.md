# Bayesian Optimization Stopping Study

This project studies when Bayesian optimization can stop while still returning an acceptably good solution. The original research plan is in `workplan.md`; see [roadmap and status](docs/roadmap_and_status.md) before interpreting the exploratory prototype as a completed phase.

## Start here

- [Workplan](workplan.md) — project question, scope, milestones, and planned experiments.
- [Roadmap and status](docs/roadmap_and_status.md) — maps the current work to Peter's original phases and identifies the next planned step.
- [Phase 1 overview](PHASE1_README.md) — core concepts, reading list, worked BO example, results, and presentation.
- [Phase 2 overview](PHASE2_README.md) — benchmark bank, checks, and landscape gallery.
- [Phase 3 overview](PHASE3_README.md) — exploratory acquisition/configuration characterization and its limits.
- [Phase 3 prototype report](docs/phase3_characterization.md) — detailed methods, results, and exploratory stopping diagnostics.
- [Documentation index](docs/README.md) — all written materials grouped by phase.

## Code map

- `examples/bo_intro.py` and `examples/01_bo_intro.ipynb`: Phase 1 one-dimensional BO loop.
- `examples/02_lengthscale_sensitivity.py`: Phase 1 GP misspecification example.
- `src/benchmarks.py`: Phase 2 target functions, all maximization on normalized inputs.
- `experiments/phase3_characterization.py`: reproducible NumPy exact-GP Phase 3 screening prototype over 9 systems and 12 configurations.
- `experiments/phase3_external_algorithms.py`: official external-solver adapter screen on six continuous 2D systems.
- `tests/`: correctness checks for the BO example and benchmark bank.
- `results/phase1/`: saved Phase 1 trace and numerical summary.
- `results/phase2/gallery/`: generated benchmark plots (recreate using the Phase 2 README).
- `results/phase3/characterization/`: Phase 3 per-step traces, per-system summaries, paired deltas, exploratory stop-diagnostic outcomes, and comparison plot.
- `results/phase3/mrf_bo_pilot/`: retained first MRF pilot, with a different protocol from the later matrix; do not pool the two result sets.
- `results/phase3/external_algorithms/`: separate continuous-domain CMA-ES, SNOBFIT, NOMAD, BAM, and random screen; do not pool it with the finite-catalogue BO prototype.

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python -m pytest -q
python -m examples.phase2_gallery
~~~

The gallery visualizes test-function landscapes; it is not an optimization result. The Phase 3 characterization is a small fixed-hyperparameter NumPy screen, not the planned BoTorch study. It helps identify conditions and failure cases for follow-up; it does not identify a universally best method.
