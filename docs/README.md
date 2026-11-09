# Documentation index

- [Roadmap and status](roadmap_and_status.md) — maps completed and exploratory work to Peter's original phase plan.

## Phase 1 — foundations and worked example

- [BO primer](bo_primer.md)
- [Concept map and worked calculation](bo_concept_map.md)
- [Detailed GP, hyperparameter, and stopping-rule guide](bo_gp_stopping_rules_study_guide.md)
- [Annotated reading list](reading_list.md)
- [Phase 1 results and limitations](phase1_results.md)
- [Cleaned English handwritten-note outline](handwritten_note_english.md)

## Phase 2 — benchmark bank

- [Benchmark catalog, API, and validation notes](benchmarks.md)
- [Benchmark implementation](../src/benchmarks.py)
- [Benchmark tests](../tests/test_benchmarks.py)
- [Auxiliary MRF tests](../tests/test_mrf_benchmark.py)
- [Landscape gallery generator](../examples/phase2_gallery.py)
- [2D benchmark gallery](../results/phase2/gallery/benchmark_gallery_2d.png)
- [3D slices gallery, x3 = 0.25](../results/phase2/gallery/benchmark_gallery_3d_slices_x3_025.png)
- [3D slices gallery, x3 = 0.50](../results/phase2/gallery/benchmark_gallery_3d_slices.png)
- [3D slices gallery, x3 = 0.75](../results/phase2/gallery/benchmark_gallery_3d_slices_x3_075.png)

## Phase 3 — exploratory NumPy characterization prototype

- [Methods, MRF explanation, results, and limitations](phase3_characterization.md); this prototype does not complete the planned BoTorch Phase 3.
- [Reproducible Phase 3 experiment](../experiments/phase3_characterization.py)
- [Extended-budget BO policy results](phase3_budget_profiles.md)
- [External derivative-free algorithm results](phase3_external_algorithms.md)
- [Full comparison matrix plan](phase3_full_matrix_plan.md)
- [Per-system/configuration summary](../results/phase3/characterization/summary.csv)
- [Paired changes versus nominal GP](../results/phase3/characterization/comparison_deltas.csv)
- [Exploratory stopping diagnostics](../results/phase3/characterization/stopping_rules.csv)
- [Full evaluation trajectories](../results/phase3/characterization/trajectories.csv)
- [Summary plot](../results/phase3/characterization/characterization_summary.png)
- [External algorithm audit table](../results/phase3/external_algorithms/run_status.csv)
- [External success-versus-budget plot](../results/phase3/external_algorithms/success_by_budget.png)

## Project navigation

- [Repository overview](../README.md)
- [Workplan](../workplan.md)
- [Phase 1 overview](../PHASE1_README.md)
- [Phase 2 overview](../PHASE2_README.md)
- [Phase 3 overview](../PHASE3_README.md)
