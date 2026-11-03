# Phase 1: Bayesian Optimization Primer

This branch completes the first learning milestone in `workplan.md`:

- `docs/reading_list.md` — annotated ten-paper reading list.
- `docs/bo_primer.md` — compact guide to BO, Gaussian processes, acquisition,
  recommendation, and stopping.
- `docs/bo_gp_stopping_rules_study_guide.md` — detailed study guide with a
  worked GP calculation, hyperparameter failure modes, model diagnostics,
  stopping rules, and the next implementation plan.
- `docs/bo_basics_cn.md` — Chinese explanation of the core concepts, formulas,
  hyperparameters, BoTorch steps, and project logic.
- `docs/phase1_results.md` — completion matrix, verified numerical result,
  limitations, and questions carried into Phase 2.
- `docs/bo_concept_map.md` — a visual map and a worked numerical example.
- `docs/handwritten_note_english.md` — cleaned English notes based on the
  original handwritten study outline.
- `examples/01_bo_intro.ipynb` — visual BoTorch closed-loop example.
- `examples/bo_intro.py` — script version of the same experiment.
- `examples/02_lengthscale_sensitivity.py` — model-misspecification example.
- `tests/test_bo_intro.py` — inexpensive correctness checks.
- `results/phase1/` — exact trace, numerical summary, and verified versions.
- `assets/` — regenerated figures for the BO run and length-scale diagnostic.
- `presentations/bo_project_phase1_bilingual.pptx` — ten-slide project overview
  with English and Chinese speaker notes.
- `presentations/bo_project_bilingual_speaker_guide.docx` — page-by-page
  English and Chinese speaking script.

## Recommended reading order

1. Start with `docs/bo_concept_map.md` for the relationship among the main
   terms.
2. Read `docs/bo_primer.md` for the Phase 1 introduction.
3. Use `docs/bo_gp_stopping_rules_study_guide.md` for the numerical example,
   hyperparameter effects, diagnostics, and stopping-rule details.
4. Use `docs/bo_basics_cn.md` for the corresponding Chinese explanation.
5. Review `docs/reading_list.md` when a concept requires its original source.
6. Run `examples/01_bo_intro.ipynb`, then use
   `examples/02_lengthscale_sensitivity.py` as a misspecification stress test.
7. Read `docs/phase1_results.md` for the verified output and Phase 2 handoff.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-phase1.txt
python examples/bo_intro.py
python examples/02_lengthscale_sensitivity.py
pytest -q
```

The toy objective is only evaluated at points selected by the optimizer. Its
dense curve is used for visualization and retrospective scoring, not by BO.

## Verified completion

The fixed-seed run uses 3 initial points and 6 BO iterations. It returns
`x = 0.7494546` with a dense-reference simple regret of approximately
`0.0004961`. All 6 Phase 1 tests pass in the recorded environment. See
`docs/phase1_results.md` for the complete interpretation and limitations.

