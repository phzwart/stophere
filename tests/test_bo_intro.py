import importlib.util
import json
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "examples" / "bo_intro.py"
SPEC = importlib.util.spec_from_file_location("bo_intro", MODULE_PATH)
bo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bo)

EXAMPLES_DIR = MODULE_PATH.parent
sys.path.insert(0, str(EXAMPLES_DIR))
LS_SPEC = importlib.util.spec_from_file_location(
    "lengthscale_sensitivity", EXAMPLES_DIR / "02_lengthscale_sensitivity.py"
)
ls_example = importlib.util.module_from_spec(LS_SPEC)
LS_SPEC.loader.exec_module(ls_example)


def test_objective_shape_and_repeatability():
    x = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
    y1, y2 = bo.objective(x), bo.objective(x)
    assert y1.shape == (3, 1)
    assert torch.equal(y1, y2)


def test_bo_adds_exactly_one_observation_per_step():
    x, y, _, trace = bo.run_bo(seed=17, n_steps=2)
    assert x.shape == y.shape == (5, 1)
    assert len(trace) == 2
    assert torch.all((x >= 0) & (x <= 1))
    assert [row["step"] for row in trace] == [1, 2]


def test_best_so_far_is_monotone():
    _, _, _, trace = bo.run_bo(seed=17, n_steps=3)
    best = [row["best_after"] for row in trace]
    assert best == sorted(best)
    assert all(row["best_after"] >= row["best_before"] for row in trace)


def test_phase1_run_finds_a_near_optimal_observation():
    x, y, _, _ = bo.run_bo(seed=17, n_steps=6)
    grid = torch.linspace(0, 1, 20001, dtype=torch.double).unsqueeze(-1)
    reference = bo.objective(grid).max()
    simple_regret = reference - y.max()
    assert simple_regret < 0.01
    assert x.shape[0] == 9


def test_saved_summary_distinguishes_evaluations_from_reference_grid(tmp_path):
    x, y, _, trace = bo.run_bo(seed=17, n_steps=1)
    summary = bo.save_results(x, y, trace, tmp_path)
    loaded = json.loads((tmp_path / "bo_intro_summary.json").read_text())
    assert summary["objective_evaluations"] == 4
    assert loaded["bo_iterations"] == 1
    assert "retrospective" in loaded["note"]


def test_lengthscale_diagnostic_sets_the_base_kernel_parameter():
    x = torch.tensor([[0.08], [0.37], [0.72]], dtype=torch.double)
    model = ls_example.fixed_lengthscale_model(x, bo.objective(x), 0.15)
    actual = model.covar_module.base_kernel.lengthscale.detach().squeeze()
    assert torch.isclose(actual, torch.tensor(0.15, dtype=torch.double))
    posterior = model.posterior(torch.tensor([[0.50]], dtype=torch.double))
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

