import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.phase3_characterization import (
    BUDGET,
    SYSTEMS,
    catalogue,
    configuration_rows,
    kernel_matrix,
    posterior,
    run_trajectory,
    stopping_posterior,
    stopping_summaries,
)
from experiments.phase3_success_profiles import success_profile_rows
from experiments.phase3_budget_profiles import _run_task


def test_characterization_system_bank_covers_continuous_and_discrete_cases():
    names = {system.name for system in SYSTEMS}
    assert len(names) == len(SYSTEMS)
    assert "ising_mrf_3x3" in names
    assert any(system.dimension == 3 for system in SYSTEMS)
    assert all(catalogue(system).shape[1] == system.dimension for system in SYSTEMS)


def test_one_factor_configuration_list_has_no_duplicates():
    configs = configuration_rows(SYSTEMS[0])
    assert len({config[0] for config in configs}) == len(configs)
    # The baseline is represented once instead of repeated for every factor.
    assert sum(config[1:] == ("rbf", 1.0, "ucb", 2.0, 0.0) for config in configs) == 1


def test_kernels_are_symmetric_positive_semidefinite_on_simple_inputs():
    x = np.array([[0.0], [0.3], [1.0]])
    for kind in ("rbf", "matern32", "matern52"):
        matrix = kernel_matrix(x, x, kind, 0.25)
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > -1e-8


def test_stopping_diagnostic_uses_the_declared_common_rbf_model():
    system = next(s for s in SYSTEMS if s.name == "smooth_2d")
    points = catalogue(system)
    train_x = points[[0, 100, 300]]
    train_y = np.array([0.1, 0.7, 0.4])
    mu, sigma, _, _ = stopping_posterior(
        train_x, train_y, points, system, noise_fraction=0.0, value_range=1.0
    )
    expected = posterior(train_x, train_y, points, "rbf", system.base_lengthscale, 0.0)
    assert np.allclose(mu, expected[0])
    assert np.allclose(sigma, expected[1])


def test_small_trajectory_and_stopping_rules_are_bounded():
    system = next(s for s in SYSTEMS if s.name == "smooth_2d")
    config = configuration_rows(system)[0]
    trace = run_trajectory(system, config, seed=0)
    assert len(trace) == BUDGET - 4  # 5 initial points are recorded at n=5..24
    result = stopping_summaries(trace)
    assert len(result) == 4
    assert all(5 <= row["stop_at_evals"] <= BUDGET for row in result)


def test_extended_trajectory_can_skip_stopping_diagnostics():
    system = next(s for s in SYSTEMS if s.name == "smooth_2d")
    config = configuration_rows(system)[0]
    trace = run_trajectory(
        system, config, seed=0, budget=8, include_stopping_diagnostics=False
    )
    assert [row["n_eval"] for row in trace] == [5, 6, 7, 8]
    assert all(np.isnan(row["posterior_ucb_gap_over_y_sd"]) for row in trace)
    assert all(np.isnan(row["max_marginal_p_improve"]) for row in trace)


def test_success_profiles_report_pooled_rate_and_system_quantiles():
    traces = [
        {"system": "easy", "config": "nominal", "seed": seed,
         "n_eval": 5, "normalized_regret": regret}
        for seed, regret in enumerate((0.0, 0.01, 0.02, 0.03))
    ] + [
        {"system": "hard", "config": "nominal", "seed": seed,
         "n_eval": 5, "normalized_regret": regret}
        for seed, regret in enumerate((0.20, 0.20, 0.01, 0.20))
    ]
    row = success_profile_rows(traces, checkpoints=(5,))[0]
    assert row["pooled_success_rate"] == 5 / 8
    assert row["system_success_median"] == 0.625
    assert row["system_success_q25"] == 0.4375
    assert row["system_success_q75"] == 0.8125


def test_incremental_budget_runner_matches_reference_trajectory():
    system = next(s for s in SYSTEMS if s.name == "smooth_2d")
    config = configuration_rows(system)[0]
    reference = run_trajectory(
        system, config, seed=0, budget=12, include_stopping_diagnostics=False
    )
    incremental = _run_task((system, config, 0, 12))
    assert [row["n_eval"] for row in incremental] == [
        row["n_eval"] for row in reference
    ]
    assert np.allclose(
        [row["normalized_regret"] for row in incremental],
        [row["normalized_regret"] for row in reference],
    )


def test_success_profiles_carry_forward_natural_early_termination():
    traces = [
        {"system": "system", "config": "method", "seed": 0,
         "n_eval": 5, "normalized_regret": 0.2},
        {"system": "system", "config": "method", "seed": 0,
         "n_eval": 8, "normalized_regret": 0.01},
    ]
    row = success_profile_rows(
        traces, checkpoints=(10,), carry_forward=True
    )[0]
    assert row["n_eval"] == 10
    assert row["runs"] == 1
    assert row["pooled_success_rate"] == 1.0
