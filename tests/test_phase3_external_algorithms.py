import importlib.util

import pytest

from experiments.phase3_external_algorithms import (
    common_2d_systems,
    run_algorithm,
)


OPTIONAL_MODULES = {
    "cma_es": "cma",
    "snobfit": "SQSnobFit",
    "nomad": "PyNomad",
    "bam": "bam",
}


@pytest.mark.parametrize(
    "algorithm",
    ("cma_es", "snobfit", "nomad", "bam", "uniform_random"),
)
def test_external_adapter_respects_small_budget(algorithm):
    module = OPTIONAL_MODULES.get(algorithm)
    if module and importlib.util.find_spec(module) is None:
        pytest.skip(f"optional dependency {module} is not installed")
    system = common_2d_systems()[0]
    trace, _ = run_algorithm(system, algorithm, seed=0, budget=10)
    assert len(trace) == 10
    assert [row["n_eval"] for row in trace] == list(range(1, 11))
    assert all(row["within_bounds"] == 1 for row in trace)
    assert all(row["algorithm"] == algorithm for row in trace)
