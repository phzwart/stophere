import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, initial_indices, make_catalogue, run_trajectory


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "phase3_fair_protocol.yaml"


def test_protocol_freezes_paired_initial_design_and_dimension_budgets():
    protocol = Protocol.load(PROTOCOL_PATH)
    assert protocol.catalogue_size == 2048
    assert protocol.initial_size(2) == 10
    assert protocol.initial_size(3) == 15
    assert protocol.budget(2) == 100
    assert protocol.budget(3) == 150
    first = initial_indices(dimension=2, optimizer_seed=7, protocol=protocol)
    second = initial_indices(dimension=2, optimizer_seed=7, protocol=protocol)
    assert first.tolist() == second.tolist()


def test_catalogue_is_repeatable_and_inside_unit_hypercube():
    protocol = Protocol.load(PROTOCOL_PATH)
    first = make_catalogue(2, protocol)
    second = make_catalogue(2, protocol)
    assert first.dtype == torch.double
    assert torch.equal(first, second)
    assert bool(torch.all((first >= 0.0) & (first <= 1.0)))


def test_short_log_ei_run_has_no_duplicates_and_replays_exactly():
    protocol = Protocol.load(PROTOCOL_PATH)
    benchmark = get_benchmark("quadratic_peak", 2)
    first = run_trajectory(
        benchmark=benchmark,
        policy="log_ei",
        optimizer_seed=0,
        protocol=protocol,
        budget_override=11,
    )
    second = run_trajectory(
        benchmark=benchmark,
        policy="log_ei",
        optimizer_seed=0,
        protocol=protocol,
        budget_override=11,
    )
    assert [row["n_eval"] for row in first] == [10, 11]
    assert [row["last_selected_index"] for row in first] == [
        row["last_selected_index"] for row in second
    ]
    selected = json.loads(first[-1]["selected_indices"])
    assert len(selected) == len(set(selected)) == 11
    assert first[-1]["normalized_regret"] <= first[0]["normalized_regret"]

