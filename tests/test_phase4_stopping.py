import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, make_catalogue, run_trajectory
from src.stopping import StoppingProtocol


ROOT = Path(__file__).resolve().parents[1]
PHASE3_PROTOCOL = ROOT / "configs" / "phase3_fair_protocol.yaml"
PHASE4_PROTOCOL = ROOT / "configs" / "phase4_stopping_observers.yaml"


def _small_protocols():
    acquisition = Protocol.load(PHASE3_PROTOCOL)
    acquisition_raw = copy.deepcopy(acquisition.raw)
    acquisition_raw["domain"]["catalogue_size"] = 64

    stopping = StoppingProtocol.load(PHASE4_PROTOCOL)
    stopping_raw = copy.deepcopy(stopping.raw)
    stopping_raw["monte_carlo"]["num_paths"] = 32
    return Protocol(acquisition_raw), StoppingProtocol(stopping_raw)


def test_hard_budget_is_always_a_stopping_check():
    _, stopping = _small_protocols()
    assert not stopping.should_check(n_eval=10, initial_size=10, hard_budget=11)
    assert stopping.should_check(n_eval=11, initial_size=10, hard_budget=11)
    assert stopping.maximum_checks(initial_size=10, hard_budget=11) == 1


def test_observers_do_not_change_the_acquisition_trajectory():
    acquisition, stopping = _small_protocols()
    benchmark = get_benchmark("quadratic_peak", 2)
    catalogue = make_catalogue(2, acquisition)
    without_observers = run_trajectory(
        benchmark=benchmark,
        policy="log_ei",
        optimizer_seed=3,
        protocol=acquisition,
        catalogue=catalogue,
        budget_override=11,
    )
    with_observers = run_trajectory(
        benchmark=benchmark,
        policy="log_ei",
        optimizer_seed=3,
        protocol=acquisition,
        catalogue=catalogue,
        budget_override=11,
        stopping_protocol=stopping,
    )
    assert json.loads(without_observers[-1]["selected_indices"]) == json.loads(
        with_observers[-1]["selected_indices"]
    )
    final = with_observers[-1]
    assert final["fixed_budget_fire"] == 1
    assert final["fixed_budget_first_trigger"] == 11
    assert final["posterior_regret_paths"] == 32
    assert 0.0 <= final["posterior_regret_probability_lower"] <= 1.0


def test_every_observer_has_a_first_trigger_field():
    acquisition, stopping = _small_protocols()
    rows = run_trajectory(
        benchmark=get_benchmark("quadratic_peak", 2),
        policy="ucb",
        optimizer_seed=1,
        protocol=acquisition,
        budget_override=10,
        stopping_protocol=stopping,
    )
    final = rows[-1]
    for name in (
        "fixed_budget",
        "incumbent_stagnation",
        "negligible_expected_improvement",
        "global_confidence_gap",
        "posterior_acceptable_regret",
    ):
        assert f"{name}_fire" in final
        assert f"{name}_first_trigger" in final
