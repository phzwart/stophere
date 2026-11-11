from experiments.phase5_full_matrix import build_manifest
from src.evaluation import summarize_stopping_outcome


def test_phase5_manifest_has_600_unique_paired_tasks():
    tasks = build_manifest()
    assert len(tasks) == 24 * 5 * 5 == 600
    keys = {
        (
            task["system"],
            task["dimension"],
            task["policy"],
            task["optimizer_seed"],
        )
        for task in tasks
    }
    assert len(keys) == len(tasks)
    assert [task["task_index"] for task in tasks] == list(range(600))


def test_truth_is_used_only_when_scoring_a_saved_trigger():
    rows = [
        {
            "system": "example",
            "dimension": 2,
            "policy": "log_ei",
            "seed": 0,
            "n_eval": 10,
            "normalized_regret": 0.2,
            "incumbent_stagnation_first_trigger": None,
        },
        {
            "system": "example",
            "dimension": 2,
            "policy": "log_ei",
            "seed": 0,
            "n_eval": 20,
            "normalized_regret": 0.1,
            "incumbent_stagnation_first_trigger": 20,
        },
        {
            "system": "example",
            "dimension": 2,
            "policy": "log_ei",
            "seed": 0,
            "n_eval": 30,
            "normalized_regret": 0.0,
            "incumbent_stagnation_first_trigger": 20,
        },
    ]
    outcome = summarize_stopping_outcome(
        rows, rule="incumbent_stagnation", success_threshold=0.05
    )
    assert outcome["termination_evaluations"] == 20
    assert outcome["evaluations_saved"] == 10
    assert outcome["false_stop"] == 1
    assert outcome["success"] == 0
