"""Truth-aware evaluation kept separate from acquisition and stopping code."""
from __future__ import annotations

from typing import Any, Iterable


def summarize_stopping_outcome(
    rows: Iterable[dict[str, Any]],
    *,
    rule: str,
    success_threshold: float = 0.05,
) -> dict[str, Any]:
    """Score one observer on one completed hard-budget trajectory.

    Benchmark regret is intentionally used here, in the evaluator, and never in
    the online observer.  A rule that never fires uses the hard-budget fallback
    and is not counted as an early trigger.
    """
    ordered = sorted(rows, key=lambda row: int(row["n_eval"]))
    if not ordered:
        raise ValueError("rows must contain one completed trajectory")
    hard_budget = int(ordered[-1]["n_eval"])
    trigger_value = ordered[-1].get(f"{rule}_first_trigger")
    criterion_triggered = trigger_value not in (None, "")
    first_trigger = int(trigger_value) if criterion_triggered else None
    termination = first_trigger if first_trigger is not None else hard_budget
    row_by_evaluation = {int(row["n_eval"]): row for row in ordered}
    if termination not in row_by_evaluation:
        raise ValueError("the first trigger does not match a saved trajectory row")
    terminal_row = row_by_evaluation[termination]
    normalized_regret = float(terminal_row["normalized_regret"])
    success = normalized_regret <= success_threshold
    early_trigger = criterion_triggered and termination < hard_budget
    false_stop = early_trigger and not success
    return {
        "system": terminal_row["system"],
        "dimension": int(terminal_row["dimension"]),
        "policy": terminal_row["policy"],
        "seed": int(terminal_row["seed"]),
        "stopping_rule": rule,
        "criterion_triggered": int(criterion_triggered),
        "first_trigger": first_trigger,
        "termination_evaluations": termination,
        "hard_budget": hard_budget,
        "evaluations_saved": hard_budget - termination,
        "non_trigger": int(not criterion_triggered),
        "early_trigger": int(early_trigger),
        "normalized_regret_at_termination": normalized_regret,
        "success": int(success),
        "false_stop": int(false_stop),
        "unconditional_premature_stop": int(false_stop),
    }
