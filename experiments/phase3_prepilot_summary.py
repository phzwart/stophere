"""Summarize completed full-budget pre-pilot trajectories and stop outcomes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation import summarize_stopping_outcome
from src.stopping import RULES


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outcomes(outcomes: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome["stopping_rule"]].append(outcome)
    summaries = []
    for rule in RULES:
        rows = grouped[rule]
        early = [row for row in rows if row["early_trigger"]]
        summaries.append({
            "stopping_rule": rule,
            "trajectories": len(rows),
            "early_triggers": len(early),
            "early_trigger_rate": len(early) / len(rows),
            "non_trigger_rate": sum(row["non_trigger"] for row in rows) / len(rows),
            "false_stops": sum(row["false_stop"] for row in rows),
            "false_stop_rate_among_early_triggers": (
                sum(row["false_stop"] for row in early) / len(early)
                if early else 0.0
            ),
            "unconditional_premature_stop_rate": (
                sum(row["unconditional_premature_stop"] for row in rows) / len(rows)
            ),
            "mean_evaluations_saved": (
                sum(row["evaluations_saved"] for row in rows) / len(rows)
            ),
            "mean_evaluations_saved_when_early": (
                sum(row["evaluations_saved"] for row in early) / len(early)
                if early else 0.0
            ),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-directory", default="results/phase3/full_budget_prepilot/raw"
    )
    parser.add_argument(
        "--output-directory", default="results/phase3/full_budget_prepilot/summary"
    )
    parser.add_argument("--success-threshold", type=float, default=0.05)
    args = parser.parse_args()

    raw_directory = Path(args.raw_directory)
    paths = sorted(raw_directory.glob("task_*_full_budget.csv"))
    if not paths:
        raise ValueError(f"no full-budget task CSV files found in {raw_directory}")

    trajectory_summaries = []
    stopping_outcomes = []
    for path in paths:
        rows = read_rows(path)
        final = rows[-1]
        metadata_path = path.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists() else {}
        )
        selected = json.loads(final["selected_indices"])
        trajectory_summaries.append({
            "source_file": path.name,
            "system": final["system"],
            "dimension": int(final["dimension"]),
            "policy": final["policy"],
            "seed": int(final["seed"]),
            "hard_budget": int(final["n_eval"]),
            "saved_posterior_states": len(rows),
            "unique_selected_points": len(set(selected)),
            "final_normalized_regret": float(final["normalized_regret"]),
            "final_success_le_0.05": int(final["success_le_0.05"]),
            "stopping_checks": sum(int(row["stopping_checked"]) for row in rows),
            "fit_warning_count": sum(int(row["fit_warning_count"]) for row in rows),
            "runtime_seconds": metadata.get("runtime_seconds", ""),
        })
        for rule in RULES:
            stopping_outcomes.append(summarize_stopping_outcome(
                rows, rule=rule, success_threshold=args.success_threshold
            ))

    output_directory = Path(args.output_directory)
    write_rows(output_directory / "trajectory_summary.csv", trajectory_summaries)
    write_rows(output_directory / "stopping_outcomes.csv", stopping_outcomes)
    write_rows(
        output_directory / "stopping_rule_summary.csv",
        aggregate_outcomes(stopping_outcomes),
    )
    print(f"summarized {len(paths)} full-budget trajectories")
    print(f"wrote summaries to {output_directory}")


if __name__ == "__main__":
    main()
