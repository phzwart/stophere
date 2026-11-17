"""Aggregate completed full-budget pre-pilot trajectories and stop observers."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.stopping import RULES


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = []
    observer_rows = []
    for path in sorted(raw_directory.glob("task_*_full_budget.csv")):
        rows = read_rows(path)
        final = rows[-1]
        budget = int(final["n_eval"])
        task = {
            "task_file": path.name,
            "system": final["system"],
            "dimension": int(final["dimension"]),
            "policy": final["policy"],
            "seed": int(final["seed"]),
            "budget": budget,
            "final_normalized_regret": float(final["normalized_regret"]),
            "final_success_le_0.05": int(final["success_le_0.05"]),
            "fit_warning_count": max(int(row["fit_warning_count"]) for row in rows),
        }
        tasks.append(task)
        by_eval = {int(row["n_eval"]): row for row in rows}
        for rule in RULES:
            value = final.get(f"{rule}_first_trigger", "")
            trigger = int(float(value)) if value else budget
            trigger_row = by_eval[trigger]
            natural_trigger = bool(value) and rule != "fixed_budget"
            success = int(trigger_row["success_le_0.05"])
            observer_rows.append({
                "task_file": path.name,
                "system": final["system"],
                "policy": final["policy"],
                "seed": int(final["seed"]),
                "rule": rule,
                "first_trigger_or_budget": trigger,
                "triggered_before_budget": int(natural_trigger and trigger < budget),
                "evaluations_saved": budget - trigger,
                "normalized_regret_at_stop": float(trigger_row["normalized_regret"]),
                "success_at_stop": success,
                "false_stop": int(natural_trigger and trigger < budget and not success),
                "non_trigger": int(not natural_trigger and rule != "fixed_budget"),
            })
    groups = []
    for policy in sorted({row["policy"] for row in tasks}):
        rows = [row for row in tasks if row["policy"] == policy]
        per_system = []
        for system in sorted({row["system"] for row in rows}):
            system_rows = [row for row in rows if row["system"] == system]
            per_system.append(np.mean([row["final_success_le_0.05"] for row in system_rows]))
        groups.append({
            "policy": policy,
            "trajectories": len(rows),
            "systems": len(per_system),
            "pooled_success_rate": float(np.mean([row["final_success_le_0.05"] for row in rows])),
            "system_success_q25": float(np.quantile(per_system, 0.25)),
            "system_success_median": float(np.quantile(per_system, 0.5)),
            "system_success_q75": float(np.quantile(per_system, 0.75)),
            "median_final_normalized_regret": float(np.median([
                row["final_normalized_regret"] for row in rows
            ])),
            "maximum_fit_warning_count": max(row["fit_warning_count"] for row in rows),
        })
    return tasks, observer_rows, groups


def make_plot(groups: list[dict], observers: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    labels = [row["policy"] for row in groups]
    medians = np.asarray([row["system_success_median"] for row in groups])
    q25 = np.asarray([row["system_success_q25"] for row in groups])
    q75 = np.asarray([row["system_success_q75"] for row in groups])
    pooled = np.asarray([row["pooled_success_rate"] for row in groups])
    x = np.arange(len(groups))
    axes[0].errorbar(
        x, medians, yerr=np.vstack((medians - q25, q75 - medians)),
        fmt="o", capsize=5, color="#1F2937", label="System median and IQR",
    )
    axes[0].scatter(x, pooled, marker="D", color="#D97706", label="Pooled")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("A. Final success by acquisition policy")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.35)

    rules = [rule for rule in RULES if rule != "fixed_budget"]
    saved = []
    false_rate = []
    for rule in rules:
        rows = [row for row in observers if row["rule"] == rule]
        saved.append(np.mean([row["evaluations_saved"] for row in rows]))
        false_rate.append(np.mean([row["false_stop"] for row in rows]))
    axes[1].scatter(saved, false_rate, s=80, color="#2563EB")
    for rule, sx, fy in zip(rules, saved, false_rate):
        axes[1].annotate(rule.replace("_", " "), (sx, fy), xytext=(5, 4),
                         textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Mean evaluations saved")
    axes[1].set_ylabel("False-stop rate")
    axes[1].set_title("B. Provisional stopping trade-off")
    axes[1].grid(alpha=0.35)
    fig.suptitle("Four-system full-budget pre-pilot")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-directory", type=Path,
        default=Path("results/phase3/full_budget_prepilot/raw"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/phase3/full_budget_prepilot"),
    )
    args = parser.parse_args()
    tasks, observers, groups = summarize(args.raw_directory)
    if not tasks:
        raise ValueError("no completed full-budget tasks found")
    write_rows(args.output / "task_summary.csv", tasks)
    write_rows(args.output / "stopping_observer_summary.csv", observers)
    write_rows(args.output / "policy_summary.csv", groups)
    make_plot(groups, observers, args.output / "prepilot_summary.png")
    print(f"summarized trajectories={len(tasks)} policies={len(groups)}")


if __name__ == "__main__":
    main()
