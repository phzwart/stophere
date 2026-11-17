"""Audit the apparent plateau of the extended-budget random baseline."""
from __future__ import annotations

import csv
from math import exp, lgamma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase3_characterization import FIELD_SEED, SYSTEMS, catalogue
from src.benchmarks import get_benchmark


CHECKPOINTS = (5, 10, 20, 50, 100, 150)
EPSILON = 0.05


def success_probability_without_replacement(
    population: int, successes: int, draws: int
) -> float:
    """Probability of at least one success in a uniform sample without replacement."""
    if not 0 <= successes <= population:
        raise ValueError("successes must be between zero and population")
    if not 0 <= draws <= population:
        raise ValueError("draws must be between zero and population")
    if draws == 0 or successes == 0:
        return 0.0
    if draws > population - successes:
        return 1.0
    log_no_success = (
        lgamma(population - successes + 1)
        - lgamma(population - successes - draws + 1)
        - lgamma(population + 1)
        + lgamma(population - draws + 1)
    )
    return float(1.0 - exp(log_no_success))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_audit(trajectory_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    random_rows = [row for row in trajectory_rows if row["config"] == "random_baseline"]
    system_rows = []
    for system in SYSTEMS:
        if system.benchmark == "mrf_ising_3x3":
            continue
        points = catalogue(system)
        objective = get_benchmark(
            system.benchmark, system.dimension, field_seed=FIELD_SEED
        )
        values = np.asarray(objective(points), dtype=float)
        value_range = max(float(values.max() - values.min()), 1e-12)
        success_mask = (values.max() - values) / value_range <= EPSILON
        population = len(points)
        successes = int(success_mask.sum())
        for checkpoint in CHECKPOINTS:
            if checkpoint > population:
                continue
            empirical = [
                float(row["normalized_regret"]) <= EPSILON
                for row in random_rows
                if row["system"] == system.name
                and int(row["n_eval"]) == checkpoint
            ]
            system_rows.append({
                "system": system.name,
                "dimension": system.dimension,
                "catalogue_size": population,
                "successful_catalogue_points": successes,
                "successful_catalogue_fraction": successes / population,
                "n_eval": checkpoint,
                "exact_success_probability": success_probability_without_replacement(
                    population, successes, checkpoint
                ),
                "empirical_seed_count": len(empirical),
                "empirical_success_rate": (
                    float(np.mean(empirical)) if empirical else float("nan")
                ),
            })
    pooled_rows = []
    for checkpoint in CHECKPOINTS:
        rows = [row for row in system_rows if int(row["n_eval"]) == checkpoint]
        if not rows:
            continue
        empirical_values = [
            row["empirical_success_rate"] for row in rows
            if not np.isnan(row["empirical_success_rate"])
        ]
        pooled_rows.append({
            "n_eval": checkpoint,
            "systems": len(rows),
            "mean_exact_success_probability": float(np.mean([
                row["exact_success_probability"] for row in rows
            ])),
            "mean_empirical_system_success": (
                float(np.mean(empirical_values)) if empirical_values else float("nan")
            ),
        })
    return system_rows, pooled_rows


def make_plot(system_rows: list[dict], pooled_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for system in sorted({row["system"] for row in system_rows}):
        rows = sorted(
            [row for row in system_rows if row["system"] == system],
            key=lambda row: row["n_eval"],
        )
        ax.plot(
            [row["n_eval"] for row in rows],
            [row["exact_success_probability"] for row in rows],
            color="#9CA3AF", alpha=0.45, linewidth=1,
        )
    x = [row["n_eval"] for row in pooled_rows]
    ax.plot(
        x, [row["mean_exact_success_probability"] for row in pooled_rows],
        marker="o", linewidth=2.5, color="#2563EB", label="Exact random expectation",
    )
    ax.plot(
        x, [row["mean_empirical_system_success"] for row in pooled_rows],
        marker="D", linestyle="--", linewidth=2, color="#D97706",
        label="Observed mean (5 seeds per system)",
    )
    ax.set_xticks(CHECKPOINTS)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Success probability")
    ax.set_title("Random search: exact expectation versus observed runs")
    ax.grid(color="#D1D5DB", alpha=0.65)
    ax.legend(frameon=False, loc="lower right")
    fig.text(
        0.01, 0.01,
        "Grey curves are individual systems. Random search samples without replacement; "
        "a plateau is expected when a system has very few successful catalogue points.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source = Path("results/phase3/budget_profiles/trajectories.csv")
    out = Path("results/phase3/random_baseline_audit")
    system_rows, pooled_rows = build_audit(read_rows(source))
    write_rows(out / "system_success_probabilities.csv", system_rows)
    write_rows(out / "pooled_success_by_budget.csv", pooled_rows)
    make_plot(system_rows, pooled_rows, out / "random_baseline_audit.png")
    print(f"wrote random-search audit for {len(set(r['system'] for r in system_rows))} systems")


if __name__ == "__main__":
    main()
