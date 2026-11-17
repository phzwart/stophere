"""Plot formal pre-pilot measurement paths over the underlying 2D objectives."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, make_catalogue


def read_final_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    return rows[-1]


def first_trigger(row: dict) -> int | None:
    values = []
    for key, value in row.items():
        if key.endswith("_first_trigger") and key != "fixed_budget_first_trigger" and value:
            values.append(int(float(value)))
    return min(values) if values else None


def plot_trajectory(ax, row: dict, protocol: Protocol, field_seed: int) -> None:
    system = row["system"]
    dimension = int(row["dimension"])
    if dimension != 2:
        raise ValueError("trajectory overlays currently support 2D systems")
    benchmark = get_benchmark(system, dimension, field_seed=field_seed)
    axis = np.linspace(0.0, 1.0, 240)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    zz = np.asarray(benchmark(grid), dtype=float).reshape(xx.shape)
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="viridis")

    catalogue = make_catalogue(dimension, protocol).detach().cpu().numpy()
    selected = np.asarray(json.loads(row["selected_indices"]), dtype=int)
    locations = catalogue[selected]
    initial_size = protocol.initial_size(dimension)
    ax.plot(
        locations[initial_size - 1 :, 0], locations[initial_size - 1 :, 1],
        color="white", alpha=0.55, linewidth=0.8, zorder=2,
    )
    ax.scatter(
        locations[:initial_size, 0], locations[:initial_size, 1],
        s=30, facecolors="none", edgecolors="white", linewidths=1.1,
        label=f"Initial design ({initial_size})", zorder=4,
    )
    sequence = np.arange(initial_size + 1, len(selected) + 1)
    later = ax.scatter(
        locations[initial_size:, 0], locations[initial_size:, 1],
        c=sequence, cmap="plasma", s=14, alpha=0.85, zorder=3,
        label="Later evaluations",
    )
    values = np.asarray(benchmark(catalogue), dtype=float)
    optimum = catalogue[int(np.argmax(values))]
    incumbent = catalogue[int(row["incumbent_index"])]
    ax.scatter(*optimum, marker="*", s=150, color="#EF4444", edgecolor="white",
               linewidth=0.8, label="Catalogue optimum", zorder=6)
    ax.scatter(*incumbent, marker="D", s=60, color="#FBBF24", edgecolor="black",
               linewidth=0.7, label="Final recommendation", zorder=6)
    trigger = first_trigger(row)
    if trigger is not None and trigger <= len(locations):
        point = locations[trigger - 1]
        ax.scatter(*point, s=125, facecolors="none", edgecolors="#F97316",
                   linewidths=2.0, label=f"First stop signal ({trigger})", zorder=7)
    ax.set_title(
        f"{system.replace('_', ' ')}\n{row['policy']}, final regret={float(row['normalized_regret']):.3f}"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    return contour, later


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-directory", type=Path,
        default=Path("results/phase3/full_budget_prepilot/raw"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/phase3/full_budget_prepilot/trajectory_overlays_2d.png"),
    )
    parser.add_argument("--phase3", default="configs/phase3_fair_protocol.yaml")
    parser.add_argument("--field-seed", type=int, default=2026)
    parser.add_argument("--policy", default="mes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.raw_directory.glob("task_*_full_budget.csv")):
        row = read_final_row(path)
        if (
            int(row["dimension"]) == 2
            and row["policy"] == args.policy
            and int(row["seed"]) == args.seed
        ):
            rows.append(row)
    if not rows:
        raise ValueError("no completed 2D full-budget trajectories were found")
    protocol = Protocol.load(args.phase3)
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5.8), squeeze=False)
    handles = labels = None
    for ax, row in zip(axes[0], rows):
        plot_trajectory(ax, row, protocol, args.field_seed)
        handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Formal pre-pilot measurement locations and paths", fontsize=15)
    if handles:
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=3,
                   frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output} for {len(rows)} trajectories")


if __name__ == "__main__":
    main()
