"""Provisional optimization replay on the real ALS dense grid.

This tests whether the formal BoTorch/observer pipeline can consume measured
data. It deliberately does not claim to reproduce the paper's mapping task:
the current objective maximizes a transparent q-peak proxy, while the paper
used variance acquisition to map a fitted feature over the sample.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.benchmarks import Benchmark
from src.botorch_runner import Protocol, run_trajectory, write_rows
from src.policies import POLICIES
from src.stopping import StoppingProtocol


def load_catalogue(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coordinates = np.asarray(
        [[float(row["x_mm"]), float(row["y_mm"])] for row in rows], dtype=float
    )
    minimum = coordinates.min(axis=0)
    span = np.maximum(coordinates.max(axis=0) - minimum, 1e-12)
    normalized = (coordinates - minimum) / span
    values = np.asarray([float(row["q_peak_proxy"]) for row in rows], dtype=float)
    return normalized, values


def measured_benchmark(catalogue: np.ndarray, values: np.ndarray) -> Benchmark:
    lookup = {tuple(np.round(point, 12)): value for point, value in zip(catalogue, values)}

    def evaluate(points: np.ndarray) -> np.ndarray:
        if points.shape == catalogue.shape and np.allclose(points, catalogue):
            return values.copy()
        output = []
        for point in points:
            key = tuple(np.round(point, 12))
            if key not in lookup:
                raise ValueError("offline replay requested a point outside the measured grid")
            output.append(lookup[key])
        return np.asarray(output, dtype=float)

    best = int(np.argmax(values))
    return Benchmark(
        name="als_nacl_q_peak_proxy",
        dimension=2,
        evaluate=evaluate,
        description="Measured ALS dense grid with a provisional q-at-maximum-intensity response.",
        reference_x=catalogue[best],
        reference_value=float(values[best]),
    )


def make_plot(all_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for policy in POLICIES:
        rows = [row for row in all_rows if row["policy"] == policy]
        if not rows:
            continue
        groups = sorted({int(row["seed"]) for row in rows})
        traces = []
        for seed in groups:
            trace = sorted(
                [row for row in rows if int(row["seed"]) == seed],
                key=lambda row: int(row["n_eval"]),
            )
            traces.append(np.asarray([row["normalized_regret"] for row in trace], dtype=float))
        matrix = np.vstack(traces)
        x = np.asarray(sorted({int(row["n_eval"]) for row in rows}))
        median = np.median(matrix, axis=0)
        ax.plot(x, median, linewidth=2, label=policy)
        if len(groups) > 1:
            ax.fill_between(x, np.quantile(matrix, 0.25, axis=0),
                            np.quantile(matrix, 0.75, axis=0), alpha=0.12)
    ax.axhline(0.05, color="#DC2626", linestyle="--", linewidth=1.3,
               label="Provisional success threshold")
    ax.set_xlabel("Measured grid points queried")
    ax.set_ylabel("Normalized simple regret")
    ax.set_yscale("log")
    ax.set_title("Provisional real-data optimization replay")
    ax.grid(color="#D1D5DB", alpha=0.55)
    ax.legend(frameon=False)
    fig.text(
        0.01, 0.01,
        "Pipeline validation only: this maximizes a q-peak proxy and is not the paper's variance-based mapping task.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogue_csv", type=Path)
    parser.add_argument("--phase3", default="configs/phase5_als_replay_provisional.yaml")
    parser.add_argument("--phase4", default="configs/phase4_stopping_observers.yaml")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/phase5/als_offline_replay_private"),
    )
    args = parser.parse_args()
    catalogue, values = load_catalogue(args.catalogue_csv)
    protocol = Protocol.load(args.phase3)
    if len(catalogue) != protocol.catalogue_size:
        raise ValueError("catalogue size does not match the replay protocol")
    stopping = StoppingProtocol.load(args.phase4)
    benchmark = measured_benchmark(catalogue, values)
    candidates = torch.as_tensor(catalogue, dtype=torch.double)
    all_rows = []
    for policy in POLICIES:
        for seed in range(args.seeds):
            rows = run_trajectory(
                benchmark=benchmark,
                policy=policy,
                optimizer_seed=seed,
                protocol=protocol,
                catalogue=candidates,
                stopping_protocol=stopping,
            )
            all_rows.extend(rows)
            write_rows(args.output / "raw" / f"{policy}_seed_{seed}.csv", rows)
            print(
                f"policy={policy} seed={seed} final_regret={rows[-1]['normalized_regret']:.6f}",
                flush=True,
            )
    write_rows(args.output / "trajectories.csv", all_rows)
    make_plot(all_rows, args.output / "regret_by_budget.png")


if __name__ == "__main__":
    main()
