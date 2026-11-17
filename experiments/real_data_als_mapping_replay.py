"""Offline mapping replay on the real ALS 57x57 measured catalogue.

This is a transparent fixed-hyperparameter smoke test for the published
variance-acquisition task. It uses the provisional q-peak proxy catalogue and
must not be presented as a reproduction of the fitted-peak paper result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_catalogue(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    physical = np.asarray([[float(row["x_mm"]), float(row["y_mm"])] for row in rows])
    minimum = physical.min(axis=0)
    span = np.maximum(physical.max(axis=0) - minimum, 1e-12)
    coordinates = (physical - minimum) / span
    values = np.asarray([float(row["q_peak_proxy"]) for row in rows])
    return coordinates, values


def matern32(x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    scaled = (x1[:, None, :] - x2[None, :, :]) / lengthscales
    distance = np.sqrt(np.maximum(np.sum(scaled * scaled, axis=2), 0.0))
    q = np.sqrt(3.0) * distance
    return (1.0 + q) * np.exp(-q)


def posterior(
    catalogue: np.ndarray,
    values: np.ndarray,
    selected: list[int],
    lengthscales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = catalogue[selected]
    train_y = values[selected]
    mean = float(train_y.mean())
    scale = max(float(train_y.std()), 1e-8)
    standardized = (train_y - mean) / scale
    kernel = matern32(train_x, train_x, lengthscales)
    kernel.flat[:: len(kernel) + 1] += 1e-6
    chol = np.linalg.cholesky(kernel)
    cross = matern32(train_x, catalogue, lengthscales)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, standardized))
    mu = mean + scale * (cross.T @ alpha)
    solved = np.linalg.solve(chol, cross)
    variance = scale * scale * np.maximum(1.0 - np.sum(solved * solved, axis=0), 0.0)
    return mu, variance


def run(
    catalogue: np.ndarray,
    values: np.ndarray,
    policy: str,
    seed: int,
    budget: int,
    lengthscales: np.ndarray,
) -> list[dict]:
    if policy not in {"max_variance", "random"}:
        raise ValueError(policy)
    design_rng = np.random.default_rng(np.random.SeedSequence([seed, 733]))
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    selected = design_rng.choice(len(catalogue), size=25, replace=False).tolist()
    used = np.zeros(len(catalogue), dtype=bool)
    used[selected] = True
    value_range = max(float(values.max() - values.min()), 1e-12)
    rows = []
    first_std_trigger = None
    while True:
        mu, variance = posterior(catalogue, values, selected, lengthscales)
        normalized_rmse = float(np.sqrt(np.mean((mu - values) ** 2)) / value_range)
        maximum_std_over_range = float(np.sqrt(variance.max()) / value_range)
        mean_variance_over_range2 = float(variance.mean() / (value_range * value_range))
        if first_std_trigger is None and maximum_std_over_range <= 0.05:
            first_std_trigger = len(selected)
        rows.append({
            "policy": policy,
            "seed": seed,
            "n_eval": len(selected),
            "normalized_map_rmse": normalized_rmse,
            "maximum_posterior_sd_over_range": maximum_std_over_range,
            "mean_posterior_variance_over_range2": mean_variance_over_range2,
            "provisional_std_stop_first_trigger": first_std_trigger or "",
            "selected_indices": json.dumps(selected),
        })
        if len(selected) == budget:
            break
        if policy == "random":
            index = int(policy_rng.choice(np.flatnonzero(~used)))
        else:
            score = variance.copy()
            score[used] = -np.inf
            index = int(np.argmax(score))
        used[index] = True
        selected.append(index)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for policy, color in (("max_variance", "#2563EB"), ("random", "#D97706")):
        policy_rows = [row for row in rows if row["policy"] == policy]
        seeds = sorted({int(row["seed"]) for row in policy_rows})
        matrices = []
        for seed in seeds:
            trace = sorted(
                [row for row in policy_rows if int(row["seed"]) == seed],
                key=lambda row: int(row["n_eval"]),
            )
            matrices.append(trace)
        x = np.asarray([int(row["n_eval"]) for row in matrices[0]])
        for ax, metric, label in (
            (axes[0], "normalized_map_rmse", "Normalized map RMSE"),
            (axes[1], "maximum_posterior_sd_over_range", "Maximum posterior SD / range"),
        ):
            values = np.asarray([[float(row[metric]) for row in trace] for trace in matrices])
            ax.plot(x, np.median(values, axis=0), color=color, linewidth=2.2,
                    label=policy.replace("_", " "))
            if len(seeds) > 1:
                ax.fill_between(x, np.quantile(values, 0.25, axis=0),
                                np.quantile(values, 0.75, axis=0), color=color, alpha=0.14)
            ax.set_xlabel("Measurements")
            ax.set_ylabel(label)
            ax.grid(alpha=0.35)
    axes[0].set_title("Truth-aware mapping error (evaluation only)")
    axes[1].set_title("Online model uncertainty diagnostic")
    axes[1].axhline(0.05, color="#DC2626", linestyle="--", linewidth=1.2,
                    label="Provisional stop threshold")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Real ALS grid: provisional offline mapping replay")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogue_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--budget", type=int, default=99)
    args = parser.parse_args()
    catalogue, values = load_catalogue(args.catalogue_csv)
    # Published final ALS length scales (0.7027 mm, 2.144 mm), normalized by
    # the 14 mm x/y motor ranges. These are fixed for this smoke test.
    lengthscales = np.asarray([0.7027 / 14.0, 2.144 / 14.0])
    all_rows = []
    for policy in ("max_variance", "random"):
        for seed in range(args.seeds):
            trace = run(catalogue, values, policy, seed, args.budget, lengthscales)
            all_rows.extend(trace)
            print(
                f"policy={policy} seed={seed} final_rmse={trace[-1]['normalized_map_rmse']:.6f}",
                flush=True,
            )
    write_rows(args.output / "mapping_trajectories.csv", all_rows)
    make_plot(all_rows, args.output / "mapping_replay.png")


if __name__ == "__main__":
    main()
