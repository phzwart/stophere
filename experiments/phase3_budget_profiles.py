"""Run an extended-budget success-profile screen.

This is deliberately a BO-policy screen, not the planned cross-algorithm
matrix. The latter requires separately validated adapters for BAM, SNOBFIT,
MCS, CMA-ES, and NOMAD on the continuous box-constrained systems.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.phase3_characterization import (
    FIELD_SEED,
    JITTER,
    N_INITIAL,
    N_SEEDS,
    SUCCESS_EPSILON,
    SYSTEMS,
    catalogue,
    choose_next,
    configuration_rows,
    kernel_matrix,
    write_csv,
)
from experiments.phase3_success_profiles import (
    make_budget_success_plot,
    make_final_success_plot,
    success_profile_rows,
)
from src.benchmarks import get_benchmark


DEFAULT_CONFIGS = (
    "nominal",
    "acquisition=ei",
    "acquisition=pi",
    "random_baseline",
)
DEFAULT_CHECKPOINTS = (5, 10, 20, 50, 100, 150)


@lru_cache(maxsize=64)
def _problem_data(system, kernel: str, lengthscale: float):
    """Cache finite-catalogue values and covariance within each worker."""
    points = catalogue(system)
    objective = get_benchmark(system.benchmark, system.dimension, field_seed=FIELD_SEED)
    latent = np.asarray(objective(points), dtype=float)
    covariance = kernel_matrix(points, points, kernel, lengthscale)
    return points, latent, covariance


def _run_task(task: tuple) -> list[dict]:
    """Run a noiseless trajectory with an incrementally updated Cholesky factor."""
    system, config, seed, budget = task
    label, kernel, ls_mult, acquisition, beta, noise_fraction = config
    if noise_fraction:
        raise ValueError("extended success profiles currently require noiseless configs")
    lengthscale = system.base_lengthscale * ls_mult
    points, latent, covariance = _problem_data(system, kernel, lengthscale)
    best_ref = float(latent.max())
    value_range = float(latent.max() - latent.min()) or 1.0
    design_rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 4201]))
    train_ids = [int(value) for value in design_rng.choice(
        len(points), N_INITIAL, replace=False
    )]
    used = np.zeros(len(points), dtype=bool)
    used[train_ids] = True
    initial_covariance = covariance[np.ix_(train_ids, train_ids)].copy()
    initial_covariance.flat[::len(initial_covariance) + 1] += JITTER
    chol = np.linalg.cholesky(initial_covariance)
    trace = []

    while True:
        train_y = latent[train_ids]
        best_latent = float(train_y.max())
        regret = max(best_ref - best_latent, 0.0)
        trace.append({
            "system": system.name, "config": label, "kernel": kernel,
            "lengthscale_multiplier": ls_mult, "acquisition": acquisition,
            "beta": beta, "observation_noise_fraction": noise_fraction,
            "seed": seed, "n_eval": len(train_ids),
            "best_observed": best_latent,
            "best_latent_at_incumbent": best_latent,
            "regret": regret, "normalized_regret": regret / value_range,
            "posterior_ucb_gap_over_y_sd": float("nan"),
            "max_marginal_p_improve": float("nan"),
        })
        if len(train_ids) >= budget:
            break

        if acquisition == "random":
            index = int(policy_rng.choice(np.flatnonzero(~used)))
        else:
            y_mean = float(train_y.mean())
            y_scale = max(float(train_y.std()), 1e-8)
            normalized_y = (train_y - y_mean) / y_scale
            cross = covariance[train_ids, :]
            alpha = np.linalg.solve(
                chol.T, np.linalg.solve(chol, normalized_y)
            )
            mu = y_mean + y_scale * (cross.T @ alpha)
            projected = np.linalg.solve(chol, cross)
            variance = np.maximum(
                1.0 - np.sum(projected * projected, axis=0), 0.0
            )
            sigma = y_scale * np.sqrt(variance)
            index = choose_next(
                mu, sigma, best_latent, acquisition, beta, used
            )

        old_ids = train_ids.copy()
        train_ids.append(index)
        used[index] = True
        if acquisition != "random":
            cross_new = covariance[old_ids, index]
            projected_new = np.linalg.solve(chol, cross_new)
            schur = float(
                covariance[index, index] + JITTER
                - projected_new @ projected_new
            )
            if schur <= JITTER * 0.1:
                rebuilt = covariance[np.ix_(train_ids, train_ids)].copy()
                rebuilt.flat[::len(rebuilt) + 1] += JITTER
                chol = np.linalg.cholesky(rebuilt)
            else:
                expanded = np.zeros((len(train_ids), len(train_ids)), dtype=float)
                expanded[:-1, :-1] = chol
                expanded[-1, :-1] = projected_new
                expanded[-1, -1] = np.sqrt(schur)
                chol = expanded
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=max(DEFAULT_CHECKPOINTS))
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Configuration labels from phase3_characterization.configuration_rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3/budget_profiles"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget < 5:
        raise ValueError("budget must be at least the five-point initial design")
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    available = {config[0]: config for config in configuration_rows(SYSTEMS[0])}
    unknown = sorted(set(args.configs) - set(available))
    if unknown:
        raise ValueError(f"unknown configuration labels: {unknown}")
    configs = [available[label] for label in args.configs]
    tasks = [
        (system, config, seed, args.budget)
        for system in SYSTEMS
        for config in configs
        for seed in range(args.seeds)
    ]
    traces = []
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            traces.extend(_run_task(task))
            if completed % 10 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} trajectories", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                traces.extend(future.result())
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)} trajectories", flush=True)

    checkpoints = tuple(dict.fromkeys(
        value for value in (*DEFAULT_CHECKPOINTS, args.budget)
        if value <= args.budget
    ))
    profiles = success_profile_rows(
        traces, checkpoints=checkpoints, epsilon=SUCCESS_EPSILON
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "trajectories.csv", traces)
    write_csv(args.output / "success_profiles.csv", profiles)
    make_final_success_plot(
        profiles,
        args.output / "success_quantiles_final_budget.png",
        final_eval=args.budget,
    )
    make_budget_success_plot(
        profiles,
        args.output / "success_by_budget.png",
        checkpoints=checkpoints,
        configs=tuple(args.configs),
    )
    print(
        f"systems={len(SYSTEMS)} configs={len(configs)} seeds={args.seeds} "
        f"trajectories={len(tasks)} budget={args.budget} output={args.output}"
    )


if __name__ == "__main__":
    main()
