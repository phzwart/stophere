"""Screen official derivative-free optimizers on the common 2D systems.

This is an adapter-validation screen, not the final BO-versus-DFO matrix.
The existing NumPy BO prototype selects from finite catalogues, whereas these
optimizers operate in the continuous unit box. Do not pool the two protocols.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from experiments.phase3_characterization import (
    FIELD_SEED,
    SUCCESS_EPSILON,
    SYSTEMS,
    catalogue,
    write_csv,
)
from experiments.phase3_success_profiles import (
    make_budget_success_plot,
    make_final_success_plot,
    success_profile_rows,
)
from src.benchmarks import get_benchmark


ALGORITHMS = ("cma_es", "snobfit", "nomad", "bam", "uniform_random")
CHECKPOINTS = (5, 10, 20, 50, 100, 150)
PACKAGE_NAMES = {
    "cma_es": "cma",
    "snobfit": "SQSnobFit",
    "nomad": "PyNomadBBO",
    "bam": "bam-solver",
    "uniform_random": None,
}


def common_2d_systems():
    return tuple(
        system for system in SYSTEMS
        if system.dimension == 2 and system.benchmark != "mrf_ising_3x3"
    )


def package_version(algorithm: str) -> str:
    package = PACKAGE_NAMES[algorithm]
    if package is None:
        return f"numpy-{np.__version__}"
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


class EvaluationRecorder:
    def __init__(self, system, algorithm: str, seed: int, budget: int):
        self.system = system
        self.algorithm = algorithm
        self.seed = seed
        self.budget = budget
        self.objective = get_benchmark(
            system.benchmark, system.dimension, field_seed=FIELD_SEED
        )
        reference_values = np.asarray(
            self.objective(catalogue(system)), dtype=float
        )
        self.best_reference = float(reference_values.max())
        self.value_range = (
            float(reference_values.max() - reference_values.min()) or 1.0
        )
        self.best_value = -np.inf
        self.trace: list[dict] = []
        self.seen: set[tuple[float, ...]] = set()
        self.started = time.perf_counter()

    def minimize(self, x) -> float:
        if len(self.trace) >= self.budget:
            raise RuntimeError("adapter exceeded the declared evaluation budget")
        point = np.asarray(x, dtype=float).ravel()
        if point.size != self.system.dimension:
            raise ValueError(
                f"expected dimension {self.system.dimension}, got {point.size}"
            )
        within_bounds = bool(np.all((0.0 <= point) & (point <= 1.0)))
        if not within_bounds:
            raise ValueError(f"algorithm proposed point outside [0, 1]^d: {point}")
        value = float(np.asarray(
            self.objective(point.reshape(1, -1)), dtype=float
        ).ravel()[0])
        key = tuple(float(value) for value in np.round(point, 12))
        duplicate = int(key in self.seen)
        self.seen.add(key)
        self.best_value = max(self.best_value, value)
        regret = max(self.best_reference - self.best_value, 0.0)
        self.trace.append({
            "system": self.system.name,
            "config": self.algorithm,
            "algorithm": self.algorithm,
            "algorithm_version": package_version(self.algorithm),
            "seed": self.seed,
            "n_eval": len(self.trace) + 1,
            "point": json.dumps([float(v) for v in point]),
            "objective_value": value,
            "best_observed": self.best_value,
            "regret": regret,
            "normalized_regret": regret / self.value_range,
            "success_le_0.05": int(regret / self.value_range <= SUCCESS_EPSILON),
            "duplicate_evaluation": duplicate,
            "within_bounds": int(within_bounds),
            "elapsed_seconds": time.perf_counter() - self.started,
        })
        return -value


def starting_point(system, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3107]))
    return rng.uniform(0.0, 1.0, size=system.dimension)


def run_uniform_random(recorder: EvaluationRecorder, x0: np.ndarray) -> str:
    rng = np.random.default_rng(np.random.SeedSequence([recorder.seed, 4201]))
    recorder.minimize(x0)
    while len(recorder.trace) < recorder.budget:
        recorder.minimize(rng.uniform(0.0, 1.0, recorder.system.dimension))
    return "completed"


def run_cma_es(recorder: EvaluationRecorder, x0: np.ndarray) -> str:
    import cma

    population = 5
    if recorder.budget % population:
        raise ValueError("CMA-ES budget must be a multiple of population size 5")
    options = {
        "bounds": [
            [0.0] * recorder.system.dimension,
            [1.0] * recorder.system.dimension,
        ],
        "seed": recorder.seed + 1,
        "popsize": population,
        "verbose": -9,
        "verb_log": 0,
    }
    strategy = cma.CMAEvolutionStrategy(x0, 0.25, options)
    while len(recorder.trace) < recorder.budget:
        points = strategy.ask(population)
        values = [recorder.minimize(point) for point in points]
        strategy.tell(points, values)
    return "completed"


def run_snobfit(recorder: EvaluationRecorder, x0: np.ndarray) -> str:
    """Drive SQSnobFit directly so the final request cannot overshoot budget."""
    from SQSnobFit import snobfit

    np.random.seed(recorder.seed + 1)
    dimension = recorder.system.dimension
    bounds = np.column_stack((np.zeros(dimension), np.ones(dimension)))
    dx = (bounds[:, 1] - bounds[:, 0]) * 1e-5
    config = {"bounds": bounds, "nreq": 2 * dimension + 6, "p": 0.5}
    uncertainty = float(np.sqrt(np.spacing(1)))
    initial_value = recorder.minimize(x0)
    request, _, _ = snobfit(
        x0.reshape(1, dimension),
        np.asarray([[initial_value, uncertainty]], dtype=float),
        config,
        dx,
    )
    while len(recorder.trace) < recorder.budget:
        remaining = recorder.budget - len(recorder.trace)
        count = min(remaining, len(request))
        points = np.asarray(request[:count, :dimension], dtype=float)
        values = np.asarray([
            [recorder.minimize(point), uncertainty] for point in points
        ])
        if len(recorder.trace) >= recorder.budget:
            break
        if count != len(request):
            raise RuntimeError("partial SNOBFIT request before final budget")
        request, _, _ = snobfit(points, values, config)
        if not len(request):
            return "terminated_early"
    return "completed"


def run_nomad(recorder: EvaluationRecorder, x0: np.ndarray) -> str:
    import PyNomad

    def blackbox(point):
        values = [
            point.get_coord(index)
            for index in range(recorder.system.dimension)
        ]
        objective = recorder.minimize(values)
        point.setBBO(str(objective).encode("UTF-8"))
        return 1

    params = [
        "BB_OUTPUT_TYPE OBJ",
        f"MAX_BB_EVAL {recorder.budget}",
        f"SEED {recorder.seed + 1}",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
    ]
    result = PyNomad.optimize(
        blackbox,
        x0.tolist(),
        [0.0] * recorder.system.dimension,
        [1.0] * recorder.system.dimension,
        params,
    )
    if len(recorder.trace) != recorder.budget:
        return f"terminated_after_{len(recorder.trace)}"
    return str(result.get("stop_reason", "completed"))


def run_bam(recorder: EvaluationRecorder, x0: np.ndarray) -> str:
    import bam

    if recorder.system.dimension > 2:
        raise ValueError("BAM free mode supports at most two variables")
    result = bam.minimize(
        recorder.minimize,
        x0=x0,
        bounds=[(0.0, 1.0)] * recorder.system.dimension,
        options={
            "max_evals": recorder.budget,
            "print_to_screen": False,
            "max_no_gain": -1,
        },
    )
    if len(recorder.trace) != recorder.budget:
        return f"terminated_after_{len(recorder.trace)}: {result.message}"
    return str(result.message)


RUNNERS = {
    "cma_es": run_cma_es,
    "snobfit": run_snobfit,
    "nomad": run_nomad,
    "bam": run_bam,
    "uniform_random": run_uniform_random,
}


def run_algorithm(system, algorithm: str, seed: int, budget: int):
    recorder = EvaluationRecorder(system, algorithm, seed, budget)
    x0 = starting_point(system, seed)
    message = RUNNERS[algorithm](recorder, x0)
    return recorder.trace, message


def _run_task(task):
    system, algorithm, seed, budget = task
    started = time.perf_counter()
    try:
        trace, message = run_algorithm(system, algorithm, seed, budget)
        status = "completed"
        error = ""
    except Exception as exc:  # retain failures in the audit table
        trace = []
        message = ""
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    summary = {
        "system": system.name,
        "dimension": system.dimension,
        "algorithm": algorithm,
        "algorithm_version": package_version(algorithm),
        "seed": seed,
        "budget": budget,
        "evaluations_recorded": len(trace),
        "terminated_early": int(0 < len(trace) < budget),
        "status": status,
        "message": message,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
    }
    return trace, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3/external_algorithms"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    if args.budget < 5:
        raise ValueError("budget must be at least five evaluations")
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    systems = common_2d_systems()
    tasks = [
        (system, algorithm, seed, args.budget)
        for system in systems
        for algorithm in args.algorithms
        for seed in range(args.seeds)
    ]
    traces = []
    summaries = []
    if args.workers == 1:
        completed_runs = map(_run_task, tasks)
        for completed, (trace, summary) in enumerate(completed_runs, start=1):
            traces.extend(trace)
            summaries.append(summary)
            if completed % 10 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} runs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                trace, summary = future.result()
                traces.extend(trace)
                summaries.append(summary)
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)} runs", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "run_status.csv", summaries)
    if traces:
        write_csv(args.output / "trajectories.csv", traces)
    failures = [row for row in summaries if row["status"] == "failed"]
    if failures:
        print(f"incomplete_or_failed={len(failures)}; see run_status.csv")
        raise RuntimeError("external algorithm screen did not complete cleanly")

    checkpoints = tuple(
        value for value in CHECKPOINTS if value <= args.budget
    )
    if args.budget not in checkpoints:
        checkpoints = (*checkpoints, args.budget)
    profiles = success_profile_rows(
        traces,
        checkpoints=checkpoints,
        epsilon=SUCCESS_EPSILON,
        carry_forward=True,
    )
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
        configs=tuple(args.algorithms),
    )
    print(
        f"systems={len(systems)} algorithms={len(args.algorithms)} "
        f"seeds={args.seeds} runs={len(tasks)} budget={args.budget} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()

