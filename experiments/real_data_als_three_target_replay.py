"""Run the formal BO policies on three targets from the real ALS grid.

Every target is transformed to a unit-range maximization objective before GP
fitting. This keeps acquisition and provisional stopping thresholds comparable
while retaining physical-unit values in the saved output.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.benchmarks import Benchmark
from src.botorch_runner import Protocol, run_trajectory, write_rows
from src.policies import POLICIES, REFERENCE_POLICIES
from src.stopping import RULES, StoppingProtocol


POLICY_COLORS = {
    "log_ei": "#2563EB",
    "log_pi": "#D97706",
    "ucb": "#059669",
    "thompson_sampling": "#DC2626",
    "mes": "#7C3AED",
    "random": "#6B7280",
}
REPLAY_POLICIES = POLICIES + REFERENCE_POLICIES
TARGET_LABELS = {
    "integrated_intensity": "Integrated intensity (maximize)",
    "maximum_peak_height": "Maximum peak height (maximize)",
    "minimum_fwhm": "FWHM (minimize)",
}
RULE_LABELS = {
    "incumbent_stagnation": "Stagnation",
    "negligible_expected_improvement": "Negligible EI",
    "global_confidence_gap": "Global gap",
    "posterior_acceptable_regret": "Posterior regret",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_catalogue(path: Path) -> tuple[list[dict], np.ndarray]:
    rows = read_rows(path)
    coordinates = np.asarray(
        [[float(row["x_mm"]), float(row["y_mm"])] for row in rows], dtype=float
    )
    minimum = coordinates.min(axis=0)
    span = np.maximum(coordinates.max(axis=0) - minimum, 1e-12)
    return rows, (coordinates - minimum) / span


def normalize_objective(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("objective values must be finite")
    value_range = float(values.max() - values.min())
    if value_range <= 0:
        raise ValueError("objective has zero range")
    if direction == "maximize":
        return (values - values.min()) / value_range
    if direction == "minimize":
        return (values.max() - values) / value_range
    raise ValueError(f"unsupported direction {direction!r}")


def measured_benchmark(
    target_name: str,
    catalogue: np.ndarray,
    normalized_values: np.ndarray,
) -> Benchmark:
    lookup = {
        tuple(np.round(point, 12)): value
        for point, value in zip(catalogue, normalized_values)
    }

    def evaluate(points: np.ndarray) -> np.ndarray:
        if points.shape == catalogue.shape and np.allclose(points, catalogue):
            return normalized_values.copy()
        output = []
        for point in points:
            key = tuple(np.round(point, 12))
            if key not in lookup:
                raise ValueError("offline replay requested an unmeasured grid point")
            output.append(lookup[key])
        return np.asarray(output, dtype=float)

    best = int(np.argmax(normalized_values))
    return Benchmark(
        name=f"als_nacl_{target_name}",
        dimension=2,
        evaluate=evaluate,
        description=f"Measured ALS grid, normalized target {target_name}.",
        reference_x=catalogue[best],
        reference_value=float(normalized_values[best]),
    )


def physical_regret(value: float, optimum: float, direction: str) -> float:
    if direction == "maximize":
        return max(optimum - value, 0.0)
    return max(value - optimum, 0.0)


def run_one_task(
    catalogue_csv: str,
    target_name: str,
    source_column: str,
    direction: str,
    policy: str,
    seed: int,
    phase3_path: str,
    phase4_path: str,
    output_path: str,
    budget_override: int | None,
) -> dict:
    rows, catalogue = read_catalogue(Path(catalogue_csv))
    physical_values = np.asarray([float(row[source_column]) for row in rows])
    normalized_values = normalize_objective(physical_values, direction)
    benchmark = measured_benchmark(target_name, catalogue, normalized_values)
    protocol = Protocol.load(phase3_path)
    stopping = StoppingProtocol.load(phase4_path)
    candidates = torch.as_tensor(catalogue, dtype=torch.double)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trajectory = run_trajectory(
            benchmark=benchmark,
            policy=policy,
            optimizer_seed=seed,
            protocol=protocol,
            catalogue=candidates,
            budget_override=budget_override,
            stopping_protocol=stopping,
        )
    runtime_warnings = list(dict.fromkeys(
        str(item.message) for item in caught
        if not issubclass(item.category, FutureWarning)
    ))
    optimum_index = int(np.argmax(normalized_values))
    optimum_physical = float(physical_values[optimum_index])
    for row in trajectory:
        incumbent_index = int(row["incumbent_index"])
        incumbent_physical = float(physical_values[incumbent_index])
        row.update({
            "target_name": target_name,
            "target_source_column": source_column,
            "target_direction": direction,
            "physical_incumbent_value": incumbent_physical,
            "physical_reference_value": optimum_physical,
            "physical_simple_regret": physical_regret(
                incumbent_physical, optimum_physical, direction
            ),
            "reference_x_mm": float(rows[optimum_index]["x_mm"]),
            "reference_y_mm": float(rows[optimum_index]["y_mm"]),
            "incumbent_x_mm": float(rows[incumbent_index]["x_mm"]),
            "incumbent_y_mm": float(rows[incumbent_index]["y_mm"]),
            "runtime_warning_count": len(runtime_warnings),
            "runtime_warnings": " | ".join(runtime_warnings),
        })
    write_rows(output_path, trajectory)
    return {
        "target": target_name,
        "policy": policy,
        "seed": seed,
        "final_regret": float(trajectory[-1]["normalized_regret"]),
        "success": int(trajectory[-1]["success_le_0.05"]),
        "fit_warnings": max(int(row["fit_warning_count"]) for row in trajectory),
        "runtime_warnings": len(runtime_warnings),
    }


def _raw_filename(target: str, policy: str, seed: int) -> str:
    return f"{target}__{policy}__seed_{seed}.csv"


def _trajectory_complete(path: Path, expected_budget: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_rows(path)
        return (
            bool(rows)
            and int(rows[-1]["n_eval"]) == expected_budget
            and "runtime_warning_count" in rows[-1]
        )
    except (OSError, KeyError, ValueError):
        return False


def load_complete_trajectories(raw_directory: Path) -> list[list[dict]]:
    trajectories = []
    for path in sorted(raw_directory.glob("*.csv")):
        rows = read_rows(path)
        if rows:
            trajectories.append(rows)
    return trajectories


def summarize_tasks(trajectories: list[list[dict]]) -> list[dict]:
    output = []
    for rows in trajectories:
        final = rows[-1]
        first_success = next(
            (int(row["n_eval"]) for row in rows if int(row["success_le_0.05"])),
            None,
        )
        output.append({
            "target_name": final["target_name"],
            "policy": final["policy"],
            "seed": int(final["seed"]),
            "budget": int(final["n_eval"]),
            "final_normalized_regret": float(final["normalized_regret"]),
            "final_success_le_0.05": int(final["success_le_0.05"]),
            "first_success_evaluation": "" if first_success is None else first_success,
            "final_physical_value": float(final["physical_incumbent_value"]),
            "physical_reference_value": float(final["physical_reference_value"]),
            "physical_simple_regret": float(final["physical_simple_regret"]),
            "recommendation_x_mm": float(final["incumbent_x_mm"]),
            "recommendation_y_mm": float(final["incumbent_y_mm"]),
            "reference_x_mm": float(final["reference_x_mm"]),
            "reference_y_mm": float(final["reference_y_mm"]),
            "fit_warning_count": max(int(row["fit_warning_count"]) for row in rows),
            "runtime_warning_count": max(int(row["runtime_warning_count"]) for row in rows),
            "runtime_warnings": final["runtime_warnings"],
        })
    return output


def summarize_policies(tasks: list[dict], target_order: list[str]) -> list[dict]:
    output = []
    for target in target_order:
        for policy in REPLAY_POLICIES:
            rows = [
                row for row in tasks
                if row["target_name"] == target and row["policy"] == policy
            ]
            if not rows:
                continue
            regrets = np.asarray([row["final_normalized_regret"] for row in rows])
            successes = np.asarray([row["final_success_le_0.05"] for row in rows])
            times = np.asarray([
                int(row["first_success_evaluation"]) if row["first_success_evaluation"] != ""
                else int(row["budget"])
                for row in rows
            ])
            output.append({
                "target_name": target,
                "policy": policy,
                "seeds": len(rows),
                "success_rate": float(successes.mean()),
                "median_final_normalized_regret": float(np.median(regrets)),
                "q25_final_normalized_regret": float(np.quantile(regrets, 0.25)),
                "q75_final_normalized_regret": float(np.quantile(regrets, 0.75)),
                "mean_first_success_or_budget": float(times.mean()),
                "median_physical_regret": float(np.median([
                    row["physical_simple_regret"] for row in rows
                ])),
                "maximum_fit_warning_count": max(row["fit_warning_count"] for row in rows),
                "maximum_runtime_warning_count": max(row["runtime_warning_count"] for row in rows),
            })
    return output


def summarize_stopping(trajectories: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    detailed = []
    for rows in trajectories:
        final = rows[-1]
        budget = int(final["n_eval"])
        by_eval = {int(row["n_eval"]): row for row in rows}
        for rule in RULES:
            raw_trigger = final.get(f"{rule}_first_trigger", "")
            trigger = int(float(raw_trigger)) if raw_trigger else budget
            trigger_row = by_eval[trigger]
            natural_trigger = bool(raw_trigger) and rule != "fixed_budget"
            early = natural_trigger and trigger < budget
            success = int(trigger_row["success_le_0.05"])
            detailed.append({
                "target_name": final["target_name"],
                "policy": final["policy"],
                "seed": int(final["seed"]),
                "rule": rule,
                "first_trigger_or_budget": trigger,
                "triggered_before_budget": int(early),
                "evaluations_saved": budget - trigger,
                "normalized_regret_at_stop": float(trigger_row["normalized_regret"]),
                "success_at_stop": success,
                "false_stop": int(early and not success),
                "non_trigger": int(not natural_trigger and rule != "fixed_budget"),
            })
    aggregate = []
    targets = sorted({row["target_name"] for row in detailed})
    for target in targets + ["all_targets"]:
        for rule in RULES:
            rows = [
                row for row in detailed
                if row["rule"] == rule
                and (target == "all_targets" or row["target_name"] == target)
            ]
            if not rows:
                continue
            triggers = sum(row["triggered_before_budget"] for row in rows)
            false_stops = sum(row["false_stop"] for row in rows)
            aggregate.append({
                "target_name": target,
                "rule": rule,
                "trajectories": len(rows),
                "early_trigger_rate": triggers / len(rows),
                "false_stop_rate_all": false_stops / len(rows),
                "false_stop_rate_given_trigger": (
                    false_stops / triggers if triggers else 0.0
                ),
                "non_trigger_rate": sum(row["non_trigger"] for row in rows) / len(rows),
                "mean_evaluations_saved": float(np.mean([
                    row["evaluations_saved"] for row in rows
                ])),
                "success_rate_at_stop": float(np.mean([
                    row["success_at_stop"] for row in rows
                ])),
            })
    return detailed, aggregate


def summarize_success_by_budget(
    trajectories: list[list[dict]], checkpoints: list[int]
) -> list[dict]:
    output = []
    targets = sorted({rows[-1]["target_name"] for rows in trajectories})
    for target in targets:
        for policy in REPLAY_POLICIES:
            selected = [
                rows for rows in trajectories
                if rows[-1]["target_name"] == target and rows[-1]["policy"] == policy
            ]
            for checkpoint in checkpoints:
                rows_at_checkpoint = []
                for rows in selected:
                    eligible = [row for row in rows if int(row["n_eval"]) <= checkpoint]
                    rows_at_checkpoint.append(max(eligible, key=lambda row: int(row["n_eval"])))
                output.append({
                    "target_name": target,
                    "policy": policy,
                    "n_eval": checkpoint,
                    "seeds": len(rows_at_checkpoint),
                    "success_rate": float(np.mean([
                        int(row["success_le_0.05"]) for row in rows_at_checkpoint
                    ])),
                    "median_normalized_regret": float(np.median([
                        float(row["normalized_regret"]) for row in rows_at_checkpoint
                    ])),
                })
    return output


def make_regret_plot(
    trajectories: list[list[dict]], target_order: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(target_order), figsize=(18, 5.5), sharey=True)
    for ax, target in zip(axes, target_order):
        for policy in REPLAY_POLICIES:
            selected = [
                rows for rows in trajectories
                if rows[-1]["target_name"] == target and rows[-1]["policy"] == policy
            ]
            if not selected:
                continue
            x = np.asarray([int(row["n_eval"]) for row in selected[0]])
            matrix = np.asarray([
                [max(float(row["normalized_regret"]), 1e-4) for row in rows]
                for rows in selected
            ])
            median = np.median(matrix, axis=0)
            q25 = np.quantile(matrix, 0.25, axis=0)
            q75 = np.quantile(matrix, 0.75, axis=0)
            ax.plot(x, median, color=POLICY_COLORS[policy], linewidth=2, label=policy)
            ax.fill_between(x, q25, q75, color=POLICY_COLORS[policy], alpha=0.12)
        ax.axhline(0.05, color="#111827", linestyle="--", linewidth=1.2)
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Measured grid points")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Normalized simple regret")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Real ALS offline optimization: median and IQR across paired seeds")
    fig.text(0.01, 0.01, "Zero regret is displayed at 1e-4 for the log scale; dashed line is the provisional 0.05 success threshold.", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_success_heatmap(
    policy_summary: list[dict], target_order: list[str], path: Path
) -> None:
    matrix = np.zeros((len(target_order), len(REPLAY_POLICIES)))
    for row in policy_summary:
        matrix[target_order.index(row["target_name"]), REPLAY_POLICIES.index(row["policy"])] = row["success_rate"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(REPLAY_POLICIES)), REPLAY_POLICIES, rotation=25, ha="right")
    ax.set_yticks(range(len(target_order)), [TARGET_LABELS[target] for target in target_order])
    ax.set_title("Final success rate at the fixed measurement budget")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] >= 0.6 else "#111827"
            ax.text(column, row, f"{matrix[row, column]:.0%}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="Success rate across five seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_success_budget_plot(
    rows: list[dict], target_order: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(target_order), figsize=(18, 5.2), sharey=True)
    for ax, target in zip(axes, target_order):
        for policy in REPLAY_POLICIES:
            series = sorted(
                [row for row in rows if row["target_name"] == target and row["policy"] == policy],
                key=lambda row: row["n_eval"],
            )
            ax.plot(
                [row["n_eval"] for row in series],
                [row["success_rate"] for row in series],
                marker="o", linewidth=2, color=POLICY_COLORS[policy], label=policy,
            )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Measured grid points")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Success rate across five seeds")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Success versus measurement budget")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_stopping_plot(
    aggregate: list[dict], target_order: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(target_order), figsize=(18, 5.2), sharey=True)
    rules = [rule for rule in RULES if rule != "fixed_budget"]
    colors = ("#2563EB", "#D97706", "#059669", "#7C3AED")
    annotation_offsets = {
        "incumbent_stagnation": (5, 5),
        "negligible_expected_improvement": (5, 29),
        "global_confidence_gap": (5, 5),
        "posterior_acceptable_regret": (5, 17),
    }
    for ax, target in zip(axes, target_order):
        for rule, color in zip(rules, colors):
            row = next(item for item in aggregate if item["target_name"] == target and item["rule"] == rule)
            x = row["mean_evaluations_saved"]
            y = row["false_stop_rate_all"]
            ax.scatter(x, y, s=70, color=color)
            ax.annotate(
                RULE_LABELS[rule], (x, y),
                xytext=annotation_offsets[rule], textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Mean evaluations saved")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("False-stop rate across trajectories")
    fig.suptitle("Provisional stopping trade-off on real ALS targets")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_recommendation_plot(
    catalogue_rows: list[dict], tasks: list[dict], config: dict,
    target_order: list[str], path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(target_order), figsize=(18, 5.8))
    x = np.asarray([float(row["x_mm"]) for row in catalogue_rows])
    y = np.asarray([float(row["y_mm"]) for row in catalogue_rows])
    for ax, target in zip(axes, target_order):
        spec = config["objectives"][target]
        values = np.asarray([float(row[spec["source_column"]]) for row in catalogue_rows])
        cmap = "viridis" if spec["direction"] == "maximize" else "viridis_r"
        image = ax.scatter(x, y, c=values, s=17, marker="s", cmap=cmap, alpha=0.8)
        target_tasks = [row for row in tasks if row["target_name"] == target]
        for policy in REPLAY_POLICIES:
            selected = [row for row in target_tasks if row["policy"] == policy]
            ax.scatter(
                [row["recommendation_x_mm"] for row in selected],
                [row["recommendation_y_mm"] for row in selected],
                s=42, marker="o", facecolors="none", edgecolors=POLICY_COLORS[policy],
                linewidths=1.5, label=policy,
            )
        reference = target_tasks[0]
        ax.scatter([reference["reference_x_mm"]], [reference["reference_y_mm"]],
                   marker="*", s=180, color="#111827", edgecolors="white", linewidths=0.8,
                   label="catalogue optimum", zorder=5)
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("X motor position (mm)")
        ax.set_ylabel("Y motor position (mm)")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.18, 1.0))
    fig.suptitle("Final recommendations from five paired seeds")
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def objective_reference_table(
    catalogue_rows: list[dict], config: dict, target_order: list[str]
) -> list[dict]:
    output = []
    for target in target_order:
        spec = config["objectives"][target]
        values = np.asarray([float(row[spec["source_column"]]) for row in catalogue_rows])
        index = int(np.argmax(values) if spec["direction"] == "maximize" else np.argmin(values))
        row = catalogue_rows[index]
        output.append({
            "target_name": target,
            "direction": spec["direction"],
            "source_column": spec["source_column"],
            "catalogue_optimum_value": float(values[index]),
            "x_mm": float(row["x_mm"]),
            "y_mm": float(row["y_mm"]),
            "peak_q": float(row["peak_q"]),
            "peak_prominence": float(row["peak_prominence"]),
        })
    return output


def summarize_threshold_sensitivity(
    tasks: list[dict], thresholds: list[float], target_order: list[str]
) -> list[dict]:
    output = []
    for target in target_order:
        for policy in REPLAY_POLICIES:
            rows = [
                row for row in tasks
                if row["target_name"] == target and row["policy"] == policy
            ]
            for threshold in thresholds:
                output.append({
                    "target_name": target,
                    "policy": policy,
                    "normalized_regret_threshold": threshold,
                    "seeds": len(rows),
                    "success_rate": float(np.mean([
                        row["final_normalized_regret"] <= threshold for row in rows
                    ])),
                })
    return output


def make_threshold_sensitivity_plot(
    rows: list[dict], target_order: list[str], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(target_order), figsize=(18, 5.2), sharey=True)
    thresholds = sorted({row["normalized_regret_threshold"] for row in rows})
    for ax, target in zip(axes, target_order):
        for policy in REPLAY_POLICIES:
            series = sorted(
                [row for row in rows if row["target_name"] == target and row["policy"] == policy],
                key=lambda row: row["normalized_regret_threshold"],
            )
            ax.plot(
                [row["normalized_regret_threshold"] for row in series],
                [row["success_rate"] for row in series],
                marker="o", linewidth=2, color=POLICY_COLORS[policy], label=policy,
            )
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Normalized regret threshold")
        ax.set_xticks(thresholds)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Final success rate across five seeds")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensitivity to the provisional success threshold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogue_csv", type=Path)
    parser.add_argument(
        "--target-config", type=Path,
        default=Path("configs/phase5_als_optimization_targets_provisional.yaml"),
    )
    parser.add_argument(
        "--phase3", type=Path,
        default=Path("configs/phase5_als_three_target_replay_provisional.yaml"),
    )
    parser.add_argument(
        "--phase4", type=Path,
        default=Path("configs/phase4_stopping_observers.yaml"),
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-override", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.target_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol = Protocol.load(args.phase3)
    catalogue_rows, coordinates = read_catalogue(args.catalogue_csv)
    if len(catalogue_rows) != protocol.catalogue_size:
        raise ValueError("catalogue size does not match the replay protocol")
    target_order = list(config["objectives"])
    budget = args.budget_override or protocol.budget(2)
    raw_directory = args.output / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    jobs = []
    for target in target_order:
        spec = config["objectives"][target]
        for policy in REPLAY_POLICIES:
            for seed in range(args.seeds):
                path = raw_directory / _raw_filename(target, policy, seed)
                if _trajectory_complete(path, budget):
                    continue
                jobs.append((
                    str(args.catalogue_csv), target, spec["source_column"],
                    spec["direction"], policy, seed, str(args.phase3),
                    str(args.phase4), str(path), args.budget_override,
                ))
    print(f"planned={len(target_order) * len(REPLAY_POLICIES) * args.seeds} missing={len(jobs)}", flush=True)
    if args.workers <= 1:
        for job in jobs:
            result = run_one_task(*job)
            print(json.dumps(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one_task, *job) for job in jobs]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)

    trajectories = load_complete_trajectories(raw_directory)
    expected = len(target_order) * len(REPLAY_POLICIES) * args.seeds
    if len(trajectories) != expected:
        raise RuntimeError(f"expected {expected} completed trajectories, found {len(trajectories)}")
    all_rows = [row for trajectory in trajectories for row in trajectory]
    write_rows(args.output / "trajectories.csv", all_rows)
    tasks = summarize_tasks(trajectories)
    policies = summarize_policies(tasks, target_order)
    stop_detail, stop_aggregate = summarize_stopping(trajectories)
    checkpoints = sorted(set([protocol.initial_size(2), 35, 50, 75, budget]))
    budget_rows = summarize_success_by_budget(trajectories, checkpoints)
    references = objective_reference_table(catalogue_rows, config, target_order)
    thresholds = [
        float(value) for value in protocol.raw["scoring"]["sensitivity_thresholds"]
    ]
    sensitivity = summarize_threshold_sensitivity(tasks, thresholds, target_order)
    write_rows(args.output / "task_summary.csv", tasks)
    write_rows(args.output / "policy_target_summary.csv", policies)
    write_rows(args.output / "stopping_detail.csv", stop_detail)
    write_rows(args.output / "stopping_summary.csv", stop_aggregate)
    write_rows(args.output / "success_by_budget.csv", budget_rows)
    write_rows(args.output / "objective_reference.csv", references)
    write_rows(args.output / "threshold_sensitivity.csv", sensitivity)
    make_regret_plot(trajectories, target_order, args.output / "regret_curves.png")
    make_success_heatmap(policies, target_order, args.output / "final_success_heatmap.png")
    make_success_budget_plot(budget_rows, target_order, args.output / "success_by_budget.png")
    make_stopping_plot(stop_aggregate, target_order, args.output / "stopping_tradeoff.png")
    make_recommendation_plot(
        catalogue_rows, tasks, config, target_order,
        args.output / "recommendation_locations.png",
    )
    make_threshold_sensitivity_plot(
        sensitivity, target_order, args.output / "threshold_sensitivity.png"
    )
    print(f"completed={len(trajectories)}", flush=True)


if __name__ == "__main__":
    main()
