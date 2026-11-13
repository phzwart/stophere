"""Resumable four-system full-budget pre-pilot for Phase 3 and Phase 4.

The default action writes the 100-task plan only. A trajectory starts only
when ``--task-index`` is supplied, so the expensive pre-pilot cannot be
launched accidentally before collaborator sign-off.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from experiments.phase3_botorch_smoke import SMOKE_SYSTEMS
from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, run_trajectory, software_versions, write_rows
from src.policies import POLICIES
from src.stopping import StoppingProtocol


def build_prepilot_manifest(*, seeds: int = 5, field_seed: int = 2026) -> list[dict]:
    tasks = []
    for system, dimension in SMOKE_SYSTEMS:
        for policy in POLICIES:
            for seed in range(seeds):
                tasks.append({
                    "task_index": len(tasks),
                    "system": system,
                    "dimension": dimension,
                    "policy": policy,
                    "optimizer_seed": seed,
                    "field_seed": field_seed,
                })
    return tasks


def write_manifest(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3", default="configs/phase3_fair_protocol.yaml")
    parser.add_argument("--phase4", default="configs/phase4_stopping_observers.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--field-seed", type=int, default=2026)
    parser.add_argument(
        "--manifest", default="results/phase3/full_budget_prepilot/manifest.csv"
    )
    parser.add_argument(
        "--raw-directory", default="results/phase3/full_budget_prepilot/raw"
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--budget-override",
        type=int,
        help="Diagnostic only; omit for the formal 100/150 full-budget pre-pilot.",
    )
    parser.add_argument("--without-stopping", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = build_prepilot_manifest(seeds=args.seeds, field_seed=args.field_seed)
    manifest_path = Path(args.manifest)
    write_manifest(manifest_path, tasks)
    print(f"wrote {len(tasks)} planned trajectories to {manifest_path}")
    if args.task_index is None:
        print("manifest only; no pre-pilot trajectory was launched")
        return
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("task-index is outside the pre-pilot manifest")

    task = tasks[args.task_index]
    suffix = "diagnostic" if args.budget_override is not None else "full_budget"
    target = Path(args.raw_directory) / f"task_{args.task_index:03d}_{suffix}.csv"
    if target.exists() and not args.overwrite:
        print(f"kept existing completed task {target}")
        return

    phase3 = Protocol.load(args.phase3)
    phase4 = None if args.without_stopping else StoppingProtocol.load(args.phase4)
    benchmark = get_benchmark(
        task["system"], task["dimension"], field_seed=task["field_seed"]
    )
    started = time.perf_counter()
    rows = run_trajectory(
        benchmark=benchmark,
        policy=task["policy"],
        optimizer_seed=task["optimizer_seed"],
        protocol=phase3,
        budget_override=args.budget_override,
        stopping_protocol=phase4,
    )
    runtime_seconds = time.perf_counter() - started
    write_rows(target, rows)
    metadata = target.with_suffix(".json")
    metadata.write_text(
        json.dumps({
            "purpose": (
                "diagnostic pre-pilot run"
                if args.budget_override is not None
                else "formal four-system full-budget pre-pilot task"
            ),
            "task": task,
            "phase3_protocol": args.phase3,
            "phase4_protocol": None if args.without_stopping else args.phase4,
            "budget_override": args.budget_override,
            "completed_rows": len(rows),
            "runtime_seconds": runtime_seconds,
            "final_normalized_regret": rows[-1]["normalized_regret"],
            "final_success_le_0.05": rows[-1]["success_le_0.05"],
            "versions": software_versions(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"completed task {args.task_index} at {target}")
    print(f"runtime_seconds={runtime_seconds:.3f}")


if __name__ == "__main__":
    main()
