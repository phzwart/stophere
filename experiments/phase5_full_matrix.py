"""Manifest and resumable task runner for the formal Phase 5 BO matrix.

The default action only writes the 600-task manifest.  A task runs only when an
explicit ``--task-index`` is supplied, preventing accidental launch before the
protocol receives collaborator sign-off.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.benchmarks import FAMILIES, get_benchmark
from src.botorch_runner import Protocol, run_trajectory, software_versions, write_rows
from src.policies import POLICIES
from src.stopping import StoppingProtocol


def build_manifest(*, seeds: int = 5, field_seed: int = 2026) -> list[dict]:
    systems = [(name, dimension) for name in FAMILIES for dimension in (2, 3)]
    systems.extend([("branin", 2), ("hartmann3", 3)])
    tasks = []
    for name, dimension in systems:
        for policy in POLICIES:
            for seed in range(seeds):
                tasks.append({
                    "task_index": len(tasks),
                    "system": name,
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
        "--manifest", default="results/phase5/full_matrix/manifest.csv"
    )
    parser.add_argument(
        "--raw-directory", default="results/phase5/full_matrix/raw"
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing task output."
    )
    args = parser.parse_args()

    tasks = build_manifest(seeds=args.seeds, field_seed=args.field_seed)
    manifest_path = Path(args.manifest)
    write_manifest(manifest_path, tasks)
    print(f"wrote {len(tasks)} planned BO trajectories to {manifest_path}")
    if args.task_index is None:
        print("manifest only; no scientific trajectory was launched")
        return
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("task-index is outside the manifest")

    task = tasks[args.task_index]
    target = Path(args.raw_directory) / f"task_{args.task_index:04d}.csv"
    if target.exists() and not args.overwrite:
        print(f"kept existing completed task {target}")
        return

    phase3 = Protocol.load(args.phase3)
    phase4 = StoppingProtocol.load(args.phase4)
    benchmark = get_benchmark(
        task["system"], task["dimension"], field_seed=task["field_seed"]
    )
    rows = run_trajectory(
        benchmark=benchmark,
        policy=task["policy"],
        optimizer_seed=task["optimizer_seed"],
        protocol=phase3,
        stopping_protocol=phase4,
    )
    write_rows(target, rows)
    metadata = target.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "task": task,
                "phase3_protocol": args.phase3,
                "phase4_protocol": args.phase4,
                "versions": software_versions(),
                "completed_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"completed task {args.task_index} at {target}")


if __name__ == "__main__":
    main()
