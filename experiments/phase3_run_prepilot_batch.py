"""Resume all missing full-budget pre-pilot tasks with bounded parallelism."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.phase3_full_budget_prepilot import build_prepilot_manifest


def run_task(index: int, extra: list[str]) -> tuple[int, int, str]:
    command = [
        sys.executable,
        "-m",
        "experiments.phase3_full_budget_prepilot",
        "--task-index",
        str(index),
        *extra,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    message = (result.stdout + result.stderr).strip()
    return index, result.returncode, message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--raw-directory", type=Path,
        default=Path("results/phase3/full_budget_prepilot/raw"),
    )
    parser.add_argument("--include-completed", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    tasks = build_prepilot_manifest(seeds=args.seeds)
    indices = []
    for task in tasks:
        target = args.raw_directory / f"task_{task['task_index']:03d}_full_budget.csv"
        if args.include_completed or not target.exists():
            indices.append(task["task_index"])
    print(f"planned={len(tasks)} remaining={len(indices)} workers={args.workers}", flush=True)
    if not indices:
        return
    extra = ["--seeds", str(args.seeds), "--raw-directory", str(args.raw_directory)]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, index, extra): index for index in indices}
        for completed, future in enumerate(as_completed(futures), start=1):
            index, returncode, message = future.result()
            status = "ok" if returncode == 0 else "failed"
            print(f"[{completed}/{len(indices)}] task={index:03d} {status}", flush=True)
            if returncode:
                failures.append((index, message))
    if failures:
        for index, message in failures:
            print(f"task {index:03d} failed:\n{message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
