"""Four-system smoke test for all five formal Phase 3 BoTorch policies."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, run_trajectory, software_versions, write_rows
from src.policies import POLICIES


SMOKE_SYSTEMS = (
    ("quadratic_peak", 2),
    ("hidden_narrow_peak", 2),
    ("grf_rough", 2),
    ("rosenbrock", 3),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/phase3_fair_protocol.yaml"
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--catalogue-size",
        type=int,
        default=128,
        help="Reduced catalogue used only for implementation smoke testing.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=2,
        help="Steps after the 5*d initialization; smoke verification only.",
    )
    parser.add_argument(
        "--output", default="results/phase3/botorch_smoke/trajectories.csv"
    )
    parser.add_argument(
        "--audit",
        help="Per-trajectory audit CSV; defaults beside --output.",
    )
    args = parser.parse_args()

    frozen_protocol = Protocol.load(args.protocol)
    smoke_raw = copy.deepcopy(frozen_protocol.raw)
    smoke_raw["domain"]["catalogue_size"] = args.catalogue_size
    protocol = Protocol(raw=smoke_raw)
    rows = []
    audit_rows = []
    failures = []
    for name, dimension in SMOKE_SYSTEMS:
        benchmark = get_benchmark(name, dimension, field_seed=2026)
        budget = protocol.initial_size(dimension) + args.extra_steps
        for policy in POLICIES:
            for seed in range(args.seeds):
                started = time.perf_counter()
                audit = {
                    "system": name,
                    "dimension": dimension,
                    "policy": policy,
                    "seed": seed,
                    "catalogue_size": protocol.catalogue_size,
                    "budget": budget,
                }
                try:
                    trajectory = run_trajectory(
                        benchmark=benchmark,
                        policy=policy,
                        optimizer_seed=seed,
                        protocol=protocol,
                        budget_override=budget,
                    )
                    rows.extend(trajectory)
                    final = trajectory[-1]
                    audit.update({
                        "status": "complete",
                        "trace_rows": len(trajectory),
                        "final_normalized_regret": final["normalized_regret"],
                        "final_success_le_0.05": final["success_le_0.05"],
                        "fit_warning_count": sum(
                            int(row["fit_warning_count"]) for row in trajectory
                        ),
                        "error": "",
                    })
                    print(name, dimension, policy, seed, "complete")
                except Exception as error:  # pragma: no cover - audit safety net
                    audit.update({
                        "status": "failed",
                        "trace_rows": 0,
                        "final_normalized_regret": "",
                        "final_success_le_0.05": "",
                        "fit_warning_count": "",
                        "error": f"{type(error).__name__}: {error}",
                    })
                    failures.append(audit.copy())
                    print(name, dimension, policy, seed, "failed", error)
                finally:
                    audit["runtime_seconds"] = time.perf_counter() - started
                    audit_rows.append(audit)

    if rows:
        write_rows(args.output, rows)
    audit_path = Path(args.audit) if args.audit else Path(args.output).with_name(
        "run_audit.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    metadata_path = Path(args.output).with_name("metadata.json")
    metadata_path.write_text(
        json.dumps({
            "purpose": "implementation smoke test, not scientific results",
            "systems": SMOKE_SYSTEMS,
            "policies": POLICIES,
            "seeds": args.seeds,
            "extra_steps_after_initialization": args.extra_steps,
            "smoke_catalogue_size": args.catalogue_size,
            "frozen_catalogue_size": frozen_protocol.catalogue_size,
            "planned_trajectories": len(audit_rows),
            "completed_trajectories": sum(
                row["status"] == "complete" for row in audit_rows
            ),
            "failed_trajectories": len(failures),
            "audit_path": str(audit_path),
            "versions": software_versions(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} trace rows to {args.output}")
    print(f"wrote {len(audit_rows)} audit rows to {audit_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} smoke trajectories failed; see audit")


if __name__ == "__main__":
    main()

