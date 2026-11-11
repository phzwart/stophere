"""Four-system smoke test for all five formal Phase 3 BoTorch policies."""
from __future__ import annotations

import argparse
import copy
import json
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
    args = parser.parse_args()

    frozen_protocol = Protocol.load(args.protocol)
    smoke_raw = copy.deepcopy(frozen_protocol.raw)
    smoke_raw["domain"]["catalogue_size"] = args.catalogue_size
    protocol = Protocol(raw=smoke_raw)
    rows = []
    for name, dimension in SMOKE_SYSTEMS:
        benchmark = get_benchmark(name, dimension, field_seed=2026)
        budget = protocol.initial_size(dimension) + args.extra_steps
        for policy in POLICIES:
            for seed in range(args.seeds):
                rows.extend(run_trajectory(
                    benchmark=benchmark,
                    policy=policy,
                    optimizer_seed=seed,
                    protocol=protocol,
                    budget_override=budget,
                ))
                print(name, dimension, policy, seed, "complete")

    write_rows(args.output, rows)
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
            "versions": software_versions(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} trace rows to {args.output}")


if __name__ == "__main__":
    main()

