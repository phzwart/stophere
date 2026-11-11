"""Reduced Phase 4 smoke test on three contrasting landscapes."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from src.benchmarks import get_benchmark
from src.botorch_runner import Protocol, run_trajectory, write_rows
from src.stopping import RULES, StoppingProtocol


SMOKE_SYSTEMS = (
    ("quadratic_peak", 2),
    ("hidden_narrow_peak", 2),
    ("grf_long", 2),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3", default="configs/phase3_fair_protocol.yaml")
    parser.add_argument("--phase4", default="configs/phase4_stopping_observers.yaml")
    parser.add_argument("--catalogue-size", type=int, default=128)
    parser.add_argument("--extra-steps", type=int, default=12)
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="log_ei")
    parser.add_argument(
        "--output", default="results/phase4/stopping_smoke/trajectories.csv"
    )
    args = parser.parse_args()

    acquisition = Protocol.load(args.phase3)
    acquisition_raw = copy.deepcopy(acquisition.raw)
    acquisition_raw["domain"]["catalogue_size"] = args.catalogue_size
    acquisition = Protocol(acquisition_raw)

    stopping = StoppingProtocol.load(args.phase4)
    stopping_raw = copy.deepcopy(stopping.raw)
    stopping_raw["monte_carlo"]["num_paths"] = args.paths
    stopping = StoppingProtocol(stopping_raw)

    all_rows = []
    summary = []
    for name, dimension in SMOKE_SYSTEMS:
        rows = run_trajectory(
            benchmark=get_benchmark(name, dimension, field_seed=2026),
            policy=args.policy,
            optimizer_seed=args.seed,
            protocol=acquisition,
            budget_override=acquisition.initial_size(dimension) + args.extra_steps,
            stopping_protocol=stopping,
        )
        all_rows.extend(rows)
        final = rows[-1]
        summary.append({
            "system": name,
            "dimension": dimension,
            "policy": args.policy,
            "seed": args.seed,
            "budget": final["n_eval"],
            **{
                f"{rule}_first_trigger": final[f"{rule}_first_trigger"]
                for rule in RULES
            },
        })
        print(name, "complete")

    write_rows(args.output, all_rows)
    summary_path = Path(args.output).with_name("summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
