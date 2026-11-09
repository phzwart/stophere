"""Success-rate summaries with cross-system quantiles.

The pooled success rate answers "what fraction of all runs succeeded?" The
quantiles answer a different question: "how much does the system-level success
rate vary across objective landscapes?" They are descriptive spread measures,
not confidence intervals.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "nominal": "Nominal RBF/UCB",
    "kernel=matern32": "Matérn 3/2",
    "kernel=matern52": "Matérn 5/2",
    "lengthscale_x0.5": "RBF length scale ×0.5",
    "lengthscale_x2": "RBF length scale ×2",
    "beta=0.5": "UCB β=0.5",
    "beta=8": "UCB β=8",
    "acquisition=ei": "Expected improvement",
    "acquisition=pi": "Probability of improvement",
    "observation_noise=3pct": "3% observation noise",
    "matern32_lengthscale_x2": "Matérn 3/2, length scale ×2",
    "random_baseline": "Random search",
    "cma_es": "CMA-ES",
    "snobfit": "SNOBFIT",
    "nomad": "NOMAD",
    "bam": "BAM",
    "uniform_random": "Uniform random",
}

CONFIG_ORDER = tuple(DISPLAY_NAMES)
PROFILE_CONFIGS = (
    "nominal",
    "acquisition=ei",
    "acquisition=pi",
    "matern32_lengthscale_x2",
    "random_baseline",
)

PROFILE_COLORS = {
    "nominal": "#3366AA",
    "acquisition=ei": "#228833",
    "acquisition=pi": "#AA3377",
    "matern32_lengthscale_x2": "#CCBB44",
    "random_baseline": "#777777",
    "cma_es": "#3366AA",
    "snobfit": "#228833",
    "nomad": "#AA3377",
    "bam": "#CC6677",
    "uniform_random": "#777777",
}


def success_profile_rows(
    traces: list[dict],
    checkpoints: tuple[int, ...] | list[int] | None = None,
    epsilon: float = 0.05,
    carry_forward: bool = False,
) -> list[dict]:
    """Summarize pooled and cross-system success at evaluation checkpoints.

    With ``carry_forward=True``, a run that terminates naturally before a
    checkpoint contributes its last incumbent. This is appropriate for a
    maximum-budget profile: stopping early does not erase the solution already
    found, and it must not be treated as an adapter failure.
    """
    if not traces:
        raise ValueError("traces must not be empty")
    available = sorted({int(row["n_eval"]) for row in traces})
    if checkpoints is None:
        checkpoints = available
    checkpoints = tuple(dict.fromkeys(int(value) for value in checkpoints))
    missing = sorted(set(checkpoints) - set(available))
    if missing and not carry_forward:
        raise ValueError(
            f"checkpoints not present in trajectories: {missing}; "
            f"available range is {available[0]}-{available[-1]}"
        )

    configs = sorted(
        {row["config"] for row in traces},
        key=lambda value: (
            CONFIG_ORDER.index(value) if value in CONFIG_ORDER else len(CONFIG_ORDER),
            value,
        ),
    )
    output = []
    for config in configs:
        for n_eval in checkpoints:
            if carry_forward:
                config_rows = [
                    row for row in traces if row["config"] == config
                ]
                run_keys = sorted({
                    (row["system"], int(row["seed"])) for row in config_rows
                })
                rows = []
                for system, seed in run_keys:
                    candidates = [
                        row for row in config_rows
                        if row["system"] == system
                        and int(row["seed"]) == seed
                        and int(row["n_eval"]) <= n_eval
                    ]
                    if candidates:
                        rows.append(max(
                            candidates, key=lambda row: int(row["n_eval"])
                        ))
            else:
                rows = [
                    row for row in traces
                    if row["config"] == config
                    and int(row["n_eval"]) == n_eval
                ]
            if not rows:
                continue
            per_system = []
            for system in sorted({row["system"] for row in rows}):
                system_rows = [row for row in rows if row["system"] == system]
                per_system.append(float(np.mean([
                    float(row["normalized_regret"]) <= epsilon for row in system_rows
                ])))
            rates = np.asarray(per_system, dtype=float)
            pooled = float(np.mean([
                float(row["normalized_regret"]) <= epsilon for row in rows
            ]))
            output.append({
                "config": config,
                "n_eval": n_eval,
                "runs": len(rows),
                "systems": len(rates),
                "success_epsilon": epsilon,
                "pooled_success_rate": pooled,
                "system_success_q25": float(np.quantile(rates, 0.25)),
                "system_success_median": float(np.quantile(rates, 0.50)),
                "system_success_q75": float(np.quantile(rates, 0.75)),
            })
    return output


def make_final_success_plot(rows: list[dict], path: Path, final_eval: int) -> None:
    selected = [row for row in rows if int(row["n_eval"]) == final_eval]
    if not selected:
        raise ValueError(f"no success rows at n_eval={final_eval}")
    selected = sorted(
        selected,
        key=lambda row: (
            float(row["system_success_median"]),
            float(row["pooled_success_rate"]),
        ),
    )
    labels = [DISPLAY_NAMES.get(row["config"], row["config"]) for row in selected]
    median = np.asarray([row["system_success_median"] for row in selected], dtype=float)
    q25 = np.asarray([row["system_success_q25"] for row in selected], dtype=float)
    q75 = np.asarray([row["system_success_q75"] for row in selected], dtype=float)
    pooled = np.asarray([row["pooled_success_rate"] for row in selected], dtype=float)
    y = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.barh(y, median, color="#4C78A8", alpha=0.88, label="Median across systems")
    ax.errorbar(
        median, y,
        xerr=np.vstack((median - q25, q75 - median)),
        fmt="none", ecolor="#1F2937", elinewidth=1.5, capsize=4,
        label="25th-75th percentile across systems",
    )
    ax.scatter(pooled, y, marker="D", s=28, color="#D97706", zorder=3,
               label="Pooled across runs")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.04)
    ax.set_xlabel("Success rate (normalized simple regret ≤ 0.05)")
    ax.set_title(f"Success spread at {final_eval} evaluations")
    ax.grid(axis="x", color="#D1D5DB", alpha=0.65)
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.01, 0.01,
        "Bars and whiskers describe variation across benchmark systems; they are not confidence intervals.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_budget_success_plot(
    rows: list[dict],
    path: Path,
    checkpoints: tuple[int, ...] | list[int],
    configs: tuple[str, ...] = PROFILE_CONFIGS,
) -> None:
    checkpoints = tuple(int(value) for value in checkpoints)
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    plotted = 0
    for config in configs:
        series = sorted(
            (row for row in rows
             if row["config"] == config and int(row["n_eval"]) in checkpoints),
            key=lambda row: int(row["n_eval"]),
        )
        if not series:
            continue
        x = np.asarray([row["n_eval"] for row in series], dtype=float)
        pooled = np.asarray([row["pooled_success_rate"] for row in series], dtype=float)
        q25 = np.asarray([row["system_success_q25"] for row in series], dtype=float)
        q75 = np.asarray([row["system_success_q75"] for row in series], dtype=float)
        color = PROFILE_COLORS.get(config, "#3366AA")
        ax.plot(x, pooled, marker="o", linewidth=2.0, color=color,
                label=DISPLAY_NAMES.get(config, config))
        ax.fill_between(x, q25, q75, color=color, alpha=0.12)
        plotted += 1
    if not plotted:
        raise ValueError("none of the requested configurations are present")
    ax.set_xticks(checkpoints)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Pooled success rate")
    ax.set_title("Success versus evaluation budget")
    ax.grid(color="#D1D5DB", alpha=0.65)
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.01, 0.01,
        "Lines pool all runs. Shaded bands show the 25th-75th percentile of system-level success rates.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
