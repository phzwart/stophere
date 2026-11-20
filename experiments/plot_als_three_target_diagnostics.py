"""Create high-contrast ALS optimization plots and a tiny policy-selector audit.

All inputs/outputs are user-supplied paths because the ALS data and derived
results are private. The selector uses target labels and training seeds only;
it is an exploratory held-out demonstration, not a deployable controller.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGETS = ["integrated_intensity", "maximum_peak_height", "minimum_fwhm"]
TARGET_TITLES = {
    "integrated_intensity": "Integrated intensity (maximize)",
    "maximum_peak_height": "Maximum peak height (maximize)",
    "minimum_fwhm": "FWHM (minimize)",
}
POLICIES = ["log_ei", "log_pi", "ucb", "thompson_sampling", "mes", "random"]
COLORS = {
    "log_ei": "#2563EB",
    "log_pi": "#D97706",
    "ucb": "#059669",
    "thompson_sampling": "#DC2626",
    "mes": "#7C3AED",
    "random": "#64748B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def save_maps(rows: list[dict[str, str]], out: Path) -> None:
    x = np.asarray([float(row["x_mm"]) for row in rows])
    y = np.asarray([float(row["y_mm"]) for row in rows])
    for target in TARGETS:
        field = "fwhm_for_optimization" if target == "minimum_fwhm" else target
        raw = np.asarray([float(row[field]) for row in rows])
        quality = -raw if target == "minimum_fwhm" else raw
        # Percentile ranks give each target a legible, robust contrast scale
        # without letting a few extreme measurements wash out most locations.
        order = np.argsort(quality, kind="stable")
        ranks = np.empty(len(quality), dtype=float)
        ranks[order] = np.arange(len(quality), dtype=float)
        scaled = ranks / max(len(quality) - 1, 1)
        fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
        points = ax.scatter(x, y, c=scaled, s=39, marker="s", cmap="magma",
                            vmin=0, vmax=1, linewidths=0)
        ax.set(title=TARGET_TITLES[target], xlabel="X motor position (mm)",
               ylabel="Y motor position (mm)", aspect="equal")
        colorbar = fig.colorbar(points, ax=ax, label="Within-target quality percentile")
        colorbar.set_ticks(np.linspace(0, 1, 5), labels=["0%", "25%", "50%", "75%", "100%"])
        ax.text(0.01, -0.13,
                "Brighter = better rank within this target; FWHM favors narrower peaks.",
                transform=ax.transAxes, fontsize=9, color="#475569")
        fig.savefig(out / f"map_{target}_contrast.png", dpi=320, bbox_inches="tight")
        plt.close(fig)


def save_distributions(rows: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fields = ["integrated_intensity", "maximum_peak_height", "fwhm_for_optimization"]
    titles = [TARGET_TITLES[t] for t in TARGETS]
    for ax, field, title in zip(axes, fields, titles):
        values = np.asarray([float(row[field]) for row in rows])
        ax.hist(values, bins=45, color="#2563EB", edgecolor="white", linewidth=0.35)
        if field == "maximum_peak_height":
            ax.set_xscale("log")
            ax.set_xlabel("Peak height (log scale)")
        else:
            ax.set_xlabel("Measured target value")
        ax.set_title(title)
        ax.set_ylabel("Number of grid locations")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Target-value distributions across 3,249 measured locations", fontsize=15)
    fig.savefig(out / "target_distributions_log_peak.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_success_bars(rows: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True, constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        values, lower, upper = [], [], []
        for policy in POLICIES:
            selected = [r for r in rows if r["target_name"] == target and r["policy"] == policy]
            n = len(selected)
            successes = sum(float(r["final_normalized_regret"]) <= 0.05 for r in selected)
            rate = successes / n if n else 0
            lo, hi = wilson(successes, n)
            values.append(rate)
            lower.append(rate - lo)
            upper.append(hi - rate)
        xx = np.arange(len(POLICIES))
        ax.bar(xx, values, color=[COLORS[p] for p in POLICIES], width=0.72,
               yerr=np.array([lower, upper]), capsize=3, error_kw={"elinewidth": 1.1})
        for i, rate in enumerate(values):
            ax.text(i, min(rate + upper[i] + 0.035, 1.04), f"{rate:.0%}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xx, [p.replace("_", "\n") for p in POLICIES], fontsize=8)
        ax.set_title(TARGET_TITLES[target])
        ax.set_ylim(0, 1.16)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylabel("Success rate (95% Wilson interval)")
    fig.suptitle("Success by acquisition policy (5 paired seeds; regret ≤ 0.05)", fontsize=15)
    fig.savefig(out / "success_rate_bars_with_intervals.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def save_selector_audit(task_rows: list[dict[str, str]], out: Path) -> None:
    train_seeds, test_seeds = {0, 1, 2}, {3, 4}
    training = [r for r in task_rows if int(r["seed"]) in train_seeds]
    testing = [r for r in task_rows if int(r["seed"]) in test_seeds]

    def avg(rows: list[dict[str, str]], policies: set[str], targets: set[str] | None = None) -> float:
        values = [float(r["final_normalized_regret"]) for r in rows
                  if r["policy"] in policies and (targets is None or r["target_name"] in targets)]
        return float(np.mean(values))

    global_policy = min(POLICIES, key=lambda p: avg(training, {p}))
    by_target = {
        target: min(POLICIES, key=lambda p: avg(training, {p}, {target}))
        for target in TARGETS
    }
    selected = []
    for row in testing:
        row = dict(row)
        row["selector_policy"] = by_target[row["target_name"]]
        row["selector_regret"] = row["final_normalized_regret"] if row["policy"] == row["selector_policy"] else ""
        row["global_policy"] = global_policy
        row["global_selector_regret"] = row["final_normalized_regret"] if row["policy"] == global_policy else ""
        selected.append(row)
    outpath = out / "selector_holdout_detail.csv"
    with outpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    records = []
    for target in TARGETS:
        chosen = by_target[target]
        fixed_test = [r for r in testing if r["target_name"] == target and r["policy"] == global_policy]
        selector_test = [r for r in testing if r["target_name"] == target and r["policy"] == chosen]
        records.append({
            "target": target,
            "training_selected_policy": chosen,
            "heldout_seeds": "3,4",
            "heldout_selector_mean_regret": float(np.mean([float(r["final_normalized_regret"]) for r in selector_test])),
            "heldout_global_policy": global_policy,
            "heldout_global_mean_regret": float(np.mean([float(r["final_normalized_regret"]) for r in fixed_test])),
            "interpretation": "Exploratory target-conditioned selector; n=2 held-out seeds; not deployable evidence",
        })
    with (out / "selector_holdout_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    labels = [r["target"].replace("_", "\n") for r in records]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.35
    ax.bar(x - width / 2, [r["heldout_global_mean_regret"] for r in records], width,
           label=f"One fixed policy ({global_policy})", color="#64748B")
    ax.bar(x + width / 2, [r["heldout_selector_mean_regret"] for r in records], width,
           label="Target-conditioned selector", color="#2563EB")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean normalized regret on held-out seeds 3–4 (lower is better)")
    ax.set_title("Exploratory selector holdout (trained on seeds 0–2)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(out / "selector_holdout_comparison.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogue_csv", type=Path)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    catalogue = read_csv(args.catalogue_csv)
    tasks = read_csv(args.replay_dir / "task_summary.csv")
    save_maps(catalogue, args.output)
    save_distributions(catalogue, args.output)
    save_success_bars(tasks, args.output)
    save_selector_audit(tasks, args.output)


if __name__ == "__main__":
    main()
