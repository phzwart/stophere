"""Extract provisional optimization targets from the real ALS dense grid.

The input is the reduced one-dimensional scattering data, not detector images.
The definitions are deliberately explicit because the q window, background
treatment, and physical meaning still require collaborator review.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import spearmanr


POSITION = re.compile(r"_X(?P<x>-?\d+p\d+)_Y(?P<y>-?\d+p\d+)_")
TARGET_COLUMNS = (
    "integrated_intensity",
    "maximum_peak_height",
    "fwhm",
)


def decode_position(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _interpolated_q(q: np.ndarray, fractional_index: float) -> float:
    return float(np.interp(fractional_index, np.arange(len(q), dtype=float), q))


def extract_targets(
    q: np.ndarray,
    intensity: np.ndarray,
    errors: np.ndarray | None,
    settings: dict,
) -> dict:
    """Return three targets and peak-quality diagnostics for one spectrum."""
    q = np.asarray(q, dtype=float).reshape(-1)
    intensity = np.asarray(intensity, dtype=float).reshape(-1)
    if errors is None:
        errors = np.full_like(intensity, np.nan)
    else:
        errors = np.asarray(errors, dtype=float).reshape(-1)
    valid = np.isfinite(q) & np.isfinite(intensity)
    if valid.sum() < 5:
        raise ValueError("fewer than five finite q/intensity pairs")
    q = q[valid]
    intensity = intensity[valid]
    errors = errors[valid]
    order = np.argsort(q)
    q = q[order]
    intensity = intensity[order]
    errors = errors[order]
    if np.any(np.diff(q) <= 0):
        raise ValueError("q values must be strictly increasing after sorting")

    window = int(settings["savgol_window_points"])
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd")
    if window > len(intensity):
        window = len(intensity) if len(intensity) % 2 else len(intensity) - 1
    polynomial_order = min(int(settings["savgol_polynomial_order"]), window - 1)
    smoothed = savgol_filter(intensity, window, polynomial_order)
    robust_range = float(np.quantile(smoothed, 0.95) - np.quantile(smoothed, 0.05))
    finite_errors = errors[np.isfinite(errors) & (errors >= 0)]
    median_error = float(np.median(finite_errors)) if finite_errors.size else 0.0
    minimum_prominence = max(
        float(settings["peak_prominence_error_multiplier"]) * median_error,
        float(settings["peak_prominence_robust_range_fraction"]) * robust_range,
        np.finfo(float).eps,
    )
    peaks, properties = find_peaks(
        smoothed,
        prominence=minimum_prominence,
        distance=int(settings["peak_minimum_distance_points"]),
    )
    integrated = float(np.trapezoid(intensity, q))
    result = {
        "integrated_intensity": integrated,
        "maximum_peak_height": float("nan"),
        "peak_height_above_background": float("nan"),
        "peak_q": float("nan"),
        "fwhm": float("nan"),
        "peak_prominence": float("nan"),
        "minimum_peak_prominence": minimum_prominence,
        "detected_peak_count": int(len(peaks)),
        "peak_valid": 0,
    }
    if not len(peaks):
        return result

    selected_local = int(np.argmax(smoothed[peaks]))
    selected_peak = int(peaks[selected_local])
    widths, _, left_ips, right_ips = peak_widths(
        smoothed,
        np.asarray([selected_peak]),
        rel_height=0.5,
    )
    left_q = _interpolated_q(q, float(left_ips[0]))
    right_q = _interpolated_q(q, float(right_ips[0]))
    fwhm = max(right_q - left_q, 0.0)
    prominence = float(properties["prominences"][selected_local])
    result.update({
        "maximum_peak_height": float(smoothed[selected_peak]),
        "peak_height_above_background": prominence,
        "peak_q": float(q[selected_peak]),
        "fwhm": fwhm,
        "peak_prominence": prominence,
        "peak_valid": int(np.isfinite(fwhm) and fwhm > 0.0),
    })
    return result


def spectrum_row(path: Path, root: Path, settings: dict) -> dict:
    match = POSITION.search(path.name)
    if not match:
        raise ValueError(f"position is not encoded in {path.name}")
    with h5py.File(path, "r") as handle:
        q = np.asarray(handle["q"], dtype=float)
        intensity = np.asarray(handle["intensity"], dtype=float)
        errors = np.asarray(handle["errors"], dtype=float) if "errors" in handle else None
    targets = extract_targets(q, intensity, errors, settings)
    valid = np.isfinite(q) & np.isfinite(intensity)
    return {
        "x_mm": decode_position(match.group("x")),
        "y_mm": decode_position(match.group("y")),
        **targets,
        "q_min": float(np.asarray(q)[valid].min()),
        "q_max": float(np.asarray(q)[valid].max()),
        "q_points": int(valid.sum()),
        "source_file": str(path.relative_to(root)),
    }


def build_catalogue(grid_root: Path, settings: dict) -> list[dict]:
    rows = [
        spectrum_row(path, grid_root, settings)
        for path in sorted(grid_root.rglob("*_integration-azimuthal.h5"))
    ]
    if not rows:
        raise ValueError(f"no reduced spectra found under {grid_root}")
    positions = [(row["x_mm"], row["y_mm"]) for row in rows]
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate motor coordinates found")
    return rows


def penalize_invalid_fwhm(rows: list[dict], settings: dict) -> None:
    valid = np.asarray([
        row["fwhm"] for row in rows if row["peak_valid"] and np.isfinite(row["fwhm"])
    ], dtype=float)
    if not valid.size:
        raise ValueError("no valid FWHM values were extracted")
    penalty = float(valid.max() + settings["invalid_fwhm_penalty_range_fraction"] * np.ptp(valid))
    for row in rows:
        row["fwhm_for_optimization"] = (
            float(row["fwhm"]) if row["peak_valid"] else penalty
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality_summary(rows: list[dict]) -> dict:
    positions = {(row["x_mm"], row["y_mm"]) for row in rows}
    x_values = sorted({row["x_mm"] for row in rows})
    y_values = sorted({row["y_mm"] for row in rows})
    valid_peaks = sum(int(row["peak_valid"]) for row in rows)
    summary = {
        "spectra": len(rows),
        "unique_positions": len(positions),
        "unique_x": len(x_values),
        "unique_y": len(y_values),
        "complete_cartesian_grid": len(rows) == len(x_values) * len(y_values),
        "valid_peak_and_fwhm": valid_peaks,
        "invalid_peak_or_fwhm": len(rows) - valid_peaks,
        "valid_peak_fraction": valid_peaks / len(rows),
        "common_q_min": len({round(float(row["q_min"]), 10) for row in rows}) == 1,
        "common_q_max": len({round(float(row["q_max"]), 10) for row in rows}) == 1,
        "common_q_points": len({int(row["q_points"]) for row in rows}) == 1,
        "objective_quantiles": {},
    }
    for column in ("integrated_intensity", "maximum_peak_height", "fwhm_for_optimization"):
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        summary["objective_quantiles"][column] = {
            label: float(value) for label, value in zip(
                ("minimum", "q25", "median", "q75", "maximum"),
                np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]),
            )
        }
    return summary


def _grid_scatter(ax, rows: list[dict], column: str, title: str, label: str, cmap: str):
    x = np.asarray([row["x_mm"] for row in rows], dtype=float)
    y = np.asarray([row["y_mm"] for row in rows], dtype=float)
    values = np.asarray([row[column] for row in rows], dtype=float)
    image = ax.scatter(x, y, c=values, s=22, marker="s", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("X motor position (mm)")
    ax.set_ylabel("Y motor position (mm)")
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, label=label)


def make_target_maps(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    _grid_scatter(
        axes[0], rows, "integrated_intensity", "A. Integrated intensity (maximize)",
        "Integral of I(q) dq", "viridis",
    )
    _grid_scatter(
        axes[1], rows, "maximum_peak_height", "B. Maximum peak height (maximize)",
        "Smoothed peak height", "magma",
    )
    _grid_scatter(
        axes[2], rows, "fwhm_for_optimization", "C. FWHM (minimize)",
        "Peak width in q units", "cividis_r",
    )
    fig.suptitle("Real ALS NaCl grid: provisional optimization targets")
    fig.text(
        0.01, 0.01,
        "Reduced 1D spectra over the full measured q range. FWHM is measured at half prominence; "
        "spectra without a valid peak receive a documented worst-value penalty.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_quality_plot(rows: list[dict], path: Path) -> None:
    columns = (
        "integrated_intensity",
        "maximum_peak_height",
        "fwhm_for_optimization",
    )
    labels = ("Integrated intensity", "Maximum peak height", "FWHM")
    values = np.column_stack([
        np.asarray([float(row[column]) for row in rows], dtype=float)
        for column in columns
    ])
    ranks = np.column_stack([np.argsort(np.argsort(values[:, index])) for index in range(3)])
    correlation = np.asarray(spearmanr(values, axis=0).statistic, dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    colors = ("#2563EB", "#D97706", "#7C3AED")
    for index in range(3):
        axes[index].hist(values[:, index], bins=40, color=colors[index], alpha=0.82)
        axes[index].set_title(labels[index])
        axes[index].set_xlabel("Extracted value")
        axes[index].set_ylabel("Grid locations")
        axes[index].grid(axis="y", alpha=0.25)
    image = axes[3].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axes[3].set_xticks(range(3), labels, rotation=35, ha="right")
    axes[3].set_yticks(range(3), labels)
    axes[3].set_title("Spearman rank correlation")
    for row in range(3):
        for column in range(3):
            axes[3].text(column, row, f"{correlation[row, column]:.2f}",
                         ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=axes[3], fraction=0.046, pad=0.04)
    fig.suptitle("Target distributions and relationships")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_root", type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/phase3_als_optimization_targets_provisional.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    settings = config["preprocessing"]
    rows = build_catalogue(args.grid_root, settings)
    penalize_invalid_fwhm(rows, settings)
    write_csv(args.output / "als_optimization_targets.csv", rows)
    summary = quality_summary(rows)
    (args.output / "target_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_target_maps(rows, args.output / "target_maps.png")
    make_quality_plot(rows, args.output / "target_quality.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
