"""Render the final GP maps and actual measurement locations from real data."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_map(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in handle.keys()}


def measurement_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["gp_x"].shape[0])


def latest_map(folder: Path) -> Path:
    files = list(folder.glob("*.h5"))
    if not files:
        raise ValueError(f"no HDF5 GP maps found in {folder}")
    return max(files, key=measurement_count)


def draw_map(ax, data: dict[str, np.ndarray], title: str, field: str = "gp_mean"):
    mesh = ax.pcolormesh(data["x"], data["y"], data[field], shading="auto", cmap="viridis")
    locations = data["gp_x"]
    sequence = np.arange(1, len(locations) + 1)
    ax.plot(locations[:, 0], locations[:, 1], color="white", alpha=0.25, linewidth=0.5)
    ax.scatter(
        locations[:, 0], locations[:, 1], c=sequence, cmap="plasma",
        s=9, edgecolors="none", alpha=0.8,
    )
    ax.scatter(
        locations[0, 0], locations[0, 1], marker="o", s=70,
        facecolors="none", edgecolors="white", linewidths=1.4, label="First point",
    )
    ax.scatter(
        locations[-1, 0], locations[-1, 1], marker="X", s=75,
        color="#EF4444", edgecolors="white", linewidths=0.7, label="Last point",
    )
    ax.set_title(f"{title}\nN={len(locations)}")
    ax.set_xlabel("X motor position (mm)")
    ax.set_ylabel("Y motor position (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("als_gp_folder", type=Path)
    parser.add_argument("petra_gp_folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    als_path = latest_map(args.als_gp_folder)
    petra_path = latest_map(args.petra_gp_folder)
    als = load_map(als_path)
    petra = load_map(petra_path)
    columns = 3 if "gp_covariance" in petra else 2
    fig, axes = plt.subplots(1, columns, figsize=(6.2 * columns, 5.5))
    mesh = draw_map(axes[0], als, "ALS NaCl: posterior mean")
    fig.colorbar(mesh, ax=axes[0], fraction=0.046)
    mesh = draw_map(axes[1], petra, "PETRA III composite: posterior mean")
    fig.colorbar(mesh, ax=axes[1], fraction=0.046)
    if columns == 3:
        mesh = draw_map(axes[2], petra, "PETRA III: stored uncertainty", "gp_covariance")
        fig.colorbar(mesh, ax=axes[2], fraction=0.046)
    fig.suptitle("Real beamline GP maps with sequential measurement locations")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"ALS={als_path.name} PETRA={petra_path.name} output={args.output}")


if __name__ == "__main__":
    main()
