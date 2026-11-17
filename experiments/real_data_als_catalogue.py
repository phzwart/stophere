"""Build a finite ALS salt catalogue from reduced grid-scan HDF5 spectra.

The default response is a transparent q-at-maximum-intensity proxy. It is not
claimed to reproduce the paper's fitted-peak workflow; that feature definition
must be confirmed before scientific comparison.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


POSITION = re.compile(r"_X(?P<x>-?\d+p\d+)_Y(?P<y>-?\d+p\d+)_")


def decode_position(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def spectrum_row(path: Path, root: Path) -> dict:
    match = POSITION.search(path.name)
    if not match:
        raise ValueError(f"position is not encoded in {path.name}")
    with h5py.File(path, "r") as handle:
        q = np.asarray(handle["q"], dtype=float)
        intensity = np.asarray(handle["intensity"], dtype=float)
    valid = np.isfinite(q) & np.isfinite(intensity)
    if not valid.any():
        raise ValueError(f"no finite q/intensity pairs in {path}")
    q = q[valid]
    intensity = intensity[valid]
    index = int(np.argmax(intensity))
    return {
        "x_mm": decode_position(match.group("x")),
        "y_mm": decode_position(match.group("y")),
        "q_peak_proxy": float(q[index]),
        "maximum_intensity": float(intensity[index]),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "q_points": len(q),
        "source_file": str(path.relative_to(root)),
    }


def build_catalogue(grid_root: Path) -> list[dict]:
    rows = []
    for path in sorted(grid_root.rglob("*_integration-azimuthal.h5")):
        rows.append(spectrum_row(path, grid_root))
    if not rows:
        raise ValueError(f"no reduced spectra found under {grid_root}")
    positions = [(row["x_mm"], row["y_mm"]) for row in rows]
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate motor coordinates found in grid scan")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict], path: Path) -> None:
    x = np.asarray([row["x_mm"] for row in rows])
    y = np.asarray([row["y_mm"] for row in rows])
    target = np.asarray([row["q_peak_proxy"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    image = ax.scatter(x, y, c=target, cmap="viridis", s=22, marker="s")
    fig.colorbar(image, ax=ax, label="q at maximum measured intensity (proxy)")
    ax.set_xlabel("X motor position (mm)")
    ax.set_ylabel("Y motor position (mm)")
    ax.set_title("ALS NaCl dense grid: provisional scalar response")
    ax.set_aspect("equal")
    fig.text(
        0.01, 0.01,
        "This uses the raw maximum of each reduced 1D profile. It is a pipeline smoke test, "
        "not yet the paper's fitted-peak scientific target.",
        fontsize=9, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_catalogue(args.grid_root)
    write_csv(args.output / "als_nacl_catalogue.csv", rows)
    make_plot(rows, args.output / "als_nacl_q_peak_proxy.png")
    x_values = sorted({row["x_mm"] for row in rows})
    y_values = sorted({row["y_mm"] for row in rows})
    print(
        f"rows={len(rows)} unique_x={len(x_values)} unique_y={len(y_values)} "
        f"cartesian_grid={len(rows) == len(x_values) * len(y_values)}"
    )


if __name__ == "__main__":
    main()
