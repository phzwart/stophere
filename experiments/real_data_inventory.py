"""Create a privacy-safe inventory and schema audit for the scattering HDF5 data."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def dataset_rows(path: Path, root: Path) -> tuple[list[dict], str | None]:
    rows: list[dict] = []
    try:
        with h5py.File(path, "r") as handle:
            def collect(name: str, obj) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                finite_fraction = "not_scanned"
                minimum = "not_scanned"
                maximum = "not_scanned"
                if np.issubdtype(obj.dtype, np.number) and obj.size <= 2_000_000:
                    values = np.asarray(obj[...])
                    finite = np.isfinite(values)
                    finite_fraction = float(finite.mean()) if values.size else 1.0
                    if finite.any():
                        minimum = float(np.nanmin(values))
                        maximum = float(np.nanmax(values))
                rows.append({
                    "file": str(path.relative_to(root)),
                    "dataset": name,
                    "shape": "x".join(str(value) for value in obj.shape),
                    "dtype": str(obj.dtype),
                    "elements": int(obj.size),
                    "finite_fraction": finite_fraction,
                    "minimum": minimum,
                    "maximum": maximum,
                })
            handle.visititems(collect)
    except Exception as exc:  # audit should report corrupt/unreadable files
        return rows, f"{type(exc).__name__}: {exc}"
    return rows, None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(data_root: Path, output: Path) -> dict:
    files = sorted(path for path in data_root.rglob("*") if path.is_file())
    extension_counts = Counter(path.suffix.lower() or "[no extension]" for path in files)
    h5_files = [path for path in files if path.suffix.lower() in {".h5", ".hdf5"}]
    datasets: list[dict] = []
    errors = []
    schema_counts: Counter[tuple] = Counter()
    for index, path in enumerate(h5_files, start=1):
        rows, error = dataset_rows(path, data_root)
        datasets.extend(rows)
        if error:
            errors.append({"file": str(path.relative_to(data_root)), "error": error})
        schema_counts[tuple((row["dataset"], row["shape"], row["dtype"]) for row in rows)] += 1
        if index % 500 == 0:
            print(f"scanned {index}/{len(h5_files)} HDF5 files")
    write_csv(output / "hdf5_datasets.csv", datasets)
    write_csv(output / "hdf5_errors.csv", errors)
    summary = {
        "data_root": str(data_root.resolve()),
        "file_count": len(files),
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "extension_counts": dict(sorted(extension_counts.items())),
        "hdf5_file_count": len(h5_files),
        "hdf5_dataset_count": len(datasets),
        "unreadable_hdf5_count": len(errors),
        "distinct_hdf5_schemas": len(schema_counts),
        "duplicate_relative_paths": len(files) - len({str(path.relative_to(data_root)) for path in files}),
        "numeric_datasets_with_nonfinite_values": sum(
            isinstance(row["finite_fraction"], float) and row["finite_fraction"] < 1.0
            for row in datasets
        ),
        "largest_files": [
            {"file": str(path.relative_to(data_root)), "bytes": path.stat().st_size}
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
        ],
        "most_common_schemas": [
            {"files": count, "datasets": list(schema)}
            for schema, count in schema_counts.most_common(20)
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.data_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
