#!/usr/bin/env python3
"""Validate Pokémon point counts, invariants, hashes, and manifest coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pokefusion.data.pokemon_common import (
    STAT_NAMES,
    moments,
    read_points,
    sha256_file,
    stats_dict,
)
from pokefusion.omega_config import as_plain_dict, load_cli_config


def _headerless_rows(path: Path) -> int:
    count = 0
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError(f"{path}: expected exactly two columns")
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                raise ValueError(f"{path}: header or non-numeric row found") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite row")
            count += 1
    return count


def validate(config: dict, data_root: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    generation = config["generation"]
    expected_points = int(generation["points_per_sample"])
    target_path = root / "data/seed_datasets/Datasaurus_data.csv"
    target_mean, target_cov, target_stats = moments(read_points(target_path))
    del target_mean, target_cov
    manifest_path = data_root / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    expected_species = sorted({row["pokemon"]["name"] for row in manifest})
    accepted = {row["points"]["path"]: row for row in manifest if row["status"] == "accepted"}
    maxima = np.zeros(5, dtype=np.float64)
    counts: dict[str, int] = {}
    checked: set[str] = set()

    for path in sorted((data_root / "points").glob("*/*.csv")):
        rows = _headerless_rows(path)
        if rows != expected_points:
            raise ValueError(f"{path}: expected {expected_points} rows, got {rows}")
        values = read_points(path)
        _, _, stats = moments(values)
        error = np.abs(stats - target_stats)
        maxima = np.maximum(maxima, error)
        if float(error.max()) > float(generation["stats_tolerance"]):
            raise ValueError(f"{path}: invariant error {dict(zip(STAT_NAMES, error.tolist()))}")
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        record = accepted.get(relative)
        if record is None:
            raise ValueError(f"{path}: missing accepted manifest entry")
        if record["points"]["sha256"] != sha256_file(path):
            raise ValueError(f"{path}: SHA-256 differs from manifest")
        contour = root / record["contour"]["path"]
        if record["contour"]["sha256"] != sha256_file(contour):
            raise ValueError(f"{contour}: SHA-256 differs from manifest")
        checked.add(relative)
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1

    if set(counts) != set(expected_species):
        raise ValueError(f"species mismatch: expected {expected_species}, got {sorted(counts)}")
    if checked != set(accepted):
        raise ValueError("manifest/file coverage mismatch")
    if len(set(counts.values())) != 1:
        raise ValueError(f"unbalanced species counts: {counts}")
    summary = {
        "schema_version": 1,
        "valid": True,
        "files": len(checked),
        "points_per_file": expected_points,
        "counts": counts,
        "target_stats": stats_dict(target_stats),
        "max_abs_error": stats_dict(maxima),
        "accepted_manifest_entries": len(accepted),
        "rejected_manifest_entries": sum(row["status"] == "rejected" for row in manifest),
    }
    return summary


def main() -> None:
    config, _ = load_cli_config(__doc__ or "Validate Pokémon point clouds")
    data_root = Path(config.run.data_root)
    summary = validate(as_plain_dict(config), data_root)
    summary_out = data_root / "validation_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
