#!/usr/bin/env python3
"""Provide shared deterministic math and I/O for the Pokémon data pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import DictConfig, OmegaConf

STAT_NAMES = ("mean_x", "mean_y", "std_x", "std_y", "corr")


def load_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True, throw_on_missing=True)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return config


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json_bytes(parts)).digest()
    return int.from_bytes(digest[:8], "big")


def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError(f"expected a finite N x 2 array with N >= 3, got {x.shape}")
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    std = np.sqrt(np.diag(covariance))
    if np.any(std <= 0.0):
        raise ValueError("rank-deficient point cloud")
    corr = covariance[0, 1] / (std[0] * std[1])
    return mean, covariance, np.array([mean[0], mean[1], std[0], std[1], corr])


def symmetric_power(matrix: np.ndarray, power: float, epsilon: float = 1e-12) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(values.min()) <= epsilon * scale:
        raise ValueError(f"rank-deficient covariance eigenvalues: {values.tolist()}")
    powered = np.maximum(values, epsilon) ** power
    return (vectors * powered) @ vectors.T


def project_moments(x: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray) -> np.ndarray:
    """Apply the analytic sample-moment projection, followed by a precision pass."""
    mean, covariance, _ = moments(x)
    color = symmetric_power(target_cov, 0.5)
    projected = (x - mean) @ symmetric_power(covariance, -0.5) @ color + target_mean
    mean2, covariance2, _ = moments(projected)
    return (projected - mean2) @ symmetric_power(covariance2, -0.5) @ color + target_mean


def read_points(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim == 2 and len(values) and not np.isfinite(values[0, :2]).all():
        values = values[1:]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected two CSV columns")
    values = values[:, :2]
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid or non-finite point cloud")
    return values


def write_points(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float64), delimiter=",", fmt="%.10f")


def read_contours(path: Path) -> list[np.ndarray]:
    groups: dict[int, list[tuple[int, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["contour_id"]), []).append(
                (int(row["order"]), float(row["x"]), float(row["y"]))
            )
    result = []
    for contour_id in sorted(groups):
        rows = sorted(groups[contour_id])
        result.append(np.asarray([(x, y) for _, x, y in rows], dtype=np.float64))
    if not result or any(len(contour) < 3 for contour in result):
        raise ValueError(f"{path}: no valid closed contours")
    return result


def write_contours(path: Path, contours: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("contour_id", "order", "x", "y", "closed"))
        for contour_id, contour in enumerate(contours):
            for order, (x, y) in enumerate(np.asarray(contour, dtype=np.float64)):
                writer.writerow((contour_id, order, f"{x:.10f}", f"{y:.10f}", "true"))


def stats_dict(stats: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(STAT_NAMES, stats)}


def repository_revision(root: Path) -> str:
    """Hash the exact generator implementation instead of relying on dirty Git state."""
    paths = [
        root / "pokefusion/data/acquire_pokemon.py",
        root / "pokefusion/data/extract_contour.py",
        root / "pokefusion/data/generate_pokemon_dataset.py",
        root / "pokefusion/data/pokemon_common.py",
        root / "pokefusion/omega_config.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
