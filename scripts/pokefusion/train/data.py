"""Load point clouds and explain/enforce the shared-statistics contract.

Every Pokémon must have the same mean and sample covariance. Equal means and
standard deviations alone are not enough: two clouds can still have different
tilt because Pearson correlation is the off-diagonal covariance term. Checking
all five values gives every class the same first- and second-order statistics,
while leaving higher-order geometry free to encode a recognizable silhouette.

Before diffusion, we *whiten* each cloud: subtract the shared mean and multiply
by covariance^(-1/2). This makes the training distribution zero-mean with unit
covariance, matching the isotropic Gaussian noise used by DDPM and preventing
the larger y variance from dominating the loss.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Data validation remains usable in lightweight CI.
    torch = None
    Dataset = object

STAT_NAMES = ("mean_x", "mean_y", "std_x", "std_y", "corr")


def read_xy_csv(path: Path) -> np.ndarray:
    """Read a CSV as a finite float64 array with shape (points, 2)."""
    try:
        values = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns, got {values.shape}")
    values = values[:, :2]
    if len(values) and not np.isfinite(values[0]).all():  # Allow an optional x,y header.
        values = values[1:]
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        raise ValueError(f"{path}: expected N x 2 with N >= 3, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    return values


def moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, sample covariance, and (mean_x, mean_y, std_x, std_y, corr)."""
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / (len(points) - 1)
    standard_deviation = np.sqrt(np.diag(covariance))
    if np.any(standard_deviation <= 0):
        raise ValueError("zero-variance point cloud")
    correlation = covariance[0, 1] / np.prod(standard_deviation)
    statistics = np.r_[mean, standard_deviation, correlation]
    return mean, covariance, statistics


def symmetric_matrix_power(
    matrix: np.ndarray, power: float, epsilon: float = 1e-12
) -> np.ndarray:
    """Raise a positive-definite symmetric matrix to a real power."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.astype(np.float64))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) <= epsilon * scale:
        raise ValueError(f"rank-deficient covariance eigenvalues: {eigenvalues.tolist()}")
    powered = np.maximum(eigenvalues, epsilon) ** power
    return (eigenvectors * powered) @ eigenvectors.T


def project_moments(
    points: np.ndarray, target_mean: np.ndarray, target_covariance: np.ndarray
) -> np.ndarray:
    """Force exact target mean/covariance without specifying the cloud's shape.

    Diffusion is approximate, so generated samples will be close to—but not
    exactly on—the required statistics. Whitening each generated cloud by its
    own covariance and recoloring it with the target covariance enforces the
    contract analytically. A second pass removes numerical eigensolver drift.
    """
    mean, covariance, _ = moments(points)
    whiten = symmetric_matrix_power(covariance, -0.5)
    color = symmetric_matrix_power(target_covariance, 0.5)
    projected = (points - mean) @ whiten @ color + target_mean
    mean, covariance, _ = moments(projected)
    return (
        (projected - mean) @ symmetric_matrix_power(covariance, -0.5) @ color
        + target_mean
    )


def discover_data(data_dir: Path) -> tuple[list[Path], list[str], dict[str, int]]:
    paths = sorted(path for path in data_dir.rglob("*.csv") if path.is_file())
    if not paths:
        raise ValueError(f"no CSVs found under {data_dir}/<species>/*.csv")
    classes = sorted({path.parent.name for path in paths})
    return paths, classes, {name: index for index, name in enumerate(classes)}


def validate_data(
    paths: Iterable[Path],
    stats_tolerance: float,
    expected_points: int,
    target_csv: Path,
) -> tuple[int, np.ndarray, np.ndarray, dict]:
    """Reject mixed point counts and any violation of the five shared statistics."""
    paths = list(paths)
    target = read_xy_csv(target_csv)
    if len(target) != expected_points:
        raise ValueError(f"{target_csv}: target has {len(target)} rows; expected {expected_points}")
    target_mean, target_covariance, target_statistics = moments(target)
    maximum_errors = np.zeros(5, dtype=np.float64)
    counts: dict[str, int] = {}

    for path in paths:
        points = read_xy_csv(path)
        if len(points) != expected_points:
            raise ValueError(f"{path}: expected {expected_points} rows, got {len(points)}")
        _, _, statistics = moments(points)
        error = np.abs(statistics - target_statistics)
        maximum_errors = np.maximum(maximum_errors, error)
        if float(error.max()) > stats_tolerance:
            detail = ", ".join(f"{name}={value:.3g}" for name, value in zip(STAT_NAMES, error))
            raise ValueError(
                f"{path}: invariant error exceeds {stats_tolerance:g}: {detail}"
            )
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1

    report = {
        "files": len(paths),
        "points_per_file": expected_points,
        "counts": counts,
        "target_source": str(target_csv),
        "target_stats": dict(zip(STAT_NAMES, target_statistics.tolist())),
        "max_abs_error": dict(zip(STAT_NAMES, maximum_errors.tolist())),
    }
    return expected_points, target_mean, target_covariance, report


def stratified_split(paths: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Create a deterministic 80/10/10 split within every class."""
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent.name, []).append(path)

    train: list[Path] = []
    validation: list[Path] = []
    test: list[Path] = []
    for name in sorted(groups):
        group = sorted(groups[name])
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f"{name}: need at least 3 examples, got {len(group)}")
        validation_count = max(1, round(0.1 * len(group)))
        test_count = max(1, round(0.1 * len(group)))
        train_end = len(group) - validation_count - test_count
        train.extend(group[:train_end])
        validation.extend(group[train_end : len(group) - test_count])
        test.extend(group[len(group) - test_count :])
    return sorted(train), sorted(validation), sorted(test)


class PointCloudDataset(Dataset):
    """In-memory whitened clouds; training access randomly permutes point rows."""

    def __init__(
        self,
        paths: list[Path],
        class_to_id: dict[str, int],
        target_mean: np.ndarray,
        target_covariance: np.ndarray,
        permute: bool,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to construct PointCloudDataset")
        inverse_square_root = symmetric_matrix_power(target_covariance, -0.5)
        self.paths = paths
        self.permute = permute
        self.clouds = [
            torch.from_numpy(
                ((read_xy_csv(path) - target_mean) @ inverse_square_root).astype(np.float32)
            )
            for path in paths
        ]
        self.labels = [class_to_id[path.parent.name] for path in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        points = self.clouds[index]
        if self.permute:
            points = points[torch.randperm(len(points))]
        return points, torch.tensor(self.labels[index], dtype=torch.long)
