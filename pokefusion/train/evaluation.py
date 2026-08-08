"""Shape metrics and matplotlib previews for generated point clouds."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from pokefusion.train.data import read_xy_csv


def chamfer_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric squared distance between two unordered point sets."""
    distances = ((first[:, None, :] - second[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).mean() + distances.min(axis=0).mean())


def _references(paths: Iterable[str]) -> dict[str, list[np.ndarray]] | None:
    grouped: dict[str, list[np.ndarray]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            return None
        grouped.setdefault(path.parent.name, []).append(read_xy_csv(path))
    return grouped


def nearest_reference_distance(
    generated: dict[str, list[np.ndarray]], reference_paths: Iterable[str]
) -> dict[str, float] | None:
    references = _references(reference_paths)
    if references is None:
        return None
    result = {}
    for name, clouds in generated.items():
        candidates = references.get(name, [])
        if not candidates:
            return None
        nearest = [
            min(chamfer_distance(cloud, reference) for reference in candidates)
            for cloud in clouds
        ]
        result[name] = float(np.mean(nearest))
    return result


def training_match_summary(
    generated: dict[str, list[np.ndarray]], reference_paths: Iterable[str]
) -> dict[str, dict[str, float | int]] | None:
    references = _references(reference_paths)
    if references is None:
        return None
    result = {}
    for name, clouds in generated.items():
        candidates = references.get(name, [])
        if not candidates:
            return None
        nearest = np.asarray(
            [
                min(chamfer_distance(cloud, reference) for reference in candidates)
                for cloud in clouds
            ]
        )
        result[name] = {
            "mean": float(nearest.mean()),
            "minimum": float(nearest.min()),
            "maximum": float(nearest.max()),
            "exact_matches": int(np.count_nonzero(nearest <= 1e-12)),
        }
    return result


def reference_classification(
    generated: dict[str, list[np.ndarray]], reference_paths: Iterable[str]
) -> dict | None:
    """Classify each sample by its nearest held-out class reference."""
    references = _references(reference_paths)
    if references is None or set(references) != set(generated):
        return None
    confusion = {actual: {predicted: 0 for predicted in generated} for actual in generated}
    correct = 0
    total = 0
    for actual, clouds in generated.items():
        for cloud in clouds:
            scores = {
                name: min(chamfer_distance(cloud, reference) for reference in candidates)
                for name, candidates in references.items()
            }
            predicted = min(scores, key=lambda name: (scores[name], name))
            confusion[actual][predicted] += 1
            correct += int(actual == predicted)
            total += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "confusion": confusion,
    }


def save_labeled_preview(samples: dict[str, list[np.ndarray]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = len(samples)
    columns = max(len(clouds) for clouds in samples.values())
    all_points = np.concatenate([cloud for clouds in samples.values() for cloud in clouds])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.04, 1e-6)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.3 * columns, 2.3 * rows),
        squeeze=False,
        layout="constrained",
    )
    for row, (name, clouds) in enumerate(samples.items()):
        for column in range(columns):
            axis = axes[row][column]
            if column < len(clouds):
                cloud = clouds[column]
                axis.scatter(cloud[:, 0], cloud[:, 1], s=5, c="black")
            if column == 0:
                axis.set_ylabel(name)
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_blinded_preview(
    samples: dict[str, list[np.ndarray]],
    path: Path,
    key_path: Path,
    count: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    requested = min(count, sum(map(len, samples.values())))
    names = sorted(samples)
    base, remainder = divmod(requested, len(names))
    selected = []
    for class_index, name in enumerate(names):
        candidates = [(name, index, cloud) for index, cloud in enumerate(samples[name])]
        rng.shuffle(candidates)
        selected.extend(candidates[: base + int(class_index < remainder)])
    rng.shuffle(selected)

    columns = min(5, max(1, len(selected)))
    rows = math.ceil(len(selected) / columns)
    all_points = np.concatenate([cloud for _, _, cloud in selected])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.04, 1e-6)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.4 * columns, 2.4 * rows),
        squeeze=False,
        layout="constrained",
    )
    key = []
    for slot, axis in enumerate(axes.ravel()):
        if slot < len(selected):
            name, source_index, cloud = selected[slot]
            axis.scatter(cloud[:, 0], cloud[:, 1], s=5, c="black")
            axis.set_title(f"sample {slot + 1:02d}")
            key.append(
                {"sample": slot + 1, "class": name, "class_sample_index": source_index}
            )
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        if slot >= len(selected):
            axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    key_path.write_text(json.dumps({"seed": seed, "samples": key}, indent=2) + "\n")
