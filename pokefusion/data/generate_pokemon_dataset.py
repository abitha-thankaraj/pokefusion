#!/usr/bin/env python3
"""Generate deterministic exact-moment Pokémon point clouds from artwork."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pokefusion.data.acquire_pokemon import acquire_all, load_cached_sources
from pokefusion.data.extract_contour import extract_contour, save_preview
from pokefusion.data.pokemon_common import (
    canonical_json_bytes,
    moments,
    project_moments,
    read_contours,
    read_points,
    repository_revision,
    sha256_bytes,
    sha256_file,
    stable_seed,
    stats_dict,
    write_contours,
    write_points,
)
from pokefusion.omega_config import as_plain_dict, load_cli_config


def _segments(contours: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = np.concatenate(contours)
    ends = np.concatenate([np.roll(contour, -1, axis=0) for contour in contours])
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("contour has zero-length segments")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return starts, ends, lengths, cumulative


def _normalize_contours(
    contours: list[np.ndarray], target_mean: np.ndarray, target_cov: np.ndarray
) -> list[np.ndarray]:
    sizes = [len(contour) for contour in contours]
    normalized = project_moments(np.concatenate(contours), target_mean, target_cov)
    offsets = np.cumsum([0, *sizes])
    return [normalized[offsets[i] : offsets[i + 1]] for i in range(len(sizes))]


def _sample_contour(
    contours: list[np.ndarray], count: int, jitter_std: float, rng: np.random.Generator
) -> np.ndarray:
    starts, ends, lengths, cumulative = _segments(contours)
    total = cumulative[-1]
    positions = (np.arange(count, dtype=np.float64) + rng.random(count)) * total / count
    rng.shuffle(positions)
    segment = np.minimum(np.searchsorted(cumulative, positions, side="right") - 1, len(lengths) - 1)
    fraction = (positions - cumulative[segment]) / lengths[segment]
    values = starts[segment] + fraction[:, None] * (ends[segment] - starts[segment])
    if jitter_std:
        values += rng.normal(0.0, jitter_std, size=values.shape)
    return values


def _shape_metrics(
    points: np.ndarray, contours: list[np.ndarray], coverage_bins: int
) -> dict[str, float]:
    starts, ends, lengths, cumulative = _segments(contours)
    direction = ends - starts
    denominator = np.sum(direction * direction, axis=1)
    delta = points[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(delta * direction[None, :, :], axis=2) / denominator[None, :], 0.0, 1.0)
    projected = starts[None, :, :] + t[..., None] * direction[None, :, :]
    squared = np.sum((points[:, None, :] - projected) ** 2, axis=2)
    nearest = np.argmin(squared, axis=1)
    distance = np.sqrt(squared[np.arange(len(points)), nearest])
    arc = cumulative[nearest] + t[np.arange(len(points)), nearest] * lengths[nearest]
    bins = np.minimum((arc / cumulative[-1] * coverage_bins).astype(int), coverage_bins - 1)
    return {
        "mean_contour_distance": float(distance.mean()),
        "max_contour_distance": float(distance.max()),
        "coverage": float(len(np.unique(bins)) / coverage_bins),
    }


def _repo_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def generate_dataset(
    config: dict[str, Any],
    data_root: Path,
    samples_per_species: int | None = None,
    acquire: bool = True,
    refresh_sources: bool = False,
) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    extraction_config = config["extraction"]
    generation_config = config["generation"]
    sample_count = int(samples_per_species or generation_config["samples_per_species"])
    if acquire:
        sources = acquire_all(config, data_root, refresh_sources)
    else:
        sources = load_cached_sources(config, data_root)

    target_path = root / "data/seed_datasets/Datasaurus_data.csv"
    target = read_points(target_path)
    target_mean, target_cov, target_stats = moments(target)
    expected_points = int(generation_config["points_per_sample"])
    if len(target) != expected_points:
        raise ValueError(f"target dataset has {len(target)} rows, expected {expected_points}")

    config_hash = sha256_bytes(canonical_json_bytes(config))
    revision = repository_revision(root)
    manifest: list[dict[str, Any]] = []

    for source in sources:
        pokemon = source["pokemon"]
        name = pokemon["name"]
        source_path = root / source["local_path"]
        mask, contours, diagnostics = extract_contour(source_path, extraction_config)
        mask_path = data_root / "masks" / f"{name}.png"
        contour_path = data_root / "contours" / f"{name}.csv"
        preview_path = data_root / "previews" / f"{name}.png"
        diagnostics_path = data_root / "contours" / f"{name}.diagnostics.json"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        write_contours(contour_path, contours)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        save_preview(source_path, mask, contours, preview_path)
        contours = read_contours(contour_path)
        normalized_contours = _normalize_contours(contours, target_mean, target_cov)

        output_folder = data_root / "points" / name
        output_folder.mkdir(parents=True, exist_ok=True)
        for stale in output_folder.glob("sample_*.csv"):
            stale.unlink()

        accepted = 0
        attempt = 0
        maximum_attempts = sample_count * int(generation_config["maximum_attempts_per_sample"])
        while accepted < sample_count and attempt < maximum_attempts:
            seed = stable_seed(config["global_seed"], int(pokemon["id"]), accepted, attempt)
            rng = np.random.default_rng(seed)
            candidate = _sample_contour(
                normalized_contours,
                expected_points,
                float(generation_config["jitter_std"]),
                rng,
            )
            try:
                candidate = project_moments(candidate, target_mean, target_cov)
                _, _, measured = moments(candidate)
                errors = np.abs(measured - target_stats)
                shape = _shape_metrics(
                    candidate, normalized_contours, int(generation_config["coverage_bins"])
                )
                reason = None
                if float(errors.max()) > float(generation_config["stats_tolerance"]):
                    reason = "statistics tolerance exceeded"
                elif shape["mean_contour_distance"] > float(
                    generation_config["maximum_mean_contour_distance"]
                ):
                    reason = "shape distance threshold exceeded"
                elif shape["coverage"] < float(generation_config["minimum_coverage"]):
                    reason = "coverage threshold not met"
            except ValueError as exc:
                measured = np.full(5, np.nan)
                errors = np.full(5, np.inf)
                shape = {}
                reason = str(exc)

            base = {
                "schema_version": 1,
                "pokemon": {"id": int(pokemon["id"]), "name": name, "form": "default"},
                "source": {
                    "url": source["asset_url"],
                    "sha256": source["sha256"],
                    "sprite_field": source["sprite_field"],
                },
                "mask": {
                    "path": _repo_path(root, mask_path),
                    "sha256": sha256_file(mask_path),
                    "foreground_fraction": diagnostics["foreground_fraction"],
                },
                "contour": {
                    "path": _repo_path(root, contour_path),
                    "sha256": sha256_file(contour_path),
                    "segments": int(sum(map(len, contours))),
                },
                "target_stats": stats_dict(target_stats),
                "measured_stats": stats_dict(measured),
                "max_stats_error": float(errors.max()),
                "shape_metrics": shape,
                "config_sha256": config_hash,
                "code_revision": revision,
            }
            if reason is not None:
                manifest.append(
                    {
                        **base,
                        "status": "rejected",
                        "reason": reason,
                        "points": {"path": None, "count": expected_points, "seed": seed},
                    }
                )
                attempt += 1
                continue

            output_path = output_folder / f"sample_{accepted:03d}.csv"
            write_points(output_path, candidate)
            serialized = read_points(output_path)
            _, _, serialized_stats = moments(serialized)
            serialized_errors = np.abs(serialized_stats - target_stats)
            if float(serialized_errors.max()) > float(generation_config["stats_tolerance"]):
                raise RuntimeError(f"serialization violated target moments for {output_path}")
            manifest.append(
                {
                    **base,
                    "status": "accepted",
                    "measured_stats": stats_dict(serialized_stats),
                    "max_stats_error": float(serialized_errors.max()),
                    "points": {
                        "path": _repo_path(root, output_path),
                        "sha256": sha256_file(output_path),
                        "count": expected_points,
                        "seed": seed,
                    },
                }
            )
            accepted += 1
            attempt += 1
        if accepted != sample_count:
            raise RuntimeError(f"{name}: accepted {accepted}/{sample_count} after {attempt} attempts")

    manifest_path = data_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest))
    return manifest


def main() -> None:
    config, _ = load_cli_config(__doc__ or "Generate Pokémon point clouds")
    plain = as_plain_dict(config)
    data_root = Path(config.run.data_root)
    manifest = generate_dataset(
        plain,
        data_root,
        int(config.generation.samples_per_species),
        acquire=bool(config.run.acquire),
        refresh_sources=bool(config.run.refresh_sources),
    )
    accepted = [row for row in manifest if row["status"] == "accepted"]
    rejected = [row for row in manifest if row["status"] == "rejected"]
    print(
        json.dumps(
            {"accepted": len(accepted), "rejected": len(rejected), "manifest": str(data_root / "manifest.jsonl")},
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
