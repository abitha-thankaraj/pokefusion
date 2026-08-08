"""Generate, project, serialize, and evaluate samples from a trained checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pokefusion.omega_config import load_cli_config
from pokefusion.train.data import (
    STAT_NAMES,
    moments,
    project_moments,
    read_xy_csv,
    symmetric_matrix_power,
)
from pokefusion.train.diffusion import Diffusion
from pokefusion.train.evaluation import (
    chamfer_distance,
    nearest_reference_distance,
    reference_classification,
    save_blinded_preview,
    save_labeled_preview,
    training_match_summary,
)
from pokefusion.train.model import ModelConfig, PointDenoiser
from pokefusion.train.runtime import choose_device, seed_everything


def main() -> None:
    config, _ = load_cli_config(__doc__ or "Sample the DDPM")
    seed_everything(int(config.seed))
    device = choose_device(str(config.device))
    checkpoint = torch.load(Path(config.checkpoint), map_location=device, weights_only=False)

    model_config = ModelConfig(**checkpoint["model_config"])
    classes: list[str] = checkpoint["classes"]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
    target_covariance = np.asarray(checkpoint["target_cov"], dtype=np.float64)
    model = PointDenoiser(model_config).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()

    diffusion = Diffusion(model_config, device)
    labels = torch.arange(len(classes), device=device).repeat_interleave(
        int(config.samples_per_class)
    )
    normalized = diffusion.sample(model, labels).cpu().numpy().astype(np.float64)

    # Training clouds were whitened to unit covariance. Recolor the generated
    # clouds to Datasaurus coordinates, then project each one exactly because a
    # learned DDPM only approximates the target moments.
    color = symmetric_matrix_power(target_covariance, 0.5)
    world_clouds = normalized @ color + target_mean
    output = Path(config.out)
    output.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[np.ndarray]] = {name: [] for name in classes}
    stats_errors = []
    target_standard_deviation = np.sqrt(np.diag(target_covariance))
    target_correlation = target_covariance[0, 1] / np.prod(target_standard_deviation)
    target_statistics = np.r_[target_mean, target_standard_deviation, target_correlation]

    for index, cloud in enumerate(world_clouds):
        name = classes[int(labels[index].item())]
        cloud = project_moments(cloud, target_mean, target_covariance)
        folder = output / name
        folder.mkdir(exist_ok=True)
        sample_path = folder / f"sample_{len(grouped[name]):03d}.csv"
        np.savetxt(sample_path, cloud, delimiter=",", fmt="%.10f")

        # Validate the serialized bytes, not just the higher-precision array in memory.
        serialized = read_xy_csv(sample_path)
        grouped[name].append(serialized)
        _, _, statistics = moments(serialized)
        error = np.abs(statistics - target_statistics)
        if float(error.max()) > float(config.stats_tolerance):
            raise ValueError(
                f"{sample_path}: serialized invariant error {error.max():.3g} exceeds "
                f"{float(config.stats_tolerance):g}"
            )
        stats_errors.append(error)

    diversity = {}
    for name, clouds in grouped.items():
        pairs = [
            chamfer_distance(clouds[first], clouds[second])
            for first in range(len(clouds))
            for second in range(first)
        ]
        diversity[name] = float(np.mean(pairs)) if pairs else None

    split = checkpoint.get("split", {})
    maximum_errors = np.max(np.asarray(stats_errors), axis=0)
    metrics = {
        "classes": classes,
        "samples_per_class": int(config.samples_per_class),
        "serialized_stats_tolerance": float(config.stats_tolerance),
        "max_abs_stats_error": dict(zip(STAT_NAMES, maximum_errors.tolist())),
        "within_class_pairwise_chamfer": diversity,
        "nearest_test_chamfer": nearest_reference_distance(grouped, split.get("test", [])),
        "nearest_train_chamfer": nearest_reference_distance(grouped, split.get("train", [])),
        "training_reference_match_summary": training_match_summary(
            grouped, split.get("train", [])
        ),
        "nearest_reference_classification": reference_classification(
            grouped, split.get("test", [])
        ),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_labeled_preview(grouped, output / "preview.png")
    save_blinded_preview(
        grouped,
        output / "blinded_preview.png",
        output / "blind_key.json",
        int(config.blind_grid_samples),
        int(config.seed),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
