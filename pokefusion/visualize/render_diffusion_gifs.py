"""Render one reverse-DDPM trajectory GIF per checkpoint class."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from pokefusion.omega_config import load_cli_config
from pokefusion.train.data import project_moments, symmetric_matrix_power
from pokefusion.train.diffusion import Diffusion
from pokefusion.train.model import ModelConfig, PointDenoiser
from pokefusion.train.runtime import choose_device, seed_everything


def main() -> None:
    config, _ = load_cli_config(__doc__ or "Render denoising GIFs")
    frame_stride = int(config.frame_stride)
    frames_per_second = int(config.fps)
    if frame_stride <= 0 or frames_per_second <= 0:
        raise ValueError("frame_stride and fps must be positive integers")

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
    process = Diffusion(model_config, device)
    labels = torch.arange(len(classes), device=device)

    trajectory: list[tuple[int, np.ndarray]] = []

    def record(completed: int, points: torch.Tensor) -> None:
        if completed % frame_stride == 0 or completed == model_config.timesteps:
            trajectory.append((completed, points.detach().cpu().numpy().astype(np.float64)))

    process.sample(model, labels, on_step=record)

    color = symmetric_matrix_power(target_covariance, 0.5)
    world_trajectory = [
        (completed, normalized @ color + target_mean) for completed, normalized in trajectory
    ]
    final_step, final_clouds = world_trajectory[-1]
    final_clouds = np.stack(
        [project_moments(cloud, target_mean, target_covariance) for cloud in final_clouds]
    )
    world_trajectory[-1] = (final_step, final_clouds)

    target_std = np.sqrt(np.diag(target_covariance))
    x_limits = (
        target_mean[0] - 5.0 * target_std[0],
        target_mean[0] + 5.0 * target_std[0],
    )
    y_limits = (
        target_mean[1] - 3.2 * target_std[1],
        target_mean[1] + 3.2 * target_std[1],
    )
    output = Path(config.out)
    output.mkdir(parents=True, exist_ok=True)

    for class_id, name in enumerate(classes):
        # Duplicate the last frame so viewers have one second to inspect it.
        frames = world_trajectory + [world_trajectory[-1]] * frames_per_second
        figure, axis = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
        scatter = axis.scatter([], [], s=12, c="black", linewidths=0)
        title = axis.set_title("")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

        def update(frame_index: int):
            completed, clouds = frames[frame_index]
            scatter.set_offsets(clouds[class_id])
            title.set_text(f"{name} · denoising {completed}/{model_config.timesteps}")
            return scatter, title

        movie = animation.FuncAnimation(
            figure,
            update,
            frames=len(frames),
            interval=1000 / frames_per_second,
            blit=True,
        )
        movie.save(
            output / f"{name}_denoising.gif",
            writer=animation.PillowWriter(fps=frames_per_second),
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
