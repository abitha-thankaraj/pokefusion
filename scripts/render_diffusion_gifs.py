#!/usr/bin/env python3
"""Render one reverse-DDPM trajectory GIF per Pokémon class."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pokemon_point_diffusion as diffusion  # noqa: E402


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--frame-stride", type=positive_int, default=4)
    parser.add_argument("--fps", type=positive_int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    diffusion.seed_everything(args.seed)
    device = diffusion.choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = diffusion.ModelConfig(**checkpoint["model_config"])
    classes = checkpoint["classes"]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
    target_cov = np.asarray(checkpoint["target_cov"], dtype=np.float64)
    color = diffusion.symmetric_power(target_cov, 0.5)

    model = diffusion.PointDenoiser(cfg).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    process = diffusion.Diffusion(cfg, device)
    labels = torch.arange(len(classes), device=device)

    x = torch.randn(len(classes), cfg.n_points, 2, device=device)
    trajectory: list[tuple[int, np.ndarray]] = [(0, x.cpu().numpy().astype(np.float64))]
    with torch.no_grad():
        for step in reversed(range(cfg.timesteps)):
            timestep = torch.full((len(classes),), step, device=device, dtype=torch.long)
            predicted_noise = model(x, timestep, labels)
            alpha = process.alpha[step]
            alpha_bar = process.alpha_bar[step]
            mean = (
                x - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()
            ) / alpha.sqrt()
            if step:
                previous_alpha_bar = process.alpha_bar[step - 1]
                variance = process.beta[step] * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
                x = mean + variance.sqrt() * torch.randn_like(x)
            else:
                x = mean
            completed = cfg.timesteps - step
            if completed % args.frame_stride == 0 or completed == cfg.timesteps:
                trajectory.append((completed, x.cpu().numpy().astype(np.float64)))

    world_trajectory = [
        (completed, normalized @ color + target_mean) for completed, normalized in trajectory
    ]
    final_completed, final_clouds = world_trajectory[-1]
    final_clouds = np.stack(
        [diffusion.project_moments_np(cloud, target_mean, target_cov) for cloud in final_clouds]
    )
    world_trajectory[-1] = (final_completed, final_clouds)

    target_std = np.sqrt(np.diag(target_cov))
    x_limits = (target_mean[0] - 5.0 * target_std[0], target_mean[0] + 5.0 * target_std[0])
    y_limits = (target_mean[1] - 3.2 * target_std[1], target_mean[1] + 3.2 * target_std[1])
    args.out.mkdir(parents=True, exist_ok=True)

    for class_id, name in enumerate(classes):
        frames = world_trajectory + [world_trajectory[-1]] * args.fps
        fig, axis = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
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
            title.set_text(f"{name} · denoising {completed}/{cfg.timesteps}")
            return scatter, title

        movie = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=1000 / args.fps, blit=True
        )
        movie.save(args.out / f"{name}_denoising.gif", writer=animation.PillowWriter(fps=args.fps))
        plt.close(fig)


if __name__ == "__main__":
    main()
