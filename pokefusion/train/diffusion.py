"""The forward noising equation and reverse DDPM sampler, written explicitly."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from pokefusion.train.model import ModelConfig


class Diffusion:
    """Linear-beta denoising diffusion process.

    Training picks a random time t and constructs
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.
    The network learns to predict epsilon. Sampling starts at Gaussian x_T and
    repeatedly applies ``reverse_step`` until x_0 remains.
    """

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        self.cfg = config
        self.device = device
        self.beta = torch.linspace(
            config.beta_start, config.beta_end, config.timesteps, device=device
        )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        retained_signal = self.alpha_bar[timestep][:, None, None]
        noisy = retained_signal.sqrt() * clean + (1.0 - retained_signal).sqrt() * noise
        return noisy, noise

    # Backward-compatible name used by the original checkpoint training code.
    noisy = add_noise

    @torch.no_grad()
    def reverse_step(
        self,
        model: nn.Module,
        points: torch.Tensor,
        labels: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        timestep = torch.full((len(labels),), step, device=self.device, dtype=torch.long)
        predicted_noise = model(points, timestep, labels)
        alpha = self.alpha[step]
        alpha_bar = self.alpha_bar[step]
        mean = (
            points - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()
        ) / alpha.sqrt()
        if step == 0:
            return mean
        previous_alpha_bar = self.alpha_bar[step - 1]
        variance = self.beta[step] * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
        return mean + variance.sqrt() * torch.randn_like(points)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        points = torch.randn(len(labels), self.cfg.n_points, 2, device=self.device)
        if on_step:
            on_step(0, points)
        for step in reversed(range(self.cfg.timesteps)):
            points = self.reverse_step(model, points, labels, step)
            completed = self.cfg.timesteps - step
            if on_step:
                on_step(completed, points)
        return points
