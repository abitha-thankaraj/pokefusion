"""The class-conditional, permutation-equivariant point denoiser."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    n_points: int
    n_classes: int
    timesteps: int = 200
    width: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class SinusoidalTime(nn.Module):
    """Encode a scalar diffusion timestep at several frequencies."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timestep.device) * -scale)
        angles = timestep.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return nn.functional.pad(embedding, (0, self.width - embedding.shape[1]))


class PointDenoiser(nn.Module):
    """Predict noise for every point, conditioned on timestep and class.

    There is intentionally no point-index positional embedding. Reordering the
    142 input rows therefore reorders the 142 outputs in exactly the same way:
    the model sees a set of points, not an ordered sequence.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cfg = config
        self.input = nn.Linear(2, config.width)
        self.time = nn.Sequential(
            SinusoidalTime(config.width),
            nn.Linear(config.width, config.width * 2),
            nn.SiLU(),
            nn.Linear(config.width * 2, config.width),
        )
        self.label = nn.Embedding(config.n_classes, config.width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 2))

    def forward(
        self, points: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        condition = self.time(timestep) + self.label(label)
        hidden = self.input(points) + condition[:, None, :]
        return self.output(self.body(hidden))
