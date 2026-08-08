"""Small runtime helpers shared by training, sampling, and visualization."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """Move evaluation weights slowly toward the current training weights."""
    with torch.no_grad():
        for destination, source in zip(ema.parameters(), model.parameters()):
            destination.mul_(decay).add_(source, alpha=1.0 - decay)
