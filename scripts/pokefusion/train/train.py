"""Train the educational class-conditional point-cloud DDPM."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pokefusion.omega_config import as_plain_dict, load_cli_config
from pokefusion.train.data import (
    PointCloudDataset,
    discover_data,
    stratified_split,
    validate_data,
)
from pokefusion.train.diffusion import Diffusion
from pokefusion.train.model import ModelConfig, PointDenoiser
from pokefusion.train.runtime import choose_device, seed_everything, update_ema


@torch.no_grad()
def validation_loss(model, diffusion, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for clean, labels in loader:
        clean, labels = clean.to(device), labels.to(device)
        timestep = torch.randint(0, diffusion.cfg.timesteps, (len(clean),), device=device)
        noisy, noise = diffusion.add_noise(clean, timestep)
        total += nn.functional.mse_loss(
            model(noisy, timestep, labels), noise, reduction="sum"
        ).item()
        count += noise.numel()
    return total / max(count, 1)


def main() -> None:
    config, config_path = load_cli_config(__doc__ or "Train the DDPM")
    seed_everything(int(config.seed))
    device = choose_device(str(config.device))

    paths, classes, class_to_id = discover_data(Path(config.data_dir))
    n_points, target_mean, target_covariance, data_report = validate_data(
        paths,
        float(config.stats_tolerance),
        int(config.expected_points),
        Path(config.target_csv),
    )
    train_paths, validation_paths, test_paths = stratified_split(paths, int(config.seed))

    training = config.training
    model_config = ModelConfig(
        n_points=n_points,
        n_classes=len(classes),
        timesteps=int(training.diffusion_timesteps),
        width=int(training.model_width),
        layers=int(training.model_layers),
        heads=int(training.attention_heads),
    )
    train_dataset = PointCloudDataset(
        train_paths, class_to_id, target_mean, target_covariance, permute=True
    )
    validation_dataset = PointCloudDataset(
        validation_paths, class_to_id, target_mean, target_covariance, permute=False
    )
    generator = torch.Generator().manual_seed(int(config.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.batch_size),
        shuffle=True,
        num_workers=int(training.workers),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.batch_size),
        shuffle=False,
        num_workers=int(training.workers),
    )

    model = PointDenoiser(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(model_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.learning_rate), weight_decay=1e-4
    )
    iterator = iter(train_loader)
    history: list[dict] = []
    output = Path(config.out)
    output.mkdir(parents=True, exist_ok=True)

    # One DDPM update is deliberately written out here: sample a time, add the
    # exact amount of noise for that time, predict that noise, and regress with MSE.
    for step in range(1, int(training.steps) + 1):
        model.train()
        try:
            clean, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            clean, labels = next(iterator)
        clean, labels = clean.to(device), labels.to(device)
        timestep = torch.randint(0, model_config.timesteps, (len(clean),), device=device)
        noisy, target_noise = diffusion.add_noise(clean, timestep)
        predicted_noise = model(noisy, timestep, labels)
        loss = nn.functional.mse_loss(predicted_noise, target_noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, model, float(training.ema_decay))

        if step == 1 or step % int(training.log_every) == 0 or step == int(training.steps):
            validation = validation_loss(ema, diffusion, validation_loader, device)
            row = {"step": step, "train_loss": float(loss.item()), "val_loss": validation}
            history.append(row)
            print(json.dumps(row), flush=True)

    checkpoint = {
        "schema_version": 2,
        "model_config": asdict(model_config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "classes": classes,
        "class_to_id": class_to_id,
        "target_mean": target_mean.tolist(),
        "target_cov": target_covariance.tolist(),
        "data_report": data_report,
        "split": {
            "seed": int(config.seed),
            "train": [str(path) for path in train_paths],
            "val": [str(path) for path in validation_paths],
            "test": [str(path) for path in test_paths],
        },
        "history": history,
        "training": {
            **as_plain_dict(training),
            "seed": int(config.seed),
            "device": str(device),
            "config_path": str(config_path),
        },
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    (output / "training_metrics.json").write_text(
        json.dumps({"history": history, "data": data_report}, indent=2) + "\n"
    )
    print(f"saved {output / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
