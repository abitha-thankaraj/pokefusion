#!/usr/bin/env python3
"""Single-file, class-conditional DDPM for fixed-size 2D Pokémon point clouds.

Data layout:
    data/pokemon/points/<species>/*.csv

Every CSV must contain the same N x 2 coordinates and the same mean_x, mean_y,
sample std_x, sample std_y, and Pearson correlation within --stats-tol.

Dependencies: Python 3.10+, numpy, torch, matplotlib.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # Give a useful error even for `--help`/`check`.
    torch = None
    nn = None
    DataLoader = Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def read_xy_csv(path: Path) -> np.ndarray:
    """Read headerless or x/y-header CSV and return float64 N x 2."""
    try:
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns, got {arr.shape}")
    arr = arr[:, :2]
    if len(arr) and not np.isfinite(arr[0]).all():  # optional x,y header
        arr = arr[1:]
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError(f"{path}: expected N x 2 with N >= 3, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    return arr


def moments_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean (2,), sample covariance (2,2), and five-stat vector."""
    mean = x.mean(axis=0)
    centered = x - mean
    cov = centered.T @ centered / (len(x) - 1)
    std = np.sqrt(np.diag(cov))
    if np.any(std <= 0):
        raise ValueError("zero-variance point cloud")
    corr = cov[0, 1] / (std[0] * std[1])
    stats = np.array([mean[0], mean[1], std[0], std[1], corr])
    return mean, cov, stats


def symmetric_power(matrix: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix.astype(np.float64))
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(values.min()) <= eps * scale:
        raise ValueError(f"rank-deficient covariance eigenvalues: {values.tolist()}")
    values = np.maximum(values, eps) ** power
    return (vectors * values) @ vectors.T


def project_moments_np(x: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray) -> np.ndarray:
    """Project N x 2 rows to exact target sample mean/covariance in float64."""
    mean, cov, _ = moments_np(x)
    whiten = symmetric_power(cov, -0.5)
    color = symmetric_power(target_cov, 0.5)
    projected = (x - mean) @ whiten @ color + target_mean
    # A second pass suppresses accumulated eigendecomposition error.
    mean2, cov2, _ = moments_np(projected)
    return (projected - mean2) @ symmetric_power(cov2, -0.5) @ color + target_mean


def discover_data(data_dir: Path) -> tuple[list[Path], list[str], dict[str, int]]:
    paths = sorted(p for p in data_dir.rglob("*.csv") if p.is_file())
    if not paths:
        raise ValueError(f"no CSVs found under {data_dir}/<species>/*.csv")
    classes = sorted({p.parent.name for p in paths})
    class_to_id = {name: i for i, name in enumerate(classes)}
    return paths, classes, class_to_id


def validate_data(
    paths: Iterable[Path],
    stats_tol: float,
    expected_points: int | None = None,
    target_csv: Path | None = None,
) -> tuple[int, np.ndarray, np.ndarray, dict]:
    paths = list(paths)
    first = read_xy_csv(paths[0])
    n_points = expected_points or len(first)
    if len(first) != n_points:
        raise ValueError(f"{paths[0]}: expected {n_points} rows, got {len(first)}")
    target_path = target_csv if target_csv is not None else paths[0]
    target = read_xy_csv(target_path)
    if len(target) != n_points:
        raise ValueError(f"{target_path}: target has {len(target)} rows; expected {n_points}")
    target_mean, target_cov, target_stats = moments_np(target)
    maxima = np.zeros(5, dtype=np.float64)
    counts: dict[str, int] = {}
    for path in paths:
        x = read_xy_csv(path)
        if len(x) != n_points:
            raise ValueError(f"{path}: expected {n_points} rows, got {len(x)}")
        _, _, stats = moments_np(x)
        error = np.abs(stats - target_stats)
        maxima = np.maximum(maxima, error)
        if error.max() > stats_tol:
            names = ("mean_x", "mean_y", "std_x", "std_y", "corr")
            detail = ", ".join(f"{k}={v:.3g}" for k, v in zip(names, error))
            raise ValueError(f"{path}: invariant error exceeds {stats_tol:g}: {detail}")
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
    report = {
        "files": len(paths),
        "points_per_file": n_points,
        "counts": counts,
        "target_source": str(target_path),
        "target_stats": dict(zip(("mean_x", "mean_y", "std_x", "std_y", "corr"), target_stats.tolist())),
        "max_abs_error": dict(zip(("mean_x", "mean_y", "std_x", "std_y", "corr"), maxima.tolist())),
    }
    return n_points, target_mean, target_cov, report


def stratified_split(paths: list[Path], seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent.name, []).append(path)
    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []
    for name in sorted(groups):
        group = sorted(groups[name])
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            raise ValueError(f"{name}: need at least 3 examples for train/val/test, got {n}")
        n_val = max(1, round(0.1 * n))
        n_test = max(1, round(0.1 * n))
        train.extend(group[: n - n_val - n_test])
        val.extend(group[n - n_val - n_test : n - n_test])
        test.extend(group[n - n_test :])
    return sorted(train), sorted(val), sorted(test)


if torch is not None:

    class PointCloudDataset(Dataset):
        def __init__(
            self,
            paths: list[Path],
            class_to_id: dict[str, int],
            target_mean: np.ndarray,
            target_cov: np.ndarray,
            permute: bool,
        ) -> None:
            self.paths = paths
            self.class_to_id = class_to_id
            self.permute = permute
            inv_sqrt = symmetric_power(target_cov, -0.5)
            self.clouds = [
                torch.from_numpy(((read_xy_csv(p) - target_mean) @ inv_sqrt).astype(np.float32))
                for p in paths
            ]
            self.labels = [class_to_id[p.parent.name] for p in paths]

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.clouds[index]
            if self.permute:
                x = x[torch.randperm(len(x))]
            return x, torch.tensor(self.labels[index], dtype=torch.long)


    class SinusoidalTime(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.width = width

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.width // 2
            scale = math.log(10000) / max(half - 1, 1)
            frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
            angles = t.float()[:, None] * frequencies[None, :]
            emb = torch.cat((angles.sin(), angles.cos()), dim=1)
            return nn.functional.pad(emb, (0, self.width - emb.shape[1]))


    class PointDenoiser(nn.Module):
        """Permutation-equivariant: no index/position embedding is added."""

        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.input = nn.Linear(2, cfg.width)
            self.time = nn.Sequential(
                SinusoidalTime(cfg.width),
                nn.Linear(cfg.width, cfg.width * 2),
                nn.SiLU(),
                nn.Linear(cfg.width * 2, cfg.width),
            )
            self.label = nn.Embedding(cfg.n_classes, cfg.width)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.width,
                nhead=cfg.heads,
                dim_feedforward=cfg.width * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.body = nn.TransformerEncoder(layer, num_layers=cfg.layers)
            self.output = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 2))

        def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            condition = self.time(t) + self.label(label)
            hidden = self.input(x) + condition[:, None, :]
            return self.output(self.body(hidden))


    class Diffusion:
        def __init__(self, cfg: ModelConfig, device: torch.device) -> None:
            self.cfg = cfg
            self.device = device
            beta = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, device=device)
            self.beta = beta
            self.alpha = 1.0 - beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        def noisy(self, clean: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            noise = torch.randn_like(clean)
            a = self.alpha_bar[t][:, None, None]
            return a.sqrt() * clean + (1.0 - a).sqrt() * noise, noise

        @torch.no_grad()
        def sample(self, model: nn.Module, labels: torch.Tensor) -> torch.Tensor:
            x = torch.randn(len(labels), self.cfg.n_points, 2, device=self.device)
            for step in reversed(range(self.cfg.timesteps)):
                t = torch.full((len(labels),), step, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[step]
                alpha_bar = self.alpha_bar[step]
                mean = (x - (1.0 - alpha) * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
                if step:
                    previous_alpha_bar = self.alpha_bar[step - 1]
                    posterior_variance = self.beta[step] * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
                    x = mean + posterior_variance.sqrt() * torch.randn_like(x)
                else:
                    x = mean
            return x


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required for train/sample. Install numpy torch matplotlib. "
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )


def choose_device(requested: str) -> "torch.device":
    require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def update_ema(ema: "nn.Module", model: "nn.Module", decay: float) -> None:
    with torch.no_grad():
        for dst, src in zip(ema.parameters(), model.parameters()):
            dst.mul_(decay).add_(src, alpha=1.0 - decay)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def validation_loss(model, diffusion, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for clean, labels in loader:
        clean, labels = clean.to(device), labels.to(device)
        t = torch.randint(0, diffusion.cfg.timesteps, (len(clean),), device=device)
        noisy, noise = diffusion.noisy(clean, t)
        loss = nn.functional.mse_loss(model(noisy, t, labels), noise, reduction="sum")
        total += loss.item()
        count += noise.numel()
    return total / max(count, 1)


def cmd_check(args: argparse.Namespace) -> None:
    paths, classes, _ = discover_data(args.data_dir)
    _, _, _, report = validate_data(
        paths, args.stats_tol, args.expected_points, args.target_csv
    )
    report["classes"] = classes
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_train(args: argparse.Namespace) -> None:
    require_torch()
    seed_everything(args.seed)
    device = choose_device(args.device)
    paths, classes, class_to_id = discover_data(args.data_dir)
    n_points, target_mean, target_cov, report = validate_data(
        paths, args.stats_tol, args.expected_points, args.target_csv
    )
    train_paths, val_paths, test_paths = stratified_split(paths, args.seed)
    cfg = ModelConfig(
        n_points=n_points,
        n_classes=len(classes),
        timesteps=args.timesteps,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
    )
    train_ds = PointCloudDataset(train_paths, class_to_id, target_mean, target_cov, permute=True)
    val_ds = PointCloudDataset(val_paths, class_to_id, target_mean, target_cov, permute=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.workers, generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = PointDenoiser(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterator = iter(train_loader)
    history: list[dict] = []
    args.out.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        try:
            clean, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            clean, labels = next(iterator)
        clean, labels = clean.to(device), labels.to(device)
        t = torch.randint(0, cfg.timesteps, (len(clean),), device=device)
        noisy, noise = diffusion.noisy(clean, t)
        loss = nn.functional.mse_loss(model(noisy, t, labels), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, model, args.ema_decay)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val = validation_loss(ema, diffusion, val_loader, device)
            row = {"step": step, "train_loss": float(loss.item()), "val_loss": float(val)}
            history.append(row)
            print(json.dumps(row), flush=True)

    checkpoint = {
        "schema_version": 1,
        "model_config": asdict(cfg),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "classes": classes,
        "class_to_id": class_to_id,
        "target_mean": target_mean.tolist(),
        "target_cov": target_cov.tolist(),
        "data_report": report,
        "split": {
            "seed": args.seed,
            "train": [str(p) for p in train_paths],
            "val": [str(p) for p in val_paths],
            "test": [str(p) for p in test_paths],
        },
        "history": history,
        "training": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "ema_decay": args.ema_decay,
            "device": str(device),
        },
    }
    torch.save(checkpoint, args.out / "checkpoint.pt")
    (args.out / "training_metrics.json").write_text(json.dumps({"history": history, "data": report}, indent=2))
    print(f"saved {args.out / 'checkpoint.pt'}")


def chamfer_np(a: np.ndarray, b: np.ndarray) -> float:
    distances = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).mean() + distances.min(axis=0).mean())


def nearest_reference_chamfer(
    generated: dict[str, list[np.ndarray]], reference_paths: Iterable[str]
) -> dict[str, float] | None:
    references: dict[str, list[np.ndarray]] = {}
    for raw_path in reference_paths:
        path = Path(raw_path)
        if not path.is_file():
            return None
        references.setdefault(path.parent.name, []).append(read_xy_csv(path))
    result: dict[str, float] = {}
    for name, clouds in generated.items():
        candidates = references.get(name, [])
        if not candidates:
            return None
        nearest = [min(chamfer_np(cloud, ref) for ref in candidates) for cloud in clouds]
        result[name] = float(np.mean(nearest))
    return result


def reference_distance_summary(
    generated: dict[str, list[np.ndarray]], reference_paths: Iterable[str]
) -> dict[str, dict[str, float | int]] | None:
    references: dict[str, list[np.ndarray]] = {}
    for raw_path in reference_paths:
        path = Path(raw_path)
        if not path.is_file():
            return None
        references.setdefault(path.parent.name, []).append(read_xy_csv(path))
    result: dict[str, dict[str, float | int]] = {}
    for name, clouds in generated.items():
        candidates = references.get(name, [])
        if not candidates:
            return None
        nearest = np.asarray(
            [min(chamfer_np(cloud, reference) for reference in candidates) for cloud in clouds]
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
    references: dict[str, list[np.ndarray]] = {}
    for raw_path in reference_paths:
        path = Path(raw_path)
        if not path.is_file():
            return None
        references.setdefault(path.parent.name, []).append(read_xy_csv(path))
    if set(references) != set(generated):
        return None
    confusion = {actual: {predicted: 0 for predicted in generated} for actual in generated}
    correct = 0
    total = 0
    for actual, clouds in generated.items():
        for cloud in clouds:
            scores = {
                name: min(chamfer_np(cloud, reference) for reference in candidates)
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


def save_preview(samples: dict[str, list[np.ndarray]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = len(samples)
    cols = max(len(v) for v in samples.values())
    all_points = np.concatenate([cloud for clouds in samples.values() for cloud in clouds])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.04, 1e-6)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.3 * cols, 2.3 * rows), squeeze=False, layout="constrained"
    )
    for row, (name, clouds) in enumerate(samples.items()):
        for col in range(cols):
            ax = axes[row][col]
            if col < len(clouds):
                cloud = clouds[col]
                ax.scatter(cloud[:, 0], cloud[:, 1], s=5, c="black")
            if col == 0:
                ax.set_ylabel(name)
            ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_blinded_preview(
    samples: dict[str, list[np.ndarray]], path: Path, key_path: Path, count: int, seed: int
) -> None:
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    available = sum(map(len, samples.values()))
    requested = min(count, available)
    names = sorted(samples)
    base, remainder = divmod(requested, len(names))
    selected = []
    for class_index, name in enumerate(names):
        candidates = [(name, index, cloud) for index, cloud in enumerate(samples[name])]
        rng.shuffle(candidates)
        quota = base + int(class_index < remainder)
        selected.extend(candidates[:quota])
    rng.shuffle(selected)
    columns = min(5, max(1, len(selected)))
    rows = math.ceil(len(selected) / columns)
    all_points = np.concatenate([cloud for _, _, cloud in selected])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.04, 1e-6)
    fig, axes = plt.subplots(
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
            key.append({"sample": slot + 1, "class": name, "class_sample_index": source_index})
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        if slot >= len(selected):
            axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    key_path.write_text(json.dumps({"seed": seed, "samples": key}, indent=2) + "\n")


def cmd_sample(args: argparse.Namespace) -> None:
    require_torch()
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**checkpoint["model_config"])
    classes: list[str] = checkpoint["classes"]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
    target_cov = np.asarray(checkpoint["target_cov"], dtype=np.float64)
    model = PointDenoiser(cfg).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    diffusion = Diffusion(cfg, device)
    labels = torch.arange(len(classes), device=device).repeat_interleave(args.per_class)
    normalized = diffusion.sample(model, labels).cpu().numpy().astype(np.float64)
    color = symmetric_power(target_cov, 0.5)
    world = normalized @ color + target_mean
    args.out.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[np.ndarray]] = {name: [] for name in classes}
    stats_errors: list[list[float]] = []
    for index, cloud in enumerate(world):
        class_id = int(labels[index].item())
        name = classes[class_id]
        cloud = project_moments_np(cloud, target_mean, target_cov)
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        sample_path = folder / f"sample_{len(grouped[name]):03d}.csv"
        np.savetxt(sample_path, cloud, delimiter=",", fmt="%.10f")
        serialized = read_xy_csv(sample_path)
        grouped[name].append(serialized)
        _, _, stats = moments_np(serialized)
        target_std = np.sqrt(np.diag(target_cov))
        target_corr = target_cov[0, 1] / np.prod(target_std)
        target_stats = np.r_[target_mean, target_std, target_corr]
        error = np.abs(stats - target_stats)
        if float(error.max()) > args.stats_tol:
            raise ValueError(
                f"{sample_path}: serialized sample invariant error {error.max():.3g} "
                f"exceeds {args.stats_tol:g}"
            )
        stats_errors.append(error.tolist())

    diversity = {}
    for name, clouds in grouped.items():
        pairs = [chamfer_np(clouds[i], clouds[j]) for i in range(len(clouds)) for j in range(i)]
        diversity[name] = float(np.mean(pairs)) if pairs else None
    split = checkpoint.get("split", {})
    nearest_test = nearest_reference_chamfer(grouped, split.get("test", []))
    nearest_train = nearest_reference_chamfer(grouped, split.get("train", []))
    train_match_summary = reference_distance_summary(grouped, split.get("train", []))
    heldout_classification = reference_classification(grouped, split.get("test", []))
    maxima = np.max(np.asarray(stats_errors), axis=0)
    metrics = {
        "classes": classes,
        "samples_per_class": args.per_class,
        "serialized_stats_tolerance": args.stats_tol,
        "max_abs_stats_error": dict(zip(("mean_x", "mean_y", "std_x", "std_y", "corr"), maxima.tolist())),
        "within_class_pairwise_chamfer": diversity,
        "nearest_test_chamfer": nearest_test,
        "nearest_train_chamfer": nearest_train,
        "training_reference_match_summary": train_match_summary,
        "nearest_reference_classification": heldout_classification,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_preview(grouped, args.out / "preview.png")
    save_blinded_preview(
        grouped,
        args.out / "blinded_preview.png",
        args.out / "blind_key.json",
        args.blind_count,
        args.seed,
    )
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate point count and all five invariants")
    check.add_argument("--data-dir", type=Path, required=True)
    check.add_argument("--stats-tol", type=float, default=1e-4)
    check.add_argument("--expected-points", type=int, default=142)
    check.add_argument(
        "--target-csv", type=Path, default=Path("data/seed_datasets/Datasaurus_data.csv")
    )
    check.set_defaults(func=cmd_check)

    train = sub.add_parser("train", help="train the class-conditional DDPM")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--steps", type=int, default=30_000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--timesteps", type=int, default=200)
    train.add_argument("--width", type=int, default=128)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--ema-decay", type=float, default=0.999)
    train.add_argument("--log-every", type=int, default=500)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--stats-tol", type=float, default=1e-4)
    train.add_argument("--expected-points", type=int, default=142)
    train.add_argument(
        "--target-csv", type=Path, default=Path("data/seed_datasets/Datasaurus_data.csv")
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.set_defaults(func=cmd_train)

    sample = sub.add_parser("sample", help="sample every class and enforce exact moments")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--per-class", type=int, default=8)
    sample.add_argument("--seed", type=int, default=123)
    sample.add_argument("--device", default="auto")
    sample.add_argument("--blind-count", type=int, default=25)
    sample.add_argument("--stats-tol", type=float, default=1e-4)
    sample.set_defaults(func=cmd_sample)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
