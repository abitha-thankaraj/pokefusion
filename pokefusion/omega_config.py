"""Tiny OmegaConf command-line convention shared by every Pokémon script.

Usage:
    python -m pokefusion.<module> CONFIG.yaml key=value nested.key=value

The first argument is always a checked-in YAML file. Remaining arguments are
OmegaConf dot-list overrides, so an experiment can be reproduced by saving the
YAML plus the exact override list—without maintaining a second argument parser.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_cli_config(description: str) -> tuple[DictConfig, Path]:
    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        command = Path(sys.argv[0]).name
        print(
            f"{description}\n\n"
            f"usage: python {command} CONFIG.yaml [key=value ...]\n"
            "example override: nested.option=value"
        )
        raise SystemExit(0)
    if len(sys.argv) < 2:
        raise SystemExit("missing YAML config; run with --help for usage")

    path = Path(sys.argv[1])
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        raise SystemExit(f"first argument must be an existing YAML config, got: {path}")

    base = OmegaConf.load(path)
    overrides = OmegaConf.from_dotlist(sys.argv[2:])
    config = OmegaConf.merge(base, overrides)
    OmegaConf.resolve(config)
    return config, path


def as_plain_dict(config: DictConfig) -> dict:
    """Resolve interpolations and return ordinary containers for JSON/hash use."""
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
