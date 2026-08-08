"""Validate every point cloud before training."""

from __future__ import annotations

import json
from pathlib import Path

from pokefusion.omega_config import load_cli_config
from pokefusion.train.data import discover_data, validate_data


def main() -> None:
    config, _ = load_cli_config(__doc__ or "Validate point clouds")
    paths, classes, _ = discover_data(Path(config.data_dir))
    _, _, _, report = validate_data(
        paths,
        float(config.stats_tolerance),
        int(config.expected_points),
        Path(config.target_csv),
    )
    report["classes"] = classes
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
