#!/usr/bin/env python3
"""Check the unchanged generic extractor on one held-out PokéAPI artwork."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import requests

from pokefusion.data.extract_contour import extract_contour
from pokefusion.data.pokemon_common import canonical_json_bytes, sha256_bytes
from pokefusion.omega_config import as_plain_dict, load_cli_config


def _contour_bytes(contours) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("contour_id", "order", "x", "y", "closed"))
    for contour_id, contour in enumerate(contours):
        for order, (x, y) in enumerate(contour):
            writer.writerow((contour_id, order, f"{x:.10f}", f"{y:.10f}", "true"))
    return buffer.getvalue().encode()


def main() -> None:
    omega_config, _ = load_cli_config(__doc__ or "Check held-out artwork")
    config = as_plain_dict(omega_config)
    pokemon_id = int(omega_config.heldout.pokemon_id)
    output = Path(omega_config.heldout.out)
    source_manifest = Path(omega_config.run.data_root) / "source_manifest.jsonl"
    configured_ids = {
        int(json.loads(line)["pokemon"]["id"])
        for line in source_manifest.read_text().splitlines()
    }
    if pokemon_id in configured_ids:
        raise ValueError("held-out ID must not appear in the training configuration")
    api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    api_response = requests.get(api_url, timeout=30)
    api_response.raise_for_status()
    record = api_response.json()
    other = record["sprites"]["other"]
    sprite_field = "sprites.other.official-artwork.front_default"
    asset_url = other["official-artwork"]["front_default"]
    if not asset_url:
        sprite_field = "sprites.other.home.front_default"
        asset_url = other["home"]["front_default"]
    if not asset_url:
        raise ValueError("held-out response has no preferred or fallback artwork")
    asset_response = requests.get(asset_url, timeout=60)
    asset_response.raise_for_status()
    mask, contours, diagnostics = extract_contour(asset_response.content, config["extraction"])
    result = {
        "schema_version": 1,
        "status": "accepted",
        "pokemon": {"id": int(record["id"]), "name": record["name"], "form": "default"},
        "api_url": api_url,
        "asset_url": asset_url,
        "sprite_field": sprite_field,
        "image_sha256": sha256_bytes(asset_response.content),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "mask_foreground_fraction": float(mask.mean()),
        "contour_sha256": sha256_bytes(_contour_bytes(contours)),
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
