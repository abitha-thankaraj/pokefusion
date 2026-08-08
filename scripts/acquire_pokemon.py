#!/usr/bin/env python3
"""Acquire configured default-form artwork from PokéAPI with source metadata."""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from pokemon_common import load_config, sha256_bytes

API_ROOT = "https://pokeapi.co/api/v2/pokemon"
RIGHTS_NOTICE = (
    "Image contents are copyright The Pokémon Company. The PokeAPI sprites repository "
    "license does not grant commercial redistribution rights to the image contents."
)


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {row["pokemon"]["name"]: row for row in map(json.loads, path.read_text().splitlines())}


def _sprite(api_record: dict[str, Any]) -> tuple[str, str]:
    other = api_record["sprites"]["other"]
    candidates = (
        ("sprites.other.official-artwork.front_default", other["official-artwork"]["front_default"]),
        ("sprites.other.home.front_default", other["home"]["front_default"]),
    )
    for field, url in candidates:
        if url:
            return field, url
    raise ValueError("PokéAPI response has neither official-artwork nor home front_default")


def acquire_all(config_path: Path, data_root: Path, refresh: bool = False) -> list[dict[str, Any]]:
    config = load_config(config_path)
    sources = data_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "source_manifest.jsonl"
    previous = _read_existing(manifest_path)
    records: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "datasaurust-pokemon-i1/1"

    for pokemon in config["pokemon"]:
        name = pokemon["name"]
        old = previous.get(name)
        if old and not refresh:
            cached = data_root.parent.parent / old["local_path"]
            if cached.is_file() and sha256_bytes(cached.read_bytes()) == old["sha256"]:
                records.append(old)
                continue

        api_url = f"{API_ROOT}/{pokemon['id']}"
        api_response = session.get(api_url, timeout=30)
        api_response.raise_for_status()
        api_record = api_response.json()
        sprite_field, asset_url = _sprite(api_record)
        asset_response = session.get(asset_url, timeout=60)
        asset_response.raise_for_status()
        content = asset_response.content
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            media_type = Image.MIME.get(image.format, asset_response.headers.get("Content-Type", ""))
            suffix = ".webp" if image.format == "WEBP" else ".png" if image.format == "PNG" else ".jpg"
        destination = sources / f"{name}{suffix}"
        destination.write_bytes(content)
        records.append(
            {
                "schema_version": 1,
                "pokemon": {
                    "id": int(api_record["id"]),
                    "name": name,
                    "form": "default",
                },
                "api_url": api_url,
                "asset_url": asset_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "etag": asset_response.headers.get("ETag"),
                "last_modified": asset_response.headers.get("Last-Modified"),
                "sha256": sha256_bytes(content),
                "width": width,
                "height": height,
                "media_type": media_type,
                "sprite_field": sprite_field,
                "local_path": destination.relative_to(data_root.parent.parent).as_posix(),
                "rights_notice": RIGHTS_NOTICE,
            }
        )

    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pokemon_i1.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/pokemon"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    records = acquire_all(args.config, args.data_root, args.refresh)
    print(json.dumps({"downloaded_or_cached": len(records), "manifest": str(args.data_root / 'source_manifest.jsonl')}, indent=2))


if __name__ == "__main__":
    main()
