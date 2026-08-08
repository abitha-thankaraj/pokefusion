#!/usr/bin/env python3
"""Acquire configured Pokémon artwork from PokéAPI with source metadata."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from pokefusion.data.pokemon_common import sha256_bytes
from pokefusion.omega_config import as_plain_dict, load_cli_config

API_ROOT = "https://pokeapi.co/api/v2/pokemon"
RIGHTS_NOTICE = (
    "Image contents are copyright The Pokémon Company. The PokeAPI sprites repository "
    "license does not grant commercial redistribution rights to the image contents."
)


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for row in map(json.loads, path.read_text().splitlines()):
        result[row["pokemon"]["name"]] = row
        result[str(row["pokemon"]["id"])] = row
    return result


def pokemon_identifier(entry: Any) -> str:
    """Accept a name, an ID, or a mapping containing either one."""
    if isinstance(entry, (str, int)):
        return str(entry).strip().lower()
    if isinstance(entry, dict):
        value = entry.get("name", entry.get("id"))
        if value is not None:
            return str(value).strip().lower()
    raise ValueError(f"Pokémon entry must be a name, ID, or mapping, got: {entry!r}")


def load_cached_sources(config: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    """Return only the cached records requested by this config, in config order."""
    manifest_path = data_root / "source_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} does not exist; set run.acquire=true for the first run"
        )
    previous = _read_existing(manifest_path)
    records = []
    for entry in config["pokemon"]:
        identifier = pokemon_identifier(entry)
        record = previous.get(identifier)
        if record is None:
            raise ValueError(
                f"Pokémon {identifier!r} is absent from {manifest_path}; "
                "set run.acquire=true to resolve it through PokéAPI"
            )
        records.append(record)
    return records


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


def acquire_all(config: dict[str, Any], data_root: Path, refresh: bool = False) -> list[dict[str, Any]]:
    sources = data_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "source_manifest.jsonl"
    previous = _read_existing(manifest_path)
    records: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = "pokefusion/1"

    for entry in config["pokemon"]:
        identifier = pokemon_identifier(entry)
        old = previous.get(identifier)
        if old and not refresh:
            cached = data_root.parent.parent / old["local_path"]
            if cached.is_file() and sha256_bytes(cached.read_bytes()) == old["sha256"]:
                records.append(old)
                continue

        api_url = f"{API_ROOT}/{identifier}"
        api_response = session.get(api_url, timeout=30)
        api_response.raise_for_status()
        api_record = api_response.json()
        name = api_record["name"]
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
                    "name": api_record["name"],
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
    config, _ = load_cli_config(__doc__ or "Acquire Pokémon artwork")
    plain = as_plain_dict(config)
    data_root = Path(config.run.data_root)
    records = acquire_all(plain, data_root, bool(config.run.refresh_sources))
    print(
        json.dumps(
            {"downloaded_or_cached": len(records), "manifest": str(data_root / "source_manifest.jsonl")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
