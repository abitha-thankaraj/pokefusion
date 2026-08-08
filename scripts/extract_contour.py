#!/usr/bin/env python3
"""Generic image-to-silhouette contour extraction with no subject-specific logic."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any

import contourpy
import numpy as np
from PIL import Image
from scipy import ndimage

from pokemon_common import load_config, sha256_bytes, write_contours


class ExtractionError(ValueError):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def _image_bytes(image: bytes | bytearray | Path | str) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    value = str(image)
    if value.startswith(("http://", "https://")):
        import requests

        response = requests.get(value, timeout=60)
        response.raise_for_status()
        return response.content
    return Path(value).read_bytes()


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64) / 255.0
    value = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    xyz = value @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float64,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def _border_values(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values[0], values[-1], values[1:-1, 0], values[1:-1, -1]), axis=0)


def _background_mask(rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    lab = _srgb_to_lab(rgb)
    border = _border_values(lab)
    center = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - center, axis=1)
    q75 = float(np.quantile(border_distance, 0.75))
    if q75 > threshold:
        raise ExtractionError(
            "ambiguous border-background estimate",
            {"border_lab_distance_q75": q75, "background_lab_distance": threshold},
        )
    candidate = np.linalg.norm(lab - center, axis=2) <= threshold
    labels, _ = ndimage.label(candidate)
    border_labels = np.unique(_border_values(labels[..., None]).ravel())
    border_labels = border_labels[border_labels != 0]
    background = np.isin(labels, border_labels)
    return background, {
        "segmentation": "border_lab_flood_fill",
        "border_lab_center": center.tolist(),
        "border_lab_distance_q75": q75,
    }


def _disk(size: int) -> np.ndarray:
    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _bbox_distance(a: tuple[slice, ...], b: tuple[slice, ...]) -> float:
    def gap(left: slice, right: slice) -> int:
        if left.stop <= right.start:
            return right.start - left.stop
        if right.stop <= left.start:
            return left.start - right.stop
        return 0

    return math.hypot(gap(a[0], b[0]), gap(a[1], b[1]))


def _clean_mask(mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = mask.shape
    image_area = height * width
    kernel = max(3, round(min(height, width) * float(config["morphology_fraction"])))
    if kernel % 2 == 0:
        kernel += 1
    structure = _disk(kernel)
    mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_opening(mask, structure=structure)

    labels, count = ndimage.label(mask)
    if count == 0:
        raise ExtractionError("no stable foreground component")
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    minimum = float(config["minimum_component_fraction"]) * image_area
    valid = [index for index in range(1, count + 1) if areas[index] >= minimum]
    if not valid:
        raise ExtractionError("all foreground components are below the global area threshold")
    dominant = max(valid, key=lambda index: (areas[index], -index))
    objects = ndimage.find_objects(labels)
    dominant_box = objects[dominant - 1]
    diagonal = math.hypot(height, width)
    keep = {dominant}
    for index in valid:
        if index == dominant:
            continue
        area_ok = areas[index] >= float(config["detached_area_ratio"]) * areas[dominant]
        distance_ok = _bbox_distance(dominant_box, objects[index - 1]) <= float(
            config["detached_distance_fraction"]
        ) * diagonal
        if area_ok and distance_ok:
            keep.add(index)
    mask = np.isin(labels, sorted(keep))

    inverse_labels, inverse_count = ndimage.label(~mask)
    border_ids = set(np.unique(_border_values(inverse_labels[..., None]).ravel()).tolist())
    inverse_areas = np.bincount(inverse_labels.ravel())
    maximum_hole = float(config["fill_hole_fraction"]) * image_area
    filled_holes = 0
    for index in range(1, inverse_count + 1):
        if index not in border_ids and inverse_areas[index] < maximum_hole:
            mask[inverse_labels == index] = True
            filled_holes += 1

    fraction = float(mask.mean())
    if not float(config["foreground_fraction_min"]) <= fraction <= float(config["foreground_fraction_max"]):
        raise ExtractionError(
            "foreground fraction outside configured range",
            {"foreground_fraction": fraction},
        )
    touches = {
        "top": bool(mask[0].any()),
        "bottom": bool(mask[-1].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }
    if all(touches.values()):
        raise ExtractionError("foreground touches all four image borders", {"touches_borders": touches})
    return mask, {
        "morphology_kernel": kernel,
        "component_count": count,
        "retained_components": len(keep),
        "dominant_component_area": int(areas[dominant]),
        "filled_holes": filled_holes,
        "foreground_fraction": fraction,
        "touches_borders": touches,
    }


def _perimeter(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def _rdp_open(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    direction = end - start
    denominator = float(direction @ direction)
    if denominator == 0.0:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        t = np.clip(((points[1:-1] - start) @ direction) / denominator, 0.0, 1.0)
        distances = np.linalg.norm(points[1:-1] - (start + t[:, None] * direction), axis=1)
    if not len(distances) or float(distances.max()) <= epsilon:
        return np.stack((start, end))
    index = int(np.argmax(distances)) + 1
    return np.concatenate((_rdp_open(points[: index + 1], epsilon)[:-1], _rdp_open(points[index:], epsilon)))


def _simplify_closed(points: np.ndarray, epsilon: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if np.linalg.norm(points[0] - points[-1]) < 1e-9:
        points = points[:-1]
    start = min(range(len(points)), key=lambda index: (points[index, 0], points[index, 1], index))
    rotated = np.concatenate((points[start:], points[:start], points[start : start + 1]))
    simplified = _rdp_open(rotated, epsilon)
    if np.linalg.norm(simplified[0] - simplified[-1]) < 1e-9:
        simplified = simplified[:-1]
    return simplified if len(simplified) >= 3 else points


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.concatenate((points, points[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    positions = np.arange(count, dtype=np.float64) * cumulative[-1] / count
    segment = np.minimum(np.searchsorted(cumulative, positions, side="right") - 1, len(lengths) - 1)
    fraction = (positions - cumulative[segment]) / lengths[segment]
    return closed[segment] + fraction[:, None] * (closed[segment + 1] - closed[segment])


def _allocate(perimeters: np.ndarray, total: int, minimum: int) -> np.ndarray:
    if len(perimeters) * minimum > total:
        raise ExtractionError(f"{len(perimeters)} retained contours cannot fit minimum allocation {minimum}")
    remaining = total - len(perimeters) * minimum
    exact = perimeters / perimeters.sum() * remaining
    result = np.full(len(perimeters), minimum, dtype=int) + np.floor(exact).astype(int)
    for index in np.argsort(-(exact - np.floor(exact)), kind="stable")[: total - int(result.sum())]:
        result[index] += 1
    return result


def _extract_lines(mask: np.ndarray, config: dict[str, Any]) -> list[np.ndarray]:
    padded = np.pad(mask.astype(np.float64), 1)
    generator = contourpy.contour_generator(z=padded, line_type="Separate")
    raw = [np.asarray(line, dtype=np.float64) - 1.0 for line in generator.lines(0.5)]
    raw = [line for line in raw if len(line) >= 4 and _perimeter(line) >= 4.0]
    if not raw:
        raise ExtractionError("mask produced no stable closed contour")
    simplified = [
        _simplify_closed(line, float(config["rdp_perimeter_fraction"]) * _perimeter(line))
        for line in raw
    ]
    perimeters = np.asarray([_perimeter(line) for line in simplified])
    counts = _allocate(
        perimeters,
        int(config["contour_vertices"]),
        int(config["minimum_vertices_per_contour"]),
    )
    return [_resample_closed(line, int(count)) for line, count in zip(simplified, counts)]


def _fit_canvas(contours: list[np.ndarray], height: int, config: dict[str, Any]) -> list[np.ndarray]:
    cartesian = []
    for contour in contours:
        value = contour.copy()
        value[:, 1] = height - 1 - value[:, 1]
        cartesian.append(value)
    all_points = np.concatenate(cartesian)
    source_min = all_points.min(axis=0)
    source_max = all_points.max(axis=0)
    extent = source_max - source_min
    if np.any(extent <= 0.0):
        raise ExtractionError("degenerate contour extent")
    x_min, x_max, y_min, y_max = map(float, config["canvas"])
    padding = float(config["canvas_padding_fraction"])
    inner_min = np.array([x_min + padding * (x_max - x_min), y_min + padding * (y_max - y_min)])
    inner_max = np.array([x_max - padding * (x_max - x_min), y_max - padding * (y_max - y_min)])
    scale = float(np.min((inner_max - inner_min) / extent))
    offset = (inner_min + inner_max) / 2.0 - scale * (source_min + source_max) / 2.0
    return [contour * scale + offset for contour in cartesian]


def extract_contour(
    image: bytes | bytearray | Path | str, config: dict[str, Any]
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Return a binary subject mask, ordered closed contours, and diagnostics."""
    content = _image_bytes(image)
    with Image.open(io.BytesIO(content)) as decoded:
        rgba = np.asarray(decoded.convert("RGBA"))
    height, width = rgba.shape[:2]
    alpha = rgba[..., 3]
    threshold = int(config["alpha_threshold"])
    meaningful_alpha = bool(np.ptp(alpha) >= threshold and np.mean(alpha < 250) >= 0.001)
    if meaningful_alpha:
        mask = alpha >= threshold
        segmentation = {"segmentation": "alpha", "alpha_threshold": threshold}
    else:
        background, segmentation = _background_mask(
            rgba[..., :3], float(config["background_lab_distance"])
        )
        mask = ~background
    mask, cleaning = _clean_mask(mask, config)
    pixel_contours = _extract_lines(mask, config)
    contours = _fit_canvas(pixel_contours, height, config)
    diagnostics = {
        "schema_version": 1,
        "image_sha256": sha256_bytes(content),
        "width": width,
        "height": height,
        "contours": len(contours),
        "vertices": int(sum(map(len, contours))),
        **segmentation,
        **cleaning,
    }
    return mask, contours, diagnostics


def save_preview(source: Path, mask: np.ndarray, contours: list[np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    with Image.open(source) as image:
        rgba = image.convert("RGBA")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgba)
    axes[0].set_title("source")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("mask")
    for contour in contours:
        closed = np.concatenate((contour, contour[:1]))
        axes[2].plot(closed[:, 0], closed[:, 1], linewidth=1.0)
    axes[2].set_aspect("equal")
    axes[2].set_title("contours")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pokemon_i1.yaml"))
    parser.add_argument("--contour-out", type=Path, required=True)
    parser.add_argument("--mask-out", type=Path)
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--diagnostics-out", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)["extraction"]
    mask, contours, diagnostics = extract_contour(args.image, config)
    write_contours(args.contour_out, contours)
    if args.mask_out:
        args.mask_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(args.mask_out)
    if args.preview_out:
        save_preview(args.image, mask, contours, args.preview_out)
    if args.diagnostics_out:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_out.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
