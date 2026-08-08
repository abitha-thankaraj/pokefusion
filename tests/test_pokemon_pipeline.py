from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from extract_contour import extract_contour  # noqa: E402
from pokemon_common import load_config, moments, project_moments, read_points, stable_seed  # noqa: E402
import pokemon_point_diffusion as diffusion  # noqa: E402

ModelConfig = diffusion.ModelConfig
torch = diffusion.torch
validate_data = diffusion.validate_data
discover_data = diffusion.discover_data


class PokemonPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs/pokemon_i1.yaml")

    def test_exact_moment_projection(self) -> None:
        rng = np.random.default_rng(123)
        target = rng.normal(size=(142, 2)) @ np.array([[3.0, -0.4], [0.2, 1.5]])
        candidate = rng.normal(size=(142, 2)) @ np.array([[0.5, 1.2], [2.0, -0.3]]) + 8.0
        target_mean, target_cov, target_stats = moments(target)
        projected = project_moments(candidate, target_mean, target_cov)
        _, _, measured = moments(projected)
        np.testing.assert_allclose(measured, target_stats, rtol=0.0, atol=1e-10)

    def test_seed_derivation_is_stable_and_distinct(self) -> None:
        self.assertEqual(stable_seed(42, 1, 2), stable_seed(42, 1, 2))
        self.assertNotEqual(stable_seed(42, 1, 2), stable_seed(42, 1, 3))

    def test_one_generic_extractor_handles_five_transparent_fixtures(self) -> None:
        polygons = [
            [(28, 15), (98, 22), (105, 100), (20, 108)],
            [(64, 10), (110, 64), (64, 116), (18, 64)],
            [(18, 30), (64, 12), (110, 30), (96, 108), (32, 108)],
            [(20, 20), (108, 20), (92, 64), (108, 108), (20, 108), (36, 64)],
            [(64, 10), (78, 46), (116, 48), (86, 72), (96, 112), (64, 88), (30, 112), (42, 72), (12, 48), (50, 46)],
        ]
        for index, polygon in enumerate(polygons):
            with self.subTest(index=index):
                image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
                ImageDraw.Draw(image).polygon(polygon, fill=(30, 160, 90, 255))
                content = io.BytesIO()
                image.save(content, format="PNG")
                mask, contours, diagnostics = extract_contour(
                    content.getvalue(), self.config["extraction"]
                )
                self.assertGreater(mask.mean(), 0.02)
                self.assertEqual(sum(map(len, contours)), 512)
                self.assertEqual(diagnostics["segmentation"], "alpha")

    def test_opaque_image_uses_generic_border_background_flood_fill(self) -> None:
        image = Image.new("RGB", (160, 120), (245, 245, 240))
        ImageDraw.Draw(image).ellipse((35, 12, 125, 108), fill=(35, 120, 210))
        content = io.BytesIO()
        image.save(content, format="JPEG", quality=95)
        mask, contours, diagnostics = extract_contour(
            content.getvalue(), self.config["extraction"]
        )
        self.assertEqual(diagnostics["segmentation"], "border_lab_flood_fill")
        self.assertGreater(mask.mean(), 0.02)
        self.assertEqual(sum(map(len, contours)), 512)

    def test_extractor_has_no_fixture_specific_literals(self) -> None:
        source = (ROOT / "scripts/extract_contour.py").read_text().lower()
        for fixture_name in ("bulbasaur", "pikachu", "gengar", "lapras", "charizard"):
            self.assertNotIn(fixture_name, source)

    def test_all_configured_artworks_use_the_same_public_extractor(self) -> None:
        source_manifest = ROOT / "data/pokemon/source_manifest.jsonl"
        if not source_manifest.is_file():
            self.skipTest("downloaded source artwork is intentionally not versioned")
        sources = {
            row["pokemon"]["name"]: ROOT / row["local_path"]
            for row in map(json.loads, source_manifest.read_text().splitlines())
        }
        for fixture in self.config["pokemon"]:
            with self.subTest(fixture=fixture["name"]):
                mask, contours, diagnostics = extract_contour(
                    sources[fixture["name"]], self.config["extraction"]
                )
                self.assertGreater(mask.mean(), 0.02)
                self.assertEqual(sum(map(len, contours)), 512)
                self.assertEqual(diagnostics["segmentation"], "alpha")

    def test_diffusion_validator_rejects_mixed_point_counts_and_bad_stats(self) -> None:
        target_path = ROOT / "data/seed_datasets/Datasaurus_data.csv"
        target = read_points(target_path)
        with tempfile.TemporaryDirectory() as raw_directory:
            folder = Path(raw_directory) / "fixture"
            folder.mkdir()
            good = folder / "good.csv"
            mixed = folder / "mixed.csv"
            bad_stats = folder / "bad_stats.csv"
            np.savetxt(good, target, delimiter=",")
            np.savetxt(mixed, target[:-1], delimiter=",")
            shifted = target.copy()
            shifted[0, 0] += 1.0
            np.savetxt(bad_stats, shifted, delimiter=",")
            validate_data([good], 1e-4, 142, target_path)
            with self.assertRaisesRegex(ValueError, "expected 142 rows"):
                validate_data([good, mixed], 1e-4, 142, target_path)
            with self.assertRaisesRegex(ValueError, "invariant error exceeds"):
                validate_data([good, bad_stats], 1e-4, 142, target_path)

    def test_diffusion_loader_discovers_csvs_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            nested = root / "archive" / "pikachu"
            nested.mkdir(parents=True)
            sample = nested / "sample.csv"
            sample.write_text("1,2\n")
            paths, classes, class_to_id = discover_data(root)
            self.assertEqual(paths, [sample])
            self.assertEqual(classes, ["pikachu"])
            self.assertEqual(class_to_id, {"pikachu": 0})

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_denoiser_is_permutation_equivariant(self) -> None:
        torch.manual_seed(5)
        config = ModelConfig(n_points=8, n_classes=2, timesteps=10, width=16, layers=1, heads=4)
        model = diffusion.PointDenoiser(config).eval()
        points = torch.randn(2, 8, 2)
        timesteps = torch.tensor([2, 7])
        labels = torch.tensor([0, 1])
        permutation = torch.tensor([3, 1, 6, 0, 7, 4, 2, 5])
        with torch.no_grad():
            original = model(points, timesteps, labels)
            permuted = model(points[:, permutation], timesteps, labels)
        torch.testing.assert_close(permuted, original[:, permutation], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
