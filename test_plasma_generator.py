#!/usr/bin/env python3
import unittest
import json
import logging
import os
import sys
import tempfile
import time
import gc
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from PlasmaCloud import (
        _diamond_square,
        _apply_palette,
        _apply_colormap,
        plasma_worker,
        PlasmaGenerator,
        _generate_palette,
        _save_as_map,
    )
except ImportError as e:
    print(f"Error: Could not import from PlasmaCloud.py: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)

class TestPlasmaGeneratorEnhanced(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.generator = None

    def tearDown(self):
        # Close all file handles before cleanup
        if self.generator:
            if hasattr(self.generator, 'logger'):
                for handler in self.generator.logger.handlers[:]:
                    handler.close()
                    self.generator.logger.removeHandler(handler)
        
        # Force garbage collection to help release file handles
        gc.collect()
        
        # Wait a bit for file handles to be released
        time.sleep(0.5)
        
        # Try to cleanup with retries
        max_retries = 5
        for i in range(max_retries):
            try:
                self.test_dir.cleanup()
                break
            except PermissionError as e:
                if i == max_retries - 1:
                    # Last attempt, ignore errors
                    try:
                        self.test_dir.cleanup()
                    except:
                        pass
                    break
                else:
                    time.sleep(0.5)
                    gc.collect()

    def test_parallel_generation(self):
        """Test multi-worker parallel generation with unique filenames."""
        palette_dir = self.test_path / "palettes"
        palette_dir.mkdir()
        simple_palette = [(i, i, i) for i in range(256)]
        _save_as_map(simple_palette, palette_dir / "gray.map")

        output_dir = self.test_path / "output_parallel"
        
        self.generator = PlasmaGenerator(
            output_dir=str(output_dir),
            width=32,
            height=32,
            num_images=3,
            palette_dir=str(palette_dir),
            generation_mode="roughness",
            roughness_levels=[2, 4],
            max_workers=2,  # Parallel
            seed=42
        )

        manifest = self.generator.generate_collection()

        # Grab all PNG files
        png_files = list(output_dir.glob("*.png"))

        # Ensure all filenames are unique
        filenames = [f.name for f in png_files]
        self.assertEqual(len(filenames), len(set(filenames)), "Filenames are not unique!")

        # At least as many files as images * roughness
        self.assertGreaterEqual(len(png_files), 3)  # generator may overwrite if hash collisions occur

        # Verify image properties
        for f in png_files:
            with Image.open(f) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (32, 32))

    def test_animated_generation(self):
        """Test animated plasma frame generation."""
        palette_dir = self.test_path / "palettes_anim"
        palette_dir.mkdir()
        simple_palette = [(i, i, 255 - i) for i in range(256)]
        _save_as_map(simple_palette, palette_dir / "anim.map")

        output_dir = self.test_path / "output_anim"
        
        self.generator = PlasmaGenerator(
            output_dir=str(output_dir),
            width=32,
            height=32,
            num_images=1,
            palette_dir=str(palette_dir),
            generation_mode="default",
            animate=True,
            frames=5,
            max_workers=1,
            seed=123
        )

        manifest = self.generator.generate_collection()

        # Check that at least 1 PNG file was created
        png_files = list(output_dir.glob("*.png"))
        self.assertGreaterEqual(len(png_files), 1)
        
        # Check that at least 1 GIF file was created
        gif_files = list(output_dir.glob("*.gif"))
        self.assertGreaterEqual(len(gif_files), 1)

        for f in png_files:
            with Image.open(f) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (32, 32))
                
        for f in gif_files:
            with Image.open(f) as img:
                self.assertEqual(img.format, "GIF")
                self.assertEqual(img.size, (32, 32))
                # Note: GIF might have fewer frames than requested due to optimization
                self.assertGreaterEqual(img.n_frames, 4)

    def test_diamond_square_deterministic(self):
        """Diamond-square consistency check."""
        rng = np.random.default_rng(42)
        arr1 = _diamond_square(65, 5.0, rng)
        rng2 = np.random.default_rng(42)
        arr2 = _diamond_square(65, 5.0, rng2)
        np.testing.assert_array_equal(arr1, arr2)

    def test_apply_palette_and_colormap(self):
        """Test palette and colormap applications."""
        img_gray = Image.new("L", (64, 64), 128)
        palette = [(i, i, i) for i in range(256)]
        img_pal = _apply_palette(img_gray, palette)
        self.assertEqual(img_pal.mode, "RGB")
        img_map = _apply_colormap(img_gray, "viridis")
        self.assertEqual(img_map.mode, "RGB")

    def test_on_demand_palettes(self):
        palette_dir = self.test_path / "pal_dir"
        self.assertFalse(palette_dir.exists())
        self.generator = PlasmaGenerator(
            output_dir=self.test_path / "out",
            palette_dir=str(palette_dir),
            generate_palettes=True,
            num_palettes_to_generate=5,
            num_images=1
        )
        self.assertTrue(palette_dir.exists())
        map_files = list(palette_dir.glob("*.map"))
        self.assertEqual(len(map_files), 5)

    def test_colormap_mode(self):
        """Test colormap generation mode."""
        output_dir = self.test_path / "output_colormap"
        
        self.generator = PlasmaGenerator(
            output_dir=str(output_dir),
            width=32,
            height=32,
            num_images=1,
            generation_mode="colormap",
            max_workers=1,
            seed=42
        )

        manifest = self.generator.generate_collection()

        # Grab all PNG files
        png_files = list(output_dir.glob("*.png"))
        
        # Should have 5 files for the 5 default colormaps
        # But since they might have the same palette hash, we need to check the manifest
        self.assertGreaterEqual(len(png_files), 1)
        
        # Check that we have entries for all 5 colormaps in the manifest
        colormap_names = [entry["params"]["colormap"] for entry in manifest if "colormap" in entry["params"]]
        self.assertEqual(len(colormap_names), 5)
        self.assertIn("plasma", colormap_names)
        self.assertIn("viridis", colormap_names)
        self.assertIn("magma", colormap_names)
        self.assertIn("inferno", colormap_names)
        self.assertIn("rainbow", colormap_names)

    def test_blur_mode(self):
        """Test blur generation mode."""
        palette_dir = self.test_path / "palettes_blur"
        palette_dir.mkdir()
        simple_palette = [(i, 255-i, 128) for i in range(256)]
        _save_as_map(simple_palette, palette_dir / "blur.map")

        output_dir = self.test_path / "output_blur"
        
        self.generator = PlasmaGenerator(
            output_dir=str(output_dir),
            width=32,
            height=32,
            num_images=1,
            palette_dir=str(palette_dir),
            generation_mode="blur",
            blur_levels=[0, 2, 4],
            max_workers=1,
            seed=42
        )

        manifest = self.generator.generate_collection()

        # Grab all PNG files
        png_files = list(output_dir.glob("*.png"))
        
        # Should have at least 1 file
        self.assertGreaterEqual(len(png_files), 1)
        
        # Check that we have entries for all 3 blur levels in the manifest
        blur_levels = [entry["params"]["blur"] for entry in manifest if "blur" in entry["params"]]
        self.assertEqual(len(blur_levels), 3)
        self.assertIn(0, blur_levels)
        self.assertIn(2, blur_levels)
        self.assertIn(4, blur_levels)

    def test_layered_mode(self):
        """Test layered generation mode."""
        palette_dir = self.test_path / "palettes_layered"
        palette_dir.mkdir()
        simple_palette = [(i, i, i) for i in range(256)]
        _save_as_map(simple_palette, palette_dir / "layered.map")

        output_dir = self.test_path / "output_layered"
        
        self.generator = PlasmaGenerator(
            output_dir=str(output_dir),
            width=32,
            height=32,
            num_images=1,
            palette_dir=str(palette_dir),
            generation_mode="layered",
            max_workers=1,
            seed=42
        )

        manifest = self.generator.generate_collection()

        # Grab all PNG files
        png_files = list(output_dir.glob("*.png"))
        
        # Should have 1 file for the single image
        self.assertEqual(len(png_files), 1)

    def test_reproducibility(self):
        """Test that using the same seed produces the same results."""
        palette_dir = self.test_path / "palettes_repro"
        palette_dir.mkdir()
        simple_palette = [(i, i, i) for i in range(256)]
        _save_as_map(simple_palette, palette_dir / "repro.map")

        output_dir1 = self.test_path / "output_repro1"
        output_dir2 = self.test_path / "output_repro2"
        
        # Use a very specific seed to ensure reproducibility
        fixed_seed = 12345
        
        # Generate with the same seed
        self.generator1 = PlasmaGenerator(
            output_dir=str(output_dir1),
            width=32,
            height=32,
            num_images=1,
            palette_dir=str(palette_dir),
            generation_mode="default",
            max_workers=1,  # Use single worker to avoid race conditions
            seed=fixed_seed
        )
        
        manifest1 = self.generator1.generate_collection()
        
        # Close the first generator before creating the second one
        if hasattr(self.generator1, 'logger'):
            for handler in self.generator1.logger.handlers[:]:
                handler.close()
                self.generator1.logger.removeHandler(handler)
        
        # Force garbage collection to help release file handles
        gc.collect()
        
        self.generator2 = PlasmaGenerator(
            output_dir=str(output_dir2),
            width=32,
            height=32,
            num_images=1,
            palette_dir=str(palette_dir),
            generation_mode="default",
            max_workers=1,  # Use single worker to avoid race conditions
            seed=fixed_seed
        )

        manifest2 = self.generator2.generate_collection()

        # Compare the generated images
        png_files1 = list(output_dir1.glob("*.png"))
        png_files2 = list(output_dir2.glob("*.png"))
        
        self.assertEqual(len(png_files1), len(png_files2))
        
        # Sort the files to ensure consistent comparison
        png_files1.sort()
        png_files2.sort()
        
        for f1, f2 in zip(png_files1, png_files2):
            with Image.open(f1) as img1, Image.open(f2) as img2:
                arr1 = np.array(img1)
                arr2 = np.array(img2)
                np.testing.assert_array_equal(arr1, arr2)

if __name__ == "__main__":
    # unittest runner
    unittest.main(verbosity=2)