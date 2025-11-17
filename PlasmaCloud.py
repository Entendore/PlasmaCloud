#!/usr/bin/env python3
"""
A comprehensive, parallelized plasma cloud generator.

This script provides a command-line interface to generate various types of
plasma cloud images. It supports multiple generation modes, custom palettes,
animation, and parallel processing for efficient batch creation.

Features:
    - Multiple generation modes (roughness, blur, layered, colormap).
    - Diamond-square algorithm for realistic terrain-like textures.
    - Support for custom .map palette files and Matplotlib colormaps.
    - Optional on-demand palette generation.
    - Optional animated GIF generation.
    - Parallel processing for faster generation.
    - Detailed logging and metadata output (manifest.json).
"""

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import colorsys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Module-level Constants and Worker Functions ---

# These functions are defined at the module level because they need to be
# pickleable for multiprocessing.

def _diamond_square(size: int, roughness: float, rng: np.random.Generator) -> np.ndarray:
    """Generates a plasma texture using the diamond-square algorithm.

    Args:
        size: The width and height of the square texture to generate.
        roughness: The initial amplitude of the random displacement.
        rng: A NumPy random generator instance.

    Returns:
        A 2D NumPy array representing the grayscale plasma texture.
    """
    # Ensure size is a power of 2 + 1 for proper diamond-square algorithm
    if (size - 1) & (size - 2) != 0:
        # Find the next power of 2 + 1
        n = 1
        while n < size:
            n <<= 1
        size = n + 1
    
    arr = np.zeros((size, size), dtype=np.float32)
    step = size - 1
    arr[0, 0] = rng.random() * 255
    arr[0, step] = rng.random() * 255
    arr[step, 0] = rng.random() * 255
    arr[step, step] = rng.random() * 255

    while step > 1:
        half = step // 2
        # Diamond step
        for y in range(half, size, step):
            for x in range(half, size, step):
                avg = (
                    arr[y - half, x - half] + arr[y - half, x + half - step]
                    + arr[y + half - step, x - half] + arr[y + half - step, x + half - step]
                ) / 4
                arr[y, x] = avg + rng.uniform(-roughness, roughness)
        # Square step
        for y in range(0, size, half):
            for x in range((y + half) % step, size, step):
                total = 0
                count = 0
                for dy, dx in [(-half, 0), (half, 0), (0, -half), (0, half)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        total += arr[ny, nx]
                        count += 1
                arr[y, x] = (total / count) + rng.uniform(-roughness, roughness)
        step = half
        roughness /= 2
    return np.clip(arr, 0, 255).astype(np.uint8)


def _apply_palette(img: Image.Image, palette: Sequence[Tuple[int, int, int]]) -> Image.Image:
    """Applies a color palette to a grayscale image.

    Args:
        img: The source PIL Image (mode 'L' or 'RGB').
        palette: A sequence of (R, G, B) tuples.

    Returns:
        A new PIL Image in RGB mode with the palette applied.
    """
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    
    # Ensure we have exactly 256 colors in the palette
    if len(palette) != 256:
        # Interpolate the palette to have exactly 256 colors
        old_palette = np.array(palette)
        new_palette = np.zeros((256, 3))
        old_indices = np.linspace(0, len(palette) - 1, len(palette))
        new_indices = np.linspace(0, len(palette) - 1, 256)
        for i in range(3):  # For each RGB channel
            new_palette[:, i] = np.interp(new_indices, old_indices, old_palette[:, i])
        palette = [tuple(color) for color in new_palette.astype(int)]
    
    palette_array = np.array(palette)
    
    # Use a more direct mapping from grayscale values to palette indices
    indices = img_gray.astype(int)
    indices = np.clip(indices, 0, 255)
    
    colored_pixels = palette_array[indices]
    return Image.fromarray(colored_pixels.astype(np.uint8))


def _apply_colormap(img: Image.Image, colormap_name: str) -> Image.Image:
    """Applies a Matplotlib colormap to a grayscale image.

    Args:
        img: The source PIL Image (mode 'L' or 'RGB').
        colormap_name: The name of the Matplotlib colormap to use.

    Returns:
        A new PIL Image in RGB mode with the colormap applied.
    """
    img_gray = np.array(img.convert("L"))
    cmap = plt.get_cmap(colormap_name)
    colored_arr = cmap(img_gray / 255.0)[:, :, :3]
    return Image.fromarray((colored_arr * 255).astype(np.uint8))


def plasma_worker(task_args: Tuple) -> Dict[str, Any]:
    """Worker function to generate a single plasma image or animation.

    This function is designed to be run in a separate process.

    Args:
        task_args: A tuple containing all necessary parameters for generation.
            (img_index, params, palette, palette_hash, width, height,
             seed, output_dir, generation_mode, animate, frames)

    Returns:
        A dictionary containing metadata about the generated file(s).
    """
    (
        img_index, params, palette, palette_hash, width, height,
        seed, output_dir, generation_mode, animate, frames
    ) = task_args

    # Create a deterministic random number generator for this worker
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)

    # --- Base Generation ---
    if generation_mode == "layered":
        layers = []
        num_layers = params.get("num_layers", 3)
        roughness_values = params.get("roughness_values", [2, 8, 16])
        blur_values = params.get("blur_values", [0, 2, 4])
        
        # Ensure we have the right number of values
        if len(roughness_values) < num_layers:
            roughness_values = roughness_values + [roughness_values[-1]] * (num_layers - len(roughness_values))
        if len(blur_values) < num_layers:
            blur_values = blur_values + [blur_values[-1]] * (num_layers - len(blur_values))
            
        for i in range(num_layers):
            layer_seed = seed + img_index * 100 + i
            layer_rng = np.random.default_rng(layer_seed)
            layer_roughness = roughness_values[i]
            layer_blur = blur_values[i]
            
            arr = _diamond_square(max(width, height), layer_roughness * 16, layer_rng)
            arr = arr[:height, :width]
            layer_img = Image.fromarray(arr).convert("RGB")
            if layer_blur > 0:
                layer_img = layer_img.filter(ImageFilter.GaussianBlur(radius=layer_blur))
            layers.append(layer_img)
        
        # Blend layers
        img = Image.new("RGB", (width, height))
        for layer in layers:
            img = Image.blend(img, layer, alpha=1.0 / len(layers))
        blur = 0
    else: # All other modes
        base_roughness = params.get("roughness", 8)
        arr = _diamond_square(max(width, height), base_roughness * 16, rng)
        arr = arr[:height, :width]
        
        # Check if the generated texture has enough variation
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val < 50:  # If not enough variation, regenerate with higher roughness
            arr = _diamond_square(max(width, height), base_roughness * 32, rng)
            arr = arr[:height, :width]
        
        img = Image.fromarray(arr).convert("RGB")
        blur = params.get("blur", 0)
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # --- Coloring ---
    if palette:
        img = _apply_palette(img, palette)
    elif "colormap" in params:
        img = _apply_colormap(img, params["colormap"])

    # --- Saving ---
    # Create a more unique filename that includes parameter-specific information
    param_str = ""
    if generation_mode == "roughness":
        param_str = f"rough{params['roughness']}"
    elif generation_mode == "blur":
        param_str = f"blur{params['blur']}"
    elif generation_mode == "colormap":
        param_str = f"colormap{params['colormap']}"
    
    filename_base = f"Plasma_{img_index}_{generation_mode}_{param_str}_seed{seed}_pal{palette_hash}"
    filename = output_path / f"{filename_base}.png"
    img.save(filename)

    # --- Animation ---
    gif_filename = None
    if animate and frames > 1:
        frames_list = [img]
        for i in range(1, frames):
            # Create a slightly perturbed version for animation
            # Use deterministic perturbation based on the frame index
            perturb_radius = 0.5 + (i / frames) * 1.5  # Range from 0.5 to 2.0
            frame_img = img.filter(ImageFilter.GaussianBlur(radius=perturb_radius))
            frames_list.append(frame_img)

        gif_filename = output_path / f"{filename_base}.gif"
        frames_list[0].save(
            gif_filename,
            save_all=True,
            append_images=frames_list[1:],
            duration=100,
            loop=0,
            optimize=True
        )

    return {
        "filename": filename.name,
        "gif": gif_filename.name if gif_filename else None,
        "image_index": img_index,
        "generation_mode": generation_mode,
        "params": params,
        "seed": seed,
        "palette_hash": palette_hash,
    }


# --- Palette Generation Functions (Integrated) ---

def _generate_palette() -> List[Tuple[int, int, int]]:
    """
    Generates a single "interesting" palette of 256 RGB colors.
    The process involves:
    1. Defining random color waypoints in HSV color space for harmony.
    2. Linearly interpolating between these waypoints.
    3. Applying sine wave distortions to each RGB channel for complexity.
    """
    num_waypoints = random.randint(4, 8)
    
    # 1. Define random waypoints (position and color)
    waypoint_positions = sorted([0] + [random.randint(1, 254) for _ in range(num_waypoints - 2)] + [255])
    
    waypoints = []
    for pos in waypoint_positions:
        # Generate color in HSV for more harmonious random colors
        h = random.random()
        s = random.uniform(0.5, 1.0)
        v = random.uniform(0.4, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        waypoints.append({'pos': pos, 'color': (r * 255, g * 255, b * 255)})

    # 2. Interpolate between waypoints to get a base gradient
    base_palette = []
    for i in range(256):
        start_wp = waypoints[0]
        end_wp = waypoints[-1]
        for j in range(len(waypoints) - 1):
            if waypoints[j]['pos'] <= i <= waypoints[j+1]['pos']:
                start_wp = waypoints[j]
                end_wp = waypoints[j+1]
                break
        
        if end_wp['pos'] == start_wp['pos']:
            t = 0
        else:
            t = (i - start_wp['pos']) / (end_wp['pos'] - start_wp['pos'])
        
        r = start_wp['color'][0] * (1 - t) + end_wp['color'][0] * t
        g = start_wp['color'][1] * (1 - t) + end_wp['color'][1] * t
        b = start_wp['color'][2] * (1 - t) + end_wp['color'][2] * t
        base_palette.append((r, g, b))

    # 3. Apply sine wave distortions for "interestingness" with increased amplitude
    final_palette = []
    # Increased amplitude for more variation
    amp_r, freq_r, phase_r = random.uniform(30, 80), random.uniform(0.02, 0.2), random.uniform(0, math.pi * 2)
    amp_g, freq_g, phase_g = random.uniform(30, 80), random.uniform(0.02, 0.2), random.uniform(0, math.pi * 2)
    amp_b, freq_b, phase_b = random.uniform(30, 80), random.uniform(0.02, 0.2), random.uniform(0, math.pi * 2)

    for i, (r, g, b) in enumerate(base_palette):
        r_mod = r + amp_r * math.sin(freq_r * i + phase_r)
        g_mod = g + amp_g * math.sin(freq_g * i + phase_g)
        b_mod = b + amp_b * math.sin(freq_b * i + phase_b)
        
        final_palette.append((
            int(max(0, min(255, r_mod))),
            int(max(0, min(255, g_mod))),
            int(max(0, min(255, b_mod)))
        ))
        
    return final_palette

def _save_as_map(palette: List[Tuple[int, int, int]], filename: Path) -> None:
    """Saves a palette to a .map file (RGB 0-255 format)."""
    with filename.open('w') as f:
        for r, g, b in palette:
            f.write(f"{r} {g} {b}\n")


# --- Main Generator Class ---

class PlasmaGenerator:
    """Orchestrates the generation of a collection of plasma images."""

    # --- Class-level Constants ---
    DEFAULT_ROUGHNESS_LEVELS: List[int] = [1, 2, 4, 8, 16]
    DEFAULT_BLUR_LEVELS: List[int] = [0, 2, 4, 6]
    DEFAULT_NUM_LAYERS: int = 3

    def __init__(
        self,
        output_dir: str = "PlasmaCollection",
        width: int = 512,
        height: int = 512,
        num_images: int = 5,
        palette_dir: str = "Palettes",
        generation_mode: str = "default",
        roughness_levels: Optional[Sequence[int]] = None,
        blur_levels: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        animate: bool = False,
        frames: int = 20,
        max_workers: Optional[int] = None,
        generate_palettes: bool = False,
        num_palettes_to_generate: int = 150,
    ):
        """Initializes the PlasmaGenerator.

        Args:
            output_dir: Directory to save the generated images.
            width: Image width in pixels.
            height: Image height in pixels.
            num_images: Number of unique image sets to generate.
            palette_dir: Directory containing .map palette files.
            generation_mode: The mode of generation ('default', 'roughness', etc.).
            roughness_levels: A sequence of roughness values to use.
            blur_levels: A sequence of blur radius values to use.
            seed: A seed for reproducible randomness.
            animate: If True, generates an animated GIF for each image.
            frames: Number of frames for the animation.
            max_workers: Maximum number of parallel processes to use.
            generate_palettes: If True, generates palettes if the directory is empty.
            num_palettes_to_generate: Number of palettes to create.
        """
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.num_images = num_images
        self.palette_dir = Path(palette_dir)
        self.generation_mode = generation_mode
        self.roughness_levels = list(roughness_levels or self.DEFAULT_ROUGHNESS_LEVELS)
        self.blur_levels = list(blur_levels or self.DEFAULT_BLUR_LEVELS)
        self.seed = seed if seed is not None else int(time.time())
        self.animate = animate
        self.frames = frames
        self.max_workers = max_workers
        self.generate_palettes = generate_palettes
        self.num_palettes_to_generate = num_palettes_to_generate

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.palettes = self._load_palettes()
        
        self.logger.info(f"Initialized PlasmaGenerator in mode: '{self.generation_mode}'")
        self.logger.info(f"Output will be saved to: {self.output_dir}")
        self.logger.info(f"Loaded {len(self.palettes)} palettes.")

    def _setup_logging(self) -> None:
        """Configures logging to file and stdout."""
        log_file = self.output_dir / f"plasma_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create a logger specific to this instance
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create file handler with explicit buffer flushing
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_palettes(self) -> List[List[Tuple[int, int, int]]]:
        """Loads palettes from .map files or creates them on-demand."""
        # Check if we need to generate palettes
        if self.generate_palettes:
            if not self.palette_dir.exists() or not any(self.palette_dir.iterdir()):
                self.logger.info("Palette directory is empty or does not exist. Generating new palettes...")
                self.palette_dir.mkdir(parents=True, exist_ok=True)
                
                # Set a fixed seed for palette generation to ensure reproducibility
                if hasattr(self, 'seed'):
                    random.seed(self.seed)
                
                for i in range(self.num_palettes_to_generate):
                    palette = _generate_palette()
                    filename = self.palette_dir / f"palette_{i+1:03d}.map"
                    _save_as_map(palette, filename)
                self.logger.info(f"Generated {self.num_palettes_to_generate} palettes in {self.palette_dir}.")
            else:
                self.logger.info("--generate_palettes flag set, but palette directory already contains files. Using existing palettes.")

        # Now, proceed with the original loading logic
        palettes = []
        if self.palette_dir.is_dir():
            for path in self.palette_dir.glob("*.map"):
                try:
                    with path.open("r") as f:
                        palette = []
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split()
                            if len(parts) >= 3:
                                r, g, b = map(int, parts[:3])
                                if all(0 <= val <= 255 for val in (r, g, b)):
                                    palette.append((r, g, b))
                        if len(palette) >= 2:
                            palettes.append(palette)
                except (IOError, ValueError) as e:
                    self.logger.warning(f"Could not load palette {path.name}: {e}")

        if not palettes:
            self.logger.warning("No valid palettes found. Using default Matplotlib gradients.")
            return [self._create_gradient_palette(name) for name in ["plasma", "viridis", "magma"]]
        
        return palettes

    def _create_gradient_palette(self, colormap_name: str, size: int = 256) -> List[Tuple[int, int, int]]:
        """Creates a palette from a Matplotlib colormap."""
        cmap = plt.get_cmap(colormap_name)
        return [tuple(int(c * 255) for c in cmap(i / size)[:3]) for i in range(size)]

    def _prepare_tasks(self) -> List[Tuple]:
        """Prepares a list of tasks for the worker processes."""
        tasks = []
        
        # Set a fixed seed for palette selection to ensure reproducibility
        if hasattr(self, 'seed'):
            random.seed(self.seed)
        
        for img_index in range(self.num_images):
            palette = random.choice(self.palettes)
            palette_bytes = bytes(sum(palette, ()))
            palette_hash = hashlib.md5(palette_bytes).hexdigest()[:6]
            
            # Create a unique seed for each task based on the base seed
            task_seed = self.seed + img_index * 1000
            
            if self.generation_mode == "roughness":
                for roughness in self.roughness_levels:
                    params = {"roughness": roughness, "blur": 0}
                    tasks.append(self._create_task_tuple(img_index, params, palette, palette_hash, task_seed))
            elif self.generation_mode == "blur":
                for blur in self.blur_levels:
                    params = {"blur": blur, "roughness": 8} # Fixed roughness for blur comparison
                    tasks.append(self._create_task_tuple(img_index, params, palette, palette_hash, task_seed))
            elif self.generation_mode == "layered":
                params = {
                    "num_layers": self.DEFAULT_NUM_LAYERS,
                    "roughness_values": [2, 8, 16],
                    "blur_values": [0, 2, 4],
                }
                tasks.append(self._create_task_tuple(img_index, params, palette, palette_hash, task_seed))
            elif self.generation_mode == "colormap":
                for colormap in ["plasma", "viridis", "magma", "inferno", "rainbow"]:
                    params = {"colormap": colormap, "roughness": 8, "blur": 0}
                    tasks.append(self._create_task_tuple(img_index, params, palette, palette_hash, task_seed))
            else:  # default mode
                params = {
                    "roughness": random.choice(self.roughness_levels),
                    "blur": random.choice(self.blur_levels),
                }
                tasks.append(self._create_task_tuple(img_index, params, palette, palette_hash, task_seed))
        return tasks

    def _create_task_tuple(
        self, img_index: int, params: Dict, palette: List, palette_hash: str, seed: int
    ) -> Tuple:
        """Helper to create a standardized task tuple."""
        return (
            img_index, params, palette, palette_hash,
            self.width, self.height, seed, str(self.output_dir),
            self.generation_mode, self.animate, self.frames
        )

    def _save_manifest(self, manifest: List[Dict[str, Any]]) -> None:
        """Saves the generation metadata to a JSON file."""
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=4)
        self.logger.info(f"Saved manifest to: {manifest_path}")

    def generate_collection(self) -> List[Dict[str, Any]]:
        """Generates the entire collection of plasma images."""
        start_time = time.time()
        self.logger.info(f"Starting generation of {self.num_images} image sets...")
        
        tasks = self._prepare_tasks()
        self.logger.info(f"Prepared {len(tasks)} tasks for parallel processing.")
        
        manifest = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(plasma_worker, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    manifest.append(result)
                    self.logger.info(f"Completed: {result['filename']}")
                except Exception as e:
                    task = futures[future]
                    self.logger.error(f"Task failed: {task}. Error: {e}")
        
        self._save_manifest(manifest)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Generation complete in {elapsed_time:.2f} seconds.")
        return manifest

    def close(self):
        """Explicitly close the generator and its resources."""
        if hasattr(self, 'logger'):
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

    def __del__(self):
        """Cleanup method to close file handlers."""
        try:
            self.close()
        except:
            pass


# --- Command-Line Interface ---

def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate plasma images with various modes and options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Core arguments
    parser.add_argument("--output_dir", type=str, default="PlasmaCollection", help="Output folder.")
    parser.add_argument("--res", type=str, default="512x512", help="Resolution as WIDTHxHEIGHT.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of base images to generate.")
    parser.add_argument("--palette_dir", type=str, default="Palettes", help="Folder with .map palette files.")
    
    # Generation mode arguments
    mode_group = parser.add_argument_group("Generation Mode")
    mode_group.add_argument(
        "--mode", type=str, default="default", choices=["default", "roughness", "blur", "layered", "colormap"],
        help="Generation mode."
    )
    mode_group.add_argument("--roughness_levels", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Roughness levels.")
    mode_group.add_argument("--blur_levels", type=int, nargs="+", default=[0, 2, 4, 6], help="Blur levels.")
    
    # Animation arguments
    anim_group = parser.add_argument_group("Animation")
    anim_group.add_argument("--animate", action="store_true", help="Generate animated GIFs.")
    anim_group.add_argument("--frames", type=int, default=20, help="Number of frames for animation.")
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    perf_group.add_argument("--max_workers", type=int, default=None, help="Max parallel processes.")
    
    # Palette Generation arguments
    pal_group = parser.add_argument_group("Palette Generation")
    pal_group.add_argument(
        "--generate_palettes", 
        action="store_true", 
        help="If the palette directory is empty, generate a set of random palettes before starting."
    )
    pal_group.add_argument(
        "--num_palettes_to_generate", 
        type=int, 
        default=150, 
        help="Number of palettes to generate when --generate_palettes is used."
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    try:
        args = _parse_args()
        try:
            width, height = map(int, args.res.lower().split("x"))
        except ValueError:
            raise ValueError("Resolution must be in WIDTHxHEIGHT format, e.g., 512x512")

        generator = PlasmaGenerator(
            output_dir=args.output_dir,
            width=width,
            height=height,
            num_images=args.num_images,
            palette_dir=args.palette_dir,
            generation_mode=args.mode,
            roughness_levels=args.roughness_levels,
            blur_levels=args.blur_levels,
            seed=args.seed,
            animate=args.animate,
            frames=args.frames,
            max_workers=args.max_workers,
            generate_palettes=args.generate_palettes,
            num_palettes_to_generate=args.num_palettes_to_generate,
        )
        generator.generate_collection()
        generator.close()
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()