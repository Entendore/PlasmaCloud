import random
import math
import colorsys
import os
import sys

# --- Check for Pillow dependency ---
try:
    from PIL import Image
except ImportError:
    print("Error: The 'Pillow' library is required to generate .png visualizations.")
    print("Please install it by running: pip install Pillow")
    sys.exit(1)


def generate_palette():
    """
    Generates a single "interesting" palette of 256 RGB colors with extreme variations.
    The process involves:
    1. Defining random color waypoints in HSV color space with extreme values.
    2. Using non-linear interpolation for more dramatic transitions.
    3. Applying stronger sine wave distortions with varying frequencies.
    4. Adding random "jumps" for unexpected color shifts.
    """
    num_waypoints = random.randint(4, 8)
    
    # 1. Define random waypoints with extreme values
    waypoint_positions = sorted([0] + [random.randint(1, 254) for _ in range(num_waypoints - 2)] + [255])
    
    waypoints = []
    for pos in waypoint_positions:
        # Use more extreme HSV values for dramatic colors
        h = random.random()  # Full hue range
        # Use full saturation range for more vivid colors
        s = random.choice([random.uniform(0.0, 0.3), random.uniform(0.7, 1.0)])  # Either very desaturated or very saturated
        # Use full value range for more contrast
        v = random.choice([random.uniform(0.1, 0.4), random.uniform(0.7, 1.0)])  # Either very dark or very bright
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        waypoints.append({'pos': pos, 'color': (r * 255, g * 255, b * 255)})

    # 2. Use non-linear interpolation for more dramatic transitions
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
            # Apply non-linear interpolation for more dramatic transitions
            t = t ** random.uniform(0.3, 3.0)  # Either ease-in, ease-out, or ease-in-out
        
        r = start_wp['color'][0] * (1 - t) + end_wp['color'][0] * t
        g = start_wp['color'][1] * (1 - t) + end_wp['color'][1] * t
        b = start_wp['color'][2] * (1 - t) + end_wp['color'][2] * t
        base_palette.append((r, g, b))

    # 3. Apply stronger sine wave distortions with varying frequencies
    final_palette = []
    # Use more extreme amplitude and frequency values
    amp_r, freq_r, phase_r = random.uniform(30, 80), random.uniform(0.02, 0.3), random.uniform(0, math.pi * 2)
    amp_g, freq_g, phase_g = random.uniform(30, 80), random.uniform(0.02, 0.3), random.uniform(0, math.pi * 2)
    amp_b, freq_b, phase_b = random.uniform(30, 80), random.uniform(0.02, 0.3), random.uniform(0, math.pi * 2)

    # Add a second sine wave for each channel for more complexity
    amp_r2, freq_r2, phase_r2 = random.uniform(10, 40), random.uniform(0.05, 0.5), random.uniform(0, math.pi * 2)
    amp_g2, freq_g2, phase_g2 = random.uniform(10, 40), random.uniform(0.05, 0.5), random.uniform(0, math.pi * 2)
    amp_b2, freq_b2, phase_b2 = random.uniform(10, 40), random.uniform(0.05, 0.5), random.uniform(0, math.pi * 2)

    for i, (r, g, b) in enumerate(base_palette):
        # Apply two sine waves for each channel
        r_mod = r + amp_r * math.sin(freq_r * i + phase_r) + amp_r2 * math.sin(freq_r2 * i + phase_r2)
        g_mod = g + amp_g * math.sin(freq_g * i + phase_g) + amp_g2 * math.sin(freq_g2 * i + phase_g2)
        b_mod = b + amp_b * math.sin(freq_b * i + phase_b) + amp_b2 * math.sin(freq_b2 * i + phase_b2)
        
        # 4. Add random "jumps" for unexpected color shifts
        if random.random() < 0.05:  # 5% chance of a jump
            jump_factor = random.uniform(0.5, 1.5)
            r_mod *= jump_factor
            g_mod *= jump_factor
            b_mod *= jump_factor
        
        final_palette.append((
            int(max(0, min(255, r_mod))),
            int(max(0, min(255, g_mod))),
            int(max(0, min(255, b_mod)))
        ))
        
    return final_palette

def save_as_map(palette, filename):
    """Saves a palette to a .map file (RGB 0-255 integer format)."""
    with open(filename, 'w') as f:
        for r, g, b in palette:
            f.write(f"{r} {g} {b}\n")

def save_as_png(palette, filename, width=256, height=50):
    """Creates and saves a visual representation of the palette as a PNG image."""
    # Create a new image with the specified dimensions
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    # Fill the image column by column with the palette colors
    for i in range(width):
        color = palette[i]
        for j in range(height):
            pixels[i, j] = color
            
    img.save(filename)

if __name__ == "__main__":
    num_palettes = 150
    output_dir = "palettes"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_palettes} palettes with extreme variations in the '{output_dir}' directory...")
    
    for i in range(1, num_palettes + 1):
        # Generate the palette data
        palette = generate_palette()
        
        # Define filenames for both .map and .png formats
        base_filename = f"palette_{i:03d}"
        map_filename = os.path.join(output_dir, f"{base_filename}.map")
        png_filename = os.path.join(output_dir, f"{base_filename}.png")
        
        # Save the palette in both formats
        save_as_map(palette, map_filename)
        save_as_png(palette, png_filename)
        
        print(f"Generated {base_filename}.map and {base_filename}.png")
        
    print("\nGeneration complete!")