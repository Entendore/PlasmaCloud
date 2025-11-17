#!/bin/bash

# Create output directory
OUTPUT_DIR="PlasmaAnimation"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/roughness"
mkdir -p "$OUTPUT_DIR/blur"
mkdir -p "$OUTPUT_DIR/random"

# Set image dimensions
RESOLUTION="512x512"

# Define roughness and blur values
ROUGHNESS_VALUES=(2 4 8 16)
BLUR_VALUES=(0 2 4 6)
SEED=12345
PALETTE_DIR="palettes"

# Check if palettes directory exists
if [ ! -d "$PALETTE_DIR" ]; then
    echo "Palettes directory not found: $PALETTE_DIR"
    echo "Please create a palettes directory with .map files or run the script with --generate_palettes"
    exit 1
fi

# Count available palettes
PALETTE_COUNT=$(find "$PALETTE_DIR" -name "*.map" | wc -l)
if [ "$PALETTE_COUNT" -eq 0 ]; then
    echo "No .map files found in $PALETTE_DIR"
    echo "Please add some .map palette files or run the script with --generate_palettes"
    exit 1
fi

echo "Found $PALETTE_COUNT palettes in $PALETTE_DIR"

# Check for ffmpeg and verify it's working
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -1)
    echo "✓ Found ffmpeg: $FFMPEG_VERSION"
    
    # Test ffmpeg with a simple command
    echo "Testing ffmpeg functionality..."
    ffmpeg -f lavfi -i testsrc=duration=0.1:size=320x240 -f null - 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ ffmpeg is working properly"
        FFMPEG_AVAILABLE=true
    else
        echo "✗ ffmpeg test failed"
        echo "Please check your ffmpeg installation"
        FFMPEG_AVAILABLE=false
    fi
else
    echo "✗ ffmpeg not found"
    echo "Please install ffmpeg to create animated GIFs."
    echo ""
    echo "Installation options:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    FFMPEG_AVAILABLE=false
fi

# Function to rename files properly
rename_file() {
    local search_dir="$1"
    local suffix="$2"
    local param_name="$3"
    local param_value="$4"
    
    # Find the most recently modified file matching the pattern
    local file=$(find "$search_dir" -name "Plasma_*.png" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$file" ]; then
        local base_name=$(basename "$file" .png)
        local new_name="${base_name}_${param_name}${param_value}_seed${SEED}${suffix}.png"
        mv "$file" "$search_dir/$new_name"
        echo "Renamed to: $new_name"
        # Return the full path
        echo "$search_dir/$new_name"
    fi
}

# Function to get palette by index
get_palette_by_index() {
    local index=$1
    # Use find with head and tail to get the specific line
    find "$PALETTE_DIR" -name "*.map" | head -n "$index" | tail -1
}

# Function to collect renamed files
collect_renamed_files() {
    local search_dir="$1"
    local files_array_name="$2"
    
    # Wait a moment for file system to sync
    sleep 1
    
    # Find all renamed files with the expected pattern
    local files=()
    for file in "$search_dir"/*_seed${SEED}*.png; do
        if [ -f "$file" ]; then
            files+=("$file")
        fi
    done
    
    # Update the global array
    eval "$files_array_name=(\"\${files[@]}\")"
    
    # Debug output
    echo "Found ${#files[@]} files in $search_dir:"
    for file in "${files[@]}"; do
        echo "  - $(basename "$file")"
    done
}

# Generate 4 images with different roughness values
echo "Generating images with different roughness values..."
roughness_count=0
for roughness in "${ROUGHNESS_VALUES[@]}"; do
    # Select a different palette for each roughness value
    palette_index=$((roughness_count % PALETTE_COUNT))
    selected_palette=$(get_palette_by_index $palette_index)
    palette_name=$(basename "$selected_palette" .map)
    
    echo "Generating roughness $roughness with palette $palette_name (index $palette_index)..."
    python PlasmaCloud.py \
        --output_dir "$OUTPUT_DIR/roughness" \
        --res "$RESOLUTION" \
        --num_images 1 \
        --mode roughness \
        --roughness_levels "$roughness" \
        --palette_dir "$PALETTE_DIR" \
        --seed "$SEED"
    
    # Rename the generated file and store the path
    renamed_file=$(rename_file "$OUTPUT_DIR/roughness" "_palette${palette_name}" "roughness" "$roughness")
    if [ -n "$renamed_file" ]; then
        roughness_files+=("$renamed_file")
    fi
    
    roughness_count=$((roughness_count + 1))
done

# Collect all roughness files
echo "Collecting roughness files..."
collect_renamed_files "$OUTPUT_DIR/roughness" "roughness_files"

# Generate 4 images with different blur values
echo "Generating images with different blur values..."
blur_count=0
for blur in "${BLUR_VALUES[@]}"; do
    # Select a different palette for each blur value
    palette_index=$((blur_count % PALETTE_COUNT))
    selected_palette=$(get_palette_by_index $palette_index)
    palette_name=$(basename "$selected_palette" .map)
    
    echo "Generating blur $blur with palette $palette_name (index $palette_index)..."
    python PlasmaCloud.py \
        --output_dir "$OUTPUT_DIR/blur" \
        --res "$RESOLUTION" \
        --num_images 1 \
        --mode blur \
        --blur_levels "$blur" \
        --palette_dir "$PALETTE_DIR" \
        --seed "$SEED"
    
    # Rename the generated file and store the path
    renamed_file=$(rename_file "$OUTPUT_DIR/blur" "_palette${palette_name}" "blur" "$blur")
    if [ -n "$renamed_file" ]; then
        blur_files+=("$renamed_file")
    fi
    
    blur_count=$((blur_count + 1))
done

# Collect all blur files
echo "Collecting blur files..."
collect_renamed_files "$OUTPUT_DIR/blur" "blur_files"

# Generate 1 random image
echo "Generating random image..."
# Use the first palette for the random image
selected_palette=$(get_palette_by_index 0)
palette_name=$(basename "$selected_palette" .map)

python PlasmaCloud.py \
    --output_dir "$OUTPUT_DIR/random" \
    --res "$RESOLUTION" \
    --num_images 1 \
    --mode default \
    --palette_dir "$PALETTE_DIR" \
    --seed "$SEED"

# Rename the random image
random_file=$(rename_file "$OUTPUT_DIR/random" "_palette${palette_name}" "random" "")

# Create animated GIFs using ffmpeg
if [ "$FFMPEG_AVAILABLE" = true ]; then
    echo "Creating animated GIFs with ffmpeg..."
    
    # Create roughness animation
    echo "Creating roughness animation..."
    if [ ${#roughness_files[@]} -gt 0 ]; then
        # Create a temporary directory for frames
        TEMP_ROUGHNESS=$(mktemp -d)
        count=0
        for frame in "${roughness_files[@]}"; do
            if [ -f "$frame" ]; then
                cp "$frame" "$TEMP_ROUGHNESS/frame$(printf "%03d" $count).png"
                count=$((count + 1))
            fi
        done
        
        # Create animation if we have frames
        if [ $(ls "$TEMP_ROUGHNESS"/*.png 2>/dev/null | wc -l) -gt 0 ]; then
            # Use a simpler ffmpeg command
            ffmpeg -y -framerate 10 -i "$TEMP_ROUGHNESS/frame%03d.png" -c:v gif "$OUTPUT_DIR/roughness_animation_seed${SEED}.gif"
            echo "✓ Created roughness animation: roughness_animation_seed${SEED}.gif"
        else
            echo "⚠ No roughness frames found"
        fi
        
        # Clean up
        rm -rf "$TEMP_ROUGHNESS"
    fi
    
    # Create blur animation
    echo "Creating blur animation..."
    if [ ${#blur_files[@]} -gt 0 ]; then
        # Create a temporary directory for frames
        TEMP_BLUR=$(mktemp -d)
        count=0
        for frame in "${blur_files[@]}"; do
            if [ -f "$frame" ]; then
                cp "$frame" "$TEMP_BLUR/frame$(printf "%03d" $count).png"
                count=$((count + 1))
            fi
        done
        
        # Create animation if we have frames
        if [ $(ls "$TEMP_BLUR"/*.png 2>/dev/null | wc -l) -gt 0 ]; then
            # Use a simpler ffmpeg command
            ffmpeg -y -framerate 10 -i "$TEMP_BLUR/frame%03d.png" -c:v gif "$OUTPUT_DIR/blur_animation_seed${SEED}.gif"
            echo "✓ Created blur animation: blur_animation_seed${SEED}.gif"
        else
            echo "⚠ No blur frames found"
        fi
        
        # Clean up
        rm -rf "$TEMP_BLUR"
    fi
    
    # Create a combined animation showing all images
    echo "Creating combined animation..."
    TEMP_COMBINED=$(mktemp -d)
    
    # Copy all images to temp directory with a common prefix
    count=0
    for frame in "${roughness_files[@]}"; do
        if [ -f "$frame" ]; then
            cp "$frame" "$TEMP_COMBINED/frame$(printf "%03d" $count).png"
            count=$((count + 1))
        fi
    done
    
    for frame in "${blur_files[@]}"; do
        if [ -f "$frame" ]; then
            cp "$frame" "$TEMP_COMBINED/frame$(printf "%03d" $count).png"
            count=$((count + 1))
        fi
    done
    
    # Add random image
    if [ -f "$random_file" ]; then
        cp "$random_file" "$TEMP_COMBINED/frame$(printf "%03d" $count).png"
    fi
    
    # Create combined animation
    if [ $(ls "$TEMP_COMBINED"/*.png 2>/dev/null | wc -l) -gt 0 ]; then
        # Use a simpler ffmpeg command
        ffmpeg -y -framerate 10 -i "$TEMP_COMBINED/frame%03d.png" -c:v gif "$OUTPUT_DIR/combined_animation_seed${SEED}.gif"
        echo "✓ Created combined animation: combined_animation_seed${SEED}.gif"
    else
        echo "⚠ No frames found for combined animation"
    fi
    
    # Clean up
    rm -rf "$TEMP_COMBINED"
    
    echo "✓ All animations created in $OUTPUT_DIR"
fi

# Create a summary text file
cat > "$OUTPUT_DIR/generation_summary.txt" << EOF
Plasma Generation Summary
========================
Seed: $SEED
Resolution: $RESOLUTION
Palettes Directory: $PALETTE_DIR
Available Palettes: $PALETTE_COUNT
Generated on: $(date)

Roughness Values:
 $(printf "  - %s\n" "${ROUGHNESS_VALUES[@]}")

Blur Values:
 $(printf "  - %s\n" "${BLUR_VALUES[@]}")

Files Generated:
- Roughness images in: $OUTPUT_DIR/roughness/
- Blur images in: $OUTPUT_DIR/blur/
- Random image in: $OUTPUT_DIR/random/
- Animations (if ffmpeg available): $OUTPUT_DIR/*_animation_*.gif

Each filename includes:
- Parameter type (roughness/blur/random)
- Parameter value
- Seed value
- Palette name
EOF

echo "✓ Generation complete! Summary saved to $OUTPUT_DIR/generation_summary.txt"
echo "✓ All images generated in $OUTPUT_DIR"