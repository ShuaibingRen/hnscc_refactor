#!/usr/bin/env python
"""
Visualization tool for HNSCC pipeline outputs.

Three viewing modes:
1. view-raw: View raw input overview images (cycle/quench × channel grid)
2. view-tiles: View intermediate tiles (FOV mosaic for specific cycle/channels)  
3. view-ome: View OME-TIFF channels in grid layout
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

# Annotation settings
LABEL_COLOR = (255, 80, 80)  # Red for text
MIN_FONT_SIZE = 48
MAX_FONT_SIZE = 96


def calc_font_size(cell_width: int) -> int:
    """Calculate font size based on cell width (5% of width, clamped)."""
    size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, cell_width // 20))
    return size


def get_font(size: int = 18):
    """Get a font for annotations, fallback to default if not available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse HNSCC image filename to extract components."""
    pattern = r'^(.+?)_(cycle\d+|quench\d+)_w(\d+)_s(\d+)_t\d+\.TIF$'
    match = re.match(pattern, filename, re.IGNORECASE)
    if not match:
        return {}
    return {
        'sample': match.group(1),
        'cycle': match.group(2),
        'channel': int(match.group(3)),
        'fov': int(match.group(4))
    }


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 for display."""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [1, 99])
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def apply_colormap(img: np.ndarray, use_colormap: bool = False) -> np.ndarray:
    """
    Convert grayscale image to RGB.
    If use_colormap is True, apply inferno colormap (black -> purple -> red -> yellow -> white).
    Otherwise, return grayscale as RGB.
    """
    if not use_colormap:
        # Return grayscale as RGB
        return np.stack([img, img, img], axis=-1)
    
    # Inferno colormap LUT (256 entries)
    # Precomputed from matplotlib's inferno
    inferno_lut = np.array([
        [0, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 8], [2, 1, 10], [2, 2, 12], [2, 2, 14], [3, 2, 16],
        [4, 3, 18], [4, 3, 20], [5, 4, 23], [6, 4, 25], [7, 5, 27], [8, 5, 29], [9, 6, 32], [10, 6, 34],
        [11, 7, 36], [12, 7, 38], [14, 7, 41], [15, 8, 43], [16, 8, 45], [18, 8, 48], [19, 9, 50], [20, 9, 52],
        [22, 9, 55], [23, 9, 57], [25, 10, 59], [26, 10, 62], [28, 10, 64], [29, 10, 66], [31, 10, 68],
        [32, 10, 71], [34, 10, 73], [36, 10, 75], [37, 10, 77], [39, 10, 79], [41, 9, 81], [42, 9, 83],
        [44, 9, 85], [46, 9, 87], [47, 9, 88], [49, 9, 90], [51, 9, 91], [52, 9, 93], [54, 9, 94],
        [56, 8, 96], [57, 8, 97], [59, 8, 98], [61, 8, 99], [62, 8, 100], [64, 8, 101], [66, 8, 102],
        [67, 8, 103], [69, 8, 104], [71, 8, 105], [72, 8, 105], [74, 8, 106], [76, 8, 106], [77, 8, 107],
        [79, 8, 107], [81, 9, 108], [82, 9, 108], [84, 9, 108], [86, 10, 109], [87, 10, 109], [89, 10, 109],
        [91, 11, 109], [92, 11, 109], [94, 12, 109], [96, 12, 109], [97, 13, 109], [99, 13, 109],
        [101, 14, 109], [102, 14, 109], [104, 15, 109], [106, 15, 108], [107, 16, 108], [109, 16, 108],
        [111, 17, 107], [112, 18, 107], [114, 18, 106], [116, 19, 106], [117, 19, 105], [119, 20, 105],
        [120, 21, 104], [122, 21, 104], [124, 22, 103], [125, 23, 102], [127, 23, 101], [129, 24, 101],
        [130, 25, 100], [132, 25, 99], [133, 26, 98], [135, 27, 97], [137, 27, 96], [138, 28, 95],
        [140, 29, 94], [141, 29, 93], [143, 30, 92], [145, 31, 91], [146, 32, 90], [148, 32, 89],
        [149, 33, 88], [151, 34, 86], [152, 35, 85], [154, 35, 84], [155, 36, 83], [157, 37, 81],
        [159, 38, 80], [160, 39, 79], [161, 39, 77], [163, 40, 76], [164, 41, 75], [166, 42, 73],
        [167, 43, 72], [169, 43, 70], [170, 44, 69], [172, 45, 67], [173, 46, 66], [175, 47, 64],
        [176, 48, 63], [177, 49, 61], [179, 50, 60], [180, 51, 58], [181, 52, 57], [183, 53, 55],
        [184, 54, 53], [185, 55, 52], [187, 56, 50], [188, 57, 49], [189, 58, 47], [190, 59, 46],
        [192, 60, 44], [193, 61, 43], [194, 62, 41], [195, 63, 40], [196, 65, 38], [198, 66, 37],
        [199, 67, 35], [200, 68, 34], [201, 69, 32], [202, 71, 31], [203, 72, 30], [204, 73, 28],
        [205, 74, 27], [206, 76, 26], [207, 77, 25], [208, 78, 24], [209, 80, 23], [210, 81, 22],
        [211, 82, 21], [212, 84, 20], [213, 85, 19], [214, 87, 19], [215, 88, 18], [215, 90, 18],
        [216, 91, 17], [217, 93, 17], [218, 94, 17], [218, 96, 17], [219, 97, 17], [220, 99, 17],
        [220, 100, 17], [221, 102, 17], [221, 104, 18], [222, 105, 18], [222, 107, 19], [223, 108, 19],
        [223, 110, 20], [224, 112, 21], [224, 113, 22], [224, 115, 22], [225, 117, 23], [225, 118, 24],
        [225, 120, 25], [226, 122, 27], [226, 123, 28], [226, 125, 29], [226, 127, 30], [227, 128, 32],
        [227, 130, 33], [227, 132, 35], [227, 133, 36], [227, 135, 38], [227, 137, 39], [228, 138, 41],
        [228, 140, 43], [228, 142, 45], [228, 143, 46], [228, 145, 48], [228, 147, 50], [228, 148, 52],
        [228, 150, 54], [228, 152, 56], [228, 153, 58], [228, 155, 60], [228, 157, 62], [228, 158, 64],
        [228, 160, 66], [228, 162, 68], [228, 163, 70], [228, 165, 72], [227, 167, 74], [227, 168, 77],
        [227, 170, 79], [227, 172, 81], [227, 173, 83], [226, 175, 86], [226, 177, 88], [226, 178, 90],
        [226, 180, 93], [225, 182, 95], [225, 183, 97], [225, 185, 100], [224, 187, 102], [224, 188, 105],
        [224, 190, 107], [223, 192, 110], [223, 193, 112], [223, 195, 115], [222, 197, 117], [222, 198, 120],
        [222, 200, 123], [221, 202, 125], [221, 203, 128], [221, 205, 131], [220, 207, 133], [220, 208, 136],
        [220, 210, 139], [219, 212, 142], [219, 213, 145], [219, 215, 148], [219, 216, 151], [218, 218, 154],
        [218, 220, 157], [218, 221, 160], [218, 223, 163], [218, 224, 166], [218, 226, 170], [218, 227, 173],
        [218, 229, 176], [218, 230, 180], [219, 232, 183], [219, 233, 187], [219, 235, 190], [220, 236, 194],
        [220, 238, 197], [221, 239, 201], [222, 240, 205], [222, 242, 209], [223, 243, 212], [224, 244, 216],
        [226, 246, 220], [227, 247, 224], [228, 248, 228], [230, 249, 232], [232, 250, 236], [233, 251, 240],
        [235, 252, 244], [238, 253, 248], [240, 254, 252], [243, 255, 255]
    ], dtype=np.uint8)
    
    # Apply LUT
    return inferno_lut[img]


def create_grid_rgb(images: List[np.ndarray], ncols: int, gap: int = 4, 
                    bg_color: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    """Create an RGB grid from list of images with optional gap."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Ensure all images are RGB
    rgb_images = []
    for img in images:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        rgb_images.append(img)
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in rgb_images)
    max_w = max(img.shape[1] for img in rgb_images)
    
    nrows = (len(rgb_images) + ncols - 1) // ncols
    
    # Create output array
    out_h = nrows * max_h + (nrows - 1) * gap
    out_w = ncols * max_w + (ncols - 1) * gap
    grid = np.full((out_h, out_w, 3), bg_color, dtype=np.uint8)
    
    for idx, img in enumerate(rgb_images):
        r, c = divmod(idx, ncols)
        y = r * (max_h + gap)
        x = c * (max_w + gap)
        # Center image if smaller
        dy = (max_h - img.shape[0]) // 2
        dx = (max_w - img.shape[1]) // 2
        grid[y+dy:y+dy+img.shape[0], x+dx:x+dx+img.shape[1]] = img
    
    return grid


def add_labels(grid: np.ndarray, row_labels: List[str], col_labels: List[str],
               cell_height: int, cell_width: int, gap: int = 4) -> np.ndarray:
    """Add row and column labels to a grid image."""
    # Dynamic font size based on cell width
    font_size = calc_font_size(cell_width)
    font = get_font(font_size)
    
    # Label area sizes scale with font
    row_label_width = font_size * 5
    col_label_height = int(font_size * 1.8)
    
    # Create new image with label space
    new_h = grid.shape[0] + col_label_height
    new_w = grid.shape[1] + row_label_width
    labeled = np.full((new_h, new_w, 3), (30, 30, 30), dtype=np.uint8)
    
    # Copy grid
    labeled[col_label_height:, row_label_width:] = grid
    
    # Convert to PIL for text drawing
    pil_img = Image.fromarray(labeled)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw column labels (channel names)
    for i, label in enumerate(col_labels):
        x = row_label_width + i * (cell_width + gap) + cell_width // 2
        y = col_label_height // 2
        draw.text((x, y), label, fill=LABEL_COLOR, font=font, anchor="mm")
    
    # Draw row labels (cycle names)
    for i, label in enumerate(row_labels):
        x = row_label_width // 2
        y = col_label_height + i * (cell_height + gap) + cell_height // 2
        draw.text((x, y), label, fill=LABEL_COLOR, font=font, anchor="mm")
    
    return np.array(pil_img)


def scan_cycles(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Scan directory for cycle and quench folders, sorted by number."""
    cycles = []
    quenches = []
    
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if m := re.match(r'cycle(\d+)', name):
            cycles.append((int(m.group(1)), subdir))
        elif m := re.match(r'quench(\d+)', name):
            quenches.append((int(m.group(1)), subdir))
    
    cycles = [p for _, p in sorted(cycles)]
    quenches = [p for _, p in sorted(quenches)]
    return cycles, quenches


def get_overview_image(cycle_dir: Path, channel: int) -> Optional[np.ndarray]:
    """Get the overview image (largest FOV number) for a cycle/channel."""
    pattern = f'*_w{channel}_*.TIF'
    files = list(cycle_dir.glob(pattern))
    
    if not files:
        return None
    
    # Find max FOV (overview image)
    max_fov = -1
    overview_file = None
    for f in files:
        parsed = parse_filename(f.name)
        if parsed and parsed['fov'] > max_fov:
            max_fov = parsed['fov']
            overview_file = f
    
    if overview_file:
        img = tifffile.imread(overview_file)
        return normalize_to_uint8(img)
    return None


def view_raw_single(dirs: List[Path], channels: List[int], downsample: int,
                    output_path: Path, title: str, use_colormap: bool = False) -> None:
    """Create and save a single raw overview grid."""
    if not dirs:
        return
    
    n_channels = len(channels)
    images = []
    row_labels = []
    cell_size = None
    
    for cycle_dir in dirs:
        row_labels.append(cycle_dir.name)
        for ch_idx, ch in enumerate(channels):
            print(f"  Reading {cycle_dir.name} w{ch}...")
            img = get_overview_image(cycle_dir, ch)
            if img is None:
                img = np.zeros((100, 100), dtype=np.uint8)
            if downsample > 1:
                img = img[::downsample, ::downsample]
            # Apply colormap
            img_rgb = apply_colormap(img, use_colormap)
            images.append(img_rgb)
            
            if cell_size is None:
                cell_size = (img.shape[0], img.shape[1])
    
    # Create grid
    gap = 4
    grid = create_grid_rgb(images, ncols=n_channels, gap=gap)
    
    # Add labels
    col_labels = [f"w{ch}" for ch in channels]
    labeled = add_labels(grid, row_labels, col_labels, cell_size[0], cell_size[1], gap)
    
    # Save
    Image.fromarray(labeled).save(output_path)
    print(f"Saved: {output_path} ({labeled.shape[1]}x{labeled.shape[0]})")


def view_raw(input_dir: Path, output_path: Path, channels: List[int] = [1, 2, 3, 4],
             downsample: int = 1, use_colormap: bool = False) -> None:
    """
    View raw input: grid of overview images.
    Cycles and quenches saved separately.
    """
    print(f"Scanning {input_dir}...")
    cycles, quenches = scan_cycles(input_dir)
    
    if not cycles and not quenches:
        print("No cycle/quench directories found!", file=sys.stderr)
        return
    
    # Generate output paths
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    
    # Save cycles
    if cycles:
        cycle_output = parent / f"{stem}_cycles{suffix}"
        print(f"\n=== Cycles ({len(cycles)}) ===")
        view_raw_single(cycles, channels, downsample, cycle_output, "Cycles", use_colormap)
    
    # Save quenches separately
    if quenches:
        quench_output = parent / f"{stem}_quenches{suffix}"
        print(f"\n=== Quenches ({len(quenches)}) ===")
        view_raw_single(quenches, channels, downsample, quench_output, "Quenches", use_colormap)
    
    print(f"\nLayout: {len(cycles)} cycles, {len(quenches)} quenches × {len(channels)} channels")


def view_tiles(input_dir: Path, output_path: Path, cycle: str, 
               channels: List[int], grid_rows: int, grid_cols: int,
               tile_size: int = 256, use_colormap: bool = False) -> None:
    """
    View intermediate tiles: mosaic of FOVs for specific cycle/channels.
    FOVs arranged in raster order (top-to-bottom, left-to-right).
    """
    cycle_dir = input_dir / cycle
    if not cycle_dir.exists():
        print(f"Cycle directory not found: {cycle_dir}", file=sys.stderr)
        return
    
    n_fovs = grid_rows * grid_cols
    n_channels = len(channels)
    
    # Create sub-grid for each channel
    channel_grids = []
    
    for ch_idx, ch in enumerate(channels):
        print(f"Processing channel {ch}...")
        
        # Find FOV images for this channel
        files = list(cycle_dir.glob(f'*_w{ch}_*.TIF'))
        fov_files = {}
        for f in files:
            parsed = parse_filename(f.name)
            if parsed:
                fov_files[parsed['fov']] = f
        
        # Collect images in raster order (top-to-bottom, left-to-right)
        fov_images = []
        for col in range(grid_cols):
            for row in range(grid_rows):
                fov_num = col * grid_rows + row + 1  # 1-indexed
                if fov_num in fov_files:
                    img = tifffile.imread(fov_files[fov_num])
                    img = normalize_to_uint8(img)
                    if img.shape[0] != tile_size or img.shape[1] != tile_size:
                        pil_img = Image.fromarray(img)
                        pil_img = pil_img.resize((tile_size, tile_size), Image.LANCZOS)
                        img = np.array(pil_img)
                else:
                    img = np.zeros((tile_size, tile_size), dtype=np.uint8)
                fov_images.append(img)
        
        # Rearrange to visual grid (row-major) and apply colormap
        visual_images = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                idx = col * grid_rows + row
                img = fov_images[idx] if idx < len(fov_images) else np.zeros((tile_size, tile_size), dtype=np.uint8)
                img_rgb = apply_colormap(img, use_colormap)
                visual_images.append(img_rgb)
        
        fov_grid = create_grid_rgb(visual_images, ncols=grid_cols, gap=2)
        channel_grids.append((ch, fov_grid))
    
    # Stack channel grids horizontally with labels
    gap = 12
    max_h = max(g.shape[0] for _, g in channel_grids)
    grid_width = channel_grids[0][1].shape[1] if channel_grids else 256
    
    # Dynamic font size based on grid width
    font_size = calc_font_size(grid_width)
    header_height = int(font_size * 1.8)
    font = get_font(font_size)
    
    final_parts = []
    for ch, grid in channel_grids:
        # Create header
        header = np.full((header_height, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        header_pil = Image.fromarray(header)
        draw = ImageDraw.Draw(header_pil)
        draw.text((grid.shape[1]//2, header_height//2), f"w{ch}", 
                  fill=LABEL_COLOR, font=font, anchor="mm")
        header = np.array(header_pil)
        
        # Combine header and grid
        combined = np.vstack([header, grid])
        if combined.shape[0] < max_h + header_height:
            pad = np.full((max_h + header_height - combined.shape[0], combined.shape[1], 3), 
                         (30, 30, 30), dtype=np.uint8)
            combined = np.vstack([combined, pad])
        final_parts.append(combined)
    
    # Join with gaps
    final_grid = final_parts[0]
    for part in final_parts[1:]:
        gap_col = np.full((final_grid.shape[0], gap, 3), (30, 30, 30), dtype=np.uint8)
        final_grid = np.hstack([final_grid, gap_col, part])
    
    # Add cycle label on left
    cycle_label_width = font_size * 4
    cycle_label = np.full((final_grid.shape[0], cycle_label_width, 3), (30, 30, 30), dtype=np.uint8)
    cycle_pil = Image.fromarray(cycle_label)
    draw = ImageDraw.Draw(cycle_pil)
    draw.text((cycle_label_width//2, final_grid.shape[0]//2), cycle, 
              fill=LABEL_COLOR, font=font, anchor="mm")
    cycle_label = np.array(cycle_pil)
    final_grid = np.hstack([cycle_label, final_grid])
    
    # Save
    Image.fromarray(final_grid).save(output_path)
    print(f"Saved: {output_path} ({final_grid.shape[1]}x{final_grid.shape[0]})")
    print(f"Layout: {grid_rows}×{grid_cols} FOVs × {n_channels} channels")


def view_ome(ome_path: Path, output_path: Path, channels: List[int],
             grid_cols: int = 4, downsample: int = 0, use_colormap: bool = False,
             max_dim: int = 8000) -> None:
    """
    View OME-TIFF: grid of specified channels with index labels.
    Uses pyramidal reading for performance if available.
    
    If downsample=0, automatically calculate based on max_dim to keep
    the output image's largest dimension under max_dim pixels.
    """
    print(f"Opening {ome_path}...")
    
    with tifffile.TiffFile(ome_path) as tif:
        # Get image dimensions first for auto-downsample calculation
        if len(tif.series) > 0:
            series = tif.series[0]
            full_shape = series.shape  # (C, H, W) or (H, W)
            if len(full_shape) == 2:
                full_h, full_w = full_shape
                n_total_ch = 1
            else:
                n_total_ch, full_h, full_w = full_shape[0], full_shape[-2], full_shape[-1]
        else:
            # Fallback: read shape from first page
            full_h, full_w = tif.pages[0].shape[:2]
            n_total_ch = len(tif.pages)
        
        # Auto-calculate downsample if not specified (downsample=0)
        if downsample <= 0:
            n_channels_to_show = len(channels)
            grid_rows = (n_channels_to_show + grid_cols - 1) // grid_cols
            # Estimate output dimensions: grid of (cell_h x cell_w) images
            # cell size = full_size / ds, output = grid * cell + gaps + headers
            # Simplified: output_w ~ grid_cols * (full_w / ds), output_h ~ grid_rows * (full_h / ds)
            # We want max(output_w, output_h) <= max_dim
            # => ds >= max(grid_cols * full_w, grid_rows * full_h) / max_dim
            estimated_out_w = grid_cols * full_w
            estimated_out_h = grid_rows * full_h
            max_estimated = max(estimated_out_w, estimated_out_h)
            downsample = max(1, int(np.ceil(max_estimated / max_dim)))
            print(f"Auto downsample: {downsample} (input: {full_w}x{full_h}, "
                  f"grid: {grid_rows}x{grid_cols}, target max_dim: {max_dim})")
        
        # Check for pyramid levels
        if len(tif.series) > 0 and len(tif.series[0].levels) > 1:
            series = tif.series[0]
            level_idx = 0
            for i, level in enumerate(series.levels):
                if level.shape[-1] <= series.shape[-1] // downsample:
                    level_idx = i
                    break
            print(f"Using pyramid level {level_idx}")
            data = series.levels[level_idx].asarray()
        else:
            print("No pyramid, reading full resolution...")
            data = tif.asarray()
            if downsample > 1:
                data = data[:, ::downsample, ::downsample]
    
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    n_total_channels = data.shape[0]
    print(f"Total channels: {n_total_channels}")
    
    # Validate channel indices
    valid_channels = [c for c in channels if 0 <= c < n_total_channels]
    if len(valid_channels) < len(channels):
        invalid = set(channels) - set(valid_channels)
        print(f"Warning: Invalid channel indices ignored: {invalid}")
    
    if not valid_channels:
        print("No valid channels to display!", file=sys.stderr)
        return
    
    # Extract and normalize channels with colors
    images = []
    for i, ch in enumerate(valid_channels):
        img = normalize_to_uint8(data[ch])
        img_rgb = apply_colormap(img, use_colormap)
        images.append((ch, img_rgb))
    
    # Create grid with gaps and labels
    gap = 8
    cell_h, cell_w = images[0][1].shape[:2]
    
    # Dynamic font size based on cell width
    font_size = calc_font_size(cell_w)
    header_height = int(font_size * 1.8)
    font = get_font(font_size)
    
    nrows = (len(images) + grid_cols - 1) // grid_cols
    
    # Calculate total size
    total_h = nrows * (cell_h + header_height) + (nrows - 1) * gap
    total_w = grid_cols * cell_w + (grid_cols - 1) * gap
    
    final_grid = np.full((total_h, total_w, 3), (30, 30, 30), dtype=np.uint8)
    
    for idx, (ch, img_rgb) in enumerate(images):
        r, c = divmod(idx, grid_cols)
        y = r * (cell_h + header_height + gap)
        x = c * (cell_w + gap)
        
        # Draw header with channel index
        header = np.full((header_height, cell_w, 3), (30, 30, 30), dtype=np.uint8)
        header_pil = Image.fromarray(header)
        draw = ImageDraw.Draw(header_pil)
        draw.text((cell_w//2, header_height//2), f"ch{ch}", 
                  fill=LABEL_COLOR, font=font, anchor="mm")
        header = np.array(header_pil)
        
        # Place header and image
        final_grid[y:y+header_height, x:x+cell_w] = header
        final_grid[y+header_height:y+header_height+cell_h, x:x+cell_w] = img_rgb
    
    # Save
    Image.fromarray(final_grid).save(output_path)
    print(f"Saved: {output_path} ({final_grid.shape[1]}x{final_grid.shape[0]})")
    print(f"Layout: {len(valid_channels)} channels in {grid_cols}-column grid")


def view_correction(input_dir: Path, output_path: Path, 
                    corr_type: str = 'flatfield', use_colormap: bool = False) -> None:
    """
    View illumination correction files (flatfield/darkfield).
    Generates separate images for cycles and quenches.
    Layout: rows = cycles/quenches, cols = channels (w1, w2, w3, w4).
    """
    input_dir = Path(input_dir)
    
    # Find all correction files
    pattern = f'*_{corr_type}.tif'
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        print(f"No {corr_type} files found in {input_dir}")
        return
    
    # Parse files to get cycle/quench/channel info
    cycle_files = {}   # (cycle_num, channel) -> file
    quench_files = {}  # (quench_num, channel) -> file
    channels = set()
    
    for f in files:
        # Try cycle pattern: cycle{N}_w{channel}_{type}.tif
        match = re.match(r'cycle(\d+)_w(\d+)_' + corr_type + r'\.tif', f.name, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            channel = int(match.group(2))
            cycle_files[(num, channel)] = f
            channels.add(channel)
            continue
        
        # Try quench pattern: quench{N}_w{channel}_{type}.tif
        match = re.match(r'quench(\d+)_w(\d+)_' + corr_type + r'\.tif', f.name, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            channel = int(match.group(2))
            quench_files[(num, channel)] = f
            channels.add(channel)
    
    channels = sorted(channels)
    
    # Generate output paths
    base = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    
    # Process cycles
    if cycle_files:
        cycles = sorted(set(k[0] for k in cycle_files.keys()))
        out_path = parent / f"{base}_cycles{suffix}"
        _render_correction_grid(cycle_files, cycles, channels, "cycle", 
                                corr_type, out_path, use_colormap)
    
    # Process quenches
    if quench_files:
        quenches = sorted(set(k[0] for k in quench_files.keys()))
        out_path = parent / f"{base}_quenches{suffix}"
        _render_correction_grid(quench_files, quenches, channels, "quench",
                                corr_type, out_path, use_colormap)
    
    if not cycle_files and not quench_files:
        print(f"No valid {corr_type} files found")


def _render_correction_grid(file_map: Dict, rows_list: List[int], channels: List[int],
                            row_type: str, corr_type: str, output_path: Path,
                            use_colormap: bool) -> None:
    """Render a grid of correction images."""
    n_rows = len(rows_list)
    n_cols = len(channels)
    
    print(f"Found {len(file_map)} {corr_type} files: {n_rows} {row_type}s × {n_cols} channels")
    
    # Read first image to get dimensions
    first_file = next(iter(file_map.values()))
    sample_img = tifffile.imread(first_file)
    h, w = sample_img.shape[:2]
    
    # Downsample for visualization
    target_size = 512
    scale = max(1, max(h, w) // target_size)
    cell_h = h // scale
    cell_w = w // scale
    
    # Calculate font size
    font_size = calc_font_size(cell_w)
    header_height = int(font_size * 1.8)
    row_label_width = font_size * 4
    font = get_font(font_size)
    
    gap = 4
    total_h = header_height + n_rows * cell_h + (n_rows - 1) * gap
    total_w = row_label_width + n_cols * cell_w + (n_cols - 1) * gap
    
    grid = np.full((total_h, total_w, 3), (30, 30, 30), dtype=np.uint8)
    
    # Fill grid
    for r_idx, row_num in enumerate(rows_list):
        for c_idx, channel in enumerate(channels):
            key = (row_num, channel)
            if key not in file_map:
                continue
            
            # Read and process image
            img = tifffile.imread(file_map[key])
            img = img[::scale, ::scale]
            img_norm = normalize_to_uint8(img)
            img_rgb = apply_colormap(img_norm, use_colormap)
            
            # Calculate position
            y = header_height + r_idx * (cell_h + gap)
            x = row_label_width + c_idx * (cell_w + gap)
            
            # Place image
            ih, iw = img_rgb.shape[:2]
            grid[y:y+ih, x:x+iw] = img_rgb
    
    # Add labels
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    
    # Column headers (channels)
    for c_idx, channel in enumerate(channels):
        x = row_label_width + c_idx * (cell_w + gap) + cell_w // 2
        y = header_height // 2
        draw.text((x, y), f"w{channel}", fill=LABEL_COLOR, font=font, anchor="mm")
    
    # Row labels
    prefix = "c" if row_type == "cycle" else "q"
    for r_idx, row_num in enumerate(rows_list):
        x = row_label_width // 2
        y = header_height + r_idx * (cell_h + gap) + cell_h // 2
        draw.text((x, y), f"{prefix}{row_num}", fill=LABEL_COLOR, font=font, anchor="mm")
    
    # Save
    Image.fromarray(np.array(pil_img)).save(output_path)
    print(f"Saved: {output_path} ({total_w}x{total_h})")
    print(f"Layout: {n_rows} {row_type}s × {n_cols} channels ({corr_type})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualization tool for HNSCC pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View raw input overview images (cycles and quenches saved separately)
  python view_out.py view-raw /path/to/raw -o overview.png
  
  # View intermediate tiles for specific cycle and channels
  python view_out.py view-tiles /path/to/align -c cycle1 -w 1 2 -r 6 -C 5 -o tiles.png
  
  # View OME-TIFF channels
  python view_out.py view-ome /path/to/output.ome.tiff -c 0 1 2 3 4 5 -o channels.png
  
  # View illumination correction files
  python view_out.py view-correction /path/to/illuminate/correction -t flatfield -o flatfield.png
"""
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # view-raw subcommand
    p_raw = subparsers.add_parser('view-raw', 
        help='View raw input overview images (cycle/quench × channel grid)')
    p_raw.add_argument('input_dir', type=Path, help='Input directory')
    p_raw.add_argument('-o', '--output', type=Path, default=Path('raw_overview.png'),
                       help='Output PNG path (will add _cycles/_quenches suffix)')
    p_raw.add_argument('-w', '--channels', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='Channels to include (default: 1 2 3 4)')
    p_raw.add_argument('-d', '--downsample', type=int, default=1,
                       help='Downsample factor (default: 1)')
    p_raw.add_argument('-m', '--colormap', action='store_true',
                       help='Use inferno colormap instead of grayscale')
    
    # view-tiles subcommand
    p_tiles = subparsers.add_parser('view-tiles',
        help='View intermediate tiles (FOV mosaic for specific cycle/channels)')
    p_tiles.add_argument('input_dir', type=Path, help='Input directory (e.g., align/)')
    p_tiles.add_argument('-c', '--cycle', type=str, required=True,
                         help='Cycle name (e.g., cycle1)')
    p_tiles.add_argument('-w', '--channels', type=int, nargs='+', required=True,
                         help='Channels to view (e.g., 1 2)')
    p_tiles.add_argument('-r', '--rows', type=int, required=True,
                         help='Number of FOV rows')
    p_tiles.add_argument('-C', '--cols', type=int, required=True,
                         help='Number of FOV columns')
    p_tiles.add_argument('-s', '--tile-size', type=int, default=256,
                         help='Tile size for display (default: 256)')
    p_tiles.add_argument('-o', '--output', type=Path, default=Path('tiles.png'),
                         help='Output PNG path')
    p_tiles.add_argument('-m', '--colormap', action='store_true',
                         help='Use inferno colormap instead of grayscale')
    
    # view-ome subcommand
    p_ome = subparsers.add_parser('view-ome',
        help='View OME-TIFF channels in grid layout')
    p_ome.add_argument('ome_path', type=Path, help='OME-TIFF file path')
    p_ome.add_argument('-c', '--channels', type=int, nargs='+', required=True,
                       help='Channel indices to view (0-indexed)')
    p_ome.add_argument('-C', '--cols', type=int, default=4,
                       help='Number of grid columns (default: 4)')
    p_ome.add_argument('-d', '--downsample', type=int, default=0,
                       help='Downsample factor (0=auto, default: 0)')
    p_ome.add_argument('--max-dim', type=int, default=8000,
                       help='Max output dimension in pixels for auto-downsample (default: 8000)')
    p_ome.add_argument('-o', '--output', type=Path, default=Path('ome_channels.png'),
                       help='Output PNG path')
    p_ome.add_argument('-m', '--colormap', action='store_true',
                       help='Use inferno colormap instead of grayscale')
    
    # view-correction subcommand
    p_corr = subparsers.add_parser('view-correction',
        help='View illumination correction files (flatfield/darkfield)')
    p_corr.add_argument('input_dir', type=Path, help='Correction directory')
    p_corr.add_argument('-t', '--type', type=str, default='flatfield',
                        choices=['flatfield', 'darkfield'],
                        help='Correction type (default: flatfield)')
    p_corr.add_argument('-o', '--output', type=Path, default=Path('correction.png'),
                        help='Output PNG path')
    p_corr.add_argument('-m', '--colormap', action='store_true',
                        help='Use inferno colormap instead of grayscale')
    
    args = parser.parse_args()
    
    if args.command == 'view-raw':
        if not args.input_dir.exists():
            print(f"Error: {args.input_dir} not found", file=sys.stderr)
            return 1
        view_raw(args.input_dir, args.output, args.channels, args.downsample, args.colormap)
    
    elif args.command == 'view-tiles':
        if not args.input_dir.exists():
            print(f"Error: {args.input_dir} not found", file=sys.stderr)
            return 1
        view_tiles(args.input_dir, args.output, args.cycle, args.channels,
                   args.rows, args.cols, args.tile_size, args.colormap)
    
    elif args.command == 'view-ome':
        if not args.ome_path.exists():
            print(f"Error: {args.ome_path} not found", file=sys.stderr)
            return 1
        view_ome(args.ome_path, args.output, args.channels, args.cols, args.downsample, 
                 args.colormap, args.max_dim)
    
    elif args.command == 'view-correction':
        if not args.input_dir.exists():
            print(f"Error: {args.input_dir} not found", file=sys.stderr)
            return 1
        view_correction(args.input_dir, args.output, args.type, args.colormap)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

