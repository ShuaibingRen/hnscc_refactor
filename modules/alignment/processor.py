"""
Image Alignment Module

FOV-level alignment using ECC (Enhanced Correlation Coefficient).
Aligns all cycles and quenches to a reference cycle (default: cycle0 DAPI).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

from ...core.base import BaseProcessor
from ...core.utils import (
    scan_input_directory,
    parse_filename
)
from ...config import Config


def _ecc_align(ref_img: np.ndarray, moving_img: np.ndarray, 
               scale: float = 0.1, max_iterations: int = 5000,
               epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Align moving image to reference using ECC with Euclidean transform.
    
    Args:
        ref_img: Reference image (grayscale float32)
        moving_img: Image to align (grayscale float32)
        scale: Downscale factor for speed (0.1 = 10x faster)
        max_iterations: Maximum ECC iterations
        epsilon: Convergence threshold
        
    Returns:
        Tuple of (aligned_image, warp_matrix, success)
    """
    # Downscale for faster alignment
    h, w = ref_img.shape[:2]
    small_ref = cv2.resize(ref_img, None, fx=scale, fy=scale)
    small_moving = cv2.resize(moving_img, None, fx=scale, fy=scale)
    
    # Keep as float32 for better precision, consistent with original implementation
    # Initialize warp matrix (Euclidean: 2x3)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # ECC criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
    
    try:
        # Find transformation on downscaled images
        # inputMask=None, gaussFiltSize=15 (critical for convergence)
        _, warp_matrix = cv2.findTransformECC(
            small_ref, small_moving,
            warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 15
        )
        
        # Scale transformation back to original size
        warp_matrix[0, 2] /= scale  # tx
        warp_matrix[1, 2] /= scale  # ty
        
        # Apply to full-size image
        aligned = cv2.warpAffine(
            moving_img, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return aligned, warp_matrix, True
        
    except cv2.error as e:
        # ECC failed (e.g., non-convergence)
        return moving_img, warp_matrix, False


def _compute_overlap_ratio(warp_matrix: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """
    Compute the ratio of overlapping area after transformation.
    
    Args:
        warp_matrix: 2x3 Euclidean transformation matrix
        img_shape: Image shape (height, width)
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    h, w = img_shape
    
    # Get translation
    tx = abs(warp_matrix[0, 2])
    ty = abs(warp_matrix[1, 2])
    
    # Approximate overlap (ignoring rotation for simplicity)
    overlap_w = max(0, w - tx)
    overlap_h = max(0, h - ty)
    
    return (overlap_w * overlap_h) / (w * h)


def _crop_to_overlap(images: List[np.ndarray], 
                     warp_matrices: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple]:
    """
    Crop all images to their common overlapping region.
    
    Args:
        images: List of aligned images
        warp_matrices: List of transformation matrices
        
    Returns:
        Tuple of (cropped_images, crop_bounds)
    """
    if not images or not warp_matrices:
        return images, (0, 0, 0, 0)
    
    h, w = images[0].shape[:2]
    
    # Find maximum shifts
    max_tx = max(abs(m[0, 2]) for m in warp_matrices)
    max_ty = max(abs(m[1, 2]) for m in warp_matrices)
    
    # Determine crop bounds
    x_start = int(np.ceil(max_tx))
    y_start = int(np.ceil(max_ty))
    x_end = w - x_start
    y_end = h - y_start
    
    if x_end <= x_start or y_end <= y_start:
        return images, (0, h, 0, w)
    
    # Crop all images
    cropped = [img[y_start:y_end, x_start:x_end] for img in images]
    
    return cropped, (y_start, y_end, x_start, x_end)


class AlignmentProcessor(BaseProcessor):
    """
    FOV-level alignment processor using ECC.
    
    Aligns all cycles and quenches to a reference cycle (cycle0 DAPI).
    """
    
    def __init__(self, config: Config):
        """
        Initialize alignment processor.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.scale = config.get('alignment.scale', 0.1)
        self.mean_threshold = config.get('alignment.mean_threshold', 200)
        self.overlap_threshold = config.get('alignment.overlap_threshold', 0.2)
        self.ref_channel = config.get('alignment.reference_channel', 1)
        self.max_iterations = config.get('alignment.max_iterations', 5000)
        self.epsilon = config.get('alignment.epsilon', 1e-10)
        self.channels_per_cycle = config.get('input.channels_per_cycle', 4)
        # on_failure: 'skip' (default) or 'copy' (save unaligned)
        self.on_failure = config.get('alignment.on_failure', 'skip')
    
    def process(self, input_dir: Path, output_dir: Path, 
                reference_cycle: str = 'cycle0', **kwargs) -> Dict[str, Any]:
        """
        Align all cycles to reference cycle.
        
        Args:
            input_dir: Input directory (typically illuminate/ output)
            output_dir: Output directory for aligned images
            reference_cycle: Reference cycle name (default: cycle0)
            
        Returns:
            Dict with processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self._log_start("Image Alignment", input_dir)
        
        # Scan input
        scan_result = scan_input_directory(input_dir)
        
        # Get reference cycle directory
        ref_cycle_idx = int(reference_cycle.replace('cycle', ''))
        if ref_cycle_idx not in scan_result['cycles']:
            raise ValueError(f"Reference cycle {reference_cycle} not found")
        
        ref_cycle_dir = scan_result['cycles'][ref_cycle_idx]
        
        # Get all directories to align (including quenches)
        all_dirs = {}
        for idx, path in scan_result['cycles'].items():
            all_dirs[f'cycle{idx}'] = path
        for idx, path in scan_result['quenches'].items():
            all_dirs[f'quench{idx}'] = path
        
        self.logger.info(f"Aligning to {reference_cycle}, "
                        f"{len(all_dirs)} directories to process")
        
        # Get reference DAPI images (all TIF files with reference channel)
        ref_dapi_files = list(ref_cycle_dir.glob(f'*_w{self.ref_channel}_*.TIF'))
        ref_dapi_dict = {parse_filename(f.name)['fov']: f for f in ref_dapi_files 
                         if parse_filename(f.name)}
        
        # Process each directory
        results = {'aligned': 0, 'skipped_intensity': 0, 'skipped_overlap': 0, 'failed': 0}
        
        for dir_name, dir_path in all_dirs.items():
            self._align_directory(
                dir_path, output_dir / dir_name,
                ref_dapi_dict, reference_cycle, dir_name, results
            )
        
        self._log_complete("Image Alignment", output_dir, results)
        
        return {
            'status': 'success',
            'reference_cycle': reference_cycle,
            'stats': results
        }
    
    def _align_directory(self, src_dir: Path, dst_dir: Path,
                        ref_dapi_dict: Dict[int, Path],
                        ref_cycle: str, current_name: str,
                        results: Dict):
        """Align all FOVs in a directory to reference"""
        dst_dir = self._ensure_dir(dst_dir)
        
        self.logger.info(f"  Aligning {current_name}...")
        
        # Group files by FOV
        all_files = list(src_dir.glob('*.TIF')) + list(src_dir.glob('*.tif'))
        fov_files = {}
        for f in all_files:
            parsed = parse_filename(f.name)
            if parsed:
                fov = parsed['fov']
                if fov not in fov_files:
                    fov_files[fov] = []
                fov_files[fov].append(f)
        
        # Process each FOV
        for fov, files in fov_files.items():
            if fov not in ref_dapi_dict:
                continue
            
            # Load reference DAPI
            ref_img = cv2.imread(str(ref_dapi_dict[fov]), cv2.IMREAD_ANYDEPTH)
            if ref_img is None:
                results['failed'] += 1
                continue
            ref_img = ref_img.astype(np.float32)
            
            # QC: Check mean intensity
            if np.mean(ref_img) < self.mean_threshold:
                self.logger.warning(f"FOV {fov} skipped: Reference mean intensity {np.mean(ref_img):.1f} < {self.mean_threshold}")
                results['skipped_intensity'] += 1
                if self.on_failure == 'copy':
                    self._copy_files(files, dst_dir)
                    results['aligned'] += 1  # Count as processed if copied
                continue
            
            # Find DAPI in current cycle for alignment
            dapi_file = next((f for f in files if f'_w{self.ref_channel}_' in f.name), None)
            if dapi_file is None:
                self.logger.warning(f"FOV {fov} skipped: No DAPI (w{self.ref_channel}) file found in {current_name}")
                results['failed'] += 1
                if self.on_failure == 'copy':
                    self._copy_files(files, dst_dir)
                continue
            
            moving_dapi = cv2.imread(str(dapi_file), cv2.IMREAD_ANYDEPTH)
            if moving_dapi is None:
                self.logger.warning(f"FOV {fov} skipped: Failed to load DAPI file {dapi_file.name}")
                results['failed'] += 1
                if self.on_failure == 'copy':
                    self._copy_files(files, dst_dir)
                continue
            moving_dapi = moving_dapi.astype(np.float32)
            
            # Skip if same as reference (cycle0)
            if current_name == ref_cycle:
                # Just copy files
                self._copy_files(files, dst_dir)
                results['aligned'] += 1
                continue
            
            # Align
            _, warp_matrix, success = _ecc_align(
                ref_img, moving_dapi, 
                scale=self.scale,
                max_iterations=self.max_iterations,
                epsilon=self.epsilon
            )
            
            if not success:
                self.logger.warning(f"FOV {fov} failed: ECC alignment did not converge for {current_name}")
                results['failed'] += 1
                if self.on_failure == 'copy':
                    self._copy_files(files, dst_dir)
                    results['aligned'] += 1
                continue
            
            # QC: Check overlap ratio
            overlap = _compute_overlap_ratio(warp_matrix, ref_img.shape)
            if overlap < self.overlap_threshold:
                self.logger.warning(f"FOV {fov} skipped: Overlap ratio {overlap:.2f} < {self.overlap_threshold}")
                results['skipped_overlap'] += 1
                if self.on_failure == 'copy':
                    self._copy_files(files, dst_dir)
                    results['aligned'] += 1
                continue
            
            # Apply transformation to all channels
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
                if img is None:
                    continue
                
                h, w = img.shape
                aligned = cv2.warpAffine(
                    img.astype(np.float32), warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                
                # Save
                Image.fromarray(aligned.astype(np.uint16)).save(str(dst_dir / f.name))
                # Count will be incremented once per FOV at end of loop? No, results is per image?
                # The logic above increments 'aligned' per FOV for copy, but here we are in loop.
                # Let's fix counter logic.
            
            results['aligned'] += 1

    def _copy_files(self, files: List[Path], dst_dir: Path):
        """Helper to copy files without modification"""
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
            if img is not None:
                Image.fromarray(img).save(str(dst_dir / f.name))
