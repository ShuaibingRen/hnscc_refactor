"""
Quench Subtraction Module

Removes autofluorescence/quench signals from cycle images.
Supports two modes:
- simple: Direct subtraction (cycle - background)
- mi_optimized: Find optimal coefficient via MI minimization
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from PIL import Image

from ...core.base import BaseProcessor
from ...core.utils import (
    scan_input_directory,
    parse_filename
)
from ...config import Config


def _mutual_information_2d(x: np.ndarray, y: np.ndarray, 
                           nbins: int = 256, normalized: bool = True) -> float:
    """Compute normalized mutual information between two images."""
    EPS = np.finfo(float).eps
    
    jh = np.histogram2d(x.ravel(), y.ravel(), bins=nbins)[0]
    ndimage.gaussian_filter(jh, sigma=1, mode='constant', output=jh)
    
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / 
              np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - 
              np.sum(s1 * np.log(s1)) - 
              np.sum(s2 * np.log(s2)))
    
    return mi


def _find_optimal_coefficient(fg: np.ndarray, bg: np.ndarray,
                              coeff_range: Tuple[float, float] = (0.6, 2.0),
                              coeff_steps: int = 50,
                              polynomial_degree: int = 5) -> Tuple[float, List[float]]:
    """
    Find optimal subtraction coefficient using MI minimization.
    
    Args:
        fg: Foreground image
        bg: Background image
        coeff_range: Range of coefficients to test
        coeff_steps: Number of steps
        polynomial_degree: Degree of polynomial for fitting
        
    Returns:
        Tuple of (optimal_coefficient, mi_values)
    """
    x_arr = np.linspace(coeff_range[0], coeff_range[1], coeff_steps)
    mi_arr = []
    
    for x in x_arr:
        subtracted = np.maximum(fg - x * bg, 0)
        mi = _mutual_information_2d(subtracted.ravel(), bg.ravel())
        mi_arr.append(mi)
    
    # Fit polynomial and find minimum
    poly = np.poly1d(np.polyfit(x_arr, mi_arr, polynomial_degree))
    poly_values = poly(x_arr)
    opt_idx = np.argmin(poly_values)
    opt_coeff = x_arr[opt_idx]
    
    return opt_coeff, mi_arr


class QuenchSubtractor(BaseProcessor):
    """
    Quench/autofluorescence subtraction processor.
    
    Subtraction pairs:
    - cycle1 - cycle0
    - cycle2 - quench1
    - cycle3 - quench2
    - ...
    - cycle7 - quench6
    """
    
    # Default subtraction pairs: cycle -> background
    SUBTRACTION_PAIRS = {
        1: 'cycle0',
        2: 'quench1',
        3: 'quench2',
        4: 'quench3',
        5: 'quench4',
        6: 'quench5',
        7: 'quench6',
    }
    
    def __init__(self, config: Config, mode: str = 'simple'):
        """
        Initialize quench subtractor.
        
        Args:
            config: Pipeline configuration
            mode: Subtraction mode ('simple' or 'mi_optimized')
        """
        super().__init__(config)
        self.mode = mode
        self.coeff_range = tuple(config.get('subtraction.coeff_range', [0.6, 2.0]))
        self.coeff_steps = config.get('subtraction.coeff_steps', 50)
        self.polynomial_degree = config.get('subtraction.polynomial_degree', 5)
        self.channels_per_cycle = config.get('input.channels_per_cycle', 4)
        # DAPI channel (w1) should not be subtracted - used for alignment reference
        self.dapi_channel = config.get('alignment.reference_channel', 1)
        # Failed FOVs from alignment (quench failures affect corresponding cycles)
        self._failed_cycle_fovs: Dict[int, set] = {}  # cycle_num -> set of failed fov numbers
    
    def process(self, input_dir: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Process quench subtraction.
        
        Args:
            input_dir: Input directory (typically align/ output)
            output_dir: Output directory for subtracted images
            
        Returns:
            Dict with processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self._log_start("Quench Subtraction", input_dir)
        
        # Scan input
        scan_result = scan_input_directory(input_dir)
        
        results = {
            'processed_pairs': 0,
            'processed_images': 0,
            'optimal_coefficients': {},
            'errors': [],
            'zeroed_fovs': 0
        }
        
        # Load alignment report to check for failed quenches
        self._load_alignment_report(input_dir)
        
        # Process each subtraction pair
        for cycle_num, bg_name in self.SUBTRACTION_PAIRS.items():
            cycle_name = f'cycle{cycle_num}'
            
            # Check if cycle exists
            if cycle_num not in scan_result['cycles']:
                continue
            
            # Get background directory
            if bg_name == 'cycle0':
                if 0 not in scan_result['cycles']:
                    results['errors'].append(f"Background {bg_name} not found for {cycle_name}")
                    continue
                bg_dir = scan_result['cycles'][0]
            else:
                quench_num = int(bg_name.replace('quench', ''))
                if quench_num not in scan_result['quenches']:
                    results['errors'].append(f"Background {bg_name} not found for {cycle_name}")
                    continue
                bg_dir = scan_result['quenches'][quench_num]
            
            fg_dir = scan_result['cycles'][cycle_num]
            
            # Process pair
            pair_result = self._process_pair(
                fg_dir, bg_dir, 
                output_dir / cycle_name,
                cycle_name, bg_name, cycle_num
            )
            
            results['processed_pairs'] += 1
            results['processed_images'] += pair_result['images']
            if 'coefficient' in pair_result:
                results['optimal_coefficients'][cycle_name] = pair_result['coefficient']
        
        # Copy cycle0 without subtraction
        if 0 in scan_result['cycles']:
            cycle0_out = self._ensure_dir(output_dir / 'cycle0')
            for f in list(scan_result['cycles'][0].glob('*.TIF')):
                img = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    Image.fromarray(img).save(str(cycle0_out / f.name))
                    results['processed_images'] += 1
        
        # Save optimization log
        if results['optimal_coefficients']:
            log_path = output_dir / 'optimization_log.json'
            with open(log_path, 'w') as f:
                json.dump(results['optimal_coefficients'], f, indent=2)
        
        self._log_complete("Quench Subtraction", output_dir, 
                          {'pairs': results['processed_pairs'], 
                           'images': results['processed_images']})
        
        return {
            'status': 'success',
            'mode': self.mode,
            'stats': results
        }
    
    def _load_alignment_report(self, input_dir: Path):
        """
        Load alignment_report.json and identify failed quench FOVs.
        
        If a quench alignment failed, the corresponding cycle should output zeros.
        """
        report_path = input_dir / 'alignment_report.json'
        if not report_path.exists():
            self.logger.debug("No alignment_report.json found, skipping failed FOV check")
            return
        
        with open(report_path) as f:
            report = json.load(f)
        
        failed_fovs = report.get('failed_fovs', [])
        
        for item in failed_fovs:
            fov = item.get('fov')
            cycle_name = item.get('cycle', '')
            
            # Check if it's a quench failure
            if 'quench' in cycle_name:
                quench_num = int(cycle_name.replace('quench', ''))
                # quench N failure affects cycle N+1
                # According to SUBTRACTION_PAIRS: cycle2 - quench1, cycle3 - quench2, etc.
                affected_cycle = quench_num + 1
                
                if affected_cycle not in self._failed_cycle_fovs:
                    self._failed_cycle_fovs[affected_cycle] = set()
                self._failed_cycle_fovs[affected_cycle].add(fov)
                
                self.logger.info(f"Quench{quench_num} FOV {fov} failed -> cycle{affected_cycle} FOV {fov} will be zeroed")
            
            # Check if it's a cycle failure (cycle itself failed)
            elif 'cycle' in cycle_name:
                cycle_num = int(cycle_name.replace('cycle', ''))
                if cycle_num not in self._failed_cycle_fovs:
                    self._failed_cycle_fovs[cycle_num] = set()
                self._failed_cycle_fovs[cycle_num].add(fov)
    
    def _process_pair(self, fg_dir: Path, bg_dir: Path,
                      output_dir: Path, fg_name: str, bg_name: str, cycle_num: int) -> Dict:
        """Process a single foreground-background pair"""
        output_dir = self._ensure_dir(output_dir)
        
        self.logger.info(f"  Processing {fg_name} - {bg_name}...")
        
        result = {'images': 0, 'coefficients': {}, 'zeroed': 0}
        
        # Get failed FOVs for this cycle
        failed_fovs = self._failed_cycle_fovs.get(cycle_num, set())
        
        # Get file mapping
        fg_files = sorted(fg_dir.glob('*.TIF'))
        bg_files = sorted(bg_dir.glob('*.TIF'))
        
        # Build background file lookup (by fov and channel)
        bg_lookup = {}
        for f in bg_files:
            parsed = parse_filename(f.name)
            if parsed:
                key = (parsed['fov'], parsed['channel'])
                bg_lookup[key] = f
                
        # Group foreground files by channel
        fg_by_channel = {}
        for f in fg_files:
            parsed = parse_filename(f.name)
            if not parsed:
                continue
            chan = parsed['channel']
            if chan not in fg_by_channel:
                fg_by_channel[chan] = []
            fg_by_channel[chan].append((f, parsed))

        # Process by channel
        for channel, file_list in fg_by_channel.items():
            # 1. Determine Coefficient
            coeff = 1.0
            
            # Skip optimization for DAPI (reference) channel - though we usually don't subtract it anyway
            if self.mode == 'mi_optimized' and channel != self.dapi_channel:
                try:
                    # Find valid pairs for optimization
                    valid_pairs = []
                    for f, parsed in file_list:
                        key = (parsed['fov'], parsed['channel'])
                        if key in bg_lookup:
                            valid_pairs.append((f, bg_lookup[key]))
                    
                    if valid_pairs:
                        # Sample up to 20 pairs
                        import random
                        sample_size = min(len(valid_pairs), 20)
                        # Use a fixed seed for reproducibility if needed, or random
                        # For now, just sample
                        sampled_pairs = random.sample(valid_pairs, sample_size)
                        
                        coeffs = []
                        for fg_p, bg_p in sampled_pairs:
                            try:
                                fg_img = cv2.imread(str(fg_p), cv2.IMREAD_ANYDEPTH)
                                bg_img = cv2.imread(str(bg_p), cv2.IMREAD_ANYDEPTH)
                                
                                if fg_img is None or bg_img is None:
                                    continue
                                    
                                # QC check: skip empty/dark images to avoid noise fitting
                                if np.mean(fg_img) < 100: # Simple intensity threshold
                                    continue

                                c, _ = _find_optimal_coefficient(
                                    fg_img.astype(np.float32),
                                    bg_img.astype(np.float32),
                                    self.coeff_range,
                                    self.coeff_steps,
                                    self.polynomial_degree
                                )
                                coeffs.append(c)
                            except Exception as e:
                                continue
                        
                        if coeffs:
                            # Take median
                            coeff = float(np.median(coeffs))
                            self.logger.info(f"    Channel {channel}: Computed median coeff {coeff:.3f} from {len(coeffs)} samples")
                            result['coefficients'][f'w{channel}'] = coeff
                        else:
                            self.logger.warning(f"    Channel {channel}: Failed to compute coefficients, using default 1.0")
                            
                except Exception as e:
                    self.logger.error(f"    Error optimizing channel {channel}: {e}")
            
            # 2. Apply Subtraction
            for fg_file, parsed in file_list:
                fov = parsed['fov']
                
                # Check if this FOV should be zeroed (alignment failed)
                if fov in failed_fovs:
                    img = cv2.imread(str(fg_file), cv2.IMREAD_ANYDEPTH)
                    if img is not None:
                        zeros = np.zeros_like(img)
                        Image.fromarray(zeros).save(str(output_dir / fg_file.name))
                        result['zeroed'] += 1
                    continue
                
                # DAPI: Copy as-is
                if channel == self.dapi_channel:
                    img = cv2.imread(str(fg_file), cv2.IMREAD_ANYDEPTH)
                    if img is not None:
                        Image.fromarray(img).save(str(output_dir / fg_file.name))
                        result['images'] += 1
                    continue
                
                # Check background availability
                key = (parsed['fov'], parsed['channel'])
                if key not in bg_lookup:
                    # Copy as-is
                    img = cv2.imread(str(fg_file), cv2.IMREAD_ANYDEPTH)
                    if img is not None:
                        Image.fromarray(img).save(str(output_dir / fg_file.name))
                        result['images'] += 1
                    continue
                
                # Perform subtraction
                fg_img = cv2.imread(str(fg_file), cv2.IMREAD_ANYDEPTH)
                bg_img = cv2.imread(str(bg_lookup[key]), cv2.IMREAD_ANYDEPTH)
                
                if fg_img is None or bg_img is None:
                    continue
                
                fg_float = fg_img.astype(np.float32)
                bg_float = bg_img.astype(np.float32)
                
                subtracted = np.maximum(fg_float - coeff * bg_float, 0)
                
                Image.fromarray(subtracted.astype(np.uint16)).save(str(output_dir / fg_file.name))
                result['images'] += 1
        
        return result

