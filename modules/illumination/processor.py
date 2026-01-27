"""
Illumination Correction Module

Supports two modes:
- simple: Single-sample BaSiC correction (fast)
- hnscc: Multi-sample MI QC + averaged correction matrices (precise)
"""
import cv2
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage
from PIL import Image
from basicpy import BaSiC

from ...core.base import BaseProcessor
from ...core.utils import (
    scan_input_directory, 
    get_files_for_cycle, 
    parse_filename,
    parallel_process
)
from ...config import Config


def _compute_basic_correction(file_list: List[Path], 
                               get_darkfield: bool = True,
                               smoothness_flatfield: float = 1.0,
                               max_iterations: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute BaSiC correction matrices for a list of images.
    
    Args:
        file_list: List of image file paths
        get_darkfield: Whether to compute darkfield
        smoothness_flatfield: Smoothness parameter
        max_iterations: Maximum iterations for BaSiC
        
    Returns:
        Tuple of (flatfield, darkfield) arrays
    """
    # Load images
    images = []
    for f in file_list:
        img = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
        if img is not None:
            images.append(img.astype(np.float32))
    
    if not images:
        return None, None
    
    images = np.stack(images, axis=0)
    
    # Run BaSiC
    basic = BaSiC(
        get_darkfield=get_darkfield,
        smoothness_flatfield=smoothness_flatfield
    )
    basic.fit(images)
    
    flatfield = basic.flatfield
    darkfield = basic.darkfield if get_darkfield else np.zeros_like(flatfield)
    
    return flatfield, darkfield


def _apply_basic_correction(image: np.ndarray, 
                            flatfield: np.ndarray, 
                            darkfield: np.ndarray) -> np.ndarray:
    """Apply BaSiC illumination correction to an image"""
    corrected = (image.astype(np.float32) - darkfield) / (flatfield + 1e-10)
    # Normalize to preserve intensity range
    corrected = corrected * np.mean(flatfield)
    return np.maximum(corrected, 0)


def _mutual_information_2d(x: np.ndarray, y: np.ndarray, 
                           nbins: int = 256, normalized: bool = True) -> float:
    """
    Compute normalized mutual information between two images.
    
    Args:
        x: First image (flattened)
        y: Second image (flattened)
        nbins: Number of histogram bins
        normalized: Whether to normalize MI to [0, 1]
        
    Returns:
        Mutual information value
    """
    EPS = np.finfo(float).eps
    
    # Compute joint histogram
    jh = np.histogram2d(x.ravel(), y.ravel(), bins=nbins)[0]
    
    # Smooth with gaussian
    ndimage.gaussian_filter(jh, sigma=1, mode='constant', output=jh)
    
    # Normalize
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    
    # Marginal histograms
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


def _process_channel_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single channel of a cycle in a worker process.
    """
    try:
        cycle_dir = args['cycle_dir']
        output_dir = args['output_dir']
        correction_dir = args['correction_dir']
        channel = args['channel']
        
        cycle_name = cycle_dir.name
        cycle_output = output_dir / cycle_name
        cycle_output.mkdir(parents=True, exist_ok=True)
        
        # Get files
        files = get_files_for_cycle(cycle_dir, channel=channel, exclude_overview=True)
        
        if not files:
            return {'count': 0, 'error': None, 'channel': channel, 'cycle': cycle_name}
            
        processed_count = 0
        
        # Check if we should skip correction
        if args.get('skip_correction', False):
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    Image.fromarray(img).save(str(cycle_output / f.name))
                    processed_count += 1
            return {'count': processed_count, 'error': None, 'channel': channel, 'cycle': cycle_name}

        # Need correction
        flatfield, darkfield = _compute_basic_correction(
            files,
            get_darkfield=args.get('get_darkfield', True),
            smoothness_flatfield=args.get('smoothness', 1.0),
            max_iterations=args.get('max_iterations', 800)
        )
        
        if flatfield is None:
            return {
                'count': 0, 
                'error': f"Failed to compute correction for {cycle_name} w{channel}",
                'channel': channel,
                'cycle': cycle_name
            }

        # Save correction matrices
        Image.fromarray(flatfield.astype(np.float32)).save(
            correction_dir / f'{cycle_name}_w{channel}_flatfield.tif'
        )
        if args.get('get_darkfield', True) and darkfield is not None:
            Image.fromarray(darkfield.astype(np.float32)).save(
                correction_dir / f'{cycle_name}_w{channel}_darkfield.tif'
            )
            
        # Apply correction to each image
        dk = darkfield if darkfield is not None else np.zeros_like(flatfield)
        
        for img_path in files:
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            corrected = _apply_basic_correction(img, flatfield, dk)
            
            out_path = cycle_output / img_path.name
            Image.fromarray(corrected.astype(np.uint16)).save(str(out_path))
            processed_count += 1
            
        return {'count': processed_count, 'error': None, 'channel': channel, 'cycle': cycle_name}
        
    except Exception as e:
        return {'count': 0, 'error': str(e), 'channel': channel, 'cycle': cycle_dir.name}


def _compute_sample_correction_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BaSiC correction matrices and SAVE TO DISK directly.
    Returns file paths instead of arrays to save memory.
    """
    try:
        sample_name = args['sample_name']
        cycle_dir = Path(args['cycle_dir'])
        channel = args['channel']
        flat_out_path = args['flat_out_path']
        dark_out_path = args['dark_out_path']
        
        files = get_files_for_cycle(cycle_dir, channel=channel, exclude_overview=True)
        
        if not files:
            return {'sample_name': sample_name, 'success': False, 'error': 'No files found'}
        
        # Check if output already exists (resume capability)
        if Path(flat_out_path).exists():
            return {
                'sample_name': sample_name,
                'success': True,
                'flat_path': flat_out_path,
                'dark_path': dark_out_path if Path(dark_out_path).exists() else None,
                'num_images': len(files),
                'from_cache': True
            }

        flatfield, darkfield = _compute_basic_correction(
            files,
            get_darkfield=args.get('get_darkfield', True),
            smoothness_flatfield=args.get('smoothness', 1.0),
            max_iterations=args.get('max_iterations', 800)
        )
        
        if flatfield is None:
             return {'sample_name': sample_name, 'success': False, 'error': 'BaSiC failed'}

        # Save immediately
        Image.fromarray(flatfield.astype(np.float32)).save(flat_out_path)
        if darkfield is not None:
            Image.fromarray(darkfield.astype(np.float32)).save(dark_out_path)
        
        return {
            'sample_name': sample_name,
            'success': True,
            'flat_path': flat_out_path,
            'dark_path': dark_out_path if darkfield is not None else None,
            'num_images': len(files)
        }
        
    except Exception as e:
        return {'sample_name': args.get('sample_name'), 'success': False, 'error': str(e)}


def _apply_correction_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply correction by loading matrices FROM DISK.
    """
    try:
        sample_name = args['sample_name']
        cycle_dir = Path(args['cycle_dir'])
        output_dir = Path(args['output_dir'])
        channel = args['channel']
        flat_path = args['flatfield_path']
        dark_path = args.get('darkfield_path')
        
        # Load matrices locally
        flatfield = cv2.imread(flat_path, cv2.IMREAD_ANYDEPTH)
        if flatfield is None:
            return {'error': f"Could not load flatfield from {flat_path}", 'sample': sample_name}
        
        darkfield = None
        if dark_path and Path(dark_path).exists():
            darkfield = cv2.imread(dark_path, cv2.IMREAD_ANYDEPTH)
        
        # Cast
        flatfield = flatfield.astype(np.float32)
        if darkfield is not None:
             darkfield = darkfield.astype(np.float32)
             
        cycle_name = cycle_dir.name
        cycle_output = output_dir / sample_name / cycle_name
        cycle_output.mkdir(parents=True, exist_ok=True)
        
        files = get_files_for_cycle(cycle_dir, channel=channel, exclude_overview=True)
        
        dk = darkfield if darkfield is not None else np.zeros_like(flatfield)
        count = 0
        
        for img_path in files:
            img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
            corrected = _apply_basic_correction(img, flatfield, dk)
            out_path = cycle_output / img_path.name
            Image.fromarray(corrected.astype(np.uint16)).save(str(out_path))
            count += 1
        
        return {'count': count, 'error': None, 'sample': sample_name, 'cycle': cycle_name, 'channel': channel}
        
    except Exception as e:
        return {'count': 0, 'error': str(e), 'sample': args.get('sample_name'), 
                'cycle': Path(args.get('cycle_dir', '')).name, 'channel': args.get('channel')}


class IlluminationProcessor(BaseProcessor):
    """
    Illumination correction processor with two modes:
    
    - simple: Single-sample BaSiC correction per cycle/channel
    - hnscc: Multi-sample MI QC + averaged correction matrices (precise)
    """
    
    def __init__(self, config: Config, mode: str = 'simple'):
        """
        Initialize illumination processor.
        
        Args:
            config: Pipeline configuration
            mode: Processing mode ('simple' or 'hnscc')
        """
        super().__init__(config)
        self.mode = mode
        self.mi_threshold = config.get('illumination.mi_threshold', 0.10)
        self.max_iterations = config.get('illumination.max_iterations', 800)
        self.get_darkfield = config.get('illumination.darkfield', True)
        self.smoothness = config.get('illumination.smoothness_flatfield', 1.0)
        self.max_workers = config.get('performance.max_workers')
        self.channels_per_cycle = config.get('input.channels_per_cycle', 4)
        self.min_fov_count = config.get('illumination.min_fov_count', 60)
    
    def process(self, input_dir: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Process illumination correction.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self._log_start("Illumination Correction", input_dir)
        
        # Create output directories
        correction_dir = self._ensure_dir(output_dir / 'correction')
        
        # Process based on mode
        if self.mode == 'hnscc':
            # Check if input is multi-sample batch directory
            batch_scan = self._scan_batch_directory(input_dir)
            
            if batch_scan['is_batch']:
                result = self._process_hnscc_batch_mode(
                    input_dir, output_dir, batch_scan, correction_dir
                )
            else:
                # Single sample - fall back to simple mode behavior (but all channels)
                self.logger.warning("HNSCC mode but single sample detected, processing as single sample")
                scan_result = scan_input_directory(input_dir)
                result = self._process_simple_mode(input_dir, output_dir, scan_result, correction_dir)
        else:
            # Simple mode
            scan_result = scan_input_directory(input_dir)
            self.logger.info(f"Found {len(scan_result['cycles'])} cycles, "
                            f"{len(scan_result['quenches'])} quenches, "
                            f"{scan_result['num_fovs']} FOVs")
            result = self._process_simple_mode(input_dir, output_dir, scan_result, correction_dir)
        
        self._log_complete("Illumination Correction", output_dir, result.get('stats'))
        return result
    
    def _scan_batch_directory(self, batch_dir: Path) -> Dict[str, Any]:
        """
        Scan batch directory to identify multiple samples.
        """
        result = {
            'is_batch': False,
            'samples': {},
            'common_cycles': [],
            'common_quenches': []
        }
        
        # Check if subdirectories are samples (contain cycle directories)
        potential_samples = []
        
        for subdir in batch_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Check if this is a sample directory (has cycle subdirs)
            has_cycles = any(
                d.is_dir() and d.name.lower().startswith('cycle')
                for d in subdir.iterdir() if d.is_dir()
            )
            
            if has_cycles:
                potential_samples.append(subdir)
        
        if len(potential_samples) < 2:
            return result
        
        result['is_batch'] = True
        
        # Scan each sample
        all_cycles = []
        all_quenches = []
        
        for sample_dir in potential_samples:
            sample_name = sample_dir.name
            sample_scan = scan_input_directory(sample_dir)
            
            result['samples'][sample_name] = {
                'path': sample_dir,
                'cycles': sample_scan['cycles'],
                'quenches': sample_scan['quenches'],
                'num_fovs': sample_scan['num_fovs']
            }
            
            all_cycles.append(set(sample_scan['cycles'].keys()))
            all_quenches.append(set(sample_scan['quenches'].keys()))
        
        # Find common cycles/quenches
        if all_cycles:
            result['common_cycles'] = sorted(list(set.intersection(*all_cycles)))
        if all_quenches:
            result['common_quenches'] = sorted(list(set.intersection(*all_quenches)))
        
        self.logger.info(f"Batch directory detected with {len(result['samples'])} samples")
        self.logger.info(f"Common cycles: {result['common_cycles']}")
        self.logger.info(f"Common quenches: {result['common_quenches']}")
        
        return result
    
    def _process_simple_mode(self, input_dir: Path, output_dir: Path, 
                             scan_result: Dict, correction_dir: Path) -> Dict:
        """
        Simple mode: BaSiC correction per cycle/channel with parallel execution.
        """
        all_dirs = list(scan_result['cycles'].values()) + list(scan_result['quenches'].values())
        tasks = []
        
        for cycle_dir in all_dirs:
            for channel in range(1, self.channels_per_cycle + 1):
                task_args = {
                    'cycle_dir': cycle_dir,
                    'output_dir': output_dir,
                    'correction_dir': correction_dir,
                    'channel': channel,
                    'get_darkfield': self.get_darkfield,
                    'smoothness': self.smoothness,
                    'max_iterations': self.max_iterations,
                    'mode': 'simple',
                    'skip_correction': False
                }
                tasks.append(task_args)

        self.logger.info(f"Starting parallel processing of {len(tasks)} tasks...")
        
        results = parallel_process(
            _process_channel_task, 
            tasks, 
            max_workers=self.max_workers,
            desc="Illumination Correction"
        )
        
        processed_count = 0
        for res in results:
            if isinstance(res, dict):
                if res.get('error'):
                    self.logger.error(f"Error in {res.get('cycle')} w{res.get('channel')}: {res.get('error')}")
                processed_count += res.get('count', 0)
            else:
                self.logger.error(f"Unexpected result type: {res}")
        
        return {
            'status': 'success',
            'mode': 'simple',
            'stats': {
                'processed_images': processed_count,
                'cycles': len(scan_result['cycles']),
                'quenches': len(scan_result['quenches'])
            }
        }

    def _process_hnscc_batch_mode(self, input_dir: Path, output_dir: Path,
                                   batch_scan: Dict, correction_dir: Path) -> Dict:
        """
        HNSCC batch mode: Multi-sample aggregation with MI-based QC.
        Refactored to minimize memory usage by passing file paths instead of arrays.
        """
        import gc
        samples = batch_scan['samples']
        
        # Create additional output directories
        per_sample_corr_dir = self._ensure_dir(output_dir / 'correction_per_sample')
        qc_dir = self._ensure_dir(output_dir / 'qc')
        
        # Filter samples by min FOV count
        valid_samples = {}
        for name, info in samples.items():
            if info['num_fovs'] >= self.min_fov_count:
                valid_samples[name] = info
            else:
                self.logger.warning(
                    f"Sample {name} has {info['num_fovs']} FOVs, "
                    f"below threshold {self.min_fov_count}, skipping for averaging"
                )
        
        if len(valid_samples) == 0:
            self.logger.error("No samples meet minimum FOV count threshold")
            return {'status': 'failed', 'error': 'No valid samples'}
        
        self.logger.info(f"Processing {len(valid_samples)} valid samples for correction matrices")
        
        # Collect all cycle and quench tasks
        tasks_to_process = []
        
        # Add cycles
        all_cycles = set()
        for info in samples.values():
            all_cycles.update(info['cycles'].keys())
        for c_idx in sorted(all_cycles):
            tasks_to_process.append(('cycle', c_idx))
            
        # Add quenches
        all_quenches = set()
        for info in samples.values():
            all_quenches.update(info['quenches'].keys())
        for q_idx in sorted(all_quenches):
            tasks_to_process.append(('quench', q_idx))
        
        total_processed = 0
        
        # Process each cycle/channel combination
        for type_name, idx in tasks_to_process:
            # Determine cycle name format
            is_quench = (type_name == 'quench')
            cycle_key = 'quenches' if is_quench else 'cycles'
            cycle_name_fmt = f"{type_name}{idx}"
            
            for channel in range(1, self.channels_per_cycle + 1):
                self.logger.info(f"Processing {cycle_name_fmt} channel {channel}")
                
                # Step 1: Compute per-sample corrections in parallel
                # Workers will save matrices to disk and return paths
                tasks = []
                for sample_name, info in valid_samples.items():
                    if idx not in info[cycle_key]:
                        continue
                    
                    cycle_dir = info[cycle_key][idx]
                    
                    # Pre-define output paths for matrices
                    sample_out_dir = per_sample_corr_dir / sample_name
                    sample_out_dir.mkdir(parents=True, exist_ok=True)
                    flat_path = sample_out_dir / f'{cycle_name_fmt}_w{channel}_flatfield.tif'
                    dark_path = sample_out_dir / f'{cycle_name_fmt}_w{channel}_darkfield.tif'
                    
                    tasks.append({
                        'sample_name': sample_name,
                        'cycle_dir': cycle_dir,
                        'channel': channel,
                        'get_darkfield': self.get_darkfield,
                        'smoothness': self.smoothness,
                        'max_iterations': self.max_iterations,
                        'flat_out_path': str(flat_path),
                        'dark_out_path': str(dark_path)
                    })
                
                if not tasks:
                    continue
                
                sample_results = parallel_process(
                    _compute_sample_correction_task,
                    tasks,
                    max_workers=self.max_workers,
                    desc=f"Computing corrections for {cycle_name_fmt} w{channel}"
                )
                
                # Verify results and map paths
                correction_paths = {}
                num_images = {}
                
                for res in sample_results:
                    if res.get('success'):
                        correction_paths[res['sample_name']] = {
                            'flatfield': res['flat_path'],
                            'darkfield': res['dark_path']
                        }
                        num_images[res['sample_name']] = res['num_images']
                    elif res.get('error'):
                        self.logger.warning(
                            f"Error computing correction for {res['sample_name']}: {res['error']}"
                        )
                
                if len(correction_paths) < 1:
                    self.logger.error(f"No valid corrections for {cycle_name_fmt} w{channel}")
                    continue
                
                # Step 2: Load matrices into memory just for MI computation and Mean
                loaded_corrections = {}
                for name, paths in correction_paths.items():
                    try:
                        flat = cv2.imread(paths['flatfield'], cv2.IMREAD_ANYDEPTH)
                        dark = None
                        if paths['darkfield'] and Path(paths['darkfield']).exists():
                            dark = cv2.imread(paths['darkfield'], cv2.IMREAD_ANYDEPTH)
                        
                        if flat is not None:
                            loaded_corrections[name] = {
                                'flatfield': flat.astype(np.float32),
                                'darkfield': dark.astype(np.float32) if dark is not None else None
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to load matrices for {name}: {e}")

                if not loaded_corrections:
                    continue

                # Compute MI matrix
                sample_names = list(loaded_corrections.keys())
                mi_matrix = self._compute_mi_matrix(loaded_corrections, sample_names)
                
                # Save MI matrix
                mi_df = pd.DataFrame(mi_matrix, index=sample_names, columns=sample_names)
                mi_df.to_csv(qc_dir / f'{cycle_name_fmt}_w{channel}_mi_matrix.csv')
                
                # Step 3: Select best samples
                selected_samples = self._select_samples_by_mi(
                    mi_matrix, sample_names, num_images
                )
                
                # Save selected samples metadata
                qc_info = {
                    'cycle': cycle_name_fmt,
                    'channel': channel,
                    'all_samples': sample_names,
                    'selected_samples': selected_samples,
                    'mi_threshold': self.mi_threshold
                }
                with open(qc_dir / f'{cycle_name_fmt}_w{channel}_selected_samples.json', 'w') as f:
                    json.dump(qc_info, f, indent=2)
                
                self.logger.info(
                    f"Selected {len(selected_samples)} samples for averaging: {selected_samples}"
                )
                
                # Step 4: Compute mean correction matrices
                mean_flat, mean_dark = self._compute_mean_correction(
                    loaded_corrections, selected_samples
                )
                
                # Release memory of loaded individual matrices immediately
                del loaded_corrections
                del sample_results
                gc.collect()
                
                # Save mean correction matrices
                mean_flat_path = correction_dir / f'{cycle_name_fmt}_w{channel}_flatfield.tif'
                mean_dark_path = correction_dir / f'{cycle_name_fmt}_w{channel}_darkfield.tif'
                
                Image.fromarray(mean_flat.astype(np.float32)).save(mean_flat_path)
                if mean_dark is not None:
                    Image.fromarray(mean_dark.astype(np.float32)).save(mean_dark_path)
                
                # Release mean matrices from memory, workers will load from disk
                del mean_flat
                del mean_dark
                gc.collect()
                
                # Step 5: Apply correction to ALL samples
                apply_tasks = []
                for sample_name, info in samples.items():
                    if idx not in info[cycle_key]:
                        continue
                    
                    cycle_dir = info[cycle_key][idx]
                    apply_tasks.append({
                        'sample_name': sample_name,
                        'cycle_dir': cycle_dir,
                        'output_dir': output_dir,
                        'channel': channel,
                        'flatfield_path': str(mean_flat_path),
                        'darkfield_path': str(mean_dark_path)
                    })
                
                apply_results = parallel_process(
                    _apply_correction_task,
                    apply_tasks,
                    max_workers=self.max_workers,
                    desc=f"Applying correction for {cycle_name_fmt} w{channel}"
                )
                
                for res in apply_results:
                    if res.get('error'):
                        self.logger.error(
                            f"Error applying correction to {res.get('sample')} "
                            f"{res.get('cycle')} w{res.get('channel')}: {res.get('error')}"
                        )
                    else:
                        total_processed += res.get('count', 0)
                        
                # Explicit cleanup after each cycle/channel
                del apply_results
                gc.collect()
        
        return {
            'status': 'success',
            'mode': 'hnscc_batch',
            'stats': {
                'processed_images': total_processed,
                'total_samples': len(samples),
                'valid_samples': len(valid_samples),
                'cycles': len(batch_scan['common_cycles']),
                'quenches': len(batch_scan['common_quenches'])
            }
        }

    def _compute_mi_matrix(self, corrections: Dict[str, Dict], 
                           sample_names: List[str]) -> np.ndarray:
        """
        Compute mutual information matrix between sample flatfields.
        """
        n = len(sample_names)
        mi_matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(sample_names):
            for j, name2 in enumerate(sample_names):
                if i == j:
                    mi_matrix[i, j] = 1.0
                elif i < j:
                    flat1 = corrections[name1]['flatfield']
                    flat2 = corrections[name2]['flatfield']
                    try:
                        mi = _mutual_information_2d(flat1.ravel(), flat2.ravel())
                        mi_matrix[i, j] = mi
                        mi_matrix[j, i] = mi
                    except Exception as e:
                        self.logger.warning(f"MI computation failed for {name1} vs {name2}: {e}")
                        mi_matrix[i, j] = 0
                        mi_matrix[j, i] = 0
        
        return mi_matrix

    def _select_samples_by_mi(self, mi_matrix: np.ndarray, 
                               sample_names: List[str],
                               num_images: Dict[str, int]) -> List[str]:
        """
        Select samples using MI-based graph clustering.
        """
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not available, using all samples")
            return sample_names
        
        n = len(sample_names)
        
        if n == 1:
            return sample_names
        
        # Create binary adjacency matrix
        adj_matrix = (mi_matrix > self.mi_threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        
        # Create graph
        graph = nx.from_numpy_array(adj_matrix)
        
        # Find all maximal cliques
        all_cliques = list(nx.find_cliques(graph))
        
        if not all_cliques:
            best_sample = max(num_images.keys(), key=lambda x: num_images.get(x, 0))
            return [best_sample]
        
        # Find largest cliques
        max_len = max(len(c) for c in all_cliques)
        largest_cliques = [c for c in all_cliques if len(c) == max_len]
        
        if len(largest_cliques) == 1:
            indices = largest_cliques[0]
            return [sample_names[i] for i in indices]
        
        # Tie breaking
        clique_mis = {}
        for idx, clique in enumerate(largest_cliques):
            if len(clique) == 1:
                clique_mis[idx] = 1.0
            else:
                sub_matrix = mi_matrix[np.ix_(clique, clique)]
                mask = ~np.eye(len(clique), dtype=bool)
                clique_mis[idx] = np.mean(sub_matrix[mask])
        
        max_mi = max(clique_mis.values())
        best_cliques = [idx for idx, mi in clique_mis.items() if mi == max_mi]
        
        if len(best_cliques) == 1:
            indices = largest_cliques[best_cliques[0]]
            return [sample_names[i] for i in indices]
        
        best_sample = max(
            [sample_names[i] for i in largest_cliques[best_cliques[0]]],
            key=lambda x: num_images.get(x, 0)
        )
        return [best_sample]

    def _compute_mean_correction(self, corrections: Dict[str, Dict],
                                  selected_samples: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean flatfield and darkfield from selected samples.
        """
        flatfields = []
        darkfields = []
        
        for name in selected_samples:
            if name in corrections:
                flatfields.append(corrections[name]['flatfield'])
                if corrections[name]['darkfield'] is not None:
                    darkfields.append(corrections[name]['darkfield'])
        
        mean_flat = np.mean(np.stack(flatfields, axis=0), axis=0)
        mean_dark = np.mean(np.stack(darkfields, axis=0), axis=0) if darkfields else None
        
        return mean_flat, mean_dark
