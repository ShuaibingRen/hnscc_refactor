"""
Common utilities for HNSCC pipeline
"""
import re
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse HNSCC image filename to extract components.
    
    Expected format: {sample}_{cycle}_w{channel}_s{fov}_t1.TIF
    Example: PIO10_cycle0_w1_s1_t1.TIF
    
    Args:
        filename: Image filename (not full path)
        
    Returns:
        Dict with keys: sample, cycle, channel, fov
    """
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


def scan_input_directory(input_dir: Path) -> Dict[str, Dict]:
    """
    Scan input directory to discover cycles, quenches, and FOVs.
    
    Args:
        input_dir: Root input directory
        
    Returns:
        Dict with structure:
        {
            'cycles': {0: Path, 1: Path, ...},
            'quenches': {1: Path, 2: Path, ...},
            'sample_name': str,
            'num_fovs': int,
            'num_channels': int
        }
    """
    input_dir = Path(input_dir)
    result = {
        'cycles': {},
        'quenches': {},
        'sample_name': None,
        'num_fovs': 0,
        'num_channels': 4
    }
    
    # Find cycle and quench directories
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        name = subdir.name.lower()
        if match := re.match(r'cycle(\d+)', name):
            result['cycles'][int(match.group(1))] = subdir
        elif match := re.match(r'quench(\d+)', name):
            result['quenches'][int(match.group(1))] = subdir
    
    # Detect sample name and FOV count from first cycle
    if result['cycles']:
        first_cycle = result['cycles'][min(result['cycles'].keys())]
        tif_files = list(first_cycle.glob('*.TIF'))
        
        if tif_files:
            # Parse first file to get sample name
            parsed = parse_filename(tif_files[0].name)
            if parsed:
                result['sample_name'] = parsed['sample']
            
            # Count unique FOVs (excluding overview - last FOV)
            fov_numbers = set()
            for f in tif_files:
                p = parse_filename(f.name)
                if p:
                    fov_numbers.add(p['fov'])
            
            if fov_numbers:
                max_fov = max(fov_numbers)
                # Exclude overview (last FOV)
                result['num_fovs'] = max_fov - 1
    
    return result


def detect_grid_from_scan(scan_file: Path) -> Tuple[int, int]:
    """
    Detect grid dimensions from .scan file.
    
    Args:
        scan_file: Path to .scan file
        
    Returns:
        Tuple of (rows, cols)
    """
    try:
        with open(scan_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse stage positions
        pattern = r'"Stage\d+",\s*"Row(\d+)_Col(\d+)"'
        matches = re.findall(pattern, content)
        
        if matches:
            rows = set(int(row) for row, col in matches)
            cols = set(int(col) for row, col in matches)
            return len(rows), len(cols)
    except Exception:
        pass
    
    return 0, 0


def find_scan_file(cycle_dir: Path) -> Optional[Path]:
    """Find .scan file in cycle directory"""
    scan_files = list(cycle_dir.glob('*.scan'))
    return scan_files[0] if scan_files else None


def get_files_for_cycle(cycle_dir: Path, channel: int = None, exclude_overview: bool = True) -> List[Path]:
    """
    Get image files for a cycle, optionally filtered by channel.
    
    Args:
        cycle_dir: Cycle directory path
        channel: Optional channel number to filter (1-4)
        exclude_overview: If True, exclude last FOV (overview image)
        
    Returns:
        List of file paths, sorted by FOV number
    """
    pattern = f'*_w{channel}_*.TIF' if channel else '*.TIF'
    files = list(cycle_dir.glob(pattern))
    
    if exclude_overview and files:
        # Find and exclude max FOV number
        fov_numbers = {}
        for f in files:
            parsed = parse_filename(f.name)
            if parsed:
                fov_numbers[f] = parsed['fov']
        
        if fov_numbers:
            max_fov = max(fov_numbers.values())
            files = [f for f, fov in fov_numbers.items() if fov < max_fov]
    
    # Sort by FOV number
    def sort_key(f):
        parsed = parse_filename(f.name)
        return (parsed.get('channel', 0), parsed.get('fov', 0)) if parsed else (0, 0)
    
    return sorted(files, key=sort_key)


def parallel_process(func, items: List, max_workers: int = None, desc: str = "Processing") -> List:
    """
    Process items in parallel using ProcessPoolExecutor.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers (None = CPU count)
        desc: Description for logging
        
    Returns:
        List of results in order
    """
    if not items:
        return []
    
    results = [None] * len(items)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {'error': str(e)}
    
    return results
