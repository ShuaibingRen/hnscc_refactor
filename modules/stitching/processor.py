"""
Stitching Module

Tile stitching using ASHLAR for multi-cycle imaging data.
Produces a single merged OME-TIFF with all cycles.
"""
import re
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tifffile

from ...core.base import BaseProcessor
from ...core.utils import (
    scan_input_directory,
    find_scan_file,
    detect_grid_from_scan,
    parse_filename
)
from ...config import Config


class StitchingProcessor(BaseProcessor):
    """
    Tile stitching processor using ASHLAR.
    
    Stitches tiles from all cycles into a single registered OME-TIFF.
    """
    
    def __init__(self, config: Config):
        """
        Initialize stitching processor.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.overlap = config.get('stitching.overlap', 0.1)
        self.layout = config.get('stitching.layout', 'snake')
        self.direction = config.get('stitching.direction', 'horizontal')
        self.align_channel = config.get('stitching.align_channel', 0)
        self.pixel_size = config.get('stitching.pixel_size', 0.325)
        self.channels_per_cycle = config.get('input.channels_per_cycle', 4)
        
        # Grid parameters (auto-detected or from config)
        self.grid_rows = config.get('grid.rows')
        self.grid_cols = config.get('grid.cols')
    
    def process(self, input_dir: Path, output_dir: Path,
                sample_name: str = None, 
                marker_names: List[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Stitch tiles into OME-TIFF.
        
        Args:
            input_dir: Input directory (typically subtract/ output)
            output_dir: Output directory for stitched images
            sample_name: Sample name for output file
            marker_names: Optional list of channel names to embed in OME-TIFF
            
        Returns:
            Dict with processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        self._ensure_dir(output_dir)
        
        self._log_start("Tile Stitching", input_dir)
        
        # Scan input
        scan_result = scan_input_directory(input_dir)
        
        if not sample_name:
            sample_name = scan_result.get('sample_name', 'sample')
        
        # Auto-detect grid if needed
        self._detect_grid(input_dir, scan_result)
        
        # Order cycles for stitching (cycle0, cycle1, ..., cycle7)
        ordered_cycles = self._order_cycles(scan_result)
        
        self.logger.info(f"Stitching {len(ordered_cycles)} cycles: {[c.name for c in ordered_cycles]}")
        self.logger.info(f"Grid: {self.grid_rows}x{self.grid_cols}")
        
        # Build ASHLAR arguments
        fileseries_args = self._build_fileseries_args(ordered_cycles)
        
        # Output file
        output_file = output_dir / f"{sample_name}.ome.tiff"
        
        # Run ASHLAR
        result = self._run_ashlar(fileseries_args, output_file)
        
        # Update channel names if provided
        if result['success'] and marker_names:
            self._update_channel_names(output_file, marker_names)
        
        self._log_complete("Tile Stitching", output_dir, 
                          {'cycles': len(ordered_cycles), 'output': output_file.name})
        
        return {
            'status': 'success' if result['success'] else 'failed',
            'output_file': str(output_file),
            'cycles': len(ordered_cycles),
            'error': result.get('error')
        }
    
    def _detect_grid(self, input_dir: Path, scan_result: Dict):
        """Auto-detect grid dimensions from .scan files or file patterns"""
        if self.grid_rows and self.grid_cols:
            return
        
        # Try to find .scan file in first cycle
        if scan_result['cycles']:
            first_cycle = scan_result['cycles'][min(scan_result['cycles'].keys())]
            scan_file = find_scan_file(first_cycle)
            
            if scan_file:
                rows, cols = detect_grid_from_scan(scan_file)
                if rows and cols:
                    self.grid_rows = rows
                    self.grid_cols = cols
                    self.logger.info(f"Detected grid from .scan: {rows}x{cols}")
                    return
        
        # Fallback: estimate from file count
        if scan_result['num_fovs'] > 0:
            # Assume square-ish grid
            import math
            n = scan_result['num_fovs']
            self.grid_cols = int(math.ceil(math.sqrt(n)))
            self.grid_rows = int(math.ceil(n / self.grid_cols))
            self.logger.info(f"Estimated grid from FOV count: {self.grid_rows}x{self.grid_cols}")
    
    def _order_cycles(self, scan_result: Dict) -> List[Path]:
        """Order cycle directories for stitching (cycle0, cycle1, ..., cycle7)"""
        ordered = []
        
        # Add cycles in order
        for i in sorted(scan_result['cycles'].keys()):
            ordered.append(scan_result['cycles'][i])
        
        return ordered
    
    def _build_fileseries_args(self, cycle_dirs: List[Path]) -> List[str]:
        """Build ASHLAR fileseries arguments for each cycle"""
        args = []
        
        for cycle_dir in cycle_dirs:
            # Find sample file to detect pattern
            tif_files = [f for f in cycle_dir.glob('*.TIF') if not f.name.startswith('._')]
            if not tif_files:
                tif_files = [f for f in cycle_dir.glob('*.tif') if not f.name.startswith('._')]
            
            if not tif_files:
                continue
            
            # Build pattern: replace channel and fov numbers with placeholders
            sample_name = tif_files[0].name
            # Pattern: {sample}_{cycle}_w{channel}_s{fov}_t1.TIF
            # Replace _w1_ with _w{channel:1}_
            pattern = re.sub(r'_w\d+_', '_w{channel:1}_', sample_name)
            # Replace _s123_ with _s{series}_
            pattern = re.sub(r'_s\d+_', '_s{series}_', pattern)
            
            arg = (
                f"fileseries|{cycle_dir}|pattern={pattern}|"
                f"overlap={self.overlap}|width={self.grid_cols}|"
                f"height={self.grid_rows}|layout={self.layout}|"
                f"direction={self.direction}|pixel_size={self.pixel_size}"
            )
            args.append(arg)
        
        return args
    
    def _run_ashlar(self, fileseries_args: List[str], output_file: Path) -> Dict:
        """Run ASHLAR command"""
        if not fileseries_args:
            return {'success': False, 'error': 'No cycles to stitch'}
        
        # Remove existing output
        output_file.unlink(missing_ok=True)
        
        cmd = [
            "ashlar",
            *fileseries_args,
            "-o", str(output_file),
            "--align-channel", str(self.align_channel)
        ]
        
        self.logger.info(f"Running ASHLAR with {len(fileseries_args)} cycles...")
        self.logger.info(f"ASHLAR Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Check if output was created
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                self.logger.info(f"Created: {output_file.name} ({size_mb:.1f} MB)")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Output file not created'}
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ASHLAR failed: {e.stderr}")
            return {'success': False, 'error': e.stderr}
        except FileNotFoundError:
            self.logger.error("ASHLAR not found. Please install: pip install ashlar")
            return {'success': False, 'error': 'ASHLAR not installed'}
    
    def _update_channel_names(self, ometiff_file: Path, marker_names: List[str]):
        """Update OME-TIFF channel names with marker names"""
        self.logger.info(f"Updating channel names ({len(marker_names)} markers)")
        
        try:
            with tifffile.TiffFile(ometiff_file) as tif:
                if not tif.is_ome:
                    self.logger.warning("File is not OME-TIFF, skipping channel update")
                    return
                
                data = tif.asarray()
                ome_xml = tif.ome_metadata
            
            # Parse and update XML
            root = ET.fromstring(ome_xml)
            channels = root.findall('.//{*}Channel')
            
            if len(channels) != len(marker_names):
                self.logger.warning(
                    f"Channel count mismatch: {len(channels)} in file, "
                    f"{len(marker_names)} provided. Skipping update."
                )
                return
            
            for channel, name in zip(channels, marker_names):
                channel.set('Name', name)
            
            # Write updated file
            updated_xml = ET.tostring(root, encoding='unicode').encode('utf-8')
            
            with tifffile.TiffWriter(ometiff_file, bigtiff=True) as writer:
                writer.write(
                    data,
                    description=updated_xml,
                    photometric='minisblack',
                    metadata={'axes': 'CYX' if data.ndim == 3 else 'YX'}
                )
            
            self.logger.info("Channel names updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update channel names: {e}")
