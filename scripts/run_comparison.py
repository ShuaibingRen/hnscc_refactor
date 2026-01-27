#!/usr/bin/env python3
"""
Run comparison of three schemes for HNSCC pipeline.
"""
import sys
import os
import shutil
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hnscc_refactor.config import Config
from hnscc_refactor.modules.illumination.processor import IlluminationProcessor
from hnscc_refactor.modules.subtraction.processor import QuenchSubtractor

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = Path("/mnt/efs/fs1/aws_home/shuaibing/RAW_test/2sample/")
BASE_OUTPUT_DIR = Path("/mnt/efs/fs1/aws_home/shuaibing/RAW_test/output_comparison")

SCHEMES = [
    {
        "id": 1,
        "name": "Scheme 1 (Flat+Dark, Simple)",
        "config": "configs/scheme1.yaml"
    },
    {
        "id": 2,
        "name": "Scheme 2 (Flat+Dark, Optimized)",
        "config": "configs/scheme2.yaml"
    },
    {
        "id": 3,
        "name": "Scheme 3 (Flat Only, Optimized)",
        "config": "configs/scheme3.yaml"
    }
]

def run_scheme(scheme):
    scheme_id = scheme["id"]
    logger.info(f"=== Running {scheme['name']} ===")
    
    # Define output directory for this scheme
    output_dir = BASE_OUTPUT_DIR / f"scheme{scheme_id}"
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists.")
    else:
        output_dir.mkdir(parents=True)

    # Load config
    config_path = Path(__file__).parent.parent / scheme["config"]
    config = Config(config_path)
    
    # 1. Illumination Correction
    logger.info(f"Starting Illumination Correction for Scheme {scheme_id}...")
    ill_proc = IlluminationProcessor(config, mode=config.get('illumination.mode'))
    
    # Note: Illumination processor writes to output_dir/correction
    # For HNSCC batch mode, it might write to output_dir/correction_per_sample
    # We need to check where it writes the *corrected images* which are input to subtraction
    # The current implementation writes corrected images to output_dir/<sample>/cycleX
    
    # HOWEVER, QuenchSubtractor expects input from 'align' directory or structured input
    # In the full pipeline: raw -> illumination -> alignment -> subtraction
    # Since we are skipping alignment for this test (to isolate variables and speed up),
    # we need to trick QuenchSubtractor or make sure Illumination output structure matches what Subtraction expects.
    
    # Let's check Illumination output structure in processor.py:
    # It calls _apply_correction_task which writes to: output_dir / sample_name / cycle_name
    
    # QuenchSubtractor expects 'scan_input_directory' structure.
    # It reads from input_dir and expects cycle/quench folders directly or via sample?
    # Let's look at QuenchSubtractor.process. It calls scan_input_directory(input_dir).
    # scan_input_directory handles both flat and sample-based structures?
    # Actually IlluminationProcessor in 'simple' mode writes to output_dir/cycleX directly if input is flat,
    # or output_dir/cycleX if input is single sample?
    # Wait, the input /mnt/efs/fs1/aws_home/shuaibing/RAW_test/2sample/ has SUBDIRECTORIES (PIO10_reimage, etc.)
    # So IlluminationProcessor will likely detect batch mode or we should loop over samples?
    
    # IlluminationProcessor logic:
    # if mode == 'hnscc': _scan_batch_directory.
    # if mode == 'simple': scan_input_directory(input_dir).
    
    # Our config for all schemes sets illumination.mode = 'simple'.
    # scan_input_directory on the batch folder will probably fail to find cycles directly.
    # We should run IlluminationProcessor PER SAMPLE for 'simple' mode.
    
    for sample_dir in INPUT_DIR.iterdir():
        if not sample_dir.is_dir(): continue
        
        sample_name = sample_dir.name
        logger.info(f"Processing sample: {sample_name}")
        
        # Sample specific output
        sample_out_dir = output_dir / sample_name
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Illumination
        # Output will be in sample_out_dir/cycleX
        ill_result = ill_proc.process(sample_dir, sample_out_dir)
        
        if ill_result.get('status') == 'failed':
            logger.error(f"Illumination failed for {sample_name}")
            continue
            
        # 2. Subtraction
        # Input for subtraction is the output of illumination (since we skip alignment)
        # QuenchSubtractor.process(input_dir, output_dir)
        # Input dir should contain cycle/quench folders. which is sample_out_dir.
        
        logger.info(f"Starting Quench Subtraction for Scheme {scheme_id}, sample {sample_name}...")
        
        sub_proc = QuenchSubtractor(config, mode=config.get('subtraction.mode'))
        
        # Subtraction writes to output_dir (new) / cycleX
        # We'll write to sample_out_dir / 'subtracted'
        sub_out_dir = sample_out_dir / 'subtracted'
        sub_out_dir.mkdir(parents=True, exist_ok=True)
        
        sub_result = sub_proc.process(sample_out_dir, sub_out_dir)
        
        # Copy optimization log if exists
        if (sub_out_dir / 'optimization_log.json').exists():
            shutil.copy(sub_out_dir / 'optimization_log.json', 
                        output_dir / f'{sample_name}_optimization_log.json')

    logger.info(f"Finished Scheme {scheme_id}")

def main():
    if BASE_OUTPUT_DIR.exists():
        logger.warning(f"Cleaning previous output directory: {BASE_OUTPUT_DIR}")
        # shutil.rmtree(BASE_OUTPUT_DIR) # Safety: don't auto delete for now
    else:
        BASE_OUTPUT_DIR.mkdir(parents=True)
        
    for scheme in SCHEMES:
        try:
            run_scheme(scheme)
        except Exception as e:
            logger.error(f"Scheme {scheme['id']} failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
