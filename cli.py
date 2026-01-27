#!/usr/bin/env python
"""
HNSCC Pipeline Command-Line Interface

Usage examples:
    # Run full pipeline
    python -m hnscc_refactor.cli -i /path/to/input -o /path/to/output -s PIO10
    
    # Run with workflow
    python -m hnscc_refactor.cli -i /path/to/input -o /path/to/output --workflow full
    
    # Resume from checkpoint
    python -m hnscc_refactor.cli -i /path/to/input -o /path/to/output --resume
    
    # Run specific steps
    python -m hnscc_refactor.cli -i /path/to/input -o /path/to/output --start illuminate --end subtract
"""
import argparse
import sys
from pathlib import Path
from typing import List

from .pipeline import Pipeline
from .config import Config


def parse_marker_file(marker_file: Path) -> List[str]:
    """Parse marker names from file (one per line)"""
    markers = []
    with open(marker_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                markers.append(line)
    return markers


def main():
    parser = argparse.ArgumentParser(
        description="HNSCC Image Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflows:
  full       - illuminate -> align -> subtract -> stitch
  no_align   - illuminate -> subtract -> stitch (skip alignment)
  preprocess - illuminate -> align -> subtract (no stitching)
  stitch_only- stitch only (assumes preprocessed input)

Examples:
  # Full pipeline with default settings
  python -m hnscc_refactor.cli -i PIO10/ -o output/ -s PIO10

  # Use configuration file
  python -m hnscc_refactor.cli -i PIO10/ -o output/ -c config.yaml

  # Resume interrupted processing
  python -m hnscc_refactor.cli -i PIO10/ -o output/ --resume

  # Run specific steps only
  python -m hnscc_refactor.cli -i PIO10/ -o output/ --start align --end subtract
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input', 
        type=Path, 
        required=True,
        help='Input directory containing cycle/quench subdirs'
    )
    parser.add_argument(
        '-o', '--output', 
        type=Path, 
        required=True,
        help='Output directory'
    )
    
    # Sample identification
    parser.add_argument(
        '-s', '--sample',
        help='Sample name (auto-detected if not provided)'
    )
    
    # Configuration
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Configuration YAML file'
    )
    parser.add_argument(
        '--set',
        nargs=2,
        action='append',
        metavar=('KEY', 'VALUE'),
        help='Override config value: --set illumination.mode hnscc'
    )
    
    # Workflow control
    parser.add_argument(
        '--workflow',
        choices=['full', 'no_align', 'preprocess', 'stitch_only'],
        help='Predefined workflow to run'
    )
    parser.add_argument(
        '--start',
        choices=['illuminate', 'align', 'subtract', 'stitch'],
        help='Step to start from'
    )
    parser.add_argument(
        '--end',
        choices=['illuminate', 'align', 'subtract', 'stitch'],
        help='Step to end at'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    # Channel names
    parser.add_argument(
        '-m', '--markers',
        type=Path,
        help='Marker names file (one per line) for OME-TIFF channel labeling'
    )
    
    # Mode options
    parser.add_argument(
        '--illumination-mode',
        choices=['simple', 'hnscc'],
        help='Illumination correction mode'
    )
    parser.add_argument(
        '--subtraction-mode',
        choices=['simple', 'mi_optimized'],
        help='Quench subtraction mode'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        return 1
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Apply command-line overrides
    if args.set:
        for key, value in args.set:
            # Try to parse as JSON for complex types
            try:
                import json
                value = json.loads(value)
            except json.JSONDecodeError:
                # Keep as string
                pass
            config.set(key, value)
    
    # Mode overrides
    if args.illumination_mode:
        config.set('illumination.mode', args.illumination_mode)
    if args.subtraction_mode:
        config.set('subtraction.mode', args.subtraction_mode)
    
    # Parse marker names
    marker_names = None
    if args.markers:
        if not args.markers.exists():
            print(f"Error: Marker file not found: {args.markers}", file=sys.stderr)
            return 1
        marker_names = parse_marker_file(args.markers)
        print(f"Loaded {len(marker_names)} marker names")
    
    # Create and run pipeline
    pipeline = Pipeline(config=config)
    
    try:
        result = pipeline.run(
            input_dir=args.input,
            output_dir=args.output,
            sample_name=args.sample,
            marker_names=marker_names,
            workflow=args.workflow,
            start_step=args.start,
            end_step=args.end,
            resume=args.resume
        )
        
        if result['status'] == 'success':
            print("\n✅ Pipeline completed successfully!")
            return 0
        else:
            print("\n⚠️ Pipeline completed with errors")
            return 1
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
