#!/usr/bin/env python
"""Pipeline Output Validation Script"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def count_tifs(directory: Path) -> dict:
    """Count TIF files in subdirectories."""
    counts = {}
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir() and (subdir.name.startswith('cycle') or subdir.name.startswith('quench')):
            counts[subdir.name] = len(list(subdir.glob('*.TIF')) + list(subdir.glob('*.tif')))
    return counts


def get_fov_set(directory: Path) -> set:
    """Extract (cycle, fov) tuples from directory."""
    fovs = set()
    for subdir in directory.iterdir():
        if subdir.is_dir():
            for f in subdir.glob('*.TIF'):
                for p in f.stem.split('_'):
                    if p.startswith('s') and p[1:].isdigit():
                        fovs.add((subdir.name, int(p[1:])))
                        break
    return fovs


def validate(output_dir: Path) -> dict:
    """Run all validation checks."""
    results = {'timestamp': datetime.now().isoformat(), 'status': 'passed', 
               'warnings': [], 'errors': [], 'details': {}}
    
    print(f"\n{'='*50}\nValidating: {output_dir}\n{'='*50}")
    
    # 1. Directory structure
    steps = ['illuminate', 'align', 'subtract', 'stitch']
    missing = [s for s in steps if not (output_dir / s).exists()]
    if missing:
        results['errors'].append(f"Missing: {missing}")
        print(f"❌ Missing directories: {missing}")
    else:
        print(f"✅ All directories exist")
    
    # 2. File counts
    for step in ['illuminate', 'align', 'subtract']:
        step_dir = output_dir / step
        if step_dir.exists():
            counts = count_tifs(step_dir)
            total = sum(counts.values())
            results['details'][step] = {'total': total, 'by_dir': counts}
            print(f"   {step}: {total} files")
    
    # 3. Data loss (exclude quench when comparing to subtract, since subtract only outputs cycles)
    def get_fov_set_filtered(directory: Path, exclude_quench: bool = False) -> set:
        fovs = set()
        for subdir in directory.iterdir():
            if subdir.is_dir():
                if exclude_quench and subdir.name.startswith('quench'):
                    continue
                for f in subdir.glob('*.TIF'):
                    for p in f.stem.split('_'):
                        if p.startswith('s') and p[1:].isdigit():
                            fovs.add((subdir.name, int(p[1:])))
                            break
        return fovs
    
    fov_sets = {}
    for s in ['illuminate', 'align', 'subtract']:
        if (output_dir / s).exists():
            fov_sets[s] = get_fov_set(output_dir / s)
    
    # For align→subtract comparison, also get cycles-only version
    if (output_dir / 'align').exists():
        fov_sets['align_cycles_only'] = get_fov_set_filtered(output_dir / 'align', exclude_quench=True)
    
    for src, dst in [('illuminate', 'align'), ('align_cycles_only', 'subtract')]:
        if src in fov_sets and dst in fov_sets:
            lost = fov_sets[src] - fov_sets[dst]
            if lost:
                rate = len(lost) / len(fov_sets[src]) * 100
                msg = f"{src}→{dst}: {len(lost)} FOVs lost ({rate:.1f}%)"
                results['warnings'].append(msg)
                print(f"⚠️  {msg}")
    
    # 4. File sizes (empty check)
    empty = []
    for step in ['illuminate', 'align', 'subtract']:
        step_dir = output_dir / step
        if step_dir.exists():
            for f in step_dir.rglob('*.TIF'):
                if f.stat().st_size == 0:
                    empty.append(str(f.relative_to(output_dir)))
    if empty:
        results['errors'].append(f"{len(empty)} empty files")
        print(f"❌ {len(empty)} empty files")
    
    # 5. OME-TIFF
    stitch_dir = output_dir / 'stitch'
    if stitch_dir.exists():
        ome_files = list(stitch_dir.glob('*.ome.tiff')) + list(stitch_dir.glob('*.ome.tif'))
        if ome_files:
            try:
                import tifffile
                for f in ome_files:
                    with tifffile.TiffFile(f) as tif:
                        if tif.is_ome:
                            import xml.etree.ElementTree as ET
                            channels = ET.fromstring(tif.ome_metadata).findall('.//{*}Channel')
                            size_mb = f.stat().st_size / 1024**2
                            print(f"✅ {f.name}: {len(channels)} channels, {size_mb:.1f}MB")
                            results['details']['ome_tiff'] = {'channels': len(channels)}
            except ImportError:
                print(f"ℹ️  tifffile not installed, skipping OME check")
        else:
            results['errors'].append("No OME-TIFF output")
            print(f"❌ No OME-TIFF files")
    
    # 6. Checkpoints
    cp_dir = output_dir / 'checkpoints'
    if cp_dir.exists():
        failed = [f.stem for f in cp_dir.glob('*.json') 
                  if json.load(open(f)).get('status') == 'failed']
        if failed:
            results['errors'].append(f"Failed steps: {failed}")
            print(f"❌ Failed: {failed}")
    
    # Summary
    results['status'] = 'failed' if results['errors'] else ('warning' if results['warnings'] else 'passed')
    print(f"\n{'='*50}")
    print(f"Status: {'✅ PASSED' if results['status']=='passed' else '⚠️ WARNING' if results['status']=='warning' else '❌ FAILED'}")
    print(f"{'='*50}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline output")
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('-r', '--report', type=Path, help='JSON report path')
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: {args.output_dir} not found", file=sys.stderr)
        return 2
    
    results = validate(args.output_dir)
    
    report_path = args.report or (args.output_dir / 'validation_report.json')
    json.dump(results, open(report_path, 'w'), indent=2)
    print(f"Report: {report_path}")
    
    return 0 if results['status'] == 'passed' else (1 if results['status'] == 'warning' else 2)


if __name__ == '__main__':
    sys.exit(main())
