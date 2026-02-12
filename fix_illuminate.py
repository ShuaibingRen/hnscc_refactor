"""
Fix script: Scan and re-apply illumination correction for all incomplete sample/cycle/channel.
Uses the already-computed mean correction matrices.

Root cause: macOS resource fork files (._*) were matched by glob, causing cv2.imread
to return None and crash. This script uses the corrected file listing logic.

Usage:
    python fix_illuminate.py --scan          # Only scan and report issues
    python fix_illuminate.py --fix           # Scan and fix all issues
"""
import cv2
import os
import re
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


INPUT_DIR = Path("/mnt/efs/fs1/aws_home/shuaibing/RAW")
OUTPUT_DIR = Path("/mnt/efs/fs1/aws_home/shuaibing/RAW_i2/illuminate")
CORRECTION_DIR = OUTPUT_DIR / "correction"
SKIP_DIRS = {'correction', 'correction_per_sample', 'qc', 'checkpoints'}

# Pattern: real image files only (exclude macOS ._ resource fork files)
FILE_PATTERN = re.compile(r'^[^.].*_w(\d+)_.*\.TIF$')


def list_image_files(directory, channel=None):
    """List real image files, excluding macOS resource fork files (._*)."""
    try:
        all_files = os.listdir(str(directory))
    except OSError:
        return []
    
    result = []
    for f in all_files:
        m = FILE_PATTERN.match(f)
        if m:
            if channel is None or int(m.group(1)) == channel:
                result.append(directory / f)
    return sorted(result)


def exclude_overview(files):
    """Exclude the overview FOV (max FOV number)."""
    fov_map = {}
    for f in files:
        m = re.search(r'_s(\d+)_', f.name)
        if m:
            fov_map[f] = int(m.group(1))
    if not fov_map:
        return files
    max_fov = max(fov_map.values())
    return sorted([f for f, fov in fov_map.items() if fov < max_fov],
                  key=lambda f: fov_map[f])


def scan_issues():
    """Scan output directory and find all incomplete sample/cycle/channel."""
    issues = []
    
    for sample_name in sorted(os.listdir(str(OUTPUT_DIR))):
        sample_dir = OUTPUT_DIR / sample_name
        if not sample_dir.is_dir() or sample_name in SKIP_DIRS:
            continue
        
        for cycle_name in sorted(os.listdir(str(sample_dir))):
            cycle_dir = sample_dir / cycle_name
            if not cycle_dir.is_dir():
                continue
            
            for w in range(1, 5):
                out_files = list_image_files(cycle_dir, channel=w)

                inp_dir = INPUT_DIR / sample_name / cycle_name
                if not inp_dir.is_dir():
                    continue
                
                in_files = exclude_overview(list_image_files(inp_dir, channel=w))
                expected = len(in_files)
                actual = len(out_files)
                
                if actual < expected and expected > 0:
                    issues.append({
                        'sample': sample_name,
                        'cycle': cycle_name,
                        'channel': w,
                        'actual': actual,
                        'expected': expected,
                    })
    
    return issues


def fix_issue(sample, cycle, channel):
    """Re-apply illumination correction for one sample/cycle/channel."""
    flat_path = CORRECTION_DIR / f"{cycle}_w{channel}_flatfield.tif"
    dark_path = CORRECTION_DIR / f"{cycle}_w{channel}_darkfield.tif"
    
    if not flat_path.exists():
        print(f"  ERROR: Flatfield not found: {flat_path}")
        return 0
    
    flatfield = cv2.imread(str(flat_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    darkfield = np.zeros_like(flatfield)
    if dark_path.exists():
        darkfield = cv2.imread(str(dark_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    
    # Get input files, exclude overview and ._ files
    files = exclude_overview(list_image_files(INPUT_DIR / sample / cycle, channel=channel))
    
    out_dir = OUTPUT_DIR / sample / cycle
    out_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    skipped = 0
    for img_path in files:
        out_path = out_dir / img_path.name
        if out_path.exists():
            skipped += 1
            continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f"  WARNING: Cannot read {img_path.name}, skipping")
            continue
        
        img = img.astype(np.float32)
        corrected = (img - darkfield) / (flatfield + 1e-10)
        corrected = corrected * np.mean(flatfield)
        corrected = np.maximum(corrected, 0)
        
        Image.fromarray(corrected.astype(np.uint16)).save(str(out_path))
        success += 1
    
    print(f"  Skipped {skipped} existing, processed {success} new")
    return success


def main():
    parser = argparse.ArgumentParser(description="Fix incomplete illumination correction")
    parser.add_argument('--scan', action='store_true', help='Only scan and report issues')
    parser.add_argument('--fix', action='store_true', help='Scan and fix all issues')
    args = parser.parse_args()
    
    if not args.scan and not args.fix:
        parser.print_help()
        return
    
    print("Scanning for incomplete illumination outputs...")
    issues = scan_issues()
    
    if not issues:
        print("No issues found. All outputs are complete.")
        return
    
    print(f"\nFound {len(issues)} incomplete outputs:")
    for iss in issues:
        print(f"  {iss['sample']} {iss['cycle']} w{iss['channel']}: "
              f"{iss['actual']}/{iss['expected']} files")
    
    if args.scan:
        return
    
    print(f"\nFixing {len(issues)} issues...")
    for i, iss in enumerate(issues, 1):
        print(f"\n[{i}/{len(issues)}] Fixing {iss['sample']} {iss['cycle']} w{iss['channel']}...")
        count = fix_issue(iss['sample'], iss['cycle'], iss['channel'])
        print(f"  Done: {count} files processed (expected {iss['expected']})")
    
    print("\nAll done!")


if __name__ == '__main__':
    main()
