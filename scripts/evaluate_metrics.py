#!/usr/bin/env python3
"""
Evaluate metrics for scheme comparison.
Adapts to user's custom output paths.

Actual structure:
  46b_s*/subtract/cycleX/  - subtracted images
  46b_s*/illuminate/quenchX/  - quench images (for MI calculation)
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hnscc_refactor.modules.illumination.processor import _mutual_information_2d

# User data paths
BASE_DIR = Path("/mnt/efs/fs1/aws_home/shuaibing/batch4_refine")
OUTPUT_REPORT_DIR = BASE_DIR / "comparison_results"

# Scheme Directories - actual structure is 46b_s*/subtract/cycleX/
# s1 = Scheme 1 (Flat+Dark, Simple)
# s2 = Scheme 2 (Flat+Dark, Optimized)
# s3 = Scheme 3 (Flat Only, Optimized)
SCHEME_DIRS = {
    1: BASE_DIR / "46b_s1",
    2: BASE_DIR / "46b_s2",
    3: BASE_DIR / "46b_s3"
}

def ensure_dir(d):
    d.mkdir(parents=True, exist_ok=True)
    return d

def calculate_mi(img1, img2):
    try:
        if img1 is None or img2 is None: return np.nan
        return _mutual_information_2d(img1.ravel(), img2.ravel())
    except:
        return np.nan

def calculate_background_stats(img):
    if img is None: return np.nan, np.nan
    bg_level = np.percentile(img, 5)
    sig_level = np.percentile(img, 99)
    std_bg = np.std(img[img < np.percentile(img, 20)]) + 1e-6
    cnr = (sig_level - bg_level) / std_bg
    return bg_level, cnr

def generate_montage(scheme_dict, cycle, fov, channel="w2", output_path=None):
    images = []
    labels = []
    TARGET_H, TARGET_W = 500, 500
    
    for s_id, s_path in scheme_dict.items():
        # Path: 46b_s*/subtract/cycleX/
        cycle_dir = s_path / "subtract" / f"cycle{cycle}"
        
        candidates = []
        if cycle_dir.exists():
            candidates = list(cycle_dir.glob(f"*{channel}*{fov}*.TIF"))
                
        if not candidates:
            images.append(np.zeros((TARGET_H, TARGET_W), dtype=np.uint8))
            labels.append(f"Scheme {s_id}\n(Missing)")
            continue
            
        img_path = candidates[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        
        if img is not None:
            p99 = np.percentile(img, 99.5)
            if p99 > 0:
                img_disp = np.clip(img, 0, p99)
                img_disp = (img_disp / p99 * 255).astype(np.uint8)
            else:
                img_disp = np.zeros(img.shape, dtype=np.uint8)
                 
            if img_disp.shape[:2] != (TARGET_H, TARGET_W):
                 img_disp = cv2.resize(img_disp, (TARGET_W, TARGET_H))
                 
            images.append(img_disp)
        else:
            images.append(np.zeros((TARGET_H, TARGET_W), dtype=np.uint8))
            
        labels.append(f"Scheme {s_id}")

    if not images: return
    
    montage = np.zeros((TARGET_H + 50, TARGET_W * len(images)), dtype=np.uint8)
    
    for i, img in enumerate(images):
        montage[50:, i*TARGET_W:(i+1)*TARGET_W] = img
        cv2.putText(montage, labels[i], (i*TARGET_W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
        
    if output_path:
        cv2.imwrite(str(output_path), montage)

def evaluate():
    ensure_dir(OUTPUT_REPORT_DIR)
        
    print(f"Evaluation started...")
    print(f"Output directory: {OUTPUT_REPORT_DIR}")
    
    # Use Scheme 3 as reference for file list
    ref_path = SCHEME_DIRS[3] / "subtract"
    if not ref_path.exists():
        print(f"Error: Scheme 3 subtract path not found: {ref_path}")
        return

    results = []
    
    # Iterate cycles 2-7 (where subtraction usually happens)
    for cycle in range(2, 8):
        cycle_dir = ref_path / f"cycle{cycle}"
        if not cycle_dir.exists(): continue
        
        files = sorted(list(cycle_dir.glob("*.TIF")))
        print(f"Processing cycle {cycle} ({len(files)} files)...")
        
        # cycle2 -> quench1, etc.
        quench_num = cycle - 1
        quench_name = f"quench{quench_num}"
        
        # Limit to first 50 files per cycle for speed
        process_files = files[:50] 
        
        for f_path in process_files:
            fname = f_path.name
            
            try:
                parts = fname.split('_')
                channel = [p for p in parts if p.startswith('w')][0]
                fov = [p for p in parts if p.startswith('s')][0]
                
                row = {
                    "cycle": cycle,
                    "channel": channel,
                    "fov": fov,
                    "filename": fname
                }
                
                for s_id, s_path in SCHEME_DIRS.items():
                    # Result image: 46b_s*/subtract/cycleX/fname
                    res_path = s_path / "subtract" / f"cycle{cycle}" / fname
                    if not res_path.exists(): 
                        row[f"s{s_id}_exists"] = False
                        continue
                        
                    img_res = cv2.imread(str(res_path), cv2.IMREAD_ANYDEPTH)
                    if img_res is None: continue
                    
                    # Background Stats
                    bg, cnr = calculate_background_stats(img_res)
                    row[f"s{s_id}_bg"] = bg
                    row[f"s{s_id}_cnr"] = cnr
                    
                    # MI Calculation
                    # Quench image: 46b_s*/illuminate/quenchX/fname (with cycle replaced)
                    q_fname = fname.replace(f"cycle{cycle}", quench_name)
                    q_path = s_path / "illuminate" / quench_name / q_fname
                    
                    if q_path.exists():
                        img_q = cv2.imread(str(q_path), cv2.IMREAD_ANYDEPTH)
                        if img_q is not None:
                            mi = calculate_mi(img_res, img_q)
                            row[f"s{s_id}_mi"] = mi
                    
                results.append(row)
                
                # Generate montage for first file per cycle
                if f_path == process_files[0]:
                     m_name = f"montage_c{cycle}_{channel}_{fov}.jpg"
                     generate_montage(SCHEME_DIRS, cycle, fov, channel, OUTPUT_REPORT_DIR / m_name)

            except Exception as e:
                continue
    
    # Save Results
    df = pd.DataFrame(results)
    csv_path = OUTPUT_REPORT_DIR / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    
    # Summary
    print("\n=== Mean Metrics by Scheme ===")
    summary_cols = [c for c in df.columns if c.startswith('s') and ('_mi' in c or '_bg' in c)]
    if summary_cols:
        print(df[summary_cols].mean())
    
    # Simple Plot
    try:
        mi_cols = [f"s{i}_mi" for i in [1,2,3]]
        plt.figure(figsize=(8,6))
        df.boxplot(column=mi_cols)
        plt.title("Residual MI (Lower is Better)")
        plt.ylabel("Mutual Information")
        plt.savefig(OUTPUT_REPORT_DIR / "mi_boxplot.png")
        print("Saved plots.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    evaluate()
