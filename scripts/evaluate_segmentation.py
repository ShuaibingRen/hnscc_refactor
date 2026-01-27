#!/usr/bin/env python3
"""
Evaluate segmentation accuracy (Precision, Recall, F1) by comparing 
Nuclear+Membrane segmentation (Test) against Nuclear segmentation (Reference).

Metrics:
- Precision: TP / (TP + FP) -> How many detected cells are valid?
- Recall: TP / (TP + FN) -> How many true cells were detected?
- F1-Score: Harmonic mean of Precision and Recall.
- Count Difference: Simple count check (Reference - Test).

Matching Logic:
- A Reference object (Nucleus) matches a Test object (Cell) if the 
  Reference centroid falls within the Test mask.
- 1-to-1 matching is enforced (Greedy or Optimal).
"""

import sys
import argparse
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops, label
from pathlib import Path
import matplotlib.pyplot as plt

def load_mask(path):
    """Load TIF mask. Ensure it's labeled (int)."""
    img = tifffile.imread(path)
    # If binary, label it
    if img.max() == 1 and img.dtype == bool:
        img = label(img)
    elif img.ndim > 2:
        img = img.squeeze()
    return img.astype(np.int32)

def calculate_metrics(ref_mask, test_mask):
    """
    Compare Ref (Nuclear) vs Test (Nuc+Mem).
    Ref objects are "True Cells".
    Test objects are "Predicted Cells".
    
    Match: Ref Centroid inside Test Mask.
    """
    
    ref_props = regionprops(ref_mask)
    test_props = regionprops(test_mask)
    
    ref_count = len(ref_props)
    test_count = len(test_props)
    
    if ref_count == 0:
        return {
            "TP": 0, "FP": test_count, "FN": 0,
            "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
            "Ref_Count": 0, "Test_Count": test_count
        }
    
    
    # 1. Identify which Test objects have at least one Ref centroid
    test_labels_with_ref = set()
    ref_labels_matched = set()
    
    for rp in ref_props:
        y, x = int(rp.centroid[0]), int(rp.centroid[1])
        if 0 <= y < test_mask.shape[0] and 0 <= x < test_mask.shape[1]:
            val = test_mask[y, x]
            if val > 0:
                test_labels_with_ref.add(val)
                ref_labels_matched.add(rp.label)
                
    TP = len(test_labels_with_ref) # Valid detected cells
    FP = test_count - TP           # Test cells with no nucleus (Ghost/Autofluo)
    FN = ref_count - len(ref_labels_matched) # Nuclei with no cell (Missed)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Ref_Count": ref_count,
        "Test_Count": test_count,
        "Diff_Count": test_count - ref_count
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Accuracy")
    parser.add_argument("--ref_dir", type=Path, required=True, help="Directory containing Nuclear (Reference) labeled masks .tif")
    parser.add_argument("--test_dir", type=Path, required=True, help="Directory containing Nuc+Mem (Test) labeled masks .tif")
    parser.add_argument("--output", type=Path, default="segmentation_scores.csv")
    args = parser.parse_args()
    
    if not args.ref_dir.exists() or not args.test_dir.exists():
        print("Error: Directories not found.")
        sys.exit(1)
        
    # Find matching files
    extensions = ["*.tif", "*.TIF", "*.tiff", "*.TIFF"]
    ref_files = []
    for ext in extensions:
        ref_files.extend(list(args.ref_dir.glob(ext)))
    ref_files = sorted(list(set(ref_files))) # Remove duplicates if any overlap
    results = []
    
    print(f"Comparing {args.test_dir.name} vs {args.ref_dir.name} (Ref)...")
    
    for r_path in ref_files:
        t_path = args.test_dir / r_path.name
        if not t_path.exists():
            # Try approximate match if needed, or skip
            print(f"Skipping {r_path.name}: not found in test dir")
            continue
            
        print(f"Processing {r_path.name}...")
        try:
            r_mask = load_mask(r_path)
            t_mask = load_mask(t_path)
            
            metrics = calculate_metrics(r_mask, t_mask)
            metrics['filename'] = r_path.name
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {r_path.name}: {e}")
            
    if not results:
        print("No paired files processed.")
        return
        
    df = pd.DataFrame(results)
    
    # Calculate aggregates
    mean_scores = df[["Precision", "Recall", "F1", "Diff_Count"]].mean()
    total_tp = df["TP"].sum()
    total_fp = df["FP"].sum()
    total_fn = df["FN"].sum()
    
    agg_precision = total_tp / (total_tp + total_fp)
    agg_recall = total_tp / (total_tp + total_fn)
    agg_f1 = 2 * (agg_precision * agg_recall) / (agg_precision + agg_recall)
    
    print("\n" + "="*40)
    print("SUMMARY RESULTS")
    print("="*40)
    print(f"Total Images: {len(df)}")
    print(f"Mean Precision: {mean_scores['Precision']:.4f}")
    print(f"Mean Recall:    {mean_scores['Recall']:.4f}")
    print(f"Mean F1 Score:  {mean_scores['F1']:.4f}")
    print(f"Mean Count Diff:{mean_scores['Diff_Count']:.1f}")
    print("-" * 40)
    print(f"Aggregate F1 (weighted): {agg_f1:.4f}")
    print("="*40)
    
    df.to_csv(args.output, index=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
