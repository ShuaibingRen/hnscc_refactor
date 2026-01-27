
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def analyze_qc_selection(qc_dir):
    qc_path = Path(qc_dir)
    print(f"Scanning {qc_path}...")
    
    json_files = sorted(qc_path.glob('*_selected_samples.json'))
    
    if not json_files:
        print("No JSON files found.")
        return

    results = {}
    all_samples_global = set()
    row_sample_counts = {}  # Track how many samples were actually available per row
    
    for jf in json_files:
        try:
            with open(jf) as f:
                d = json.load(f)
                
                # Construct consistent row name: cycle_w{channel}
                # d['cycle'] usually contains "cycle0" or "quench1"
                c_name = d.get('cycle', 'unknown')
                u_chan = d.get('channel', '?')
                
                # Check formatting
                if '_w' in c_name:
                    # sometimes filename has cycle0_w1 but json cycle field might differ?
                    # let's rely on filename components if json is ambiguous, but json is safer
                    pass
                
                row_name = f"{c_name}_w{u_chan}"
                
                selected = set(d.get('selected_samples', []))
                all_s = set(d.get('all_samples', []))
                
                results[row_name] = selected
                all_samples_global.update(all_s)
                row_sample_counts[row_name] = len(all_s)
        except Exception as e:
            print(f"Skipping {jf}: {e}")

    # Sort Rows
    def sort_key(x):
        # x like cycle0_w1 or quench6_w4
        is_quench = x.startswith('quench')
        try:
            parts = x.split('_')
            # part0: cycle0 or quench6
            num_str = parts[0].replace('cycle', '').replace('quench', '')
            num = int(num_str)
            # part1: w1
            chan = int(parts[1].replace('w', ''))
        except:
            num = 999
            chan = 999
        return (is_quench, num, chan)

    rows_sorted = sorted(results.keys(), key=sort_key)
    samples_sorted = sorted(list(all_samples_global))
    
    # Build Matrix (1 = Selected, 0 = Not Selected, -1/NaN = Not Present?) 
    # Let's keep it simple: 1=Selected, 0=Not Selected (even if not present, implies not selected)
    matrix = []
    stats = []
    
    for row in rows_sorted:
        selected_set = results[row]
        # 1 if selected, 0 otherwise
        mat_row = [1 if s in selected_set else 0 for s in samples_sorted]
        matrix.append(mat_row)
        
        stats.append({
            'Cycle_Channel': row,
            'Selected_Count': len(selected_set),
            'Available_Samples': row_sample_counts.get(row, 0),
            'Selection_Rate': len(selected_set) / row_sample_counts.get(row, 1) if row_sample_counts.get(row, 0) > 0 else 0
        })
        
    df_matrix = pd.DataFrame(matrix, index=rows_sorted, columns=samples_sorted)
    df_stats = pd.DataFrame(stats)
    
    # Save CSVs
    stats_csv = qc_path / 'selection_summary.csv'
    matrix_csv = qc_path / 'selection_matrix.csv'
    df_stats.to_csv(stats_csv, index=False)
    df_matrix.to_csv(matrix_csv)
    print(f"Saved stats to {stats_csv}")
    print(f"Saved matrix to {matrix_csv}")
    
    # Plotting
    try:
        plt.figure(figsize=(max(12, len(samples_sorted)*0.3), max(8, len(rows_sorted)*0.3)))
        
        # Color map: White=0, Blue=1
        cmap = plt.cm.Blues
        
        plt.imshow(df_matrix.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Ticks
        plt.xticks(range(len(samples_sorted)), samples_sorted, rotation=90, fontsize=8)
        plt.yticks(range(len(rows_sorted)), rows_sorted, fontsize=9)
        
        # Grid lines
        plt.grid(which='major', color='gray', linestyle='-', linewidth=0.1, alpha=0.5)
        # We need to manually add grid lines effectively
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, len(samples_sorted), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows_sorted), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
        ax.tick_params(which='minor', bottom=False, left=False)

        plt.title('HNSCC Sample Selection Heatmap (Dark=Selected)', pad=20)
        plt.tight_layout()
        
        plot_path = qc_path / 'selection_heatmap.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {plot_path}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        qc_dir = sys.argv[1]
    else:
        # Default fallback
        qc_dir = "/mnt/efs/fs1/aws_home/shuaibing/RAW_i2/illuminate/qc"
    
    analyze_qc_selection(qc_dir)
