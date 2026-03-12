import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_multi_loss(metrics_files, output_file, title="Validation Loss Comparison"):
    plt.figure(figsize=(15, 10))
    
    # Sort files to ensure consistent legend order
    metrics_files = sorted(metrics_files)
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            
            # The experiment name is usually the parent directory name 
            # or in the JSON itself
            exp_name = data.get("experiment_config", {}).get("experiment_name", Path(metrics_file).parent.name)
            
            steps = data["history"]["steps"]
            val_loss = data["history"]["val_losses"]
            
            plt.plot(steps, val_loss, label=exp_name, alpha=0.8, linewidth=1.5)
        except Exception as e:
            print(f"Skipping {metrics_file}: {e}")

    plt.title(title, fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.yscale('log') # Log scale is often better for loss comparison
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()

    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"✅ Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Metrics JSON files to plot")
    parser.add_argument("--output", default="ablation_comparison.png", help="Output image file")
    parser.add_argument("--title", default="SwiGLU Ablation (10M Tokens) - First 17 Experiments", help="Plot title")
    args = parser.parse_args()
    
    if args.files:
        plot_multi_loss(args.files, args.output, args.title)
    else:
        print("No files provided.")
