import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for a premium look
sns.set_theme(style="whitegrid", context="talk", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25

def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_ablation_plot():
    # Define experiments to plot
    experiments = {
        "checkpoints/qk_norm": {"label": "With QK Norm (Base)", "color": "#1f77b4", "marker": "o"},
        "checkpoints/per_head_scaling": {"label": "Per-Head Scaling", "color": "#d62728", "marker": "^"},
        "checkpoints/k_only_norm": {"label": "K-only Norm", "color": "#9467bd", "marker": "<"},
        "checkpoints/shared_norm": {"label": "Shared Norm", "color": "#8c564b", "marker": ">"},
        "checkpoints/qk_bias": {"label": "QK Bias", "color": "#e377c2", "marker": "D"},
    }
    
    plt.figure(figsize=(14, 8))
    
    any_data = False
    for path, meta in experiments.items():
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Skipping {path}: No metrics.json found.")
            continue
        
        any_data = True
        data = load_metrics(metrics_path)
        steps = data["history"]["steps"]
        losses = data["history"]["val_losses"]
        
        plt.plot(steps, losses, marker=meta["marker"], markersize=6, linewidth=2, 
                 label=meta["label"], color=meta["color"], alpha=0.85)

    if not any_data:
        print("Error: No metrics data found for any experiment.")
        return

    # Formatting
    plt.title("QK Normalization Ablation Study", pad=20, fontsize=22, fontweight='bold')
    plt.xlabel("Training Steps", labelpad=12, fontsize=14)
    plt.ylabel("Validation Loss", labelpad=12, fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', prop={'size': 12})
    
    # Optional: zoom in on the interesting part of the loss curve if needed
    # plt.ylim(2.5, 4.5) 

    plt.tight_layout()
    save_path = "plots/ablation_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved combined ablation comparison plot to {save_path}")

    # Zoomed late-stage view to reveal small loss differences between long runs
    plt.figure(figsize=(14, 8))
    zoom_min = None
    zoom_max = None
    max_step = 0
    zoom_start_step = 1000

    for path, meta in experiments.items():
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        data = load_metrics(metrics_path)
        steps = data["history"]["steps"]
        losses = data["history"]["val_losses"]
        zoom_points = [(s, l) for s, l in zip(steps, losses) if s >= zoom_start_step]
        if not zoom_points:
            continue

        z_steps, z_losses = zip(*zoom_points)
        max_step = max(max_step, max(z_steps))
        zmin = min(z_losses)
        zmax = max(z_losses)
        zoom_min = zmin if zoom_min is None else min(zoom_min, zmin)
        zoom_max = zmax if zoom_max is None else max(zoom_max, zmax)

        plt.plot(
            z_steps,
            z_losses,
            marker=meta["marker"],
            markersize=6,
            linewidth=2,
            label=meta["label"],
            color=meta["color"],
            alpha=0.9,
        )

    if zoom_min is not None and zoom_max is not None:
        span = zoom_max - zoom_min
        pad = max(span * 0.15, 0.01)
        plt.xlim(zoom_start_step - 100, max_step + 100)
        plt.ylim(zoom_min - pad, zoom_max + pad)

    plt.title("QK Normalization Ablations (Zoomed Late Stage)", pad=20, fontsize=22, fontweight="bold")
    plt.xlabel("Training Steps", labelpad=12, fontsize=14)
    plt.ylabel("Validation Loss", labelpad=12, fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", prop={"size": 12})

    plt.tight_layout()
    zoom_save_path = "plots/ablation_comparison_zoom_long.png"
    plt.savefig(zoom_save_path, dpi=300)
    print(f"Successfully saved zoomed late-stage comparison plot to {zoom_save_path}")

    # Tokens plotting
    plt.figure(figsize=(14, 8))
    for path, meta in experiments.items():
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.exists(metrics_path): continue
        
        data = load_metrics(metrics_path)
        steps = data["history"]["steps"]
        losses = data["history"]["val_losses"]
        tokens_total = data.get("tokens_seen", 15_000_000)
        
        # Calculate tokens at each eval step
        step_tokens = [s * (tokens_total / steps[-1]) / 1e6 for s in steps]
        
        plt.plot(step_tokens, losses, marker=meta["marker"], markersize=6, linewidth=2, 
                 label=meta["label"], color=meta["color"], alpha=0.85)

    plt.title("Scaling Efficiency: QK Norm Ablations", pad=20, fontsize=22, fontweight='bold')
    plt.xlabel("Tokens Seen (Millions)", labelpad=12, fontsize=14)
    plt.ylabel("Validation Loss", labelpad=12, fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', prop={'size': 12})
    
    plt.tight_layout()
    token_save_path = "plots/ablation_tokens_comparison.png"
    plt.savefig(token_save_path, dpi=300)
    print(f"Successfully saved combined token comparison plot to {token_save_path}")

if __name__ == "__main__":
    create_ablation_plot()
