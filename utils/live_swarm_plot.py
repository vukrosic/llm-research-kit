import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme for aesthetics
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['grid.color'] = '#333333'

def generate_live_plots():
    results_dir = '/root/llm-research-kit/ablation_results/10000000tok'
    metric_files = glob.glob(os.path.join(results_dir, '**/metrics.json'), recursive=True)
    
    if not metric_files:
        print("No metrics found yet.")
        return

    all_data = []
    for f in metric_files:
        try:
            with open(f, 'r') as j:
                all_data.append(json.load(j))
        except:
            continue

    # 1. Validation Loss Plot
    plt.figure(figsize=(12, 7))
    
    # Sort by baseline first
    all_data.sort(key=lambda x: 0 if x['experiment_name'] == 'baseline' else 1)
    
    for data in all_data:
        name = data['experiment_name']
        history = data.get('history', {})
        if 'steps' in history and 'val_losses' in history:
            # We only have a few points for 10k runs, but let's plot them
            steps = history['steps']
            losses = history['val_losses']
            
            # If it's a very short run, we might only have 1 eval point. 
            # Let's add the final metric if it's not in history
            if len(losses) == 0:
                 losses = [data['final_metrics']['val_loss']]
                 steps = [data['actual_steps']]

            alpha = 1.0 if name == 'baseline' else 0.7
            linewidth = 3 if name == 'baseline' else 1.5
            
            plt.plot(steps, losses, label=name, marker='o', alpha=alpha, linewidth=linewidth)

    plt.title('Live Swarm Progress: Validation Loss (10k tokens)', fontsize=16, pad=20, color='#00d1ff')
    plt.xlabel('Steps')
    plt.ylabel('Val Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = '/root/llm-research-kit/plots/live_swarm_loss.png'
    plt.savefig(save_path, dpi=150)
    print(f"✅ Loss plot generated: {save_path}")

    # 2. Performance Comparison Bar Chart
    plt.figure(figsize=(10, 8))
    
    # Extract final losses
    comp_data = []
    baseline_loss = None
    for data in all_data:
        name = data['experiment_name']
        loss = data['final_metrics']['val_loss']
        if name == 'baseline':
            baseline_loss = loss
        comp_data.append((name, loss))
    
    # Sort by loss
    comp_data.sort(key=lambda x: x[1])
    
    names = [x[0] for x in comp_data]
    losses = [x[1] for x in comp_data]
    
    colors = ['#FFD700' if n == 'baseline' else '#4e79a7' for n in names]
    
    bars = plt.barh(names, losses, color=colors)
    plt.axvline(x=baseline_loss, color='gold', linestyle='--', alpha=0.5, label='Baseline') if baseline_loss else None
    
    plt.title('Final Validation Loss Comparison (Lower is Better)', fontsize=15, color='#ff6b6b')
    plt.xlabel('Loss')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path_bar = '/root/llm-research-kit/plots/live_swarm_comparison.png'
    plt.savefig(save_path_bar, dpi=150)
    print(f"✅ Comparison plot generated: {save_path_bar}")

if __name__ == "__main__":
    generate_live_plots()
