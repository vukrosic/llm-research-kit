"""
Cross-Architecture Comparison
===============================
Tools for comparing results across different architecture families.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "frontier_results"
QUEUE_PATH = ROOT / "frontier" / "experiments" / "queue.json"
KNOWLEDGE_PATH = ROOT / "frontier" / "knowledge" / "mechanism_graph.json"


def load_all_results(token_budget: int = 6000000) -> Dict[str, Dict]:
    """Load all frontier experiment results for a given token budget."""
    results_dir = RESULTS_DIR / f"{token_budget}tok"
    if not results_dir.exists():
        return {}

    results = {}
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            results[exp_dir.name] = data
    return results


def rank_experiments(results: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Rank all experiments by val_loss."""
    ranked = []
    for exp_id, data in results.items():
        final = data.get("final_metrics", {})
        config = data.get("experiment_config", {})
        ranked.append({
            "exp_id": exp_id,
            "val_loss": final.get("val_loss", float("inf")),
            "val_accuracy": final.get("val_accuracy", 0),
            "arch_family": config.get("arch_family", "unknown"),
            "arch_class": config.get("arch_class", "unknown"),
            "total_params": config.get("total_params", 0),
            "sequence_complexity": config.get("sequence_complexity", "unknown"),
            "recurrent_inference": config.get("recurrent_inference", False),
        })
    ranked.sort(key=lambda x: x["val_loss"])
    return ranked


def compare_families(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compare best result per architecture family."""
    family_best = {}
    for exp_id, data in results.items():
        final = data.get("final_metrics", {})
        config = data.get("experiment_config", {})
        family = config.get("arch_family", "unknown")
        val_loss = final.get("val_loss", float("inf"))

        if family not in family_best or val_loss < family_best[family]["val_loss"]:
            family_best[family] = {
                "exp_id": exp_id,
                "val_loss": val_loss,
                "val_accuracy": final.get("val_accuracy", 0),
                "arch_class": config.get("arch_class", "unknown"),
                "total_params": config.get("total_params", 0),
            }
    return family_best


def update_knowledge_from_results(results: Dict[str, Dict]):
    """Update the knowledge graph with new experimental results."""
    with open(KNOWLEDGE_PATH) as f:
        knowledge = json.load(f)

    queue_data = []
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH) as f:
            queue_data = json.load(f)

    queue_map = {e["exp_id"]: e for e in queue_data}

    # Find transformer baseline
    transformer_loss = None
    for exp_id, data in results.items():
        config = data.get("experiment_config", {})
        if config.get("arch_family") == "transformer" or exp_id == "transformer_baseline":
            transformer_loss = data.get("final_metrics", {}).get("val_loss")
            break

    # If no transformer baseline in frontier results, use ablation system's best
    if transformer_loss is None:
        transformer_loss = 4.7888  # from ablation leaderboard

    # Update best_per_family
    family_best = compare_families(results)
    knowledge["best_per_family"] = {
        family: {
            "exp_id": info["exp_id"],
            "val_loss": info["val_loss"],
            "delta_vs_transformer": (transformer_loss - info["val_loss"]) / transformer_loss * 100,
        }
        for family, info in family_best.items()
    }

    # Update mechanism statuses based on results
    for exp_id, data in results.items():
        final = data.get("final_metrics", {})
        val_loss = final.get("val_loss", float("inf"))
        config = data.get("experiment_config", {})
        arch_family = config.get("arch_family", "unknown")

        queue_entry = queue_map.get(exp_id, {})
        delta_pct = (transformer_loss - val_loss) / transformer_loss * 100

        result_entry = {
            "exp_id": exp_id,
            "val_loss": val_loss,
            "delta_vs_transformer": delta_pct,
        }

        # Find and update matching mechanism
        for mech_name, mech_info in knowledge["mechanisms"].items():
            if mech_info["family"] == arch_family:
                mech_info["results"].append(result_entry)
                # Update status
                if delta_pct > 0.5:
                    mech_info["status"] = "winner"
                elif delta_pct > -2:
                    mech_info["status"] = "neutral"
                else:
                    mech_info["status"] = "loser"
                break

    with open(KNOWLEDGE_PATH, "w") as f:
        json.dump(knowledge, f, indent=2)
        f.write("\n")

    return knowledge


def generate_leaderboard_md(results: Dict[str, Dict], transformer_loss: float = 4.7888) -> str:
    """Generate leaderboard markdown."""
    ranked = rank_experiments(results)

    lines = [
        "# Frontier Architecture Leaderboard",
        "",
        f"**Transformer baseline**: val_loss = {transformer_loss:.4f} (from ablation system, gen12_warm150)",
        "",
        "| Rank | exp_id | Family | val_loss | Δ vs transformer | Params | Complexity | Recurrent |",
        "|------|--------|--------|----------|------------------|--------|------------|-----------|",
    ]

    for i, entry in enumerate(ranked, 1):
        delta = (transformer_loss - entry["val_loss"]) / transformer_loss * 100
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        recurrent = "Yes" if entry["recurrent_inference"] else "No"
        params_str = f"{entry['total_params']/1e6:.1f}M" if entry["total_params"] > 0 else "?"

        lines.append(
            f"| {i} | {entry['exp_id']} | {entry['arch_family']} | "
            f"{entry['val_loss']:.4f} | {delta_str} | {params_str} | "
            f"{entry['sequence_complexity']} | {recurrent} |"
        )

    return "\n".join(lines)


def print_summary():
    """Print a summary of all frontier results."""
    results = load_all_results()
    if not results:
        print("No frontier results found yet.")
        return

    ranked = rank_experiments(results)
    family_best = compare_families(results)

    print(f"\n{'='*80}")
    print("FRONTIER ARCHITECTURE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal experiments: {len(results)}")
    print(f"Transformer baseline: 4.7888 (ablation system best)")

    print(f"\n{'─'*80}")
    print("Best per family:")
    for family, info in sorted(family_best.items(), key=lambda x: x[1]["val_loss"]):
        delta = (4.7888 - info["val_loss"]) / 4.7888 * 100
        sign = "+" if delta > 0 else ""
        print(f"  {family:20s}  {info['exp_id']:30s}  val_loss={info['val_loss']:.4f}  ({sign}{delta:.2f}%)")

    print(f"\n{'─'*80}")
    print("Full ranking:")
    for i, entry in enumerate(ranked[:20], 1):
        delta = (4.7888 - entry["val_loss"]) / 4.7888 * 100
        sign = "+" if delta > 0 else ""
        print(f"  {i:2d}. {entry['exp_id']:30s}  {entry['arch_family']:15s}  "
              f"val_loss={entry['val_loss']:.4f}  ({sign}{delta:.2f}%)")


if __name__ == "__main__":
    print_summary()
