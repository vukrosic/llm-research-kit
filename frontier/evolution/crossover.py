"""
Architecture Crossover
=======================
Combines winning mechanisms from different architecture families
to create novel hybrids.

This module implements the cross-pollination logic described in FRONTIER.md:
- Take winning mechanism from family A, test in family B
- Build hybrids with unique winners from different families
- Track every cross-pollination attempt in the knowledge graph
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = ROOT / "frontier" / "knowledge" / "mechanism_graph.json"


def load_knowledge():
    with open(KNOWLEDGE_PATH) as f:
        return json.load(f)


def save_knowledge(knowledge: dict):
    with open(KNOWLEDGE_PATH, "w") as f:
        json.dump(knowledge, f, indent=2)
        f.write("\n")


def find_transferable_mechanisms(knowledge: dict) -> List[Dict[str, Any]]:
    """
    Find mechanisms that won in one family and haven't been tested in others.

    Returns list of {mechanism, source_family, target_families} dicts.
    """
    winners = {
        name: info for name, info in knowledge["mechanisms"].items()
        if info["status"] == "winner"
    }

    all_families = set(
        info["family"] for info in knowledge["mechanisms"].values()
    )

    transferable = []
    for name, info in winners.items():
        source_family = info["family"]
        # Check which families haven't tested this mechanism
        tested_families = {source_family}
        for edge in knowledge.get("edges", []):
            if edge["from"] == name or edge["to"] == name:
                for mech_name, mech_info in knowledge["mechanisms"].items():
                    if mech_name in (edge["from"], edge["to"]):
                        tested_families.add(mech_info["family"])

        untested = all_families - tested_families
        if untested:
            transferable.append({
                "mechanism": name,
                "source_family": source_family,
                "target_families": list(untested),
                "original_delta": info.get("results", [{}])[-1].get("delta_vs_transformer", 0),
            })

    return transferable


def record_cross_pollination(
    knowledge: dict,
    mechanism: str,
    source_family: str,
    target_family: str,
    exp_id: str,
    result: str,  # "positive", "negative", "neutral"
    delta: float,
    notes: str = "",
):
    """Record a cross-pollination attempt in the knowledge graph."""
    entry = {
        "mechanism": mechanism,
        "source_family": source_family,
        "target_family": target_family,
        "exp_id": exp_id,
        "result": result,
        "delta": delta,
        "notes": notes,
    }
    knowledge.setdefault("cross_pollination_log", []).append(entry)

    # Add edge to graph
    knowledge.setdefault("edges", []).append({
        "from": mechanism,
        "to": f"{target_family}_application",
        "type": "synergy" if result == "positive" else "conflict" if result == "negative" else "neutral",
        "evidence": f"{exp_id}: delta={delta:.4f}. {notes}",
    })

    save_knowledge(knowledge)
    return entry


def propose_crossover_experiments(
    knowledge: dict,
    max_proposals: int = 5,
) -> List[Dict[str, Any]]:
    """
    Propose experiments that combine winning mechanisms from different families.

    Strategy:
    1. Find all winners
    2. For each pair of winners from different families, propose a hybrid
    3. Prioritize pairs where both mechanisms had large positive deltas
    """
    winners = [
        (name, info) for name, info in knowledge["mechanisms"].items()
        if info["status"] == "winner"
    ]

    if len(winners) < 2:
        return []

    proposals = []
    seen_pairs = set()

    for i, (name_a, info_a) in enumerate(winners):
        for name_b, info_b in winners[i+1:]:
            if info_a["family"] == info_b["family"]:
                continue  # same family, not a crossover

            pair_key = tuple(sorted([name_a, name_b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Estimate priority based on combined delta
            delta_a = info_a.get("results", [{}])[-1].get("delta_vs_transformer", 0)
            delta_b = info_b.get("results", [{}])[-1].get("delta_vs_transformer", 0)

            proposals.append({
                "mechanism_a": name_a,
                "family_a": info_a["family"],
                "mechanism_b": name_b,
                "family_b": info_b["family"],
                "combined_delta_estimate": delta_a + delta_b,
                "hypothesis": f"Combine {name_a} ({info_a['family']}) with {name_b} ({info_b['family']}) — independent wins may stack",
            })

    # Sort by estimated combined delta (best first)
    proposals.sort(key=lambda p: p["combined_delta_estimate"], reverse=True)
    return proposals[:max_proposals]
