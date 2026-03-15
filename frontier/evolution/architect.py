"""
Architecture Evolution System
===============================
Reads the knowledge graph and generates new experiment candidates.

This is the "brain" of the self-building research system. It analyzes
what's been tried, what worked, what failed, and proposes the most
promising next experiments.

Usage:
    python -m frontier.evolution.architect --batch-size 15

    # Or from Python:
    from frontier.evolution.architect import generate_next_batch
    experiments = generate_next_batch(batch_size=15)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KNOWLEDGE_DIR = ROOT / "frontier" / "knowledge"
QUEUE_PATH = ROOT / "frontier" / "experiments" / "queue.json"


@dataclass
class ExperimentProposal:
    exp_id: str
    arch_family: str
    arch_class: str
    hypothesis: str
    source: str  # "cross_pollination", "hybrid", "novel", "exploitation"
    parent_exp: str
    expected_delta: str
    priority: int
    arch_config: Dict[str, Any]
    tags: List[str]


def load_knowledge():
    """Load the full knowledge graph."""
    path = KNOWLEDGE_DIR / "mechanism_graph.json"
    with open(path) as f:
        return json.load(f)


def load_queue():
    """Load current experiment queue."""
    with open(QUEUE_PATH) as f:
        return json.load(f)


def get_untested_mechanisms(knowledge: dict) -> List[str]:
    """Find mechanisms that haven't been tested yet."""
    return [
        name for name, info in knowledge["mechanisms"].items()
        if info["status"] == "untested"
    ]


def get_winning_mechanisms(knowledge: dict) -> List[str]:
    """Find mechanisms that showed positive results."""
    return [
        name for name, info in knowledge["mechanisms"].items()
        if info["status"] == "winner"
    ]


def get_done_experiments(queue: list) -> set:
    """Get set of completed experiment IDs."""
    return {e["exp_id"] for e in queue if e.get("status") in ("done", "running")}


def propose_tier1_baselines(knowledge: dict, done: set) -> List[ExperimentProposal]:
    """
    Phase 1: Propose baseline experiments for each Tier 1 architecture family.
    One canonical experiment per family that hasn't been run yet.
    """
    proposals = []

    tier1_baselines = [
        ExperimentProposal(
            exp_id="ssm_mamba_baseline",
            arch_family="state_space",
            arch_class="MambaLM",
            hypothesis="Selective SSM provides competitive quality with O(n) complexity",
            source="tier1_baseline",
            parent_exp="transformer_baseline",
            expected_delta="within 5% of transformer",
            priority=1,
            arch_config={
                "d_model": 512, "n_layers": 24, "d_ff": 2048,
                "d_state": 64, "d_conv": 4, "expand_factor": 2,
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["state-space", "tier1", "baseline"],
        ),
        ExperimentProposal(
            exp_id="linattn_elu_baseline",
            arch_family="linear_attention",
            arch_class="LinearAttnLM",
            hypothesis="Linear attention with ELU map establishes a lower bound for linear methods",
            source="tier1_baseline",
            parent_exp="transformer_baseline",
            expected_delta="5-15% worse than transformer",
            priority=1,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "n_heads": 8, "feature_map": "elu",
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["linear-attention", "tier1", "baseline"],
        ),
        ExperimentProposal(
            exp_id="gla_baseline",
            arch_family="linear_attention",
            arch_class="GLALM",
            hypothesis="Gated linear attention improves over basic linear attention via learned decay",
            source="tier1_baseline",
            parent_exp="linattn_elu_baseline",
            expected_delta="2-5% better than basic linear attn",
            priority=1,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
            },
            tags=["linear-attention", "gla", "tier1", "baseline"],
        ),
        ExperimentProposal(
            exp_id="retnet_baseline",
            arch_family="retention",
            arch_class="RetNetLM",
            hypothesis="Multi-scale retention provides good quality with recurrent inference capability",
            source="tier1_baseline",
            parent_exp="transformer_baseline",
            expected_delta="within 3% of transformer",
            priority=1,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
            },
            tags=["retention", "tier1", "baseline"],
        ),
        ExperimentProposal(
            exp_id="rwkv_baseline",
            arch_family="rwkv",
            arch_class="RWKVLM",
            hypothesis="RWKV's WKV operator provides competitive language modeling with pure RNN inference",
            source="tier1_baseline",
            parent_exp="transformer_baseline",
            expected_delta="within 5% of transformer",
            priority=1,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["rwkv", "tier1", "baseline"],
        ),
    ]

    for p in tier1_baselines:
        if p.exp_id not in done:
            proposals.append(p)

    return proposals


def propose_tier2_experiments(knowledge: dict, done: set) -> List[ExperimentProposal]:
    """Phase 1 continued: Tier 2 and experimental baselines."""
    proposals = []

    tier2 = [
        ExperimentProposal(
            exp_id="hyena_baseline",
            arch_family="convolution",
            arch_class="HyenaLM",
            hypothesis="Hyena long convolutions capture sequential patterns without attention",
            source="tier2_baseline",
            parent_exp="transformer_baseline",
            expected_delta="within 10% of transformer",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "order": 2, "kernel_size": 128,
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["convolution", "hyena", "tier2", "baseline"],
        ),
        ExperimentProposal(
            exp_id="multires_baseline",
            arch_family="convolution",
            arch_class="MultiResLM",
            hypothesis="Multi-resolution convolutions capture patterns at multiple scales",
            source="tier2_baseline",
            parent_exp="transformer_baseline",
            expected_delta="within 10% of transformer",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "kernel_sizes": [3, 7, 15, 31],
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["convolution", "multi-resolution", "tier2", "baseline"],
        ),
        ExperimentProposal(
            exp_id="diffattn_baseline",
            arch_family="experimental",
            arch_class="DiffAttnLM",
            hypothesis="Differential attention cancels noise in attention patterns",
            source="novel",
            parent_exp="transformer_baseline",
            expected_delta="±2% of transformer",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
            },
            tags=["experimental", "differential-attention", "baseline"],
        ),
        ExperimentProposal(
            exp_id="freqmix_baseline",
            arch_family="experimental",
            arch_class="FreqMixerLM",
            hypothesis="Frequency-domain mixing provides global context efficiently",
            source="novel",
            parent_exp="transformer_baseline",
            expected_delta="5-15% worse than transformer",
            priority=3,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["experimental", "frequency", "baseline"],
        ),
        ExperimentProposal(
            exp_id="evolving_state_baseline",
            arch_family="experimental",
            arch_class="EvolvingStateLM",
            hypothesis="Full-rank evolving state matrix provides higher capacity than vector state SSMs",
            source="novel",
            parent_exp="ssm_mamba_baseline",
            expected_delta="±5% of Mamba",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "d_state": 32, "residual_scale": 1.0, "use_bias": True,
            },
            tags=["experimental", "evolving-state", "baseline"],
        ),
        ExperimentProposal(
            exp_id="polyattn_deg2_baseline",
            arch_family="experimental",
            arch_class="PolyAttnLM",
            hypothesis="Quadratic polynomial attention captures feature interactions better than softmax",
            source="novel",
            parent_exp="transformer_baseline",
            expected_delta="±3% of transformer",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "n_heads": 8, "degree": 2,
                "residual_scale": 1.0, "use_bias": True,
            },
            tags=["experimental", "polynomial-attention", "baseline"],
        ),
    ]

    for p in tier2:
        if p.exp_id not in done:
            proposals.append(p)

    return proposals


def propose_hybrids(knowledge: dict, done: set) -> List[ExperimentProposal]:
    """Propose hybrid architectures mixing different layer types."""
    proposals = []

    hybrids = [
        ExperimentProposal(
            exp_id="hybrid_ssm_attn_alt",
            arch_family="hybrid",
            arch_class="HybridLM",
            hypothesis="Alternating SSM+attention layers combine local tracking with global retrieval",
            source="hybrid",
            parent_exp="ssm_mamba_baseline",
            expected_delta="better than both pure SSM and pure attention",
            priority=2,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "pattern": "alternating_ssm_attn",
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
                "ssm_config": {"d_state": 64, "d_conv": 4, "expand_factor": 2},
            },
            tags=["hybrid", "ssm", "attention"],
        ),
        ExperimentProposal(
            exp_id="hybrid_ssm_heavy",
            arch_family="hybrid",
            arch_class="HybridLM",
            hypothesis="3:1 SSM-to-attention ratio minimizes attention cost while keeping retrieval",
            source="hybrid",
            parent_exp="hybrid_ssm_attn_alt",
            expected_delta="similar quality to alternating with less compute",
            priority=3,
            arch_config={
                "d_model": 512, "n_layers": 24, "d_ff": 2048,
                "pattern": "ssm_heavy",
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
                "ssm_config": {"d_state": 64, "d_conv": 4, "expand_factor": 2},
            },
            tags=["hybrid", "ssm", "attention", "efficiency"],
        ),
        ExperimentProposal(
            exp_id="hybrid_progressive",
            arch_family="hybrid",
            arch_class="HybridLM",
            hypothesis="SSM for local features early, attention for global reasoning late",
            source="hybrid",
            parent_exp="hybrid_ssm_attn_alt",
            expected_delta="better gradient flow than alternating",
            priority=3,
            arch_config={
                "d_model": 512, "n_layers": 24, "d_ff": 2048,
                "pattern": "progressive",
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
                "ssm_config": {"d_state": 64, "d_conv": 4, "expand_factor": 2},
            },
            tags=["hybrid", "ssm", "attention", "progressive"],
        ),
        ExperimentProposal(
            exp_id="hybrid_gla_attn",
            arch_family="hybrid",
            arch_class="HybridLM",
            hypothesis="GLA + attention hybrid: gated linear attn for most layers, full attn for retrieval",
            source="hybrid",
            parent_exp="gla_baseline",
            expected_delta="better than pure GLA",
            priority=3,
            arch_config={
                "d_model": 512, "n_layers": 22, "d_ff": 2048,
                "pattern": "gla_attn",
                "n_heads": 8, "residual_scale": 1.0, "use_bias": True,
                "ssm_config": {},
            },
            tags=["hybrid", "gla", "attention"],
        ),
    ]

    for p in hybrids:
        if p.exp_id not in done:
            proposals.append(p)

    return proposals


def generate_next_batch(batch_size: int = 15) -> List[Dict[str, Any]]:
    """
    Generate the next batch of frontier experiments.

    Strategy:
    1. If Tier 1 baselines aren't done → prioritize those
    2. If baselines are done → mix exploitation + cross-pollination + novel
    3. Always include at least 2 hybrid experiments
    """
    knowledge = load_knowledge()
    queue = load_queue()
    done = get_done_experiments(queue)

    proposals = []

    # Phase 1: Baselines
    tier1 = propose_tier1_baselines(knowledge, done)
    tier2 = propose_tier2_experiments(knowledge, done)
    hybrids = propose_hybrids(knowledge, done)

    # If there are untested Tier 1 baselines, prioritize them
    if tier1:
        proposals.extend(tier1)
        # Fill remaining slots with Tier 2 and hybrids
        remaining = batch_size - len(proposals)
        proposals.extend(tier2[:remaining // 2])
        proposals.extend(hybrids[:remaining - remaining // 2])
    else:
        # All baselines done — shift to exploitation and cross-pollination
        proposals.extend(tier2)
        proposals.extend(hybrids)

    # Trim to batch size
    proposals = proposals[:batch_size]

    # Convert to queue entries
    entries = []
    for p in proposals:
        entries.append({
            "exp_id": p.exp_id,
            "arch_family": p.arch_family,
            "arch_class": p.arch_class,
            "hypothesis": p.hypothesis,
            "source": p.source,
            "parent_exp": p.parent_exp,
            "expected_delta": p.expected_delta,
            "priority": p.priority,
            "token_budget": 6000000,
            "arch_config": p.arch_config,
            "status": "pending",
            "added_by": "architect",
            "tags": p.tags,
        })

    return entries


def update_queue_with_proposals(entries: List[Dict[str, Any]]):
    """Add proposed experiments to the queue (skip duplicates)."""
    queue = load_queue()
    existing_ids = {e["exp_id"] for e in queue}

    added = 0
    for entry in entries:
        if entry["exp_id"] not in existing_ids:
            queue.append(entry)
            added += 1

    with open(QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)
        f.write("\n")

    return added


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate next batch of frontier experiments")
    parser.add_argument("--batch-size", type=int, default=15, help="Number of experiments to propose")
    parser.add_argument("--dry-run", action="store_true", help="Print proposals without adding to queue")
    args = parser.parse_args()

    entries = generate_next_batch(args.batch_size)

    if args.dry_run:
        print(f"\nProposed {len(entries)} experiments:\n")
        for e in entries:
            print(f"  [{e['priority']}] {e['exp_id']:30s} {e['arch_class']:20s} — {e['hypothesis'][:60]}")
        return

    added = update_queue_with_proposals(entries)
    print(f"\nAdded {added} new experiments to frontier queue ({len(entries)} proposed, {len(entries) - added} already existed)")


if __name__ == "__main__":
    main()
