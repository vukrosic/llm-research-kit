#!/usr/bin/env python3
"""
Experiment analytics for LLM architecture research.

Reads all metrics.json files and queue.json to compute:
- Per-generation hit rates and effect size distributions
- Exploration vs exploitation ROI
- Improvement velocity and saturation metrics
- Frontier density (plateau detection)
- Mechanism orthogonality (synergy vs interference)
- Learning curve analysis from training history
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("ablation_results/6000000tok")
QUEUE_PATH = Path("experiments/queue.json")
LEADERBOARD_PATH = Path("experiments/leaderboard.md")

# Known invalid experiments (from CLAUDE.md)
EXCLUDED = {
    "attn_pool_k4", "attn_pool_k8",  # artificial perplexity
    "gen11_x_hilo_025",               # broken eval
}
EXCLUDED_PREFIXES = ("g2_",)  # config dispatch bug


def load_all_experiments():
    """Load all metrics.json files, excluding known-bad experiments."""
    experiments = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name in EXCLUDED or any(name.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            # Only include completed runs with valid metrics
            if data.get("tokens_seen", 0) >= 5_000_000 and "final_metrics" in data:
                vl = data["final_metrics"].get("val_loss")
                if vl is None or (isinstance(vl, float) and (vl != vl)):  # NaN check
                    continue
                experiments[name] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return experiments


def load_queue():
    """Load queue.json and index by exp_id."""
    if not QUEUE_PATH.exists():
        return {}
    with open(QUEUE_PATH) as f:
        entries = json.load(f)
    return {e["exp_id"]: e for e in entries}


def _build_generation_map(queue):
    """Build exp_id -> generation mapping from queue.json parent_exp chains."""
    gen_map = {}

    # Map parent_exp to generation
    parent_to_gen = {
        "baseline": "gen0",
        "attn_qk_layernorm": "gen3",
        "combo_deepnorm_bilinear": "gen4",
        "opt_linear_residual_stack": "gen6",
        "g6_muon_lr_018": "gen7",
        "g7_use_bias": "gen8",
        "g9_gated_residual": "gen9",
        "g9_muon_warm_mom": "gen10",
        "muon_warm_row_rms": "gen11",
        "gen11_x_singlu": "gen12",
        "gen12_warm150": "gen13",
    }
    # Also map baselines that share the same parent
    parent_to_gen["g7_use_bias"] = "gen9"  # gen8 and gen9 share baseline

    for entry in queue.values():
        parent = entry.get("parent_exp", "")
        if parent in parent_to_gen:
            gen_map[entry["exp_id"]] = parent_to_gen[parent]

    return gen_map


_GEN_MAP_CACHE = None


def infer_generation(name, queue=None):
    """Infer generation from experiment name prefix or queue.json parent_exp."""
    global _GEN_MAP_CACHE

    # Named generation prefixes (most reliable)
    m = re.match(r"^(gen\d+|g\d+)", name)
    if m:
        prefix = m.group(1)
        if prefix.startswith("g") and not prefix.startswith("gen"):
            num = prefix[1:]
            return f"gen{num}"
        return prefix

    # Check queue-based mapping
    if _GEN_MAP_CACHE and name in _GEN_MAP_CACHE:
        return _GEN_MAP_CACHE[name]

    # Fallback: known prefix patterns
    prefix_map = [
        ("muon_warm_row", "gen10"), ("muon_warm_rms", "gen10"),
        ("muon_warm_cautious", "gen10"), ("muon_warm_ema", "gen10"),
        ("muon_warm_update", "gen10"), ("muon_warm_frob", "gen10"),
        ("muon_warm_sign", "gen10"), ("muon_warm_half", "gen10"),
        ("muon_warm_gated", "gen10"),
        ("ffn_elu", "gen10"), ("ffn_gauss", "gen10"), ("ffn_softplus", "gen10"),
        ("ffn_cubic", "gen10"), ("ffn_mish", "gen10"), ("ffn_star", "gen10"),
        ("ffn_sqr", "gen10"), ("ffn_sq_silu", "gen10"),
        ("gated_res", "gen10"), ("gated_cautious", "gen10"),
        ("gated_trust", "gen10"), ("gated_update", "gen10"),
        ("pos_q_rope", "gen10"),
        ("residual_scale", "gen10"),
        ("combo_", "gen3"),
        ("opt_", "gen4"),
        ("new_", "gen3"),
    ]
    for prefix, gen in prefix_map:
        if name.startswith(prefix):
            return gen

    return "gen0"


def get_generation_baseline(gen):
    """Map each generation to its baseline val_loss."""
    baselines = {
        "gen0":  5.0611,   # original baseline
        "gen3":  5.0306,   # attn_qk_layernorm
        "gen4":  4.9869,   # combo_deepnorm_bilinear
        "gen6":  4.9241,   # opt_linear_residual_stack
        "gen7":  4.9133,   # g6_muon_lr_018
        "gen8":  4.8948,   # g7_use_bias
        "gen9":  4.8948,   # g7_use_bias (same baseline as gen8)
        "gen10": 4.8488,   # g9_muon_warm_mom
        "gen11": 4.8109,   # muon_warm_row_rms
        "gen12": 4.8012,   # gen11_x_singlu
        "gen13": 4.7888,   # gen12_warm150
    }
    return baselines.get(gen)


def classify_result(val_loss, baseline_loss, noise=0.002):
    """Classify experiment as winner/neutral/loser."""
    delta = baseline_loss - val_loss
    if delta > noise:
        return "winner"
    elif delta < -noise:
        return "loser"
    else:
        return "neutral"


# ─── Report sections ───────────────────────────────────────────────


def report_generation_stats(experiments, queue):
    """Per-generation hit rates and effect distributions."""
    print("=" * 70)
    print("PER-GENERATION HIT RATES")
    print("=" * 70)

    by_gen = defaultdict(list)
    for name, data in experiments.items():
        gen = infer_generation(name)
        by_gen[gen].append((name, data))

    # Determine source from queue
    def get_source(name):
        if name in queue:
            return queue[name].get("source", "unknown")
        return "unknown"

    sorted_gens = sorted(by_gen.keys(), key=lambda g: int(re.search(r"\d+", g).group()) if re.search(r"\d+", g) else -1)

    totals = {"winner": 0, "neutral": 0, "loser": 0}

    for gen in sorted_gens:
        exps = by_gen[gen]
        baseline = get_generation_baseline(gen)
        if baseline is None:
            continue

        winners, neutrals, losers = [], [], []
        deltas = []
        explore_w, explore_total = 0, 0
        exploit_w, exploit_total = 0, 0

        for name, data in exps:
            vl = data["final_metrics"]["val_loss"]
            delta = baseline - vl
            deltas.append(delta)
            cat = classify_result(vl, baseline)

            if cat == "winner":
                winners.append((name, delta))
            elif cat == "neutral":
                neutrals.append((name, delta))
            else:
                losers.append((name, delta))

            source = get_source(name)
            if source == "exploration":
                explore_total += 1
                if cat == "winner":
                    explore_w += 1
            elif source == "exploitation":
                exploit_total += 1
                if cat == "winner":
                    exploit_w += 1

        n = len(exps)
        totals["winner"] += len(winners)
        totals["neutral"] += len(neutrals)
        totals["loser"] += len(losers)

        deltas.sort()
        median_d = deltas[len(deltas) // 2] if deltas else 0
        best_d = max(deltas) if deltas else 0
        worst_d = min(deltas) if deltas else 0

        print(f"\n{gen} ({n} experiments, baseline {baseline:.4f})")
        print(f"  Winners: {len(winners):3d} ({100*len(winners)/n:5.1f}%)  "
              f"Neutral: {len(neutrals):3d} ({100*len(neutrals)/n:5.1f}%)  "
              f"Losers: {len(losers):3d} ({100*len(losers)/n:5.1f}%)")
        print(f"  Deltas — best: {best_d:+.4f}  median: {median_d:+.4f}  worst: {worst_d:+.4f}")

        if explore_total > 0 or exploit_total > 0:
            exr = f"{explore_w}/{explore_total}" if explore_total else "n/a"
            ext = f"{exploit_w}/{exploit_total}" if exploit_total else "n/a"
            print(f"  Exploration win rate: {exr}   Exploitation win rate: {ext}")

        if winners:
            winners.sort(key=lambda x: -x[1])
            top = winners[:3]
            print(f"  Top winners: {', '.join(f'{n} ({d:+.4f})' for n, d in top)}")

    total_n = sum(totals.values())
    print(f"\n{'─'*70}")
    print(f"ALL GENS: {totals['winner']} winners ({100*totals['winner']/total_n:.1f}%), "
          f"{totals['neutral']} neutral ({100*totals['neutral']/total_n:.1f}%), "
          f"{totals['loser']} losers ({100*totals['loser']/total_n:.1f}%) "
          f"out of {total_n} total")


def report_exploration_roi(experiments, queue):
    """Compare exploration vs exploitation efficiency."""
    print("\n" + "=" * 70)
    print("EXPLORATION vs EXPLOITATION ROI")
    print("=" * 70)

    explore = {"total": 0, "winners": 0, "total_delta": 0.0, "best_delta": 0.0, "best_name": ""}
    exploit = {"total": 0, "winners": 0, "total_delta": 0.0, "best_delta": 0.0, "best_name": ""}

    for name, data in experiments.items():
        if name not in queue:
            continue
        source = queue[name].get("source", "unknown")
        if source not in ("exploration", "exploitation"):
            continue

        gen = infer_generation(name)
        baseline = get_generation_baseline(gen)
        if baseline is None:
            continue

        vl = data["final_metrics"]["val_loss"]
        delta = baseline - vl
        bucket = explore if source == "exploration" else exploit
        bucket["total"] += 1
        if delta > 0.002:
            bucket["winners"] += 1
        # Only count positive deltas for ROI (improvements)
        if delta > 0:
            bucket["total_delta"] += delta
        if delta > bucket["best_delta"]:
            bucket["best_delta"] = delta
            bucket["best_name"] = name

    for label, b in [("Exploration", explore), ("Exploitation", exploit)]:
        if b["total"] == 0:
            continue
        win_rate = 100 * b["winners"] / b["total"]
        avg_gain = b["total_delta"] / b["total"]
        print(f"\n  {label}:")
        print(f"    Experiments run:    {b['total']}")
        print(f"    Winners:            {b['winners']} ({win_rate:.1f}%)")
        print(f"    Avg positive gain:  {avg_gain:.4f} per experiment")
        print(f"    Best single result: {b['best_name']} ({b['best_delta']:+.4f})")

    if explore["total"] > 0 and exploit["total"] > 0:
        explore_roi = explore["total_delta"] / explore["total"]
        exploit_roi = exploit["total_delta"] / exploit["total"]
        ratio = explore_roi / exploit_roi if exploit_roi > 0 else float("inf")
        print(f"\n  ROI ratio (explore/exploit): {ratio:.2f}x")
        if ratio > 1.2:
            print("  → Exploration is MORE efficient. Consider increasing exploration ratio.")
        elif ratio < 0.8:
            print("  → Exploitation is MORE efficient. Current ratio is appropriate.")
        else:
            print("  → Roughly equal efficiency. Current 70/30 ratio is reasonable.")


def report_improvement_velocity(experiments):
    """Track improvement per experiment over time (saturation detection)."""
    print("\n" + "=" * 70)
    print("IMPROVEMENT VELOCITY (saturation detection)")
    print("=" * 70)

    # Leaderboard progression (hardcoded from leaderboard.md since it's the ground truth)
    records = [
        ("attn_qk_layernorm", 5.0306, "gen0"),
        ("new_deepnorm_07", 5.0066, "gen3"),
        ("combo_deepnorm_bilinear", 4.9869, "gen3"),
        ("opt_linear_combo", 4.9328, "gen4"),
        ("opt_linear_residual_stack", 4.9241, "gen4"),
        ("g6_muon_lr_018", 4.9133, "gen6"),
        ("g7_use_bias", 4.8948, "gen7"),
        ("g9_gated_residual", 4.8844, "gen9"),
        ("g9_muon_rms_norm", 4.8802, "gen9"),
        ("g9_muon_row_norm", 4.8613, "gen9"),
        ("g9_muon_warm_mom", 4.8488, "gen9"),
        ("muon_warm_row_rms", 4.8109, "gen10"),
        ("gen11_x_singlu", 4.8012, "gen11"),
        ("gen12_warm150", 4.7888, "gen12"),
    ]

    # Count experiments per generation
    by_gen = defaultdict(int)
    for name in experiments:
        gen = infer_generation(name)
        by_gen[gen] += 1

    print(f"\n  Record progression (cumulative Δ from original baseline 5.0611):\n")
    print(f"  {'Record':<30s} {'val_loss':>9s} {'Δ cumul':>9s} {'Δ step':>9s} {'Gen':>6s} {'Exps in gen':>12s}")
    print(f"  {'─'*30} {'─'*9} {'─'*9} {'─'*9} {'─'*6} {'─'*12}")

    prev_loss = 5.0611
    for name, vl, gen in records:
        cumul = 5.0611 - vl
        step = prev_loss - vl
        n_exps = by_gen.get(gen, "?")
        print(f"  {name:<30s} {vl:9.4f} {cumul:+9.4f} {step:+9.4f} {gen:>6s} {str(n_exps):>12s}")
        prev_loss = vl

    # Per-generation velocity
    print(f"\n  Per-generation improvement velocity:\n")
    sorted_gens = sorted(by_gen.keys(), key=lambda g: int(re.search(r"\d+", g).group()) if re.search(r"\d+", g) else -1)

    gen_improvements = {}
    for name, vl, gen in records:
        if gen not in gen_improvements:
            gen_improvements[gen] = 0
        gen_improvements[gen] += (get_generation_baseline(gen) or 5.0611) - vl if vl < (get_generation_baseline(gen) or 5.0611) else 0

    # Actually compute: best improvement found in each generation
    for gen in sorted_gens:
        baseline = get_generation_baseline(gen)
        if baseline is None:
            continue
        gen_exps = [(n, d) for n, d in experiments.items() if infer_generation(n) == gen]
        if not gen_exps:
            continue
        best_vl = min(d["final_metrics"]["val_loss"] for _, d in gen_exps)
        best_delta = baseline - best_vl
        n = len(gen_exps)
        velocity = best_delta / n if n > 0 else 0
        print(f"  {gen:>6s}: {n:3d} experiments → best Δ {best_delta:+.4f} → velocity {velocity:.5f} per experiment")


def report_frontier_density(experiments):
    """How many experiments cluster near the best — plateau detection."""
    print("\n" + "=" * 70)
    print("FRONTIER DENSITY (plateau detection)")
    print("=" * 70)

    all_losses = [(n, d["final_metrics"]["val_loss"]) for n, d in experiments.items()]
    all_losses.sort(key=lambda x: x[1])
    best_loss = all_losses[0][1]

    thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
    print(f"\n  Best val_loss: {best_loss:.4f} ({all_losses[0][0]})")
    print(f"  Total valid experiments: {len(all_losses)}\n")

    for t in thresholds:
        count = sum(1 for _, vl in all_losses if vl - best_loss <= t)
        pct = 100 * count / len(all_losses)
        print(f"  Within {t:.3f} of best: {count:3d} experiments ({pct:5.1f}%)")

    # Per-generation frontier density (within 0.01 of gen's best)
    print(f"\n  Per-generation density (within 0.01 of gen best):\n")
    by_gen = defaultdict(list)
    for name, data in experiments.items():
        gen = infer_generation(name)
        by_gen[gen].append(data["final_metrics"]["val_loss"])

    sorted_gens = sorted(by_gen.keys(), key=lambda g: int(re.search(r"\d+", g).group()) if re.search(r"\d+", g) else -1)
    for gen in sorted_gens:
        losses = by_gen[gen]
        if not losses:
            continue
        gen_best = min(losses)
        near = sum(1 for vl in losses if vl - gen_best <= 0.01)
        pct = 100 * near / len(losses)
        print(f"  {gen:>6s}: {near:3d}/{len(losses):3d} ({pct:5.1f}%) within 0.01 of gen best {gen_best:.4f}")

    # Trend: is density increasing? (sign of plateau)
    recent_gens = [g for g in sorted_gens if re.search(r"\d+", g) and int(re.search(r"\d+", g).group()) >= 10]
    if len(recent_gens) >= 2:
        densities = []
        for gen in recent_gens:
            losses = by_gen[gen]
            gen_best = min(losses)
            near = sum(1 for vl in losses if vl - gen_best <= 0.01)
            densities.append(100 * near / len(losses))
        if densities[-1] > densities[0] + 5:
            print(f"\n  ⚠ Frontier density is INCREASING ({densities[0]:.0f}% → {densities[-1]:.0f}%). "
                  f"Landscape is flattening — shift toward exploration.")
        elif densities[-1] < densities[0] - 5:
            print(f"\n  ✓ Frontier density is DECREASING ({densities[0]:.0f}% → {densities[-1]:.0f}%). "
                  f"Still finding differentiated results.")


def report_mechanism_orthogonality(experiments):
    """Check whether combining two mechanisms gives synergy, additivity, or interference."""
    print("\n" + "=" * 70)
    print("MECHANISM ORTHOGONALITY (synergy vs interference)")
    print("=" * 70)

    # Known individual and combined results from leaderboard/strategy
    combos = [
        {
            "name": "row_norm + rms_norm",
            "individual_a": ("g9_muon_row_norm", 0.0189),    # row_norm alone
            "individual_b": ("g9_muon_rms_norm", 0.0042),    # rms_norm alone
            "combined": ("muon_warm_row_rms", 0.0379),       # vs g9_muon_warm_mom baseline
            "note": "row+rms on warm_mom baseline",
        },
        {
            "name": "bilinear + deepnorm",
            "individual_a": ("new_bilinear", 0.0254),        # bilinear vs qk_layernorm
            "individual_b": ("new_deepnorm_07", 0.0240),     # deepnorm vs qk_layernorm
            "combined": ("combo_deepnorm_bilinear", 0.0437), # vs qk_layernorm
            "note": "on qk_layernorm baseline",
        },
        {
            "name": "linear_schedule + residual_scale",
            "individual_a": ("opt_linear_combo", 0.0541),    # linear schedule
            "individual_b": ("residual_scale=0.5", 0.0240),  # deepnorm
            "combined": ("opt_linear_residual_stack", 0.0628),
            "note": "on combo_deepnorm_bilinear baseline (cumulative)",
        },
    ]

    print(f"\n  {'Combination':<30s} {'A alone':>9s} {'B alone':>9s} {'Expected':>9s} {'Actual':>9s} {'Type':>12s}")
    print(f"  {'─'*30} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*12}")

    for c in combos:
        a = c["individual_a"][1]
        b = c["individual_b"][1]
        expected = a + b
        actual = c["combined"][1]
        ratio = actual / expected if expected > 0 else 0

        if ratio > 1.1:
            typ = "SYNERGY"
        elif ratio > 0.9:
            typ = "additive"
        elif ratio > 0.5:
            typ = "partial"
        else:
            typ = "INTERFERENCE"

        print(f"  {c['name']:<30s} {a:+9.4f} {b:+9.4f} {expected:+9.4f} {actual:+9.4f} {typ:>12s} ({ratio:.2f}x)")

    print(f"\n  Interpretation:")
    print(f"    >1.1x = synergy (mechanisms amplify each other)")
    print(f"    0.9-1.1x = additive (independent improvements stack)")
    print(f"    0.5-0.9x = partial interference (diminishing returns)")
    print(f"    <0.5x = interference (mechanisms conflict)")


def report_learning_curves(experiments):
    """Analyze training dynamics from history data."""
    print("\n" + "=" * 70)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 70)

    # Compare early vs late performance for recent experiments
    # This detects experiments that start strong but plateau, or start weak but catch up
    recent_gens = ["gen11", "gen12", "gen13"]
    recent = {n: d for n, d in experiments.items()
              if infer_generation(n) in recent_gens and "history" in d}

    if not recent:
        print("\n  No recent experiments with history data.")
        return

    print(f"\n  Analyzing {len(recent)} experiments from {', '.join(recent_gens)}")

    # For each experiment, compute rank at step ~100 vs rank at final step
    step_100_losses = {}
    final_losses = {}

    for name, data in recent.items():
        hist = data["history"]
        steps = hist.get("steps", [])
        vl = hist.get("val_losses", [])
        if len(steps) < 3 or len(vl) < 3:
            continue

        # Find val_loss closest to step 100
        for i, s in enumerate(steps):
            if s >= 100:
                step_100_losses[name] = vl[i]
                break

        final_losses[name] = data["final_metrics"]["val_loss"]

    if len(step_100_losses) < 5:
        print("\n  Not enough experiments with step-100 data.")
        return

    # Rank at step 100 vs rank at final
    names_100 = sorted(step_100_losses.keys(), key=lambda n: step_100_losses[n])
    names_final = sorted(final_losses.keys(), key=lambda n: final_losses[n])

    rank_100 = {n: i for i, n in enumerate(names_100)}
    rank_final = {n: i for i, n in enumerate(names_final)}

    # Find experiments whose rank changed most
    rank_changes = []
    for name in step_100_losses:
        if name in rank_final:
            change = rank_100[name] - rank_final[name]  # positive = improved rank
            rank_changes.append((name, rank_100[name], rank_final[name], change))

    rank_changes.sort(key=lambda x: -abs(x[3]))

    print(f"\n  Biggest rank changes (step 100 → final):")
    print(f"  {'Experiment':<35s} {'Rank@100':>9s} {'Rank@end':>9s} {'Change':>8s} {'VL@100':>9s} {'VL@end':>9s}")
    print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*8} {'─'*9} {'─'*9}")

    for name, r100, rfinal, change in rank_changes[:10]:
        direction = "↑" if change > 0 else "↓" if change < 0 else "="
        vl100 = step_100_losses[name]
        vlfinal = final_losses[name]
        print(f"  {name:<35s} {r100+1:>9d} {rfinal+1:>9d} {direction}{abs(change):>7d} {vl100:>9.4f} {vlfinal:>9.4f}")

    # Correlation between early and final performance
    common = [n for n in step_100_losses if n in rank_final]
    if len(common) >= 10:
        # Spearman rank correlation (manual, no scipy dependency)
        n = len(common)
        d_sq_sum = sum((rank_100[name] - rank_final[name]) ** 2 for name in common)
        spearman = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
        print(f"\n  Rank correlation (step 100 vs final): {spearman:.3f}")
        if spearman > 0.8:
            print("  → Strong correlation: early performance is predictive. "
                  "Could use early stopping to filter experiments faster.")
        elif spearman > 0.5:
            print("  → Moderate correlation: some experiments change trajectory mid-training. "
                  "Keep full runs but monitor for late bloomers.")
        else:
            print("  → Weak correlation: early performance is NOT predictive. "
                  "Must run full training to evaluate. Do NOT use early stopping to filter.")


def report_summary(experiments, queue):
    """High-level summary with actionable recommendations."""
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    total = len(experiments)
    best_loss = min(d["final_metrics"]["val_loss"] for d in experiments.values())
    best_name = min(experiments, key=lambda n: experiments[n]["final_metrics"]["val_loss"])

    # Count queue stats
    q_explore = sum(1 for e in queue.values() if e.get("source") == "exploration")
    q_exploit = sum(1 for e in queue.values() if e.get("source") == "exploitation")
    q_done = sum(1 for e in queue.values() if e.get("status") == "done")

    print(f"\n  Total valid experiments: {total}")
    print(f"  Best result: {best_name} ({best_loss:.4f})")
    print(f"  Total improvement: {5.0611 - best_loss:+.4f} ({100*(5.0611-best_loss)/5.0611:.2f}%)")
    print(f"  Queue: {q_exploit} exploitation, {q_explore} exploration ({100*q_explore/(q_explore+q_exploit):.0f}% explore)")

    # Compute recent win rate
    recent_exps = {n: d for n, d in experiments.items()
                   if infer_generation(n) in ("gen11", "gen12", "gen13")}
    if recent_exps:
        recent_winners = 0
        for n, d in recent_exps.items():
            gen = infer_generation(n)
            bl = get_generation_baseline(gen)
            if bl and classify_result(d["final_metrics"]["val_loss"], bl) == "winner":
                recent_winners += 1
        recent_rate = 100 * recent_winners / len(recent_exps)
        print(f"  Recent win rate (gen11-13): {recent_winners}/{len(recent_exps)} ({recent_rate:.0f}%)")

    # Recommendations
    print(f"\n  Actionable recommendations:")

    # Check if improvements are shrinking
    last_two_deltas = [0.0097, 0.0124]  # singlu, warm150
    avg_recent = sum(last_two_deltas) / len(last_two_deltas)
    if avg_recent < 0.015:
        print(f"  1. SATURATION WARNING: Last 2 records averaged Δ={avg_recent:.4f}.")
        print(f"     Consider: increase exploration ratio, try fundamentally different axes,")
        print(f"     or increase token budget for promising experiments.")

    # Check exploration ratio
    if q_explore + q_exploit > 0:
        explore_pct = 100 * q_explore / (q_explore + q_exploit)
        if explore_pct < 25:
            print(f"  2. Exploration at {explore_pct:.0f}% — below the 30% minimum. Add more exploration experiments.")
        elif explore_pct > 40:
            print(f"  2. Exploration at {explore_pct:.0f}% — above standard 30%. Appropriate if recent gains are stalling.")


def main():
    global _GEN_MAP_CACHE
    print("Loading experiments...")
    experiments = load_all_experiments()
    queue = load_queue()
    _GEN_MAP_CACHE = _build_generation_map(queue)
    print(f"Loaded {len(experiments)} valid experiments, {len(queue)} queue entries.\n")

    report_generation_stats(experiments, queue)
    report_exploration_roi(experiments, queue)
    report_improvement_velocity(experiments)
    report_frontier_density(experiments)
    report_mechanism_orthogonality(experiments)
    report_learning_curves(experiments)
    report_summary(experiments, queue)


if __name__ == "__main__":
    main()
