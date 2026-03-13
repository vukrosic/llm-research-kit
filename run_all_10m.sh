#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Run ALL pending 6M token experiments with crash resilience
# - Skips experiments that already have metrics.json
# - On crash: saves CRASH.log, cleans GPU memory, continues to next
# - Logs everything to logs/run_all_6m.log
# ═══════════════════════════════════════════════════════════════════════════

set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

TOKENS=6000000
OUTPUT_DIR="./ablation_results"
LOG_DIR="./logs"
MAIN_LOG="${LOG_DIR}/run_all_10m.log"

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════" | tee -a "$MAIN_LOG"
echo "  STARTING FULL 10M SWEEP — $(date)" | tee -a "$MAIN_LOG"
echo "═══════════════════════════════════════════════════════" | tee -a "$MAIN_LOG"

# Read pending experiments
EXPERIMENTS=$(python3 -c "
import os
from configs.ablation_configs import ABLATION_CONFIGS
results_dir = '${OUTPUT_DIR}/${TOKENS}tok'
existing = set()
if os.path.isdir(results_dir):
    for d in os.listdir(results_dir):
        mf = os.path.join(results_dir, d, 'metrics.json')
        if os.path.exists(mf):
            existing.add(d)
to_run = [n for n in sorted(ABLATION_CONFIGS.keys()) if n not in existing]
print(' '.join(to_run))
")

TOTAL=$(echo $EXPERIMENTS | wc -w)
echo "Experiments to run: $TOTAL" | tee -a "$MAIN_LOG"

COMPLETED=0
CRASHED=0
SKIPPED=0

for EXP in $EXPERIMENTS; do
    COMPLETED=$((COMPLETED + 1))
    METRICS_PATH="${OUTPUT_DIR}/${TOKENS}tok/${EXP}/metrics.json"

    # Double-check skip (in case a prior iteration in THIS run already created it)
    if [ -f "$METRICS_PATH" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$COMPLETED/$TOTAL] SKIP $EXP (already done)" | tee -a "$MAIN_LOG"
        continue
    fi

    echo "" | tee -a "$MAIN_LOG"
    echo "[$COMPLETED/$TOTAL] RUNNING: $EXP  ($(date '+%H:%M:%S'))" | tee -a "$MAIN_LOG"

    # Run single experiment in a subprocess for isolation
    EXP_LOG="${LOG_DIR}/exp_${EXP}.log"
    timeout 1800 python3 run_ablations.py \
        --tokens $TOKENS \
        --experiments "$EXP" \
        --output_dir "$OUTPUT_DIR" \
        --compile \
        > "$EXP_LOG" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] && [ -f "$METRICS_PATH" ]; then
        # Extract val_loss from metrics
        VAL_LOSS=$(python3 -c "import json; d=json.load(open('$METRICS_PATH')); print(f\"{d['final_metrics']['val_loss']:.4f}\")" 2>/dev/null || echo "???")
        echo "  ✅ $EXP done — val_loss=$VAL_LOSS" | tee -a "$MAIN_LOG"
    else
        CRASHED=$((CRASHED + 1))
        echo "  ❌ $EXP CRASHED (exit=$EXIT_CODE) — see $EXP_LOG" | tee -a "$MAIN_LOG"

        # Save crash info
        CRASH_DIR="${OUTPUT_DIR}/${TOKENS}tok/${EXP}"
        mkdir -p "$CRASH_DIR"
        tail -50 "$EXP_LOG" > "${CRASH_DIR}/CRASH.log" 2>/dev/null

        # If timeout killed it
        if [ $EXIT_CODE -eq 124 ]; then
            echo "  ⏰ TIMEOUT after 30 minutes" | tee -a "$MAIN_LOG"
            echo "TIMEOUT after 30 minutes" >> "${CRASH_DIR}/CRASH.log"
        fi
    fi

    # Aggressive GPU cleanup between experiments
    python3 -c "
import torch, gc, torch._dynamo
try: torch._dynamo.reset()
except: pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
" 2>/dev/null

    sleep 2
done

echo "" | tee -a "$MAIN_LOG"
echo "═══════════════════════════════════════════════════════" | tee -a "$MAIN_LOG"
echo "  COMPLETE — $(date)" | tee -a "$MAIN_LOG"
echo "  Total: $TOTAL  Completed: $((COMPLETED - CRASHED - SKIPPED))  Crashed: $CRASHED  Skipped: $SKIPPED" | tee -a "$MAIN_LOG"
echo "═══════════════════════════════════════════════════════" | tee -a "$MAIN_LOG"

# Show crash summary if any
if [ $CRASHED -gt 0 ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "CRASHED EXPERIMENTS:" | tee -a "$MAIN_LOG"
    find "${OUTPUT_DIR}/${TOKENS}tok" -name "CRASH.log" -exec echo "  - {}" \; | tee -a "$MAIN_LOG"
fi

# Generate final comparison report
python3 -c "
import json, os
results_dir = '${OUTPUT_DIR}/${TOKENS}tok'
results = []
for d in sorted(os.listdir(results_dir)):
    mf = os.path.join(results_dir, d, 'metrics.json')
    if os.path.exists(mf):
        with open(mf) as f:
            results.append(json.load(f))
print(f'Total experiments with results: {len(results)}')
" 2>/dev/null | tee -a "$MAIN_LOG"
