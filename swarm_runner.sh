#!/bin/bash

# Swarm Runner v2: Executes ablation experiments in separate processes 
# with memory-fragmentation fixes and auto-fallback for compilation failures.

TOKENS=$1
SHIFT_ARGS="${@:2}"

if [ -z "$TOKENS" ]; then
  echo "Usage: ./swarm_runner.sh <tokens> [experiments...]"
  exit 1
fi

# Optimization: reduce CUDA fragmentation during JIT compilation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"

EXPERIMENTS=$SHIFT_ARGS

echo "🚀 Starting Swarm Runner v2 for $TOKENS tokens"
echo "🔬 Experiments: $EXPERIMENTS"

for EXP in $EXPERIMENTS; do
  echo ""
  echo "######################################################################"
  echo "  SWARM EXECUTION: $EXP"
  echo "######################################################################"
  
  # Try first with compilation
  python run_ablations.py --tokens "$TOKENS" --experiments "$EXP" --compile
  EXIT_CODE=$?

  if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ [COMPILE FAILED] Experiment $EXP failed with code $EXIT_CODE (likely OOM in torch.compile)."
    echo "🔄 RETRYING $EXP WITHOUT COMPILATION..."
    
    # Retry without compile to at least get the data
    python run_ablations.py --tokens "$TOKENS" --experiments "$EXP"
    RETRY_CODE=$?
    
    if [ $RETRY_CODE -ne 0 ]; then
      echo "❌ [ERROR] Experiment $EXP failed even without compilation ($RETRY_CODE). Moving on..."
    else
      echo "✅ [SUCCESS] Experiment $EXP completed (no-compile fallback)."
    fi
  else
    echo "✅ [SUCCESS] Experiment $EXP completed with torch.compile."
  fi
  
  # Ensure clean slate for next process
  sleep 3
done

echo ""
echo "🏁 Swarm Runner finished all attempted experiments."
