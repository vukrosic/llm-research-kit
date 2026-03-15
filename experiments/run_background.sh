#!/bin/bash
# Run the experiment queue in the background and ignore hangup signals
nohup python experiments/run_queue.py > experiments/queue_run.log 2>&1 &
echo "Experiment queue started in background with PID $!"
echo "Logging to experiments/queue_run.log"
