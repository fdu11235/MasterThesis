#!/bin/bash
# Overnight orchestrator: wait for run_pipeline.py to finish, then run
# run_es_correction_net.py and run_real_pipeline.py in sequence.
# Each step's exit code is logged. Stops the chain on any failure.

set -u
cd /home/fdu/GitHub/MasterThesis

LOG=logs/_overnight_chain.log
echo "[CHAIN $(date -Is)] orchestrator started" >> "$LOG"

# Step 1: wait for run_pipeline.py to finish (poll every 30s)
while pgrep -f "python run_pipeline.py" > /dev/null; do
  sleep 30
done
echo "[CHAIN $(date -Is)] run_pipeline.py finished" >> "$LOG"

# Sanity: confirm the synthetic-eval log ended cleanly
if grep -qE "Traceback|Killed|Error" logs/run_pipeline_strict.log 2>/dev/null; then
  echo "[CHAIN $(date -Is)] run_pipeline.py log shows errors — stopping chain" >> "$LOG"
  exit 1
fi

# Step 2: ES correction network retrain
echo "[CHAIN $(date -Is)] starting run_es_correction_net.py" >> "$LOG"
python run_es_correction_net.py --config config/default.yaml \
    > logs/run_es_correction_strict.log 2>&1
EX1=$?
echo "[CHAIN $(date -Is)] run_es_correction_net.py exited $EX1" >> "$LOG"
if [ "$EX1" -ne 0 ]; then
  echo "[CHAIN $(date -Is)] correction net failed — stopping chain" >> "$LOG"
  exit 1
fi

# Step 3: Real pipeline
echo "[CHAIN $(date -Is)] starting run_real_pipeline.py" >> "$LOG"
python run_real_pipeline.py --config config/default.yaml \
    > logs/run_real_pipeline_strict.log 2>&1
EX2=$?
echo "[CHAIN $(date -Is)] run_real_pipeline.py exited $EX2" >> "$LOG"

echo "[CHAIN $(date -Is)] orchestrator complete (correction=$EX1 real=$EX2)" >> "$LOG"
