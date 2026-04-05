#!/usr/bin/env bash
# Run all pipelines end-to-end. Intended for overnight execution.
#
# Usage:
#   nohup bash run_all_overnight.sh > logs/overnight.log 2>&1 &
#
# Or in a tmux/screen session:
#   bash run_all_overnight.sh 2>&1 | tee logs/overnight.log

set -e  # stop on first error
cd "$(dirname "$0")"

mkdir -p logs outputs/data

echo "=== $(date) === Starting overnight pipeline run ==="

# ── 1. Remove stale model checkpoints (force retrain with GPU) ────────
echo "--- Removing stale model checkpoints ---"
rm -f outputs/checkpoints/model_regression.pt
rm -f outputs/data/perturbed_diags_*.pkl
rm -f outputs/perturbation_results.pkl
rm -f outputs/data/real_datasets_loss.pkl outputs/data/real_datasets_profit.pkl
rm -f outputs/data/real_diagnostics_loss.pkl outputs/data/real_diagnostics_profit.pkl
rm -f outputs/real_results_loss.pkl outputs/real_results_profit.pkl

# ── 2. Synthetic pipeline (reuses cached diagnostics, trains CNN on GPU)
echo "=== $(date) === [1/3] Synthetic pipeline ==="
python run_pipeline.py --config config/default.yaml --n-jobs -1 \
    2>&1 | tee logs/pipeline_synthetic.log

# ── 3. Perturbation experiment (requires trained CNN from step 2) ──────
echo "=== $(date) === [2/3] Perturbation experiment ==="
python run_perturbation_experiment.py --config config/default.yaml --n-jobs -1 \
    2>&1 | tee logs/pipeline_perturbation.log

# ── 4. Real-data pipeline (independent, includes sign-split) ──────────
echo "=== $(date) === [3/3] Real-data pipeline ==="
python run_real_pipeline.py --config config/default.yaml --fresh --n-jobs -1 \
    2>&1 | tee logs/pipeline_real.log

echo "=== $(date) === All pipelines complete ==="
