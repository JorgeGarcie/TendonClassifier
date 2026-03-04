#!/usr/bin/env bash
# Overnight training: baseline re-run + wandb sweep
# Usage: nohup ./run_overnight.sh > overnight.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/aquabot/miniforge3/envs/VISIONFT/bin/python"

echo "=== Phase 1: Baseline spatial_combined with token_self_attention ==="
echo "Started: $(date)"
$PYTHON train_v2.py --config configs/spatial_combined.yaml
echo "Phase 1 complete: $(date)"

echo ""
echo "=== Phase 2: Create wandb sweep ==="
SWEEP_OUTPUT=$(wandb sweep configs/sweep_spatial.yaml 2>&1)
echo "$SWEEP_OUTPUT"
# Extract sweep ID from output (format: "Created sweep with ID: <id>")
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \S+' | head -1 | awk '{print $NF}')
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID from output"
    exit 1
fi
echo "Sweep ID: $SWEEP_ID"

echo ""
echo "=== Phase 3: Run wandb agent ==="
echo "Started: $(date)"
wandb agent "$SWEEP_ID"
echo "Phase 3 complete: $(date)"
