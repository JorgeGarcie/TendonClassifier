#!/usr/bin/env bash
# Run wandb sweep only (Phase 1 baseline already complete)
# Usage: nohup ./run_sweep.sh > overnight.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env for wandb CLI
export PATH="/home/aquabot/miniforge3/envs/VISIONFT/bin:$PATH"

echo "=== Create wandb sweep ==="
SWEEP_OUTPUT=$(wandb sweep configs/sweep_spatial.yaml 2>&1)
echo "$SWEEP_OUTPUT"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \S+' | head -1 | awk '{print $NF}')
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID from output"
    exit 1
fi
echo "Sweep ID: $SWEEP_ID"

echo ""
echo "=== Run wandb agent ==="
echo "Started: $(date)"
wandb agent "$SWEEP_ID"
echo "Complete: $(date)"
