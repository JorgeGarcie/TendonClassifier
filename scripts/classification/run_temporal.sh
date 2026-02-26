#!/bin/bash
# Run temporal ablation experiments
# Usage: ./run_temporal.sh

set -e

cd "$(dirname "$0")"

echo "=== Temporal Ablation Experiments ==="
echo ""

# Run ablation experiments sequentially
echo "[1/3] Training: Force-only temporal model..."
python train_v2.py --config configs/temporal_force_only.yaml

echo "[2/3] Training: Image-only temporal model..."
python train_v2.py --config configs/temporal_image_only.yaml

echo "[3/3] Training: Combined temporal model..."
python train_v2.py --config configs/temporal_combined.yaml

echo ""
echo "=== Temporal ablation experiments complete ==="
