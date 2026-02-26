#!/bin/bash
# Run all 6 ablation experiments (spatial + temporal)
# Usage: ./run_ablation.sh

set -e

cd "$(dirname "$0")"

PYTHON="/home/aquabot/miniforge3/envs/VISIONFT/bin/python"

echo "=== Ablation Experiments (6 configs) ==="
echo ""

echo "[1/6] Training: Spatial Force-only..."
$PYTHON train_v2.py --config configs/spatial_force_only.yaml

echo "[2/6] Training: Spatial Image-only..."
$PYTHON train_v2.py --config configs/spatial_image_only.yaml

echo "[3/6] Training: Spatial Combined..."
$PYTHON train_v2.py --config configs/spatial_combined.yaml

echo "[4/6] Training: Temporal Force-only..."
$PYTHON train_v2.py --config configs/temporal_force_only.yaml

echo "[5/6] Training: Temporal Image-only..."
$PYTHON train_v2.py --config configs/temporal_image_only.yaml

echo "[6/6] Training: Temporal Combined..."
$PYTHON train_v2.py --config configs/temporal_combined.yaml

echo ""
echo "=== All 6 ablation experiments complete ==="
