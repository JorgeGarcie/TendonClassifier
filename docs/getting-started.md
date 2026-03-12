# Getting Started

## Environment Setup

```bash
# Conda environment
conda activate VISIONFT
# Or use the full path:
/home/aquabot/miniforge3/envs/VISIONFT/bin/python

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

Two main modules:
- `scripts/labeling/` — Data pipeline (raw recordings → GT dataset)
- `scripts/classification/` — Models, training, evaluation

## Labeling Pipeline (Run Once)

```bash
cd scripts/labeling

# Stage 1: Discover and index runs
python discover_and_index.py

# Stage 2: Extract contact windows
python extract_valid_windows.py

# Stage 3: Generate GT grids (per phantom)
python gt_labeler.py

# Stage 4: Generate dataset
python generate_gt.py
```

Output: `scripts/labeling/output/gt_dataset/gt_manifest.csv` + `images/`

## Training

```bash
cd scripts/classification

# Train a single model
python train_v2.py --config configs/spatial_combined.yaml

# Run all 6 ablation configs
./run_ablation.sh

# Override parameters from CLI
python train_v2.py --config configs/spatial_combined.yaml \
    --override training.epochs=50 training.lr=0.0005

# Disable wandb (offline mode)
WANDB_MODE=disabled python train_v2.py --config configs/spatial_combined.yaml
```

## Evaluation

```bash
cd scripts/classification

# Evaluate on held-out test split
python eval_test_set.py --config configs/spatial_combined.yaml

# Cross-phantom generalization
python eval_generalization.py \
    --checkpoint checkpoints/spatial_combined/best.pth \
    --phantoms p1

# Batch inference
python run_inference.py --frames test_frames/ --model spatial_combined
```

## Hyperparameter Sweep

```bash
cd scripts/classification

# Create sweep
wandb sweep configs/sweep_spatial.yaml

# Launch agent
wandb agent <sweep_id>
```

## Tests

```bash
python -m pytest tests/ -v
```

## Key Files

| File | What it does |
|------|-------------|
| `scripts/classification/config.py` | Config dataclasses + YAML loading |
| `scripts/classification/configs/default.yaml` | Full config schema with all options |
| `scripts/classification/DESCRIPTION.md` | Detailed architecture documentation |
| `scripts/classification/BACKLOG.md` | Historical notes, known bugs, TODOs |
| `scripts/classification/EXPERIMENT_LOG.md` | Sweep results, Sparsh experiment details |

## GPU

RTX 3090 (25.4 GB). Spatial models use ~2-4 GB, temporal models use ~6-8 GB at batch_size=16.
