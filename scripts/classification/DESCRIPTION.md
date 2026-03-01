# TendonClassifier v2 - Architecture & Usage

## Overview

A deep learning system for classifying tendon types from tactile images and force/torque sensor data.

- **Input**: Tactile images (224x224) + 6-axis force/torque data [fx, fy, fz, tx, ty, tz]
- **Output**: Classification of tendon type (0=none, 1=single, 2=crossed, 3=double)

## Architecture

### Experimental Design

The project follows a clean 2x3 experimental matrix:

```
                         MODALITY
               +------------+------------+------------+
               | force_only | image_only |  combined  |
    +----------+------------+------------+------------+
    | spatial  |     Y      |     Y      |     Y      |
TASK+----------+------------+------------+------------+
    | temporal |     Y      |     Y      |     Y      |
    +----------+------------+------------+------------+
```

**Task axis:**
- `spatial`: Single-frame classification
- `temporal`: Multi-frame, uses causal self-attention to aggregate context

**Modality axis:**
- `force_only`: Just force/torque data (ablation baseline)
- `image_only`: Just tactile images (ablation baseline)
- `combined`: Image + Force fusion (multimodal)

### Model Types

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `spatial` | Single-frame image +/- force | Default, image-based classification |
| `spatial_force` | Force-only MLP | Ablation: force contribution |
| `temporal` | Multi-frame sequence + force | Detect temporal patterns |
| `temporal_force` | Force sequence only | Ablation: temporal force patterns |

### Vision Encoders (Frozen)

| Encoder | Output Dim | Notes |
|---------|------------|-------|
| `resnet18` | 512 | Default, fast |
| `dinov2_small` | 384 | Self-supervised |
| `dinov2_base` | 768 | Larger DinoV2 |
| `clip_vit_b16` | 512 | CLIP visual encoder |

Encoders are always frozen (not fine-tuned). Dataset is too small to fine-tune (~200K trainable params vs 11M+ total).

### Fusion Methods

- `concat`: Simple concatenation + MLP
- `attention`: Cross-modal attention (force attends to image)
- `token_self_attention`: Modality tokens + self-attention (used in `traverse_combined.yaml`)

### Temporal Aggregation

Transformer-based causal self-attention with a learnable aggregation token. When a run has fewer frames than `num_frames`, early positions are padded with the first frame and masked.

## Current Dataset

All 6 trained models use **4 traverse runs only** (`include_run_regex: "traverse"` in every config), totaling **7,904 frames** across all 4 classes:

| Run | Phantom | Label | Frames |
|-----|---------|-------|--------|
| `none_traverse-2026-02-24` | none | 0 (none) | 2,938 |
| `p1_single_traverse-2026-02-24` | p1 | 1 (single) | 2,260 |
| `p4_crossed_traverse-2026-02-24` | p4 | 2 (crossed) | 2,234 |
| `p5_double_traverse-2026-02-24` | p5 | 3 (double) | 472 |

Split: **80/10/10** frame-level (`split_by: "frame"`, `val_ratio: 0.1`, `test_ratio: 0.1`). Subtraction is off. `WeightedRandomSampler` handles class imbalance; primary metric is **macro-F1**.

The earlier 32-run stratified split strategy is documented in `BACKLOG.md` but is not currently used.

## File Structure

```
scripts/classification/
+-- configs/                       # YAML configurations
|   +-- default.yaml              # Full schema with all options
|   +-- spatial_force_only.yaml   # Force MLP only
|   +-- spatial_image_only.yaml   # Image only
|   +-- spatial_combined.yaml     # Image + Force (multimodal)
|   +-- temporal_force_only.yaml  # Force sequence only
|   +-- temporal_image_only.yaml  # Image sequence
|   +-- temporal_combined.yaml    # Image sequence + Force
|   +-- traverse_combined.yaml    # Temporal combined with token_self_attention fusion
|   +-- sweep_spatial.yaml        # Wandb hyperparameter sweep
+-- config.py                     # Dataclasses + YAML loading
+-- encoders.py                   # Vision encoder factory (ResNet/DinoV2/CLIP)
+-- attention.py                  # Attention mechanisms (CrossModal, Temporal)
+-- models_v2.py                  # SpatialModel, TemporalModel, etc.
+-- dataset.py                    # TendonDatasetV2 with ImageNet norm, temporal
+-- wandb_logger.py               # Wandb integration wrapper
+-- train_v2.py                   # Main training script
+-- train_sweep.py                # Script for wandb sweeps
+-- eval_test_set.py              # Evaluate checkpoint on held-out test split
+-- eval_new_frames.py            # Evaluate checkpoint on new unseen frames
+-- run_inference.py              # Single-image inference from a checkpoint
+-- export_force_stats.py         # Export force statistics to JSON
+-- run_ablation.sh               # Run all ablation experiments
+-- analyze_failures.py           # Confusion matrix + per-class F1
+-- inspect_failures.py           # Failure image grids + sequence plots
+-- inspect_run_spatial.py        # Spatial scatter: GT vs predictions
+-- inspect_run_imagemap.py       # Thumbnails at spatial positions
+-- dump_run_frames.py            # Grid of all frames sorted by frame_idx
+-- train_utils.py                # Utilities (device, checkpoints, plots)
```

## Usage

### Train Single Model
```bash
cd scripts/classification
python train_v2.py --config configs/spatial_combined.yaml
```

### Run Ablation Experiments
```bash
./run_ablation.sh
```

### Hyperparameter Sweep
```bash
wandb sweep configs/sweep_spatial.yaml
wandb agent <sweep_id>
```

### Disable Wandb
```bash
WANDB_MODE=disabled python train_v2.py --config configs/spatial_combined.yaml
```

## Configuration Schema

```yaml
experiment:
  name: "experiment_name"
  project: "TendonClassifier"
  seed: 42

model:
  type: "spatial"  # spatial | spatial_force | temporal | temporal_force
  encoder:
    name: "resnet18"
    pretrained: true
    freeze: true
  temporal:
    num_frames: 5
    aggregation: "attention"
  fusion:
    type: "attention"
    hidden_dim: 128
  num_classes: 4
  use_force: true
  use_depth_head: false

data:
  manifest: "../labeling/output/gt_dataset/gt_manifest.csv"
  img_size: 224
  include_run_regex: "traverse"  # regex filter on run names (all current configs use this)
  normalization:
    type: "imagenet"
  subtraction:
    enabled: false
    reference: "first_frame"  # first_frame | pre_contact | /path/to/image.png
  augmentation:
    enabled: false

training:
  epochs: 40
  batch_size: 32
  lr: 0.0001
  optimizer: "adam"
  scheduler:
    type: "cosine"
    warmup_epochs: 5
  balanced_sampling: true     # WeightedRandomSampler (do NOT also enable class_weights)
  loss:
    class_weights: "balanced"  # auto-disabled when balanced_sampling=true
  val_ratio: 0.1
  split_by: "frame"
  split:
    test_ratio: 0.1

checkpoint:
  dir: "checkpoints"
  save_best: true    # best val macro-F1
  save_last: true
  save_every_n_epochs: 5

logging:
  wandb:
    enabled: true
  csv:
    enabled: true
```

## Key Notes

- **Checkpoints are the source of truth**: Each `.pth` file stores its full config, model state, optimizer state, and metrics. Load with `torch.load()` to inspect.
- **Experiment results live in wandb**: Don't maintain results tables in docs — query wandb instead.
- **ImageNet normalization**: Required for pretrained encoders. Applied after optional subtraction.
- **Balanced sampling**: Use `balanced_sampling: true` (WeightedRandomSampler) — do NOT also enable `class_weights: "balanced"`. Using both double-corrects the imbalance.
- **Reproducibility**: Fixed seed (default 42) controls split, weight init, and shuffling.
- **Boundary exclusion**: For p4 and p5, frames within 3mm of GT label transitions are excluded from training (hardcoded in `dataset.py`).

*Last updated: 2026-02-28 — restructured; historical data strategy and results moved to BACKLOG.md*
