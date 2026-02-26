# TendonClassifier v2 - Project Description

## Overview

A deep learning system for classifying tendon types from tactile images and force/torque sensor data. Built for systematic ablation studies and hyperparameter optimization.

## Problem Statement

- **Input**: Tactile images (224x224) + 6-axis force/torque data [fx, fy, fz, tx, ty, tz]
- **Output**: Classification of tendon type (0=none, 1=single, 2=crossed, 3=double)
- **Dataset**: See "Data Strategy" section below for full breakdown

## Architecture

### Experimental Design

The project follows a clean 2×3 experimental matrix:

```
                         MODALITY
               ┌────────────┬────────────┬────────────┐
               │ force_only │ image_only │  combined  │
    ┌──────────┼────────────┼────────────┼────────────┤
    │ spatial  │     ✓      │     ✓      │     ✓      │
TASK├──────────┼────────────┼────────────┼────────────┤
    │ temporal │     ✓      │     ✓      │     ✓      │
    └──────────┴────────────┴────────────┴────────────┘
```

**Task axis:**
- `spatial`: Single-frame classification
- `temporal`: Multi-frame, uses temporal attention to aggregate context

**Modality axis:**
- `force_only`: Just force/torque data (ablation baseline)
- `image_only`: Just tactile images (ablation baseline)
- `combined`: Image + Force fusion (multimodal)

**Additional options (for image_only & combined):**
- **Encoder**: Swappable backbone (ResNet18, DinoV2, etc.)
- **Fusion** (combined only): concat vs cross-modal attention

### Model Types

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `spatial` | Single-frame image ± force | Default, image-based classification |
| `spatial_force` | Force-only MLP | Ablation: force contribution |
| `temporal` | Multi-frame sequence + force | Detect temporal patterns |
| `temporal_force` | Force sequence only | Ablation: temporal force patterns |

### Vision Encoders (Frozen)

| Encoder | Output Dim | Notes |
|---------|------------|-------|
| `resnet18` | 512 | Default, fast |
| `dinov2_small` | 384 | Self-supervised, better features |
| `dinov2_base` | 768 | Larger DinoV2 |
| `clip_vit_b16` | 512 | CLIP visual encoder |

**Critical**: Encoders are frozen (not fine-tuned). Dataset is too small to fine-tune.

### Fusion Methods

- `concat`: Simple concatenation + MLP
- `attention`: Cross-modal attention (force attends to image)

### Temporal Aggregation

- `attention`: Transformer-based (default)

## File Structure

```
scripts/classification/
├── configs/                    # YAML configurations
│   ├── default.yaml           # Full schema with all options
│   │
│   │   # Spatial (single-frame)
│   ├── spatial_force_only.yaml   # Force MLP only, no encoder
│   ├── spatial_image_only.yaml   # Image only
│   ├── spatial_combined.yaml     # Image + Force (multimodal)
│   │
│   │   # Temporal (multi-frame)
│   ├── temporal_force_only.yaml  # Force sequence only
│   ├── temporal_image_only.yaml  # Image sequence
│   ├── temporal_combined.yaml    # Image sequence + Force
│   │
│   └── sweep_spatial.yaml     # Wandb hyperparameter sweep
├── config.py                  # Dataclasses + YAML loading
├── encoders.py                # Vision encoder factory (ResNet/DinoV2/CLIP)
├── attention.py               # Attention mechanisms (CrossModal, Temporal)
├── models_v2.py               # SpatialModel, TemporalModel, etc.
├── dataset.py                 # TendonDatasetV2 with ImageNet norm, temporal
├── wandb_logger.py            # Wandb integration wrapper
├── train_v2.py                # Main training script
├── train_sweep.py             # Script for wandb sweeps
├── run_ablation.sh            # Run all ablation experiments + sweep
├── analyze_failures.py        # Confusion matrix + per-class F1 across all experiments
├── inspect_failures.py        # Failure image grids + sequence plots per run
├── inspect_run_spatial.py     # Spatial (TCP x,y) scatter: GT vs predictions
├── dump_run_frames.py         # Grid of all frames sorted by frame_idx (GT + pred bars)
├── inspect_run_imagemap.py    # Thumbnails at spatial positions (matplotlib OffsetImage)
├── models.py                  # [Legacy] Original models
├── train.py                   # [Legacy] Original training
├── validate.py                # [Legacy] Validation script
└── train_utils.py             # Utilities (device, checkpoints, plots)
```

## Key Features

### 1. ImageNet Normalization
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```
Required for pretrained encoders. Previous code used simple 0-1 scaling.

### 2. Data Strategy

#### Phantom inventory

| Phantom | Runs | Frames | Classes present | Role in Phase 1 |
|---------|------|--------|-----------------|-----------------|
| **none** (forced-label) | 1 | 2,938 | none only | Core: "nothing here" baseline, full 0–180° rotation |
| **p1** (5 original + 1 forced-label) | 6 | 3,509 | none + single | Core: none↔single boundary + full-rotation single |
| **p2** | 6 | 1,794 | single only | Redundant for Phase 1 — all single, no class boundaries |
| **p3** | 5 | 1,461 | single only | Redundant for Phase 1 — all single, no class boundaries |
| **p4** (5 original + 1 forced-label) | 6 | 3,608 | single + crossed | Core: single↔crossed boundary + full-rotation crossed |
| **p5** (7 original + 1 forced-label) | 8 | 2,580 | single + double | Core: single↔double boundary + full-rotation double |

**Total**: 32 runs (after excluding 10 nat runs), ~15,890 frames.

#### Data pruning strategy

Phase 1 classifies: none / single / crossed / double. To learn these boundaries, the model needs:
- **none vs single** → p1 (has both) + none_traverse
- **single vs crossed** → p4 (has both) + p4_crossed_traverse
- **single vs double** → p5 (has both) + p5_double_traverse
- **Diverse "single" examples** → p1_single_traverse (2,260 frames, full 0–180° rotation)

p2 and p3 contribute **only single-class frames** from phantoms with depth/width variation. This variation is invisible in single-frame classification (it requires temporal along-tendon scans — Phase 2). For Phase 1, p2/p3 just inflate the "single" class without helping distinguish single from crossed/double/none.

**Strategy**: Keep 1 train + 1 val run per phantom for p2/p3 (generalization check), exclude the rest. This is controlled via `train_n_override` in the YAML configs.

#### Forced-label runs (the Feb 2026 additions)

4 pure-class recordings with **full 0–180° probe rotation** (the robot swept orientations continuously):

| Run | Phantom | Label | Frames | Notes |
|-----|---------|-------|--------|-------|
| `none_traverse-2026-02-24` | none | 0 | 2,938 | Pure silicone, no tendon |
| `p1_single_traverse-2026-02-24` | p1 | 1 | 2,260 | Single tendon, cleaned frames |
| `p4_crossed_traverse-2026-02-24` | p4 | 2 | 2,234 | Crossed tendons, cleaned frames |
| `p5_double_traverse-2026-02-24` | p5 | 3 | 472 | Double tendons, heavily cleaned |

These runs use **hard-coded GT labels** (`forced_label` in `run_manifest.json`). No bounding-box lookup needed — every frame is the stated class. Wrench data is interpolated by timestamp. Generated by `generate_gt.py` via `process_run_forced()`.

Lab assistant manually cleaned frames (removed start/end transitions, partial-contact, and bad images). This is why p5_double has only 472 frames — most frames showed insufficient double-tendon visibility.

#### Train/Val split

Controlled by `training.split` in each YAML config. Three mechanisms:

1. **`frame_split_phantoms: ["none"]`** — The none_traverse run (2,938 frames) is split 50/50 by frame between train and val. All frames are absence-of-tendon on pure silicone, so they're nearly identical — random frame-level split is fine.

2. **`val_n_override`** — How many runs go to val per phantom. Multi-class runs are selected first (diversity sort).

3. **`train_n_override: {"p2": 1, "p3": 1}`** — Limits training runs per phantom. Excess runs are excluded entirely. Keeps the dataset lean and focused on diverse, discriminative data.

**Current config (all 6 YAMLs):**
```yaml
split:
  frame_split_phantoms: ["none"]
  val_n_override: { p2: 1, p3: 1, p4: 2, p5: 2 }
  train_n_override: { p2: 1, p3: 1 }
```

#### Exact split assignment (seed=42)

**TRAIN (~10,443 frames before boundary exclusion)**

| Phantom | Run | Frames | Classes |
|---------|-----|--------|---------|
| **none** | `none_traverse` (50% frames) | 1,469 | none=1469 |
| **p1** | `p1_n2t2n_25N` | 302 | none=185, single=117 |
| | `p1_single_traverse` | 2,260 | single=2260 |
| | `p1_t2n_35N` | 100 | none=75, single=25 |
| | `p1_t_0_str` | 301 | single=301 |
| | `p1_t_90_str` | 308 | single=308 |
| **p2** | `p2_d2s_35N` | 300 | single=300 |
| **p3** | `p3_thicker2thinner_25N` | 293 | single=293 |
| **p4** | `p4_crossed_traverse` | 2,234 | crossed=2234 |
| | `p4_m2s_25N` | 291 | single=137, crossed=154 |
| | `p4_m2s_45N` | 310 | single=140, crossed=170 |
| | `p4_s2m_0_str` | 297 | single=167, crossed=130 |
| **p5** | `p5_double_traverse` | 472 | double=472 |
| | `p5_m2s_25N` | 291 | single=138, double=153 |
| | `p5_m2s_35N` | 305 | single=137, double=168 |
| | `p5_s2m_35N` (17.47) | 298 | single=157, double=141 |
| | `p5_s2m_35N` (17.50) | 308 | single=169, double=139 |
| | `p5_s2m_90_str` | 304 | single=178, double=126 |

**VAL (~3,370 frames before boundary exclusion)**

| Phantom | Run | Frames | Classes |
|---------|-----|--------|---------|
| **none** | `none_traverse` (50% frames) | 1,469 | none=1469 |
| **p1** | `p1_n2t_25N` | 238 | none=95, single=143 |
| **p2** | `p2_s2d_25N` | 290 | single=290 |
| **p3** | `p3_thicker2thinner_0_str` | 295 | single=295 |
| **p4** | `p4_m2s_35N` | 175 | single=9, crossed=166 |
| | `p4_s2m_90_str` | 301 | single=169, crossed=132 |
| **p5** | `p5_s2m_0_str` | 298 | single=167, double=131 |
| | `p5_s2m_0_str35` | 304 | single=178, double=126 |

**EXCLUDED (2,077 frames)** — 4 p2 runs + 3 p3 runs (all single-only, low diversity)

#### Known issues
- **p1_t2n_35N GT is inverted**: This run's GT labels are flipped (none↔single). Noted, deferred — only 100 frames.
- **p5_double_traverse low count**: Only 472 frames survived cleaning. Double class relies heavily on original p5 traversal runs.

### 3. Balanced Sampling + No Class-Weighted Loss
The dataset is heavily imbalanced (70.6% single). Two strategies exist to handle this:
- **Class-weighted loss**: Multiplies the CE loss for minority classes
- **`WeightedRandomSampler`** (`balanced_sampling: true`): Oversamples minority classes in each training batch so all classes appear at equal frequency

**We use only the sampler, not both.** Using both simultaneously double-corrects the imbalance, causing the model to drastically over-predict minority classes and collapse val accuracy. When `balanced_sampling: true`, class-weighted loss is automatically disabled.

**Boundary exclusion**: For p4 and p5, frames within 3mm of the single↔crossed/double transition (gy≈0) are excluded from training. At the boundary the probe sees both regions simultaneously, making GT labels ambiguous. This drops ~130 frames (~10-13 per run) and is hardcoded in `dataset.py`.

**Primary metric**: Use **macro-F1** (equal weight per class) rather than overall accuracy. A model trained with balanced sampling that achieves 25% macro-recall across all classes is better than a biased model that gets 70% accuracy by always predicting "single".

### 4. Metrics Logging
Logged to Wandb at end of training:
- Per-class precision, recall, F1
- Macro-averaged F1, precision, recall
- Confusion matrix visualization

### 5. Subtraction Trick
Subtract reference frame to highlight deformations:
- `first_frame`: First frame of each run
- `pre_contact`: Frame before contact (needs `contact_frame` column)
- `/path/to/image.png`: Global reference image
Order: subtraction is applied in raw pixel space **before** ImageNet normalization.

### 6. Checkpointing
- `best.pth`: Best validation **macro-F1** (not accuracy — macro-F1 is the primary metric)
- `last.pth`: Most recent epoch
- `epoch_XXX.pth`: Every N epochs (default: 5)

### 7. Wandb Integration
- Automatic logging of metrics, learning curves
- Confusion matrix visualization
- Hyperparameter sweep support (Bayesian optimization + Hyperband)

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
Runs: force-only → image-only → combined → hyperparameter sweep

### Hyperparameter Sweep (Manual)
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
    freeze: true  # CRITICAL for small dataset
  temporal:
    num_frames: 5
    aggregation: "attention"
  fusion:
    type: "attention"
    hidden_dim: 128
  num_classes: 4
  use_force: true
  use_depth_head: false  # Classification only, no depth regression

data:
  manifest: "../labeling/output/gt_dataset/gt_manifest.csv"
  img_size: 224
  normalization:
    type: "imagenet"
  subtraction:
    enabled: false
    reference: "first_frame"
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
  balanced_sampling: true    # WeightedRandomSampler (recommended)
  loss:
    class_weights: "balanced"  # Auto-DISABLED when balanced_sampling=true

checkpoint:
  save_best: true
  save_last: true
  save_every_n_epochs: 5

logging:
  wandb:
    enabled: true
  csv:
    enabled: true
```

## Sweep Parameters

Hyperparameters searched in `sweep_spatial.yaml`:
- `model_type`: [spatial, spatial_force]
- `encoder_name`: [resnet18, dinov2_small]
- `use_force`: [true, false]
- `fusion_type`: [concat, attention]
- `fusion_hidden_dim`: [64, 128, 256]
- `lr`: log-uniform [1e-5, 1e-2]
- `batch_size`: [16, 32, 64]
- `optimizer`: [adam, adamw]
- `weight_decay`: log-uniform [1e-5, 1e-2]

## Ablation Study

The ablation study compares modalities against each other to attribute performance to the right signal. Force-only serves as the baseline.

### Spatial (single-frame)

| Modality | Role | Expected |
|----------|------|----------|
| Force only | **Baseline** — how well can raw force/torque alone classify? | ~60% |
| Image only | Does a single tactile image capture tendon type? | ~99% |
| Combined | Does adding force help on top of image? | ~99% |

**Hypothesis**: For single frames, the tactile image already captures enough visual information. Force alone is weak, and combining doesn't add much.

### Temporal (multi-frame)

| Modality | Role | Expected |
|----------|------|----------|
| Force only | Baseline — temporal force patterns alone | TBD |
| Image only | Temporal image deformation alone | TBD |
| Combined | Full multimodal with temporal context | TBD |

**Hypothesis**: Temporal context is where combining force and image should outperform either modality alone, since how force changes over time together with image deformation provides richer information.

### Temporal Model Status
- **Causal attention**: Implemented — `TemporalAttentionAggregator` uses a causal mask so each frame attends only to itself and earlier frames.
- **BUG — Temporal combined uses single-frame force**: `TemporalModel.forward()` receives `force` as `(B, 6)` — only the current frame's force. The `return_force_sequence=True` flag is only set for `temporal_force` model type, NOT for `temporal` (combined). So the temporal combined model is really: **temporal images + single-frame force**. The force branch has no temporal context, which likely explains why temporal_combined (0.711) barely matches temporal_image_only (0.713) — force adds noise when it only sees one frame. **Fix needed**: wire `return_force_sequence=True` for temporal combined, add `TemporalForceBranch` to `TemporalModel`, and run both branches through the temporal aggregator before fusion.

## Dependencies

```
torch
torchvision
opencv-python
pandas
numpy
pyyaml
wandb
scikit-learn
matplotlib
```

Optional for DinoV2/CLIP:
```
# DinoV2: loaded via torch.hub (no install needed)
# CLIP: pip install git+https://github.com/openai/CLIP.git
```

## Notes

- **Frozen encoders**: Always freeze. Only ~200K trainable params vs 11M+ total. Dataset (~15K samples) is still too small to fine-tune.
- **Image preprocessing**: Currently using resize (may distort aspect ratio). Consider padding if edge regions are important.
- **Class imbalance**: Use `balanced_sampling: true` (WeightedRandomSampler) — do NOT also enable `class_weights: "balanced"`. Using both double-corrects the imbalance and causes the model to over-predict minority classes.
- **Reproducibility**: Fixed seed (default 42) controls train/val split, weight init, and shuffling. Set via `experiment.seed` in config.
- **Temporal padding & masking**: When a run has fewer frames than `num_frames`, early positions are padded with the first frame. A mask tensor marks which frames are real vs padded, and attention mechanisms ignore padded frames.

## System Vision: Two-Phase Tendon Mapping

This classifier is **Phase 1** of a larger tendon mapping pipeline. The two phases are:

### Phase 1 — Coarse Map Builder (this project)
A per-frame greedy classifier that scans the phantom surface and builds a spatial map of tendon presence and type. At each probe position it answers: "what is under here right now?" The output is a 2D map indexed by TCP (x, y) position.

- Input: single frame (image ± force) or short lookback window (temporal)
- Output: none / single / crossed / double at each grid point
- Role: spatial prior — where are tendons?

**Current limitation**: p2 and p3 (100% "single" in the dataset) add no discriminative value here. They contribute only to learning "something is present", not to distinguishing type. Their classification value is in Phase 2.

### Phase 2 — Tendon Characterizer (future work)
Once Phase 1 identifies where a tendon is, the robot executes a **purposeful straight-line scan directly along the tendon axis**. The resulting image sequence is temporally coherent (sweeping along the tendon from one end to the other). A transformer-based model then characterizes:

- **p2-type**: single tendon with gradually changing depth (tapered — shallow→deep)
- **p3-type**: single tendon with changing width/stiffness (thicker→thinner)
- Cross-tendon relationships, depth profile, texture variation along length

The key insight: p2 and p3 signals **only appear in the temporal context of a directed scan**. In a random frame they look identical to any other "single tendon" moment. The temporal transformer can attend over the full scan trajectory and classify the tendon *type* from its longitudinal profile.

**Two-phase pipeline**:
```
Phase 1: coarse scan → tendon map (where?)
              ↓
Phase 2: directed along-tendon scan → tendon characterization (what kind?)
```
This mirrors how a clinician works: quick overall palpation to find tendons, then targeted examination of each one.

## Future Work (Draft)

Inspired by TaF-Adapter (arxiv:2601.20321). These are architectural improvements to explore after the initial ablation study is complete.

### 1. VQ-VAE Force Quantization (Exploratory)
Discretize force signals into a learned codebook of K force primitives before feeding into the temporal model. The idea: instead of the model seeing raw (noisy) force vectors, it sees one of K learned prototypes — denoising by discretization.
```
force (B,T,6) → MLP encoder → continuous z → snap to nearest codebook entry → discrete c*
                                                    ↓
                                    use c* as force feature for fusion
```
The codebook is pre-trained with a VQ-VAE objective (reconstruction + commitment loss) and then frozen. Requires a separate pre-training stage. Likely overkill at current dataset scale (~4,690 samples) — revisit if dataset grows or force noise becomes a bottleneck.

---

## Ablation Results Summary

### Baseline — Original dataset only (28 runs, 7,986 frames, 1080px/40ep)

| Experiment | Val macro-F1 |
|---|---|
| spatial_force_only | 0.382 |
| spatial_image_only | 0.715 |
| spatial_combined | 0.724 |
| temporal_force_only | 0.396 |
| temporal_image_only | 0.793 |
| temporal_combined | **0.804** |

### Run 2 — Pruned dataset, frame-split none (32 runs, 13,813 frames used, 224px/40ep)

| Experiment | Val macro-F1 | none F1 | single F1 | crossed F1 | double F1 |
|---|---|---|---|---|---|
| spatial_force_only | 0.494 | 0.934 | 0.724 | 0.319 | 0.000 |
| spatial_image_only | 0.671 | 0.995 | 0.830 | 0.485 | 0.375 |
| spatial_combined | 0.678 | 0.998 | 0.847 | 0.484 | 0.382 |
| temporal_force_only | 0.394 | 0.878 | 0.551 | 0.000 | 0.148 |
| temporal_image_only | **0.713** | 0.999 | 0.833 | 0.620 | 0.402 |
| temporal_combined | 0.711 | 0.994 | 0.767 | 0.618 | 0.467 |

**Key findings:**
- **None is solved** — 99%+ F1 across all image models.
- **Double is the bottleneck** — best F1 is 0.467 (temporal_combined). Main failure: double→single confusion. Only 472 pure double training frames.
- **Crossed benefits from temporal** — jumped from 0.485 (spatial) to 0.620 (temporal image-only).
- **Force adds nothing to temporal** — temporal_combined ≈ temporal_image_only. Root cause: temporal combined model only sees single-frame force `(B, 6)`, not the force sequence. See "Temporal Model Status" bug above.
- **Val set composition changed** — 47% none (from frame split) makes macro-F1 not directly comparable to baseline. Per-class F1 is the fair comparison.

**Current deployment model**: `spatial_image_only` — image only, no force sensor needed at inference. Subtraction enabled (first_frame reference).

---

## Next Steps

### For the classification agent (this repo)

1. ~~Add data pruning config~~ **Done** — `split` block in all 6 YAMLs
2. ~~Run ablation~~ **Done** — see Run 2 results above
3. **Fix temporal combined force bug** — wire `return_force_sequence=True` for temporal combined, use `TemporalForceBranch` + temporal aggregation on force, then fuse both temporal streams. Re-run temporal_combined only.
4. **Improve double class** — collect more p5_double_traverse data, or try augmentation (rotation/flip) on existing double frames
5. **Export best model** — copy `best.pth` + config to inference directory

### For the ROS2 inference agent (Flexiv computer)

**Inference node requirements:**

1. **Model**: `spatial_image_only` — ResNet18 frozen encoder + classification head
2. **Input**: Single camera frame from `/camera/image_raw` (sensor_msgs/Image)
3. **Preprocessing pipeline** (must match training exactly):
   - Center crop to 1080×1080 pixels
   - Resize to 224×224
   - Subtraction: subtract the **first contact frame** (save frame when force > threshold, subtract from all subsequent frames). This is done in raw uint8 pixel space (clip to 0).
   - Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Convert to tensor (C, H, W) float32
4. **Output**: argmax of 4-class logits → 0=none, 1=single, 2=crossed, 3=double
5. **Files needed on Flexiv** (`~/tendon_classifier_inference/`):
   - `best.pth` — model checkpoint
   - `spatial_image_only.yaml` — config (for loading model architecture)
   - `config.py` — config dataclass definitions
   - `models_v2.py` — model definitions
   - `encoders.py` — encoder factory
   - `attention.py` — attention modules
6. **Force sensor**: NOT needed for `spatial_image_only`. The `/coinft/wrench` topic can be ignored.

**Loading the model in inference code:**
```python
import torch
from config import load_config
from models_v2 import build_model

config = load_config("spatial_image_only.yaml")
model = build_model(config)
ckpt = torch.load("best.pth", map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# inference
with torch.no_grad():
    logits = model(image_tensor)           # (1, 4)
    prediction = logits.argmax(dim=1).item()  # 0/1/2/3
```

*Last updated: 2026-02-25 — data strategy overhaul, forced-label runs, p2/p3 exclusion recommendation, inference spec*
