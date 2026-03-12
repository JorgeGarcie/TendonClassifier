# Architecture

## Module Dependency Graph

```
                    ┌─────────────────────┐
                    │   YAML Configs       │
                    │  configs/*.yaml      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │     config.py         │
                    │  Config dataclasses   │
                    │  YAML → typed Config  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
    │   dataset.py   │  │ models_v2  │  │ train_v2.py │
    │ TendonDatasetV2│  │ Spatial/   │  │ Training    │
    │ temporal idx   │  │ Temporal   │  │ loop, split │
    │ subtraction    │  │ Model      │  │ metrics     │
    └────────────────┘  └─────┬──────┘  └──────┬──────┘
                              │                │
                 ┌────────────┼────────┐       │
                 │            │        │       │
          ┌──────▼───┐ ┌─────▼────┐ ┌─▼───────▼──────┐
          │encoders  │ │attention │ │ train_utils.py  │
          │ResNet18  │ │CrossModal│ │ device, ckpt    │
          │DinoV2    │ │Temporal  │ │ plot, save      │
          │CLIP      │ │TokenSelf │ └─────────────────┘
          │Sparsh    │ │Attention │
          └──────────┘ └──────────┘
```

## Data Flow

```
Raw Recordings (camera + F/T + TCP pose)
    │
    ├── [labeling/discover_and_index.py] ──→ run_manifest.json
    │
    ├── [labeling/extract_valid_windows.py] ──→ valid_frames.json
    │       (force thresholding, timestamp matching, F/T resampling)
    │
    ├── [labeling/gt_labeler.py] ──→ output/gt_grids/*.npz
    │       (STL coords → bbox label grids)
    │
    └── [labeling/generate_gt.py] ──→ gt_manifest.csv + images/
            (TCP → STL transform, label lookup, center crop 1080px)

gt_manifest.csv
    │
    └── [classification/dataset.py] TendonDatasetV2
            │   - Filter by include/exclude regex
            │   - Boundary exclusion (3mm for p4/p5)
            │   - Optional subtraction (simple or sparsh)
            │   - ImageNet normalization
            │   - Temporal frame indexing (num_frames window)
            │   - Force z-score normalization
            │
            └── [classification/train_v2.py]
                    │   - split_frame_contiguous() or split_by_run()
                    │   - WeightedRandomSampler
                    │   - Training loop (CE loss + optional depth MSE)
                    │   - Cosine scheduler + warmup
                    │   - Wandb logging
                    │
                    └── checkpoints/{name}/best.pth
                            │
                            └── [eval_*.py] scripts
                                    - eval_test_set.py (test split)
                                    - eval_generalization.py (cross-phantom)
                                    - run_inference.py (deployment)
```

## Model Architecture (Spatial Combined)

```
Image (B, 3, 224, 224)
    │
    └── ResNet-18 (frozen) ──→ (B, 512)
                                   │
                                   ├── CrossModalAttention ──→ (B, 128)
                                   │       ↑                       │
Force (B, 6)                       │   Force queries,              │
    │                              │   Image keys/values           │
    └── ForceBranch MLP ──→ (B, 64)                               │
            6 → 64 → 128 → 64                                     │
                                                            ClassificationHead
                                                            128 → 64 → 4
                                                                   │
                                                            logits (B, 4)
```

## Model Architecture (Temporal Combined)

```
Images (B, T, 3, 224, 224)
    │
    └── [ResNet-18 per frame] ──→ (B, T, 512)
                                       │
                                       ├── Per-frame CrossModalAttention ──→ (B, T, 128)
                                       │       ↑
Force (B, 6)*                          │   [BUG: receives single frame,
    │                                  │    not (B, T, 6) sequence]
    └── ForceBranch MLP ──→ (B, 64)
                                                    │
                                        ┌───────────▼───────────┐
                                        │ TemporalAttention     │
                                        │ Aggregator            │
                                        │ - Learnable agg token │
                                        │ - Causal mask         │
                                        │ - Positional encoding │
                                        └───────────┬───────────┘
                                                    │
                                             (B, 128)  [agg token output]
                                                    │
                                             ClassificationHead
                                                    │
                                             logits (B, 4)
```

## Dependency Rules

- `config.py` imports nothing from the project (leaf dependency)
- `encoders.py` and `attention.py` import only PyTorch (no project imports)
- `models_v2.py` imports from `encoders.py` and `attention.py`
- `dataset.py` imports from `config.py` only
- `train_v2.py` orchestrates everything — imports config, dataset, models, utils, wandb
- Labeling scripts are fully independent of classification (shared only via `gt_manifest.csv`)
- `sparsh_vit.py` is vendored (no external dependency beyond torch + safetensors)

## Integration Points

| Boundary | Format | Producer | Consumer |
|----------|--------|----------|----------|
| Raw data → Pipeline | CSV + JPG files | ROS bag recordings | labeling scripts |
| Pipeline → Dataset | `gt_manifest.csv` | `generate_gt.py` | `TendonDatasetV2` |
| Config → All | YAML → Config dataclass | Human / sweep | train, eval, models |
| Train → Eval | `.pth` checkpoint | `train_v2.py` | `eval_*.py`, `run_inference.py` |
| Train → Tracking | API calls | `wandb_logger.py` | Wandb cloud |
