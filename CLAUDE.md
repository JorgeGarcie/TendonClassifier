# TendonClassifier

Classifies tendon type (none/single/crossed/double) beneath silicone phantoms using tactile images + force/torque data from a robot arm probe.

## Repository Map

```
TendonClassifier/
├── scripts/
│   ├── classification/          # Training, models, evaluation, configs
│   │   ├── config.py            # Config dataclasses + YAML loading
│   │   ├── models_v2.py         # SpatialModel, TemporalModel, force variants
│   │   ├── dataset.py           # TendonDatasetV2 (temporal, subtraction, Sparsh)
│   │   ├── encoders.py          # Vision encoder factory (ResNet/DinoV2/CLIP/Sparsh)
│   │   ├── attention.py         # CrossModalAttention, TemporalAttentionAggregator
│   │   ├── train_v2.py          # Main training loop + splitting
│   │   ├── train_sweep.py       # Wandb sweep launcher
│   │   ├── wandb_logger.py      # Wandb integration wrapper
│   │   ├── train_utils.py       # Device, checkpoint, plot utilities
│   │   ├── eval_test_set.py     # Evaluate on contiguous test split
│   │   ├── eval_generalization.py # Cross-phantom evaluation
│   │   ├── run_inference.py     # Single-image or batch inference
│   │   ├── inspect_*.py / analyze_*.py / dump_*.py  # Visualization tools
│   │   ├── configs/             # 15 YAML experiment configs
│   │   └── sparsh_vit.py        # Sparsh ViT-B (vendored from Meta)
│   ├── labeling/                # Data pipeline: raw recordings -> GT dataset
│   │   ├── config.py            # Pipeline constants (force threshold, crop, etc.)
│   │   ├── discover_and_index.py    # Stage 1: scan runs -> run_manifest.json
│   │   ├── extract_valid_windows.py # Stage 2: contact detection -> valid_frames.json
│   │   ├── gt_labeler.py            # Stage 3: bbox GT grids -> .npz
│   │   ├── generate_gt.py          # Stage 4: dataset -> gt_manifest.csv + images/
│   │   └── configs/             # phantom_configs.json, manifests
│   └── reconstruction/          # Legacy module (not active)
├── tests/                       # pytest: test_config.py, test_models.py
├── docs/                        # Design docs, exec plans, experiments
├── checkpoints/                 # Saved model weights
└── ARCHITECTURE.md              # Module diagram + data flow
```

## Key Concepts

### Four independent axes define every experiment

1. **Pipeline** — how many frames the model wrapper processes:
   - **Single-frame** (`model.type: "spatial"`): one image in, one prediction out
   - **Sequence** (`model.type: "temporal"`): T images in, aggregated into one prediction

2. **Modality** — which sensor inputs are used:
   - **Force-only**: 6-DoF force/torque MLP, no vision encoder (separate model class)
   - **Image-only**: vision encoder only, no force branch
   - **Combined**: vision encoder + force branch, merged via a fusion module

3. **Encoder** — the frozen vision backbone (only for image-only and combined):
   - **ResNet-18** (512-d) — works with both single-frame and sequence pipelines
   - **Sparsh ViT-B/14** (768-d, 3ch) — single-frame pipeline only
   - **Sparsh ViT-B/16** (768-d, 6ch) — two frames channel-stacked at encoder input; still single-frame pipeline (temporal context baked into encoder, not sequence aggregation)

4. **Fusion** — how image + force features are merged (only for combined):
   - `token_self_attention`, `cross_attention`, `concat` — see `attention.py`

**Naming note**: "spatial" = single-frame pipeline, "temporal" = sequence pipeline in configs/code. Sparsh 6ch is temporally-aware but uses `model.type: "spatial"` because its temporal context comes from channel concatenation at the encoder input, not from the sequence aggregation pipeline.

### Other key concepts

- **Frozen encoders**: Dataset too small (~6.5K train) to fine-tune any backbone
- **Frame-level split**: 80/10/10 contiguous per-run with purge gaps (not run-level)
- **Balanced sampling**: `WeightedRandomSampler` for class imbalance; never combine with class_weights
- **Config-driven**: All experiments via YAML configs in `scripts/classification/configs/`

## Conventions

- **Python env**: `/home/aquabot/miniforge3/envs/VISIONFT/bin/python`
- **Run training**: `cd scripts/classification && python train_v2.py --config configs/<name>.yaml`
- **Plots**: Always `--save DIR`, never `plt.show()` (SSH)
- **Primary metric**: Macro-F1 (handles class imbalance)
- **Checkpoints store config**: Each `.pth` embeds its full config. Load with `torch.load()`
- **Results via wandb**: Query wandb MCP, don't maintain result tables in docs

## Golden Principles

1. **One fact, one place** — constants, config values, shared logic have a single source of truth
2. **Centralize, don't duplicate** — extract repeated patterns into shared utilities
3. **Validate at boundaries** — assert shapes/types/ranges where data enters a module
4. **Fail loud, not silent** — log warnings, don't swallow errors
5. **Config drives experiments** — all hyperparameters in YAML configs, not hardcoded in source
6. **Frozen encoder, learned head** — never fine-tune backbone on this dataset size

## ExecPlans & Experiments

- **ExecPlans** (pipeline/implementation changes): `docs/exec-plans/active/` and `completed/`
- **Experiments** (hypotheses + results): `docs/experiments/active/` and `completed/`
- Cross-reference between them with relative markdown links
- **Doc convention**: Exec plans and experiment docs intentionally omit architecture descriptions to avoid duplicating CLAUDE.md. They link here for context. This keeps "one fact, one place" across the doc hierarchy.
- **Experiment lifecycle**: When an experiment grows in scope or spawns follow-up questions, create a child experiment doc that links back to the parent. Close the parent with conclusions and move to `completed/`.

### Active Work

| Name | ExecPlan | Experiment | Description |
|------|----------|------------|-------------|
| Temporal v2 | [Implementation Complete](docs/exec-plans/active/temporal-v2-architecture.md) | [Ready to run](docs/experiments/active/temporal-v2-fusion-ablation.md) | Causal conv force + Sparsh 6ch + 4 fusion variants. **Read before any temporal/fusion changes.** |

## Deeper Context

| Topic | File |
|-------|------|
| Module dependencies + data flow | `ARCHITECTURE.md` |
| Design rationale | `docs/design-docs/design.md` |
| Golden principles (expanded) | `docs/design-docs/golden-principles.md` |
| Lessons learned pipeline | `docs/design-docs/lessons-learned.md` |
| Tech debt tracker | `docs/exec-plans/tech-debt-tracker.md` |
| Module quality grades | `docs/quality-score.md` |
| How to run things | `docs/getting-started.md` |
| ExecPlan template | `docs/exec-plans/plans.md` |
| Experiment template | `docs/experiments/experiment-template.md` |
| Classification architecture | `scripts/classification/DESCRIPTION.md` |
| Labeling pipeline | `scripts/labeling/DESCRIPTION.md` |
| Historical notes + TODOs | `scripts/classification/BACKLOG.md` |
| Experiment log (sweeps, Sparsh) | `scripts/classification/EXPERIMENT_LOG.md` |
| Paper references (canonical) | `docs/references/ref.md` |
