# Design Decisions

## Why Frozen Encoders

Dataset is ~4,700 training samples. ResNet-18 alone has 11.2M parameters. Fine-tuning would overfit immediately. By freezing the encoder, the model learns only through the classification head (~200K params), which is appropriately sized for the dataset. All encoders (ResNet-18, DinoV2, CLIP, Sparsh) are always frozen.

## Why Frame-Level Split (Not Run-Level)

With only 4 traverse runs (one per class), run-level splitting would put entire classes in val/test. Frame-level contiguous splitting takes 80/10/10 from each run with purge gaps (temporal_frames - 1) to prevent temporal leakage. This ensures all classes appear in all splits while maintaining temporal integrity.

## Why WeightedRandomSampler (Not Class Weights)

Class distribution is heavily skewed (double: 472 frames vs none: 2,938). Two correction mechanisms exist:
- `WeightedRandomSampler`: oversamples minority classes per batch
- `class_weights` in CrossEntropyLoss: upweights minority class gradients

Using both **double-corrects** — the model over-predicts minority classes and val accuracy collapses. Decision: use only the sampler (`balanced_sampling: true`).

## Why Causal Temporal Attention

The temporal aggregator uses causal self-attention (each frame attends only to itself and earlier frames) with a learnable aggregation token that attends to all frames. This design:
- Prevents future information leakage (important for sequential prediction)
- The aggregation token acts as soft temporal pooling
- More expressive than mean pooling or LSTM for small sequences (T=5)

## Why Cross-Modal Fusion (Force Queries Image)

Force data is low-dimensional (6-d) and noisy. Image data is high-dimensional (512-d) and information-rich. Cross-modal attention lets force "ask questions" of the image representation, learning which visual features are relevant given the current force state. This outperforms simple concatenation on generalization tasks.

## Why Bbox GT (Not Raycasting)

The labeling pipeline uses bounding-box lookup in STL coordinates rather than full 3D raycasting. For the current phantom geometries (parallel tendons, rectangular bounds), bbox is sufficient and much simpler. Raycasting would be needed for complex curved geometries (Phase 2).

## Why Config-Driven Architecture

All experiment parameters live in YAML files. This enables:
- Reproducibility: checkpoint embeds its config
- Sweep integration: wandb overlays on base config
- Ablation: systematic parameter variation across 6+ configs
- No code changes needed for new experiments

## Why Macro-F1 (Not Accuracy)

With 4 classes and significant imbalance, accuracy is dominated by the majority class. Macro-F1 gives equal weight to each class, making it sensitive to minority class performance (especially double, which has only 472 frames).

## Why Subtraction Preprocessing

Subtracting a reference "no contact" frame removes static sensor artifacts (markers, lighting gradients) and isolates the deformation signal. The sweep confirmed: all top runs used subtraction. Two types:
- `simple`: `clip(img - ref, 0, 255)` in uint8 space
- `sparsh`: `clip((img - ref) + 0.5, 0, 1)` in float space (Meta's convention)

## Why Separate Labeling and Classification Modules

The labeling pipeline (raw recordings → GT dataset) and classification (GT dataset → trained model) are completely decoupled. They share only `gt_manifest.csv`. This means:
- Labeling can be re-run without touching classification code
- New labeling strategies don't require model changes
- The classification module can be tested with any CSV in the right format
