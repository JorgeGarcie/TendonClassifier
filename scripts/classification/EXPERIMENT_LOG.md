# Experiment Log — Sweep & Ablation (March 2026)

## Sweep Winner

**Run**: `true-sweep-2` (`xwrzqhgu`) from Bayesian sweep `csx5cmvk` (27 runs, project `TendonClassifier-scripts_classification`).

| Parameter | Value |
|-----------|-------|
| Fusion | token_self_attention (128-d, 4 heads, 1 layer) |
| LR / BS / WD | 4.36e-05 / 16 / 3.33e-04 |
| Optimizer | adam, cosine schedule, 5 warmup epochs |
| Subtraction | enabled (first_frame) |
| Encoder | ResNet-18 (frozen) |

All top sweep runs used subtraction. Non-subtraction runs were consistently worse.

## Results

### Traverse Test Set (793 frames, same distribution)

| Model | Test Acc | Test F1 | Val F1 | Best Epoch |
|-------|----------|---------|--------|------------|
| Force Only | 66.2% | 0.575 | 0.804 | 9 |
| Image Only | 100.0% | 1.000 | 0.997 | 33 |
| **Combined** | **100.0%** | **1.000** | **1.000** | **7** |

### Generalization — p1 Unseen Trajectory (299 single-tendon frames)

`p1_t_0_str` frames 930-1228, subtraction ref: `p1_t_0_nat frame_00594` (verified clean none).

| Model | Accuracy | Predictions (none/single/crossed/double) |
|-------|----------|------------------------------------------|
| Force Only | 14.4% | 0 / 43 / 256 / 0 |
| Image Only | 55.5% | 2 / 166 / 6 / 125 |
| **Combined** | **75.6%** | 1 / 226 / 6 / 66 |

The 66 "double" predictions cluster at high force (32-42N) where single-tendon deformation visually resembles double. Manual inspection confirms the ambiguity is genuine.

## Key Takeaway

On traverse data, image alone saturates at 100%. On unseen trajectories, image-only drops to 55.5% while combined holds at 75.6%. **Force provides trajectory-invariant signal that is critical for generalization.** Combined also converges 4x faster (epoch 7 vs 33).

## Evaluation Tools

- `eval_generalization.py` — evaluate on held-out phantoms with `--checkpoint`, `--phantoms`, `--subtraction-ref`
- `eval_test_frames.py` — evaluate on `test_frames/` folder (no force data, zero-force fallback)
- `configs/ablation_force_only.yaml`, `configs/ablation_image_only.yaml` — match sweep winner HPs

## Checkpoints

All embed full config. Load with `torch.load()`.

| Model | Checkpoint |
|-------|------------|
| Combined | `checkpoints/sweep/xwrzqhgu/best.pth` |
| Force Only | `checkpoints/ablation_force_only/best.pth` |
| Image Only | `checkpoints/ablation_image_only/best.pth` |

## Limitations

- p1-p5 non-traverse GT labels are unreliable — use manually verified frame ranges for eval
- High-force (>30N) single/double visual ambiguity needs more training data
- Generalization tested on one phantom (p1), one trajectory — broader eval needed

---

## Next: Tactile-Pretrained Backbone (Sparsh)

### Motivation

The current encoder (frozen ImageNet ResNet-18) was trained on natural images — dogs, cars, etc. Tactile images have fundamentally different visual structure (deformation patterns, contact geometry, controlled lighting). A backbone pretrained on tactile data should extract more relevant features.

**Sparsh** (Meta FAIR, 2024) is a family of SSL models trained on ~460K tactile images from DIGIT and GelSight sensors using DINO self-supervision. Key finding: DINO outperforms MAE for tactile images because tactile images have simpler visual structure, making pixel reconstruction less informative than latent-space self-distillation.

Paper: [arXiv 2410.24090](https://arxiv.org/abs/2410.24090)

### Key Definitions

- **Spatial**: Single frame → single prediction. No temporal context. This is what all our current trained models use.
- **Temporal**: Multiple frames → single prediction. Our existing temporal model uses causal self-attention over 5 frames with a learnable aggregation token.
- **Image subtraction**: Subtract a reference "no contact" frame from each image in raw pixel space before normalization. Removes static sensor artifacts (markers, lighting), isolates deformation signal. Both our pipeline and Sparsh use this.
- **Temporal tokenization (Sparsh)**: Concatenate two background-subtracted frames along channel dim: `cat(img_t, img_{t-5}) → 6 channels`. At 60Hz this is ~80ms — roughly human slip detection reaction time. The encoder sees *change*, not just a snapshot.

### Two Planned Experiments

#### Experiment A: Spatial — DINOv2 trained on tactile data (BACKLOG)

Train DINOv2 ourselves on Sparsh's open tactile datasets (~460K images) with **3-channel single-frame input**. This gives a fair spatial comparison:

| Encoder | Pretraining | Input | Mode |
|---------|------------|-------|------|
| ResNet-18 | ImageNet | 3ch | spatial (current) |
| DINOv2 | Tactile (ours) | 3ch | spatial (planned) |

Isolates the effect of tactile pretraining without changing mode. Requires ~1-2 days of GPU time on RTX 3090. **Deferred — needs server access.**

Datasets to download: YCB-Slide (~180K, DIGIT), Touch-and-Go (~220K, GelSight), ObjectFolder (~81K).

#### Experiment B: Temporal — Sparsh pretrained weights (ACTIVE)

Use Sparsh's pretrained ViT-B (6-channel, DINO) directly as a frozen temporal encoder. This follows exactly how Sparsh was trained: two background-subtracted frames concatenated.

| Encoder | Pretraining | Input | Mode |
|---------|------------|-------|------|
| Sparsh ViT-B | Tactile (Meta) | 6ch (t, t-5) | temporal |

Our addition on top: **force/torque fusion via token self-attention** — something Sparsh never had. They infer force from the image; we measure it directly with a 6-axis F/T sensor.

Architecture:
```
img_t (subtracted) ⊕ img_{t-5} (subtracted) → 6ch → Sparsh ViT (frozen, 768-d) → token ─┐
                                                                                            ├→ Self-Attention → classifier
Force (6-d) → MLP (6→64-d) ──────────────────────────────────────────────────────→ token ─┘
```

Sensor differences to note:
- Sparsh trained at 60Hz, our sensor runs at ~30Hz → temporal stride of 5 = ~166ms (vs Sparsh's 80ms)
- Sparsh trained on DIGIT/GelSight; ours is a reflective dome variant (similar to DenseTact)
- Domain gap is expected — the experiment tests whether low-level tactile features transfer

*Last updated: 2026-03-03*
