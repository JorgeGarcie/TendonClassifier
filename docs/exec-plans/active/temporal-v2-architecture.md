# ExecPlan: Temporal v2 Architecture

**Status**: Implementation Complete
**Created**: 2026-03-12
**Motivated by**: [Experiment: Temporal v2 Fusion Ablation](../../experiments/active/temporal-v2-fusion-ablation.md)

## Goal

Build a new temporal architecture using causal conv force encoding + Sparsh ViT-B/16 image encoding, with 4 fusion variants for ablation.

## Context

See [CLAUDE.md](../../../CLAUDE.md) "Key Concepts" for architecture context (encoders, pipelines, fusion axes).

The legacy temporal pipeline has two problems that motivate this work:
1. ResNet-18 is pretrained on ImageNet — poor domain match for tactile images
2. The 2-token fusion (1 image vector + 1 force vector) is degenerate — softmax over 2 tokens collapses to a trivial scalar weight

This plan replaces both branches with temporally-aware alternatives (causal conv force, Sparsh 6ch image) and tests 4 fusion variants. Temporal context lives inside each branch — no sequence aggregation pipeline needed.

## References

Full citations in [docs/references/ref.md](../../references/ref.md). Key papers for this plan:

- **Lee 2019** — causal conv force encoder design; concat+MLP fusion baseline
- **UMI-FT 2026** — self-attention over modality tokens with CoinFT; motivates token-level fusion
- **FiLM 2018** — per-channel affine modulation from conditioning signal
- **VTT 2022** — cross-modal attention between vision and tactile
- **MBT 2021** — bottleneck tokens; evidence that mid-fusion outperforms late fusion
- **Sparsh 2024** — frozen ViT-B tactile encoder; 6ch temporal channel-stacking

## Plan

### Step 1: Causal convolutional force encoder
- [x] Implement `CausalConvForceEncoder` in `models_v2.py`
- [x] 5-layer causal conv, stride 2, input (B, 32, 6) → output (B, 64)
- [x] Shape assertions verified in smoke test

### Step 2: Sparsh encoder — expose unpooled patch tokens
- [x] Modify encoder wrapper in `encoders.py` to optionally return (B, 196, 768) instead of (B, 768)
- [x] Needed for self-attention and cross-attention fusion variants
- [x] Mean-pooled path still works for concat and FiLM variants

### Step 3: Dataset changes — 32-step force window
- [x] Precompute script `scripts/labeling/precompute_force_windows.py` generates `force_windows.npy` (N, 32, 6)
- [x] `dataset.py` loads via mmap, indexes by `_orig_row` to survive filtering
- [x] Zero-padding for early frames handled in precompute

### Step 4: Implement 4 fusion modules
- [x] `SimpleFusion` (concat): reused existing, cat(pool(image), force) → MLP
- [x] `FiLMFusion`: force → (γ, β), apply to pooled image vector
- [x] `PatchSelfAttentionFusion`: 196 image patches + 1 force token → transformer encoder layer → readout
- [x] `PatchCrossAttentionFusion`: force queries, image patch K/V → attended force → head

### Step 5: New model class + configs
- [x] `TemporalV2Model` + `TemporalV2ForceModel` in `models_v2.py`
- [x] 5 new YAML configs: tv2_force_only, tv2_concat, tv2_film, tv2_self_attn, tv2_cross_attn (image-only reuses existing `sparsh_temporal_image_only.yaml`)
- [x] All 5 configs verified with 1-epoch smoke tests

### Step 6: Train and evaluate
- [ ] Run all 5 new configs + existing image-only (full 40 epochs)
- [ ] Log to wandb, record results in experiment doc

## Files to Change

| File | Change |
|------|--------|
| `scripts/classification/attention.py` | Add FiLMFusion, PatchSelfAttentionFusion, PatchCrossAttentionFusion + factory entries |
| `scripts/classification/encoders.py` | Add `pool` param to SparshEncoder for unpooled patch tokens |
| `scripts/classification/dataset.py` | Add `_orig_row` tracking, `force_window_path` loading via mmap |
| `scripts/classification/models_v2.py` | Add CausalConvForceEncoder, TemporalV2Model, TemporalV2ForceModel + factory entries |
| `scripts/classification/config.py` | Add temporal_v2/temporal_v2_force types, new fusion types, force_window_path |
| `scripts/classification/configs/` | 5 new YAML configs (tv2_*); image-only reuses sparsh_temporal_image_only.yaml |
| `scripts/labeling/precompute_force_windows.py` | New script: extracts 32-step 300Hz force windows aligned to manifest |
| `scripts/classification/train_v2.py` | Add temporal_v2/temporal_v2_force dispatch in run_epoch + collect_predictions |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-12 | Causal conv 32 steps at 300Hz | Matches Lee et al. design; ~107ms context |
| 2026-03-12 | Sparsh stride 3 (not 5) | 30Hz sensor; 3 frames ≈ 100ms ≈ 80ms slip reaction |
| 2026-03-12 | 4 fusion variants | Span complexity spectrum; eliminate bad options early with limited data |
| 2026-03-12 | Keep legacy temporal pipeline | Not removing, just building alongside |

## Surprises

- CausalConvForceEncoder placed in `models_v2.py` (not `attention.py`) — it's a model component, not a fusion/attention module
- Force windows precomputed offline (`precompute_force_windows.py`) rather than computed in dataset `__getitem__` — avoids loading raw wrench CSVs at training time, keeps dataset code simple, ~15 MB output
- Image-only config not duplicated — existing `sparsh_temporal_image_only.yaml` already uses Sparsh 6ch stride 3, so only 5 new configs needed (not 6)

## Progress

- [x] Step 1: Causal conv force encoder
- [x] Step 2: Sparsh unpooled patch tokens
- [x] Step 3: Dataset 32-step force window
- [x] Step 4: Fusion modules
- [x] Step 5: Model class + configs
- [ ] Step 6: Train and evaluate (full runs)
- [ ] Docs updated

## Outcomes

**Result**: (filled when completed)
**Lessons**: (what would you do differently?)
