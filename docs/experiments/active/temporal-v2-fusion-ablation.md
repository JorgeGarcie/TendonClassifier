# Experiment: Temporal v2 Fusion Ablation

**Status**: Ready to run
**Created**: 2026-03-12
**Depends on**: [ExecPlan: Temporal v2 Architecture](../../exec-plans/active/temporal-v2-architecture.md) (implementation complete)

## Hypothesis

Structured fusion mechanisms that preserve spatial information from Sparsh patch tokens (cross-attention, self-attention over 197 tokens) will outperform pooled-vector fusion (concat MLP, FiLM) on the combined image+force model, because force signals should selectively attend to deformed regions rather than treating the image as a single global descriptor.

## Caveat: Limited Data

Current dataset is ~6.5K frames from 4 traverse runs. This is not enough for a clean ablation — results may not generalize. The goal here is to **eliminate clearly bad options** early and identify which fusion mechanisms warrant further investigation once more data is collected.

## Architecture Context

See [CLAUDE.md](../../../CLAUDE.md) "Key Concepts" and the [exec plan](../../exec-plans/active/temporal-v2-architecture.md) for full architecture details. Fusion mechanism is the variable under test.

## Method

### Configs (4 fusion variants for the combined model + 2 unimodal baselines)

| Config | Fusion | Image repr | Force repr | Reference |
|--------|--------|------------|------------|-----------|
| `tv2_force_only` | N/A | N/A | Causal conv → 64-d | Lee et al. 2019 |
| `tv2_image_only` | N/A | Sparsh ViT-B/16 mean-pooled → 768-d | N/A | Sparsh paper |
| `tv2_concat` | Concat + 2-layer MLP | 768-d pooled | 64-d | Lee et al. 2019 |
| `tv2_film` | FiLM (force conditions image) | 768-d pooled | 64-d → γ, β | Perez et al. 2018 |
| `tv2_self_attn` | Self-attention, 197 tokens | 196 patch tokens × 768-d | 64-d → projected to 768-d | UMI-FT pattern |
| `tv2_cross_attn` | Cross-attention, 197 tokens | 196 patches as K/V | Force token as Q | VTT (Chen 2022) |

### Baseline

- Legacy spatial_combined (ResNet + token_self_attention): test macro-F1 = 0.921
- Legacy spatial_image_only (ResNet): test macro-F1 = 0.835

### Variables

- **Independent**: fusion mechanism (4 levels)
- **Dependent**: macro-F1 (primary), accuracy, per-class F1
- **Controlled**: encoder (Sparsh ViT-B/16 frozen), force encoder (causal conv, 32 steps), dataset, split, training hyperparams

## Results (append-only)

| Date | Run | Config | Macro-F1 | Accuracy | Notes |
|------|-----|--------|----------|----------|-------|

## Analysis

(pending)

## Conclusions

(pending)

## References

Full citations in [docs/references/ref.md](../../references/ref.md). Per-config mapping:

- **Lee 2019** — `tv2_force_only`, `tv2_concat` (causal conv encoder; concat+MLP baseline)
- **FiLM 2018** — `tv2_film` (per-channel affine modulation)
- **UMI-FT 2026** — `tv2_self_attn` (self-attention over modality tokens)
- **VTT 2022** — `tv2_cross_attn` (cross-modal attention, vision+tactile)
- **MBT 2021** — theoretical motivation (mid-fusion outperforms late fusion)
- **Sparsh 2024** — all image configs (frozen ViT-B tactile encoder, 6ch temporal)
