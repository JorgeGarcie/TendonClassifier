# References

Canonical citation list for papers referenced across exec plans and experiments. Use the **Key** column as shorthand in other docs (e.g. "Lee 2019").

| Key | Citation | Domain |
|-----|----------|--------|
| Lee 2019 | Lee et al., "Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks", ICRA 2019 (extended T-RO 2020) | Visuo-tactile fusion; causal conv force encoder (WaveNet-style); concat+MLP baseline |
| FiLM 2018 | Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018 | Per-channel affine modulation (γ·x + β) from a conditioning signal |
| UMI-FT 2026 | Choi et al., "In-the-Wild Compliant Manipulation with UMI-FT", 2026 | Self-attention over ~4 modality tokens including CoinFT force |
| VTT 2022 | Chen et al., "Visuo-Tactile Transformers for Manipulation", CoRL 2022 | Cross-modal attention between vision and tactile; spatial attention heatmaps |
| MBT 2021 | Nagrani et al., "Attention Bottlenecks for Multimodal Fusion", NeurIPS 2021 | Bottleneck tokens for cross-modal attention; mid-fusion outperforms late fusion |
| Sparsh 2024 | Higuera et al., "Sparsh: Self-supervised Touch Representations for Vision-Based Tactile Sensing", CoRL 2024 | Frozen ViT-B encoder pretrained on tactile data; 6ch temporal channel-stacking |
