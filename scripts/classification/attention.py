"""Attention mechanisms for TendonClassifier v2.

Provides attention modules for:
- Cross-modal fusion (image + force)
- Temporal aggregation (multi-frame sequences)
- Self-attention (standard transformer attention)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T) or (B, T, T)

        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.proj(out)

        return out


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing image and force features.

    Uses the force features to attend over image features, producing
    a force-aware image representation.
    """

    def __init__(self, image_dim: int, force_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both modalities to same dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.force_proj = nn.Linear(force_dim, hidden_dim)

        # Cross-attention: force queries, image keys/values
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_feat: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_feat: Image features of shape (B, D_img) or (B, T, D_img)
            force_feat: Force features of shape (B, D_force) or (B, T, D_force)

        Returns:
            Fused features of shape (B, hidden_dim) or (B, T, hidden_dim)
        """
        temporal = image_feat.dim() == 3
        if temporal:
            B, T, _ = image_feat.shape
            image_feat = image_feat.reshape(B * T, -1)
            force_feat = force_feat.reshape(B * T, -1)

        N = image_feat.size(0)

        # Project to hidden dim
        img = self.image_proj(image_feat)  # (N, H)
        force = self.force_proj(force_feat)  # (N, H)

        # Reshape for attention (treat as single-token sequences)
        img = img.unsqueeze(1)  # (N, 1, H)
        force = force.unsqueeze(1)  # (N, 1, H)

        # Cross-attention: force attends to image
        q = self.q_proj(force)  # (N, 1, H)
        k = self.k_proj(img)
        v = self.v_proj(img)

        # Reshape for multi-head attention
        q = q.view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        attended = (attn @ v).transpose(1, 2).reshape(N, 1, self.hidden_dim)
        attended = self.out_proj(attended).squeeze(1)  # (N, H)

        # Residual + layer norm
        force_attended = self.layer_norm(force.squeeze(1) + attended)

        # Fuse attended force with original image features
        fused = self.fusion_mlp(torch.cat([img.squeeze(1), force_attended], dim=-1))

        if temporal:
            fused = fused.view(B, T, self.hidden_dim)

        return fused


class TemporalAttentionAggregator(nn.Module):
    """Temporal attention for aggregating multi-frame sequences.

    Uses self-attention over frame features to learn which frames
    are most informative for the task.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim

        # Positional encoding for temporal order
        self.pos_encoding = PositionalEncoding(feature_dim, max_len=100)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable aggregation token
        self.agg_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Final projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask with aggregation token.

        The agg token (position 0) can attend to all frames.
        Each frame can only attend to itself and earlier frames (causal).

        Returns:
            Float mask of shape (T+1, T+1) where -inf = blocked.
        """
        size = T + 1  # +1 for agg token
        # Start with all blocked
        mask = torch.full((size, size), float("-inf"), device=device)
        # Agg token (row 0) attends to everything
        mask[0, :] = 0.0
        # Frame tokens: causal (lower triangular for positions 1..T)
        for i in range(1, size):
            mask[i, 1:i+1] = 0.0
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Frame features of shape (B, T, D)
            mask: Optional mask of shape (B, T) indicating valid frames

        Returns:
            Aggregated features of shape (B, D)
        """
        B, T, D = x.shape

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepend aggregation token
        agg_tokens = self.agg_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([agg_tokens, x], dim=1)  # (B, T+1, D)

        # Build causal attention mask
        causal_mask = self._build_causal_mask(T, x.device)

        # Update padding mask if provided
        if mask is not None:
            # Add True for agg token
            agg_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([agg_mask, mask], dim=1)
            # Convert to attention mask format (True = ignore)
            mask = ~mask.bool()

        # Apply transformer with causal mask
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=mask)

        # Extract aggregation token output
        agg_out = x[:, 0, :]  # (B, D)

        return self.out_proj(agg_out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Output tensor of shape (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TokenSelfAttentionFusion(nn.Module):
    """Per-frame token self-attention fusion.

    Creates [force_token, image_token] with learned modality embeddings,
    runs self-attention, returns the fused force token.
    Works for both spatial (B, D) and temporal (B, T, D) inputs.
    """

    def __init__(self, image_dim: int, force_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.force_proj = nn.Linear(force_dim, hidden_dim)
        self.modality_embed = nn.Embedding(2, hidden_dim)  # 0=force, 1=image

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, image_feat: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_feat: Image features of shape (B, D_img) or (B, T, D_img)
            force_feat: Force features of shape (B, D_force) or (B, T, D_force)

        Returns:
            Fused features of shape (B, hidden_dim) or (B, T, hidden_dim)
        """
        # Modality indices
        force_idx = torch.zeros(1, dtype=torch.long, device=image_feat.device)
        image_idx = torch.ones(1, dtype=torch.long, device=image_feat.device)

        # Project + add modality embeddings
        img = self.image_proj(image_feat) + self.modality_embed(image_idx)
        frc = self.force_proj(force_feat) + self.modality_embed(force_idx)

        # Handle spatial (B, D) vs temporal (B, T, D)
        if img.dim() == 2:
            tokens = torch.stack([frc, img], dim=1)  # (B, 2, H)
            out = self.transformer(tokens)
            return out[:, 0, :]  # (B, H) — force token
        else:
            B, T, H = img.shape
            tokens = torch.stack([frc, img], dim=2).view(B * T, 2, H)
            out = self.transformer(tokens)  # (B*T, 2, H)
            return out[:, 0, :].view(B, T, H)  # (B, T, H) — force token


class SimpleFusion(nn.Module):
    """Simple concatenation-based fusion with MLP."""

    def __init__(self, image_dim: int, force_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + force_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_feat: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_feat: Image features of shape (B, D_img) or (B, T, D_img)
            force_feat: Force features of shape (B, D_force) or (B, T, D_force)

        Returns:
            Fused features of shape (B, hidden_dim) or (B, T, hidden_dim)
        """
        temporal = image_feat.dim() == 3
        if temporal:
            B, T, _ = image_feat.shape
            image_feat = image_feat.reshape(B * T, -1)
            force_feat = force_feat.reshape(B * T, -1)

        out = self.fusion(torch.cat([image_feat, force_feat], dim=-1))

        if temporal:
            out = out.view(B, T, self.hidden_dim)

        return out


class MeanAggregator(nn.Module):
    """Simple mean aggregation for temporal sequences."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Frame features of shape (B, T, D)
            mask: Optional mask of shape (B, T) indicating valid frames

        Returns:
            Aggregated features of shape (B, D)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            x = x * mask
            return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)


class LSTMAggregator(nn.Module):
    """LSTM-based temporal aggregation."""

    def __init__(self, feature_dim: int, hidden_dim: int = None,
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = hidden_dim or feature_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.out_proj = nn.Linear(out_dim, feature_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Frame features of shape (B, T, D)
            mask: Optional mask of shape (B, T) indicating valid frames

        Returns:
            Aggregated features of shape (B, D)
        """
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        # Concatenate forward and backward hidden states
        if self.lstm.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_n = h_n[-1]

        return self.out_proj(h_n)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation fusion.

    Force features generate scale (gamma) and shift (beta) parameters
    that modulate the pooled image vector: gamma * image + beta.
    """

    def __init__(self, image_dim: int, force_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Force -> (gamma, beta) for image modulation
        self.film_gen = nn.Linear(force_dim, image_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_feat: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_feat: Pooled image features (B, D_img)
            force_feat: Force features (B, D_force)

        Returns:
            Fused features (B, hidden_dim)
        """
        film_params = self.film_gen(force_feat)  # (B, D_img * 2)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, D_img)
        modulated = gamma * image_feat + beta  # (B, D_img)
        return self.proj(modulated)  # (B, hidden_dim)


class PatchSelfAttentionFusion(nn.Module):
    """Self-attention over 196 image patches + 1 force token.

    Projects force to patch dim, adds modality embeddings, concatenates
    with image patches -> 197 tokens. Runs 1-layer TransformerEncoder.
    Reads out the force token (position 0).
    """

    def __init__(self, patch_dim: int, force_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_dim = patch_dim

        self.force_proj = nn.Linear(force_dim, patch_dim)
        self.modality_embed = nn.Embedding(2, patch_dim)  # 0=force, 1=image

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_dim,
            nhead=num_heads,
            dim_feedforward=patch_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(patch_dim, hidden_dim)

    def forward(self, image_patches: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_patches: (B, N, patch_dim) — unpooled patch tokens
            force_feat: (B, force_dim)

        Returns:
            Fused features (B, hidden_dim)
        """
        B = force_feat.size(0)
        device = force_feat.device

        force_token = self.force_proj(force_feat).unsqueeze(1)  # (B, 1, D)

        # Add modality embeddings
        force_idx = torch.zeros(1, dtype=torch.long, device=device)
        image_idx = torch.ones(1, dtype=torch.long, device=device)
        force_token = force_token + self.modality_embed(force_idx)
        image_patches = image_patches + self.modality_embed(image_idx)

        tokens = torch.cat([force_token, image_patches], dim=1)  # (B, N+1, D)
        out = self.transformer(tokens)  # (B, N+1, D)
        return self.out_proj(out[:, 0, :])  # (B, hidden_dim) — force token readout


class PatchCrossAttentionFusion(nn.Module):
    """Cross-attention: force queries over image patch keys/values.

    Force feature projected to patch dim as single query.
    196 image patches as K/V. nn.MultiheadAttention.
    """

    def __init__(self, patch_dim: int, force_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_dim = patch_dim

        self.force_proj = nn.Linear(force_dim, patch_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(patch_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, image_patches: torch.Tensor,
                force_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image_patches: (B, N, patch_dim) — unpooled patch tokens
            force_feat: (B, force_dim)

        Returns:
            Fused features (B, hidden_dim)
        """
        query = self.force_proj(force_feat).unsqueeze(1)  # (B, 1, D)
        attn_out, _ = self.cross_attn(query, image_patches, image_patches)  # (B, 1, D)
        attn_out = self.layer_norm(query + attn_out)  # residual
        return self.out_proj(attn_out.squeeze(1))  # (B, hidden_dim)


def get_fusion_module(fusion_type: str, image_dim: int, force_dim: int,
                      hidden_dim: int = 128, num_heads: int = 4,
                      num_layers: int = 1, dropout: float = 0.1,
                      patch_dim: int = 0) -> nn.Module:
    """Factory for fusion modules.

    Args:
        fusion_type: One of "concat", "attention", "cross_attention", "token_self_attention"
        image_dim: Image feature dimension
        force_dim: Force feature dimension
        hidden_dim: Hidden dimension for fusion
        num_heads: Number of attention heads (token_self_attention)
        num_layers: Number of transformer layers (token_self_attention)
        dropout: Dropout rate (token_self_attention)

    Returns:
        Fusion module
    """
    if fusion_type == "concat":
        return SimpleFusion(image_dim, force_dim, hidden_dim)
    elif fusion_type in ("attention", "cross_attention"):
        return CrossModalAttention(image_dim, force_dim, hidden_dim)
    elif fusion_type == "token_self_attention":
        return TokenSelfAttentionFusion(
            image_dim, force_dim, hidden_dim,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )
    elif fusion_type == "film":
        return FiLMFusion(image_dim, force_dim, hidden_dim, dropout=dropout)
    elif fusion_type == "patch_self_attention":
        assert patch_dim > 0, "patch_dim required for patch_self_attention fusion"
        return PatchSelfAttentionFusion(
            patch_dim, force_dim, hidden_dim,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )
    elif fusion_type == "patch_cross_attention":
        assert patch_dim > 0, "patch_dim required for patch_cross_attention fusion"
        return PatchCrossAttentionFusion(
            patch_dim, force_dim, hidden_dim,
            num_heads=num_heads, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def get_temporal_aggregator(aggregation: str, feature_dim: int) -> nn.Module:
    """Factory for temporal aggregation modules.

    Args:
        aggregation: One of "attention", "mean", "lstm"
        feature_dim: Feature dimension

    Returns:
        Aggregation module
    """
    if aggregation == "attention":
        return TemporalAttentionAggregator(feature_dim)
    elif aggregation == "mean":
        return MeanAggregator(feature_dim)
    elif aggregation == "lstm":
        return LSTMAggregator(feature_dim)
    else:
        raise ValueError(f"Unknown temporal aggregation: {aggregation}")
