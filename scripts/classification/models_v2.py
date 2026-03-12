"""Model architectures for TendonClassifier v2.

Provides two main model types:
- SpatialModel: Single-frame classification with optional force fusion
- TemporalModel: Multi-frame sequence classification with temporal aggregation

Both support multiple vision encoders and attention-based fusion.
"""

import torch
import torch.nn as nn

from encoders import get_encoder, get_encoder_dim
from attention import get_fusion_module, get_temporal_aggregator, SimpleFusion


class ForceBranch(nn.Module):
    """MLP branch for processing force/torque sensor data."""

    def __init__(self, input_dim: int = 6, hidden_dims: list = None,
                 output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Force tensor of shape (B, 6)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        return self.mlp(x)


class TemporalForceBranch(nn.Module):
    """MLP branch for processing temporal force/torque sequences."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.output_dim = output_dim

        # Per-frame MLP
        self.frame_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Force tensor of shape (B, T, 6)

        Returns:
            Feature tensor of shape (B, T, hidden_dim)
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        out_flat = self.frame_mlp(x_flat)
        return out_flat.view(B, T, -1)


class CausalConvForceEncoder(nn.Module):
    """5-layer causal 1D convolution for force/torque temporal encoding.

    Processes 32 raw wrench readings at 300Hz (~107ms context).
    Architecture from Lee et al. "Making Sense of Vision and Touch" (ICRA 2019).

    Input: (B, 32, 6) -> Output: (B, 64)
    Sequence length: 32->16->8->4->2->1, squeeze final dim.
    """

    def __init__(self, input_channels: int = 6):
        super().__init__()
        self.output_dim = 64
        channels = [input_channels, 16, 32, 64, 64, 64]
        kernel_size = 3

        layers = []
        for i in range(5):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            pad = kernel_size - 1  # causal: left-pad by k-1
            layers.append(nn.ConstantPad1d((pad, 0), 0.0))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=2, padding=0))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Force window tensor of shape (B, 32, 6)

        Returns:
            Feature tensor of shape (B, 64)
        """
        x = x.permute(0, 2, 1)  # (B, 6, 32)
        x = self.net(x)  # (B, 64, 1)
        return x.squeeze(-1)  # (B, 64)


class ClassificationHead(nn.Module):
    """Classification and optional depth regression head."""

    def __init__(self, input_dim: int, num_classes: int = 4,
                 use_depth: bool = True, dropout: float = 0.5):
        super().__init__()
        self.use_depth = use_depth

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Linear(128, num_classes)
        if use_depth:
            self.depth_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Feature tensor of shape (B, input_dim)

        Returns:
            If use_depth: (cls_logits, depth_pred) tuple
            Else: cls_logits only
        """
        feat = self.shared(x)
        cls_logits = self.cls_head(feat)

        if self.use_depth:
            depth_pred = self.depth_head(feat).squeeze(-1)
            return cls_logits, depth_pred
        return cls_logits


class SpatialForceModel(nn.Module):
    """Force-only spatial model (no images).

    Simple MLP for force/torque classification + depth regression.
    Useful for ablation studies to isolate force contribution.
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_dim: int = 6,
        hidden_dims: list = None,
        dropout: float = 0.3,
        use_depth_head: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.use_depth_head = use_depth_head

        # MLP backbone
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, num_classes)
        if use_depth_head:
            self.depth_head = nn.Linear(prev_dim, 1)

    def forward(self, force: torch.Tensor):
        """Forward pass.

        Args:
            force: Force tensor of shape (B, 6)

        Returns:
            (cls_logits, depth_pred) tuple or cls_logits only
        """
        feat = self.backbone(force)
        cls_logits = self.cls_head(feat)

        if self.use_depth_head:
            depth_pred = self.depth_head(feat).squeeze(-1)
            return cls_logits, depth_pred
        return cls_logits


class SpatialModel(nn.Module):
    """Single-frame spatial model with optional force fusion.

    Supports multiple vision encoders and attention-based fusion.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        num_classes: int = 4,
        freeze_encoder: bool = True,
        use_force: bool = True,
        fusion_type: str = "attention",
        fusion_hidden_dim: int = 128,
        use_depth_head: bool = True,
        pretrained: bool = True,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_force = use_force
        self.encoder_name = encoder_name

        # Vision encoder
        self.encoder = get_encoder(encoder_name, pretrained=pretrained,
                                   freeze=freeze_encoder)
        encoder_dim = self.encoder.output_dim

        # Force branch (if used)
        if use_force:
            self.force_branch = ForceBranch(input_dim=6, output_dim=64)
            force_dim = self.force_branch.output_dim

            # Fusion module
            self.fusion = get_fusion_module(
                fusion_type, encoder_dim, force_dim, fusion_hidden_dim,
                num_heads=fusion_num_heads, num_layers=fusion_num_layers,
                dropout=fusion_dropout,
            )
            head_input_dim = fusion_hidden_dim
        else:
            # No force, just project encoder output
            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, fusion_hidden_dim),
                nn.ReLU(),
            )
            head_input_dim = fusion_hidden_dim

        # Classification head
        self.head = ClassificationHead(
            head_input_dim, num_classes, use_depth=use_depth_head
        )

    def forward(self, image: torch.Tensor, force: torch.Tensor = None):
        """Forward pass.

        Args:
            image: Image tensor of shape (B, 3, H, W)
            force: Optional force tensor of shape (B, 6)

        Returns:
            (cls_logits, depth_pred) tuple or cls_logits only
        """
        # Extract image features
        img_feat = self.encoder(image)

        if self.use_force and force is not None:
            # Extract force features and fuse
            force_feat = self.force_branch(force)
            fused = self.fusion(img_feat, force_feat)
        else:
            # Just project image features
            fused = self.proj(img_feat)

        return self.head(fused)


class TemporalModel(nn.Module):
    """Multi-frame temporal model with sequence aggregation.

    Processes a sequence of frames and aggregates them using
    attention, mean pooling, or LSTM.

    When use_force=True, fusion happens per-frame BEFORE temporal
    aggregation so that force context informs every frame's representation.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        num_classes: int = 4,
        freeze_encoder: bool = True,
        num_frames: int = 5,
        aggregation: str = "attention",
        use_force: bool = True,
        fusion_type: str = "attention",
        fusion_hidden_dim: int = 128,
        use_depth_head: bool = True,
        pretrained: bool = True,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_force = use_force
        self.encoder_name = encoder_name

        # Vision encoder (shared across frames)
        self.encoder = get_encoder(encoder_name, pretrained=pretrained,
                                   freeze=freeze_encoder)
        encoder_dim = self.encoder.output_dim

        # Force branch (if used) — per-frame temporal force
        if use_force:
            self.force_branch = TemporalForceBranch(
                input_dim=6, hidden_dim=64, output_dim=64,
            )
            force_dim = self.force_branch.output_dim

            # Per-frame fusion module (operates on (B, T, D) tensors)
            self.fusion = get_fusion_module(
                fusion_type, encoder_dim, force_dim, fusion_hidden_dim,
                num_heads=fusion_num_heads, num_layers=fusion_num_layers,
                dropout=fusion_dropout,
            )
            # Temporal aggregator operates on fused dim
            self.temporal_agg = get_temporal_aggregator(aggregation, fusion_hidden_dim)
            self.proj = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(),
            )
            head_input_dim = fusion_hidden_dim
        else:
            # Temporal aggregator operates on encoder dim
            self.temporal_agg = get_temporal_aggregator(aggregation, encoder_dim)
            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, fusion_hidden_dim),
                nn.ReLU(),
            )
            head_input_dim = fusion_hidden_dim

        # Classification head
        self.head = ClassificationHead(
            head_input_dim, num_classes, use_depth=use_depth_head
        )

    def forward(self, images: torch.Tensor, force: torch.Tensor = None,
                mask: torch.Tensor = None):
        """Forward pass.

        Args:
            images: Image sequence tensor of shape (B, T, 3, H, W)
            force: Optional force tensor of shape (B, T, 6) - per-frame force sequence
            mask: Optional mask of shape (B, T) indicating valid frames

        Returns:
            (cls_logits, depth_pred) tuple or cls_logits only
        """
        B, T, C, H, W = images.shape

        # Encode all frames
        images_flat = images.view(B * T, C, H, W)
        features_flat = self.encoder(images_flat)
        features = features_flat.view(B, T, -1)  # (B, T, D_enc)

        if self.use_force and force is not None:
            # Per-frame fusion: fuse image + force at each timestep
            force_features = self.force_branch(force)  # (B, T, D_force)
            fused = self.fusion(features, force_features)  # (B, T, fusion_hidden_dim)
            agg_feat = self.temporal_agg(fused, mask)  # (B, fusion_hidden_dim)
            out = self.proj(agg_feat)
        else:
            agg_feat = self.temporal_agg(features, mask)  # (B, D_enc)
            out = self.proj(agg_feat)

        return self.head(out)


class TemporalForceModel(nn.Module):
    """Force-only temporal model with sequence aggregation.

    Processes a sequence of force readings without images.
    Useful for ablation studies and when images aren't available.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_frames: int = 5,
        aggregation: str = "attention",
        force_hidden_dim: int = 64,
        use_depth_head: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames

        # Force branch (per-frame)
        self.force_branch = TemporalForceBranch(
            input_dim=6, hidden_dim=force_hidden_dim, output_dim=force_hidden_dim
        )

        # Temporal aggregator
        self.temporal_agg = get_temporal_aggregator(aggregation, force_hidden_dim)

        # Classification head
        self.head = ClassificationHead(
            force_hidden_dim, num_classes, use_depth=use_depth_head
        )

    def forward(self, forces: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass.

        Args:
            forces: Force sequence tensor of shape (B, T, 6)
            mask: Optional mask of shape (B, T) indicating valid frames

        Returns:
            (cls_logits, depth_pred) tuple or cls_logits only
        """
        # Process force sequence
        force_features = self.force_branch(forces)  # (B, T, D)

        # Temporal aggregation
        agg_feat = self.temporal_agg(force_features, mask)  # (B, D)

        return self.head(agg_feat)


class TemporalV2ForceModel(nn.Module):
    """Temporal v2 force-only model: CausalConvForceEncoder -> ClassificationHead.

    Processes 32-step force windows (300Hz) through causal convolutions.
    No vision encoder — force/torque only.
    """

    def __init__(self, num_classes: int = 4, use_depth_head: bool = True):
        super().__init__()
        self.force_encoder = CausalConvForceEncoder()
        self.head = ClassificationHead(
            self.force_encoder.output_dim, num_classes, use_depth=use_depth_head,
        )

    def forward(self, force_window: torch.Tensor):
        """Forward pass.

        Args:
            force_window: (B, 32, 6) raw wrench window

        Returns:
            (cls_logits, depth_pred) or cls_logits
        """
        feat = self.force_encoder(force_window)
        return self.head(feat)


class TemporalV2Model(nn.Module):
    """Temporal v2 combined/image-only model.

    Image: Sparsh ViT-B/16 (6ch, stride 3) — pooled or unpooled depending on fusion.
    Force: CausalConvForceEncoder (32 steps, 300Hz) when use_force=True.
    Fusion: concat/film (pooled image) or patch_self_attention/patch_cross_attention (unpooled).
    """

    # Fusion types that need unpooled (B, N, D) patch tokens
    PATCH_FUSIONS = {"patch_self_attention", "patch_cross_attention"}

    def __init__(
        self,
        encoder_name: str = "sparsh_vitb16",
        num_classes: int = 4,
        freeze_encoder: bool = True,
        use_force: bool = True,
        fusion_type: str = "concat",
        fusion_hidden_dim: int = 128,
        use_depth_head: bool = True,
        pretrained: bool = True,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_force = use_force
        self.encoder_name = encoder_name
        self.fusion_type = fusion_type

        # Vision encoder — unpooled for patch-level fusions
        needs_patches = fusion_type in self.PATCH_FUSIONS and use_force
        self.encoder = get_encoder(
            encoder_name, pretrained=pretrained,
            freeze=freeze_encoder, pool=not needs_patches,
        )
        encoder_dim = self.encoder.output_dim  # 768

        if use_force:
            self.force_encoder = CausalConvForceEncoder()
            force_dim = self.force_encoder.output_dim  # 64

            self.fusion = get_fusion_module(
                fusion_type, encoder_dim, force_dim, fusion_hidden_dim,
                num_heads=fusion_num_heads, num_layers=fusion_num_layers,
                dropout=fusion_dropout, patch_dim=encoder_dim if needs_patches else 0,
            )
            head_input_dim = fusion_hidden_dim
        else:
            self.proj = nn.Sequential(
                nn.Linear(encoder_dim, fusion_hidden_dim),
                nn.ReLU(),
            )
            head_input_dim = fusion_hidden_dim

        self.head = ClassificationHead(
            head_input_dim, num_classes, use_depth=use_depth_head,
        )

    def forward(self, image: torch.Tensor, force_window: torch.Tensor = None):
        """Forward pass.

        Args:
            image: (B, 6, H, W) — Sparsh 6ch temporal pair
            force_window: (B, 32, 6) — raw wrench window (when use_force)

        Returns:
            (cls_logits, depth_pred) or cls_logits
        """
        img_feat = self.encoder(image)  # (B, 768) or (B, N, 768)

        if self.use_force and force_window is not None:
            force_feat = self.force_encoder(force_window)  # (B, 64)

            if self.fusion_type in self.PATCH_FUSIONS:
                # Patch-level fusion: img_feat is (B, N, 768)
                fused = self.fusion(img_feat, force_feat)
            else:
                # Pooled fusion (concat/film): img_feat is (B, 768)
                fused = self.fusion(img_feat, force_feat)
        else:
            # Image-only: pool if not already pooled
            if img_feat.dim() == 3:
                img_feat = img_feat.mean(dim=1)
            fused = self.proj(img_feat)

        return self.head(fused)


def get_model_v2(config) -> nn.Module:
    """Factory function to create models from config.

    Args:
        config: ModelConfig dataclass or dict with model settings

    Returns:
        Model instance
    """
    # Handle both dataclass and dict
    if hasattr(config, "type"):
        model_type = config.type
        encoder_name = config.encoder.name
        freeze_encoder = config.encoder.freeze
        pretrained = config.encoder.pretrained
        num_classes = config.num_classes
        use_force = config.use_force
        use_depth_head = config.use_depth_head
        fusion_type = config.fusion.type
        fusion_hidden_dim = config.fusion.hidden_dim
        fusion_num_heads = getattr(config.fusion, "num_heads", 4)
        fusion_num_layers = getattr(config.fusion, "num_layers", 1)
        fusion_dropout = getattr(config.fusion, "dropout", 0.1)
        num_frames = config.temporal.num_frames
        aggregation = config.temporal.aggregation
    else:
        model_type = config.get("type", "spatial")
        encoder_cfg = config.get("encoder", {})
        encoder_name = encoder_cfg.get("name", "resnet18")
        freeze_encoder = encoder_cfg.get("freeze", True)
        pretrained = encoder_cfg.get("pretrained", True)
        num_classes = config.get("num_classes", 4)
        use_force = config.get("use_force", True)
        use_depth_head = config.get("use_depth_head", True)
        fusion_cfg = config.get("fusion", {})
        fusion_type = fusion_cfg.get("type", "attention")
        fusion_hidden_dim = fusion_cfg.get("hidden_dim", 128)
        fusion_num_heads = fusion_cfg.get("num_heads", 4)
        fusion_num_layers = fusion_cfg.get("num_layers", 1)
        fusion_dropout = fusion_cfg.get("dropout", 0.1)
        temporal_cfg = config.get("temporal", {})
        num_frames = temporal_cfg.get("num_frames", 5)
        aggregation = temporal_cfg.get("aggregation", "attention")

    if model_type == "spatial":
        return SpatialModel(
            encoder_name=encoder_name,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            use_force=use_force,
            fusion_type=fusion_type,
            fusion_hidden_dim=fusion_hidden_dim,
            use_depth_head=use_depth_head,
            pretrained=pretrained,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_dropout=fusion_dropout,
        )
    elif model_type == "spatial_force":
        # Get hidden_dims from config if available
        if hasattr(config, "force_hidden_dims"):
            hidden_dims = config.force_hidden_dims
        else:
            hidden_dims = [64, 128, 64]
        return SpatialForceModel(
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            use_depth_head=use_depth_head,
        )
    elif model_type == "temporal":
        return TemporalModel(
            encoder_name=encoder_name,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            num_frames=num_frames,
            aggregation=aggregation,
            use_force=use_force,
            fusion_type=fusion_type,
            fusion_hidden_dim=fusion_hidden_dim,
            use_depth_head=use_depth_head,
            pretrained=pretrained,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_dropout=fusion_dropout,
        )
    elif model_type == "temporal_force":
        return TemporalForceModel(
            num_classes=num_classes,
            num_frames=num_frames,
            aggregation=aggregation,
            force_hidden_dim=fusion_hidden_dim,
            use_depth_head=use_depth_head,
        )
    elif model_type == "temporal_v2":
        return TemporalV2Model(
            encoder_name=encoder_name,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            use_force=use_force,
            fusion_type=fusion_type,
            fusion_hidden_dim=fusion_hidden_dim,
            use_depth_head=use_depth_head,
            pretrained=pretrained,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_dropout=fusion_dropout,
        )
    elif model_type == "temporal_v2_force":
        return TemporalV2ForceModel(
            num_classes=num_classes,
            use_depth_head=use_depth_head,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def verify_frozen_encoder(model: nn.Module) -> bool:
    """Verify that encoder weights are frozen (no gradients).

    Args:
        model: Model with encoder attribute

    Returns:
        True if all encoder parameters have requires_grad=False
    """
    if not hasattr(model, "encoder"):
        return True

    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            print(f"Warning: {name} has requires_grad=True")
            return False
    return True
