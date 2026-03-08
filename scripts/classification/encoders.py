"""Vision encoders for TendonClassifier v2.

Provides a factory for creating vision encoders with support for:
- ResNet18 (torchvision)
- DinoV2 (torch.hub)
- CLIP (openai/CLIP)

All encoders output a feature vector and support freezing for transfer learning.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


# Encoder output dimensions
ENCODER_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "dinov2_small": 384,
    "dinov2_base": 768,
    "dinov2_large": 1024,
    "clip_vit_b16": 512,
    "clip_vit_b32": 512,
    "clip_vit_l14": 768,
    "sparsh_vitb16": 768,
    "sparsh_vitb14_3ch": 768,
}


class ResNetEncoder(nn.Module):
    """ResNet encoder that outputs feature vectors."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True,
                 freeze: bool = True):
        super().__init__()
        self.name = name
        self.output_dim = ENCODER_DIMS[name]

        # Load pretrained model
        weights = "IMAGENET1K_V1" if pretrained else None
        if name == "resnet18":
            backbone = models.resnet18(weights=weights)
        elif name == "resnet34":
            backbone = models.resnet34(weights=weights)
        elif name == "resnet50":
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown ResNet variant: {name}")

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze:
            self._freeze()

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        feat = self.backbone(x)
        return feat.flatten(1)


class DinoV2Encoder(nn.Module):
    """DinoV2 encoder using torch.hub."""

    def __init__(self, name: str = "dinov2_small", freeze: bool = True):
        super().__init__()
        self.name = name
        self.output_dim = ENCODER_DIMS[name]

        # Map our names to torch.hub model names
        hub_names = {
            "dinov2_small": "dinov2_vits14",
            "dinov2_base": "dinov2_vitb14",
            "dinov2_large": "dinov2_vitl14",
        }

        if name not in hub_names:
            raise ValueError(f"Unknown DinoV2 variant: {name}")

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            hub_names[name],
            pretrained=True,
        )

        if freeze:
            self._freeze()

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)
                Note: DinoV2 expects 224x224 or 518x518 images

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        # DinoV2 returns CLS token features directly
        return self.backbone(x)


class CLIPEncoder(nn.Module):
    """CLIP visual encoder."""

    def __init__(self, name: str = "clip_vit_b16", freeze: bool = True):
        super().__init__()
        self.name = name
        self.output_dim = ENCODER_DIMS[name]

        # Map our names to CLIP model names
        clip_names = {
            "clip_vit_b16": "ViT-B/16",
            "clip_vit_b32": "ViT-B/32",
            "clip_vit_l14": "ViT-L/14",
        }

        if name not in clip_names:
            raise ValueError(f"Unknown CLIP variant: {name}")

        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        model, _ = clip.load(clip_names[name], device="cpu")
        self.backbone = model.visual

        # CLIP uses its own preprocessing; we need to match it
        # Note: CLIP expects images normalized with its own mean/std
        # but ImageNet normalization is close enough for transfer learning

        if freeze:
            self._freeze()

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        # CLIP visual encoder outputs features
        return self.backbone(x.type(self.backbone.conv1.weight.dtype)).float()


class SparshEncoder(nn.Module):
    """Sparsh ViT-B/16 encoder for 6-channel temporal pair input."""

    WEIGHTS_PATH = "checkpoints/sparsh/dinov2_vitbase.safetensors"

    def __init__(self, freeze: bool = True):
        super().__init__()
        from sparsh_vit import vit_base
        self.name = "sparsh_vitb16"
        self.output_dim = ENCODER_DIMS[self.name]

        self.backbone = vit_base(
            patch_size=16, in_chans=6, img_size=224,
            pos_embed_fn="sinusoidal", num_register_tokens=1,
        )
        self._load_weights()
        if freeze:
            self._freeze()

    def _load_weights(self):
        from safetensors.torch import load_file
        from pathlib import Path

        weights_path = Path(__file__).parent / self.WEIGHTS_PATH
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Sparsh weights not found at {weights_path}. "
                "Download from huggingface.co/facebook/sparsh-dinov2-base"
            )
        state_dict = load_file(str(weights_path))
        # The checkpoint may have keys with "teacher.backbone." prefix
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("teacher.backbone.", "").replace("teacher.", "")
            cleaned[k] = v

        # patch_embed.proj expects 6 input channels but pretrained weights have 3.
        # Tile the conv1 weight across the channel dimension (3ch -> 6ch).
        proj_key = "patch_embed.proj.weight"
        if proj_key in cleaned and cleaned[proj_key].shape[1] == 3:
            w = cleaned[proj_key]  # (768, 3, 16, 16)
            cleaned[proj_key] = w.repeat(1, 2, 1, 1) / 2.0  # avg preserves scale

        msg = self.backbone.load_state_dict(cleaned, strict=False)
        if msg.missing_keys:
            print(f"Sparsh: missing keys (expected for sinusoidal pos_embed): "
                  f"{msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Sparsh: unexpected keys (ignored): {msg.unexpected_keys}")

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 6, H, W) — concatenated temporal pair

        Returns:
            Feature tensor of shape (B, 768) — mean-pooled patch tokens
        """
        patch_tokens = self.backbone(x)  # (B, N, 768)
        return patch_tokens.mean(dim=1)  # (B, 768)


class SparshSpatialEncoder(nn.Module):
    """Sparsh ViT-B/14 encoder for 3-channel single-frame input.

    Trained with DINOv2 SSL on ~385K tactile frames (same data as Sparsh,
    but 3ch single frames instead of 6ch temporal pairs).
    """

    WEIGHTS_PATH = "checkpoints/sparsh/dinov2_vitb14_3ch.safetensors"

    def __init__(self, freeze: bool = True):
        super().__init__()
        from sparsh_vit import vit_base
        self.name = "sparsh_vitb14_3ch"
        self.output_dim = ENCODER_DIMS[self.name]

        self.backbone = vit_base(
            patch_size=14, in_chans=3, img_size=224,
            pos_embed_fn="sinusoidal", num_register_tokens=1,
        )
        self._load_weights()
        if freeze:
            self._freeze()

    def _load_weights(self):
        from safetensors.torch import load_file
        from pathlib import Path

        weights_path = Path(__file__).parent / self.WEIGHTS_PATH
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Sparsh 3ch weights not found at {weights_path}. "
                "Transfer from BDML-server: scp jorge@BDML-server.stanford.edu:"
                "/home/jorge/VisionFT/checkpoints/dinov2_vitb14_3ch.safetensors "
                "checkpoints/sparsh/"
            )
        state_dict = load_file(str(weights_path))
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("teacher.backbone.", "").replace("teacher.", "")
            cleaned[k] = v

        msg = self.backbone.load_state_dict(cleaned, strict=False)
        if msg.missing_keys:
            print(f"Sparsh 3ch: missing keys (expected for sinusoidal pos_embed): "
                  f"{msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"Sparsh 3ch: unexpected keys (ignored): {msg.unexpected_keys}")

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W) — single subtracted frame

        Returns:
            Feature tensor of shape (B, 768) — mean-pooled patch tokens
        """
        patch_tokens = self.backbone(x)  # (B, N, 768)
        return patch_tokens.mean(dim=1)  # (B, 768)


def get_encoder(name: str, pretrained: bool = True,
                freeze: bool = True) -> nn.Module:
    """Factory function to create vision encoders.

    Args:
        name: Encoder name. One of:
            - resnet18, resnet34, resnet50
            - dinov2_small, dinov2_base, dinov2_large
            - clip_vit_b16, clip_vit_b32, clip_vit_l14
        pretrained: Whether to load pretrained weights.
        freeze: Whether to freeze encoder weights (recommended for small datasets).

    Returns:
        Encoder module with `output_dim` attribute.
    """
    if name.startswith("resnet"):
        return ResNetEncoder(name, pretrained, freeze)
    elif name.startswith("dinov2"):
        return DinoV2Encoder(name, freeze)
    elif name.startswith("clip"):
        return CLIPEncoder(name, freeze)
    elif name == "sparsh_vitb16":
        return SparshEncoder(freeze=freeze)
    elif name == "sparsh_vitb14_3ch":
        return SparshSpatialEncoder(freeze=freeze)
    else:
        raise ValueError(
            f"Unknown encoder: {name}. "
            f"Available: {list(ENCODER_DIMS.keys())}"
        )


def get_encoder_dim(name: str) -> int:
    """Get output dimension for an encoder without instantiating it."""
    if name not in ENCODER_DIMS:
        raise ValueError(f"Unknown encoder: {name}")
    return ENCODER_DIMS[name]
