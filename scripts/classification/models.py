import torch
import torch.nn as nn
import torchvision.models as models


class ForceClassifier(nn.Module):
    """MLP for 6-axis force/torque: classification + depth regression."""

    def __init__(self, input_dim=6, hidden_dims=[64, 32], num_classes=3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, num_classes)
        self.depth_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.depth_head(feat).squeeze(-1)


class ImageClassifier(nn.Module):
    """ResNet18 image classifier + depth regression."""

    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.cls_head = nn.Linear(in_features, num_classes)
        self.depth_head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.depth_head(feat).squeeze(-1)


class CombinedClassifier(nn.Module):
    """Late-fusion image + force/torque: classification + depth regression."""

    def __init__(self, sensor_dim=6, num_classes=3,
                 pretrained=True, freeze_backbone=False):
        super().__init__()

        # Image branch (ResNet18)
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        image_feat_dim = 512

        # Sensor branch
        self.sensor_branch = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        sensor_feat_dim = 64

        # Shared fusion layer
        combined_dim = image_feat_dim + sensor_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.cls_head = nn.Linear(128, num_classes)
        self.depth_head = nn.Linear(128, 1)

    def forward(self, image, sensor):
        img_feat = self.image_features(image).flatten(1)
        sensor_feat = self.sensor_branch(sensor)
        fused = self.fusion(torch.cat([img_feat, sensor_feat], dim=1))
        return self.cls_head(fused), self.depth_head(fused).squeeze(-1)


def get_model(name="combined", **kwargs):
    """Factory function for model selection."""
    if name == "force":
        return ForceClassifier(**kwargs)
    elif name == "image":
        return ImageClassifier(**kwargs)
    elif name == "combined":
        return CombinedClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
