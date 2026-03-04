"""Tests for model architectures (models.py).

Verifies that each model variant:
- Instantiates without error
- Produces correct output shapes for classification + depth regression
- Works with the get_model factory function
"""

import pytest
import torch

import sys
from pathlib import Path

# Add classification scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "classification"))

from models import ForceClassifier, ImageClassifier, CombinedClassifier, get_model


BATCH_SIZE = 4
NUM_CLASSES = 4


class TestForceClassifier:
    def test_output_shapes(self):
        model = ForceClassifier(input_dim=6, num_classes=NUM_CLASSES)
        model.eval()
        force = torch.randn(BATCH_SIZE, 6)
        cls_logits, depth = model(force)

        assert cls_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert depth.shape == (BATCH_SIZE,)

    def test_custom_hidden_dims(self):
        model = ForceClassifier(input_dim=6, hidden_dims=[128, 64, 32], num_classes=3)
        model.eval()
        force = torch.randn(BATCH_SIZE, 6)
        cls_logits, depth = model(force)

        assert cls_logits.shape == (BATCH_SIZE, 3)


class TestImageClassifier:
    def test_output_shapes(self):
        model = ImageClassifier(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        image = torch.randn(BATCH_SIZE, 3, 224, 224)
        cls_logits, depth = model(image)

        assert cls_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert depth.shape == (BATCH_SIZE,)

    def test_frozen_backbone(self):
        model = ImageClassifier(pretrained=False, freeze_backbone=True)
        for param in model.backbone.parameters():
            assert not param.requires_grad


class TestCombinedClassifier:
    def test_output_shapes(self):
        model = CombinedClassifier(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        image = torch.randn(BATCH_SIZE, 3, 224, 224)
        force = torch.randn(BATCH_SIZE, 6)
        cls_logits, depth = model(image, force)

        assert cls_logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert depth.shape == (BATCH_SIZE,)


class TestGetModel:
    def test_force_model(self):
        model = get_model("force")
        assert isinstance(model, ForceClassifier)

    def test_image_model(self):
        model = get_model("image", pretrained=False)
        assert isinstance(model, ImageClassifier)

    def test_combined_model(self):
        model = get_model("combined", pretrained=False)
        assert isinstance(model, CombinedClassifier)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")
