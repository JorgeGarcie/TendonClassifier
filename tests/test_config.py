"""Tests for configuration loading and defaults (config.py).

Verifies that:
- Default config has expected values
- YAML loading produces valid Config objects
- Missing config file raises FileNotFoundError
"""

import pytest
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "classification"))

from config import (
    Config, ExperimentConfig, ModelConfig, DataConfig, TrainingConfig,
    load_config, config_to_dict,
)


class TestDefaults:
    def test_default_config_creates(self):
        config = Config()
        assert config.experiment.seed == 42
        assert config.model.num_classes == 4
        assert config.training.epochs == 100
        assert config.training.batch_size == 32

    def test_default_model_config(self):
        mc = ModelConfig()
        assert mc.type == "spatial"
        assert mc.encoder.name == "resnet18"
        assert mc.encoder.pretrained is True
        assert mc.encoder.freeze is True
        assert mc.use_force is True
        assert mc.use_depth_head is True

    def test_default_data_config(self):
        dc = DataConfig()
        assert dc.img_size == 224
        assert dc.normalization.type == "imagenet"
        assert dc.subtraction.enabled is False
        assert dc.augmentation.enabled is False


class TestYAMLLoading:
    def test_load_default_yaml(self):
        yaml_path = (
            Path(__file__).resolve().parent.parent
            / "scripts" / "classification" / "configs" / "default.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("default.yaml not found")

        config = load_config(str(yaml_path))

        assert isinstance(config, Config)
        assert config.model.encoder.name == "resnet18"
        assert config.training.lr == 0.0001
        assert config.data.normalization.type == "imagenet"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_minimal_yaml(self):
        """A minimal YAML should fill in defaults for missing fields."""
        minimal_yaml = "experiment:\n  name: test_run\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(minimal_yaml)
            f.flush()
            config = load_config(f.name)

        assert config.experiment.name == "test_run"
        # Everything else should be defaults
        assert config.model.num_classes == 4
        assert config.training.epochs == 100


class TestConfigToDict:
    def test_roundtrip(self):
        config = Config()
        d = config_to_dict(config)

        assert isinstance(d, dict)
        assert d["experiment"]["seed"] == 42
        assert d["model"]["encoder"]["name"] == "resnet18"
        assert d["training"]["batch_size"] == 32
