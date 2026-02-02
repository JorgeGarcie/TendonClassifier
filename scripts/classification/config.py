"""Configuration management for TendonClassifier v2.

Provides dataclasses for type-safe configuration and YAML loading utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    project: str = "TendonClassifier"
    seed: int = 42


@dataclass
class EncoderConfig:
    name: str = "resnet18"  # resnet18 | dinov2_small | dinov2_base | clip_vit_b16
    pretrained: bool = True
    freeze: bool = True


@dataclass
class TemporalConfig:
    num_frames: int = 5
    aggregation: str = "attention"  # attention | mean | lstm


@dataclass
class FusionConfig:
    type: str = "attention"  # concat | attention | cross_attention
    hidden_dim: int = 128


@dataclass
class ModelConfig:
    type: str = "spatial"  # spatial | temporal
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    num_classes: int = 4
    use_force: bool = True
    use_depth_head: bool = True


@dataclass
class NormalizationConfig:
    type: str = "imagenet"  # imagenet | simple
    mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class SubtractionConfig:
    enabled: bool = False
    reference: str = "first_frame"  # first_frame | untouched


@dataclass
class ColorJitterConfig:
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1


@dataclass
class AugmentationConfig:
    enabled: bool = False
    horizontal_flip: bool = True
    rotation_degrees: float = 10
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)


@dataclass
class DataConfig:
    manifest: str = "../labeling/output/gt_dataset/gt_manifest.csv"
    img_size: int = 224
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    subtraction: SubtractionConfig = field(default_factory=SubtractionConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    exclude_phantoms: Optional[list] = None


@dataclass
class SchedulerConfig:
    type: str = "cosine"  # cosine | step | none
    warmup_epochs: int = 5


@dataclass
class LossConfig:
    class_weights: Any = "balanced"  # balanced | none | list of weights
    depth_weight: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # adam | adamw | sgd
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    val_ratio: float = 0.2
    split_by: str = "run"  # run | random


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_every_n_epochs: int = 5
    resume: Optional[str] = None


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""


@dataclass
class LoggingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    csv: dict = field(default_factory=lambda: {"enabled": True})
    print_every: int = 1


@dataclass
class Config:
    """Top-level configuration container."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass instance."""
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is type(None):
            kwargs[key] = value
        # Handle nested dataclasses
        elif hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config dataclass with all settings.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Build Config from nested dicts
    config = Config(
        experiment=_dict_to_dataclass(
            ExperimentConfig, data.get("experiment", {})
        ),
        model=_build_model_config(data.get("model", {})),
        data=_build_data_config(data.get("data", {})),
        training=_build_training_config(data.get("training", {})),
        checkpoint=_dict_to_dataclass(
            CheckpointConfig, data.get("checkpoint", {})
        ),
        logging=_build_logging_config(data.get("logging", {})),
    )

    return config


def _build_model_config(data: dict) -> ModelConfig:
    """Build ModelConfig with nested dataclasses."""
    if not data:
        return ModelConfig()

    return ModelConfig(
        type=data.get("type", "spatial"),
        encoder=_dict_to_dataclass(EncoderConfig, data.get("encoder", {})),
        temporal=_dict_to_dataclass(TemporalConfig, data.get("temporal", {})),
        fusion=_dict_to_dataclass(FusionConfig, data.get("fusion", {})),
        num_classes=data.get("num_classes", 4),
        use_force=data.get("use_force", True),
        use_depth_head=data.get("use_depth_head", True),
    )


def _build_data_config(data: dict) -> DataConfig:
    """Build DataConfig with nested dataclasses."""
    if not data:
        return DataConfig()

    aug_data = data.get("augmentation", {})
    augmentation = AugmentationConfig(
        enabled=aug_data.get("enabled", False),
        horizontal_flip=aug_data.get("horizontal_flip", True),
        rotation_degrees=aug_data.get("rotation_degrees", 10),
        color_jitter=_dict_to_dataclass(
            ColorJitterConfig, aug_data.get("color_jitter", {})
        ),
    )

    return DataConfig(
        manifest=data.get("manifest", "../labeling/output/gt_dataset/gt_manifest.csv"),
        img_size=data.get("img_size", 224),
        normalization=_dict_to_dataclass(
            NormalizationConfig, data.get("normalization", {})
        ),
        subtraction=_dict_to_dataclass(
            SubtractionConfig, data.get("subtraction", {})
        ),
        augmentation=augmentation,
        exclude_phantoms=data.get("exclude_phantoms"),
    )


def _build_training_config(data: dict) -> TrainingConfig:
    """Build TrainingConfig with nested dataclasses."""
    if not data:
        return TrainingConfig()

    return TrainingConfig(
        epochs=data.get("epochs", 100),
        batch_size=data.get("batch_size", 32),
        lr=data.get("lr", 1e-4),
        weight_decay=data.get("weight_decay", 1e-4),
        optimizer=data.get("optimizer", "adam"),
        scheduler=_dict_to_dataclass(SchedulerConfig, data.get("scheduler", {})),
        loss=_dict_to_dataclass(LossConfig, data.get("loss", {})),
        val_ratio=data.get("val_ratio", 0.2),
        split_by=data.get("split_by", "run"),
    )


def _build_logging_config(data: dict) -> LoggingConfig:
    """Build LoggingConfig with nested dataclasses."""
    if not data:
        return LoggingConfig()

    return LoggingConfig(
        wandb=_dict_to_dataclass(WandbConfig, data.get("wandb", {})),
        csv=data.get("csv", {"enabled": True}),
        print_every=data.get("print_every", 1),
    )


def config_to_dict(config: Config) -> dict:
    """Convert a Config dataclass to a nested dict (for wandb logging)."""
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        else:
            return obj

    return _to_dict(config)
