"""Weights & Biases logging utilities for TendonClassifier v2.

Provides a WandbLogger class that wraps wandb functionality with
graceful fallback when wandb is not available or disabled.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class WandbLogger:
    """Wrapper for Weights & Biases logging.

    Provides a consistent interface regardless of whether wandb is
    enabled, making it easy to switch between local and cloud logging.
    """

    def __init__(
        self,
        enabled: bool = True,
        project: str = "TendonClassifier",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        save_code: bool = True,
    ):
        """Initialize wandb logger.

        Args:
            enabled: Whether to enable wandb logging
            project: Project name
            name: Run name (auto-generated if None)
            config: Configuration dict to log
            entity: Wandb username or team
            tags: List of tags for the run
            notes: Notes for the run
            save_code: Whether to save code to wandb
        """
        self.enabled = enabled
        self._run = None

        if not enabled:
            print("WandbLogger: Disabled, logging locally only")
            return

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print("WandbLogger: wandb not installed, logging locally only")
            self.enabled = False
            return

        # Initialize wandb
        try:
            # Finish any existing run before starting a new one
            if wandb.run is not None:
                wandb.finish()

            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                tags=tags or [],
                notes=notes or "",
                save_code=save_code,
            )
            print(f"WandbLogger: Initialized run '{self._run.name}'")
            print(f"  URL: {self._run.url}")
        except Exception as e:
            print(f"WandbLogger: Failed to initialize: {e}")
            self.enabled = False

    @property
    def run(self):
        """Get the wandb run object."""
        return self._run

    @property
    def run_name(self) -> str:
        """Get the run name."""
        if self._run:
            return self._run.name
        return "local"

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        if self._run:
            return self._run.id
        return "local"

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None,
            commit: bool = True):
        """Log metrics.

        Args:
            metrics: Dict of metric names to values
            step: Optional step number
            commit: Whether to commit the log (advance step counter)
        """
        if not self.enabled:
            return

        self._wandb.log(metrics, step=step, commit=commit)

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float], lr: Optional[float] = None):
        """Log metrics for an epoch.

        Args:
            epoch: Epoch number (0-indexed)
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            lr: Optional learning rate to log
        """
        metrics = {"epoch": epoch + 1}

        for key, value in train_metrics.items():
            metrics[f"train/{key}"] = value

        for key, value in val_metrics.items():
            metrics[f"val/{key}"] = value

        if lr is not None:
            metrics["lr"] = lr

        self.log(metrics)  # No explicit step - let wandb auto-increment

    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: list, title: str = "Confusion Matrix"):
        """Log a confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the plot
        """
        if not self.enabled:
            return

        self._wandb.log({
            title: self._wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram.

        Args:
            name: Metric name
            values: Array of values
            step: Optional step number
        """
        if not self.enabled:
            return

        self._wandb.log({name: self._wandb.Histogram(values)}, step=step)

    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None,
                  caption: Optional[str] = None):
        """Log an image.

        Args:
            name: Image name
            image: Image array (H, W, C) or (H, W)
            step: Optional step number
            caption: Optional caption
        """
        if not self.enabled:
            return

        self._wandb.log({
            name: self._wandb.Image(image, caption=caption)
        }, step=step)

    def log_model(self, model_path: str, name: str = "model",
                  metadata: Optional[Dict] = None):
        """Log a model artifact.

        Args:
            model_path: Path to model checkpoint
            name: Artifact name
            metadata: Optional metadata dict
        """
        if not self.enabled:
            return

        artifact = self._wandb.Artifact(name, type="model", metadata=metadata)
        artifact.add_file(model_path)
        self._run.log_artifact(artifact)

    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch a model for gradient/parameter logging.

        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency (batches)
        """
        if not self.enabled:
            return

        self._wandb.watch(model, log=log, log_freq=log_freq)

    def save(self, filepath: str):
        """Save a file to wandb.

        Args:
            filepath: Path to file to save
        """
        if not self.enabled:
            return

        self._wandb.save(filepath)

    def finish(self):
        """Finish the wandb run."""
        if self._run:
            self._run.finish()
            print(f"WandbLogger: Finished run '{self._run.name}'")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_logger(config) -> WandbLogger:
    """Create a WandbLogger from config.

    Args:
        config: Config dataclass or dict

    Returns:
        WandbLogger instance
    """
    # Handle both dataclass and dict
    if hasattr(config, "logging"):
        wandb_cfg = config.logging.wandb
        enabled = wandb_cfg.enabled
        entity = wandb_cfg.entity
        tags = wandb_cfg.tags
        notes = wandb_cfg.notes
        project = config.experiment.project
        name = config.experiment.name
    else:
        logging_cfg = config.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        enabled = wandb_cfg.get("enabled", True)
        entity = wandb_cfg.get("entity")
        tags = wandb_cfg.get("tags", [])
        notes = wandb_cfg.get("notes", "")
        exp_cfg = config.get("experiment", {})
        project = exp_cfg.get("project", "TendonClassifier")
        name = exp_cfg.get("name")

    # Import config_to_dict if available
    try:
        from config import config_to_dict
        config_dict = config_to_dict(config) if hasattr(config, "experiment") else config
    except ImportError:
        config_dict = config if isinstance(config, dict) else {}

    return WandbLogger(
        enabled=enabled,
        project=project,
        name=name,
        config=config_dict,
        entity=entity,
        tags=tags,
        notes=notes,
    )
