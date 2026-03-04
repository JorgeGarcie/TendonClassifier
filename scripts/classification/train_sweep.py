"""Sweep training wrapper for TendonClassifier v2.

Thin wrapper around train_v2.run_training() for use with wandb sweeps.
The base YAML config carries all "fixed" settings (split strategy, balanced
sampling, scheduler, etc.) — only swept params get overridden.

Usage:
    # Create sweep
    wandb sweep configs/sweep_spatial.yaml

    # Run agent (use the sweep_id from above)
    wandb agent <entity>/<project>/<sweep_id>
"""

import argparse

import yaml
import wandb

from config import load_config_from_dict
from train_v2 import run_training


# Maps flat sweep parameter names → dotted config paths
SWEEP_KEY_MAP = {
    "lr": "training.lr",
    "batch_size": "training.batch_size",
    "weight_decay": "training.weight_decay",
    "optimizer": "training.optimizer",
    "epochs": "training.epochs",
    "fusion_type": "model.fusion.type",
    "fusion_hidden_dim": "model.fusion.hidden_dim",
    "fusion_num_heads": "model.fusion.num_heads",
    "fusion_num_layers": "model.fusion.num_layers",
    "fusion_dropout": "model.fusion.dropout",
    "subtraction_enabled": "data.subtraction.enabled",
}


def set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict via dotted key path."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Sweep training wrapper")
    parser.add_argument("--base-config", type=str,
                        default="configs/spatial_combined.yaml",
                        help="Path to base YAML config")
    args = parser.parse_args()

    # Sweep agent calls wandb.init() which injects sweep params
    wandb.init()

    # Load base config as raw dict
    with open(args.base_config) as f:
        yaml_dict = yaml.safe_load(f)

    # Overlay sweep params onto base dict
    sweep_config = dict(wandb.config)
    for sweep_key, value in sweep_config.items():
        if sweep_key in SWEEP_KEY_MAP:
            set_nested(yaml_dict, SWEEP_KEY_MAP[sweep_key], value)

    # Convert to Config dataclass and run training
    config = load_config_from_dict(yaml_dict)
    run_training(config, sweep_mode=True)


if __name__ == "__main__":
    main()
