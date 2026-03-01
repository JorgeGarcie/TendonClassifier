"""Export force normalization statistics from the training split.

Loads the temporal_combined config, builds the dataset with the same
include/exclude filters used during training, runs the contiguous frame
split to get training indices, computes per-channel mean/std from those
indices, and saves force_stats.json to the classification directory.

Usage:
    cd scripts/classification
    python export_force_stats.py --config configs/temporal_combined.yaml
    # Outputs: scripts/classification/force_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow imports from the classification directory
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from dataset import TendonDatasetV2


def parse_args():
    p = argparse.ArgumentParser(description="Export force normalization stats")
    p.add_argument("--config", type=str, default="configs/temporal_combined.yaml",
                   help="Path to YAML config file")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: same dir as config, force_stats.json)")
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    data_cfg = config.data if hasattr(config, "data") else config.get("data", {})
    train_cfg = config.training if hasattr(config, "training") else config.get("training", {})

    if hasattr(data_cfg, "manifest"):
        manifest = data_cfg.manifest
        include_regex = getattr(data_cfg, "include_run_regex", None)
        exclude_regex = getattr(data_cfg, "exclude_run_regex", None)
        img_size = getattr(data_cfg, "img_size", 224)
    else:
        manifest = data_cfg.get("manifest", "../labeling/output/gt_dataset/gt_manifest.csv")
        include_regex = data_cfg.get("include_run_regex", None)
        exclude_regex = data_cfg.get("exclude_run_regex", None)
        img_size = data_cfg.get("img_size", 224)

    if hasattr(train_cfg, "split"):
        split_cfg = train_cfg.split
        train_ratio = getattr(split_cfg, "train_ratio", 0.8)
        val_ratio = getattr(split_cfg, "val_ratio", 0.1)
        test_ratio = getattr(split_cfg, "test_ratio", 0.1)
        purge_frames = getattr(split_cfg, "purge_frames", 0)
    else:
        split_cfg = train_cfg.get("split", {}) if hasattr(train_cfg, "get") else {}
        train_ratio = split_cfg.get("train_ratio", 0.8)
        val_ratio = split_cfg.get("val_ratio", 0.1)
        test_ratio = split_cfg.get("test_ratio", 0.1)
        purge_frames = split_cfg.get("purge_frames", 0)

    print(f"Loading dataset from: {manifest}")
    print(f"  include_run_regex: {include_regex}")
    print(f"  exclude_run_regex: {exclude_regex}")

    dataset = TendonDatasetV2(
        manifest_csv=manifest,
        img_size=(img_size, img_size) if isinstance(img_size, int) else img_size,
        include_run_regex=include_regex,
        exclude_run_regex=exclude_regex,
        normalization="imagenet",
        temporal_frames=1,  # Just need the force data
    )

    print(f"Dataset size: {len(dataset)} frames")

    # Replicate the contiguous split to get training indices
    df = dataset.df
    train_idx = []

    for run_id, group in sorted(df.groupby("run_id"), key=lambda x: x[0]):
        sorted_group = group.sort_values("frame_idx")
        indices = sorted_group.index.tolist()
        n = len(indices)

        n_train = int(n * train_ratio)
        train_end = n_train
        train_idx.extend(indices[:train_end])

    print(f"Training frames: {len(train_idx)}")

    # Compute force stats from training frames only
    FORCE_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]
    force_data = df.iloc[train_idx][FORCE_COLS].values.astype(np.float32)

    mean = force_data.mean(axis=0)
    std = force_data.std(axis=0)
    std = np.maximum(std, 1e-8)

    print("\nForce z-score stats (training split):")
    for i, col in enumerate(FORCE_COLS):
        print(f"  {col}: mean={mean[i]:.6f}, std={std[i]:.6f}")

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "cols": FORCE_COLS,
        "n_train_frames": len(train_idx),
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / "force_stats.json"

    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved force_stats.json to: {out_path}")


if __name__ == "__main__":
    main()
