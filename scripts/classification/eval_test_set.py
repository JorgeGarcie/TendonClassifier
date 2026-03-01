"""Evaluate saved checkpoints on the test split.

Reconstructs the exact same contiguous frame split (seed=42) used during
training and evaluates best.pth on the test portion.

Usage:
    python eval_test_set.py --config configs/spatial_combined.yaml
    python eval_test_set.py --config configs/spatial_combined.yaml configs/spatial_image_only.yaml configs/spatial_force_only.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import load_config
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_v2 import split_frame_contiguous, collect_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True,
                   help="One or more YAML config paths")
    return p.parse_args()


def eval_config(config_path: str):
    print(f"\n{'='*60}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path("checkpoints") / config.experiment.name
    best_ckpt = ckpt_dir / "best.pth"
    if not best_ckpt.exists():
        print(f"ERROR: {best_ckpt} not found, skipping.")
        return

    # ---- Dataset (same as training) ----
    model_type = config.model.type
    temporal_frames = 1
    if model_type in ("temporal", "temporal_force"):
        temporal_frames = config.model.temporal.num_frames

    return_force_sequence = model_type == "temporal_force" or (
        model_type == "temporal" and config.model.use_force
    )

    dataset = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        temporal_frames=temporal_frames,
        return_force_sequence=return_force_sequence,
        normalization=config.data.normalization.type if config.data.normalization else "imagenet",
        exclude_run_regex=config.data.exclude_run_regex,
        include_run_regex=config.data.include_run_regex,
        subtraction_enabled=config.data.subtraction.enabled if config.data.subtraction else False,
    )
    print(f"Dataset: {len(dataset)} samples")

    # ---- Split (identical seed to training) ----
    split_cfg = config.training.split
    test_ratio = split_cfg.test_ratio
    val_ratio = config.training.val_ratio
    train_ratio = 1.0 - val_ratio - test_ratio
    purge_frames = (temporal_frames - 1) if temporal_frames > 1 else 0

    train_ds, _, test_ds = split_frame_contiguous(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=config.experiment.seed,
        purge_frames=purge_frames,
    )

    # Apply force z-score normalization from training split (identical to training)
    dataset.compute_force_stats(train_ds.indices)

    if test_ds is None or len(test_ds) == 0:
        print("No test split available.")
        return

    test_loader = DataLoader(
        test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=4
    )

    # ---- Model ----
    model = get_model_v2(config.model)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val macro-F1={ckpt.get('val_macro_f1', float('nan')):.4f}")

    # ---- Evaluate ----
    test_pred, test_true = collect_predictions(model, test_loader, device, config)

    class_names = ["none", "single", "crossed", "double"]

    macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    acc = np.mean(np.array(test_true) == np.array(test_pred))
    print(f"\nAccuracy:  {acc*100:.1f}%")
    print(f"Macro F1:  {macro_f1:.4f}")

    cm = confusion_matrix(test_true, test_pred)
    print("\nTest Confusion Matrix:")
    print(f"{'':>12} " + " ".join(f"{name:>10}" for name in class_names))
    for i, row_vals in enumerate(cm):
        print(f"{class_names[i]:>12} " + " ".join(f"{val:>10}" for val in row_vals))

    report = classification_report(test_true, test_pred, target_names=class_names, digits=4)
    print(f"\nTest Classification Report:\n{report}")

    # Save confusion matrix plot
    out_path = ckpt_dir / "confusion_test.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title=f"{config.experiment.name}\nTest Set — Macro F1={macro_f1:.3f}")
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    for cfg in args.config:
        eval_config(cfg)


if __name__ == "__main__":
    main()
