"""Evaluate a trained model on the test_frames folder.

test_frames/ has class-labeled subfolders (none, single, crossing, double)
with raw images — no manifest, no force data. This script:
1. Loads images from the folder structure
2. Subtracts a random 'none' image as reference (like first-frame subtraction)
3. Feeds zero force (model still needs the input, but force is uninformative here)
4. Reports overall + per-class metrics

Usage:
    python eval_test_frames.py --checkpoint checkpoints/sweep/<run_id>/best.pth
    python eval_test_frames.py --checkpoint checkpoints/sweep/<run_id>/best.pth --test-dir test_frames
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import load_config_from_dict
from models_v2 import get_model_v2
from train_utils import get_device

# Map folder names to class indices (matching training labels)
FOLDER_TO_LABEL = {
    "none": 0,
    "single": 1,
    "crossing": 2,
    "double": 3,
}
CLASS_NAMES = ["none", "single", "crossing", "double"]

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on test_frames folder")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-dir", type=str, default="test_frames",
                   help="Path to test_frames directory")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def load_image(path, img_size=(224, 224)):
    """Load image as float tensor (0-1), no normalization."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0


def main():
    args = parse_args()

    # Load checkpoint
    device = get_device()
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = load_config_from_dict(ckpt["config"])

    print(f"  Model type: {config.model.type}")
    print(f"  Fusion: {config.model.fusion.type}")
    print(f"  Subtraction: {config.data.subtraction.enabled}")

    # Build and load model
    model = get_model_v2(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_size = (config.data.img_size, config.data.img_size)
    subtraction = config.data.subtraction.enabled

    # Discover test images
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = Path(__file__).parent / test_dir

    images_info = []  # (path, label)
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder = test_dir / folder_name
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping")
            continue
        for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
            images_info.append((img_path, label))

    print(f"\nTest set: {len(images_info)} images")
    for name, label in FOLDER_TO_LABEL.items():
        n = sum(1 for _, l in images_info if l == label)
        print(f"  {name}: {n}")

    if not images_info:
        print("ERROR: No images found")
        return

    # Pick a reference image from 'none' for subtraction
    ref_img = None
    if subtraction:
        none_dir = test_dir / "none"
        ref_path = sorted(none_dir.glob("*.jpg"))[0]  # First none image
        ref_img = load_image(ref_path, img_size)
        print(f"\nSubtraction reference: {ref_path.name}")

    # Run inference in batches
    all_preds = []
    all_labels = []
    batch_imgs = []
    batch_labels = []

    # Zero force vector (no force data available)
    zero_force = torch.zeros(6, dtype=torch.float32)

    with torch.no_grad():
        for i, (img_path, label) in enumerate(images_info):
            img = load_image(img_path, img_size)

            if subtraction and ref_img is not None:
                img = img - ref_img

            # ImageNet normalize
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == args.batch_size or i == len(images_info) - 1:
                imgs_t = torch.stack(batch_imgs).to(device)
                forces_t = zero_force.unsqueeze(0).expand(len(batch_imgs), -1).to(device)

                # Forward (spatial combined model)
                if config.model.use_force:
                    output = model(imgs_t, forces_t)
                else:
                    output = model(imgs_t)

                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.extend(batch_labels)

                batch_imgs = []
                batch_labels = []

    all_preds = np.concatenate(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Results  ({len(all_preds)} samples)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10} " + " ".join(f"{n:>8}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>10} " + " ".join(f"{v:>8}" for v in row))

    report = classification_report(
        all_labels, all_preds, labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    # Save results
    save_dir = Path(args.save_dir) if args.save_dir else ckpt_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(
        all_labels, all_preds, labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    results = {
        "checkpoint": str(ckpt_path),
        "test_dir": str(test_dir),
        "subtraction": subtraction,
        "n_samples": int(len(all_preds)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in CLASS_NAMES
        },
    }
    out_path = save_dir / "test_frames_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
