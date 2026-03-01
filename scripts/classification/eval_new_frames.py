"""Evaluate spatial models on new test frames organised by class folder.

Expected layout:
    test_frames/
    ├── none/
    ├── single/
    ├── crossing/     (maps to class 2 "crossed")
    └── double/

Preprocessing: resize shortest side to 256, center crop 224×224.
NOTE: training used direct resize to 224×224, so there may be a small
distribution shift. If numbers look unexpectedly low, try --no-centercrop.

Outputs (all saved to --out-dir, default test_frames_results/):
    results.csv            per-image predictions for every model
    confusion_<model>.png  confusion matrix
    confidence_<model>.png confidence histogram per class
    failures_<model>.png   grid of worst-k misclassified images

Usage:
    cd scripts/classification
    python eval_new_frames.py
    python eval_new_frames.py --model spatial_image_only
    python eval_new_frames.py --no-centercrop   # use resize like training
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from models_v2 import get_model_v2

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_LABELS  = ["none", "single", "crossed", "double"]

# folder name → class id
FOLDER_TO_CLASS = {
    "none":     0,
    "single":   1,
    "crossing": 2,
    "crossed":  2,
    "double":   3,
}

MODELS = {
    "spatial_image_only": "configs/spatial_image_only.yaml",
    "spatial_combined":   "configs/spatial_combined.yaml",
    "spatial_force_only": "configs/spatial_force_only.yaml",
}
CHECKPOINTS = {k: f"checkpoints/{k}/best.pth" for k in MODELS}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames",        default="test_frames")
    p.add_argument("--model",         default=None, choices=list(MODELS.keys()))
    p.add_argument("--out-dir",       default="test_frames_results")
    p.add_argument("--stats",         default="force_stats.json")
    p.add_argument("--worst-k",       type=int, default=16,
                   help="Number of failure images to save in the failure grid")
    return p.parse_args()


# ── Preprocessing ──────────────────────────────────────────────────────────────
CROP_SIZE = 1080  # matches labeling pipeline config.py CROP_SIZE


def center_crop(bgr: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Center crop to size×size — matches generate_gt.py exactly."""
    h, w = bgr.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    return bgr[y1:y2, x1:x2]


def preprocess(bgr: np.ndarray, device) -> torch.Tensor:
    """Match training preprocessing: center crop 1080×1080, resize to 224×224."""
    bgr = center_crop(bgr, CROP_SIZE)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(name, device):
    cfg = load_config(MODELS[name])
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg.get("model", {})
    model = get_model_v2(model_cfg).to(device)
    ckpt = torch.load(CHECKPOINTS[name], map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_force_stats(path):
    with open(path) as f:
        s = json.load(f)
    return (torch.tensor(s["mean"], dtype=torch.float32),
            torch.tensor(s["std"],  dtype=torch.float32))


# ── Inference ──────────────────────────────────────────────────────────────────
def run_model(model, name, img_tensor, force_mean, force_std, device):
    zeros = torch.zeros(1, 6, device=device)
    with torch.no_grad():
        if name == "spatial_image_only":
            out = model(img_tensor)
        elif name == "spatial_force_only":
            f = ((zeros - force_mean.to(device)) / force_std.to(device))
            out = model(f)
        else:
            f = ((zeros - force_mean.to(device)) / force_std.to(device))
            out = model(img_tensor, f)
        logits = out[0] if isinstance(out, (tuple, list)) else out
    probs = F.softmax(logits, dim=-1)[0].cpu()
    class_id = int(probs.argmax())
    return class_id, float(probs[class_id]), probs.tolist()


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(4), yticks=range(4),
           xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
           xlabel="Predicted", ylabel="True",
           title=f"{model_name}\nConfusion Matrix")
    thresh = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confidence_histograms(records, model_name, out_path):
    """One confidence histogram per true class."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(f"{model_name} — Confidence Histograms")
    for cls_id, ax in enumerate(axes):
        confs = [r["confidence"] for r in records if r["true_class"] == cls_id]
        correct = [r["confidence"] for r in records
                   if r["true_class"] == cls_id and r["pred_class"] == cls_id]
        wrong = [r["confidence"] for r in records
                 if r["true_class"] == cls_id and r["pred_class"] != cls_id]
        ax.hist(correct, bins=20, range=(0, 1), color="steelblue",
                alpha=0.7, label=f"correct ({len(correct)})")
        ax.hist(wrong,   bins=20, range=(0, 1), color="tomato",
                alpha=0.7, label=f"wrong ({len(wrong)})")
        ax.set_title(CLASS_LABELS[cls_id])
        ax.set_xlabel("Confidence")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_failure_grid(failures, model_name, out_path, k=16):
    """Grid of worst-k failures (lowest confidence correct / most confused)."""
    # Sort by confidence descending (model was most confident but wrong)
    failures = sorted(failures, key=lambda r: -r["confidence"])[:k]
    if not failures:
        return

    cols = 4
    rows = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, r in zip(axes, failures):
        img = cv2.imread(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(
            f"GT:{CLASS_LABELS[r['true_class']]}\n"
            f"Pred:{CLASS_LABELS[r['pred_class']]} ({r['confidence']:.2f})",
            fontsize=7
        )
        ax.axis("off")

    fig.suptitle(f"{model_name} — Top Failures (most confident wrong)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_dir = Path(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing: center crop {CROP_SIZE}×{CROP_SIZE} → resize 224×224 (matches training)")

    # Discover images
    samples = []  # (path, true_class_id)
    for folder, cls_id in FOLDER_TO_CLASS.items():
        d = frames_dir / folder
        if not d.exists():
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in imgs:
            samples.append((p, cls_id))
    if not samples:
        print(f"[ERROR] No images found under {frames_dir}")
        sys.exit(1)
    print(f"Found {len(samples)} images across "
          f"{len(set(c for _, c in samples))} classes\n")

    force_mean, force_std = load_force_stats(args.stats)
    model_names = [args.model] if args.model else list(MODELS.keys())

    # Load models
    models = {}
    for name in model_names:
        print(f"Loading {name}...")
        models[name] = load_model(name, device)
    print()

    # Run inference + collect records
    all_records = defaultdict(list)  # model_name → list of dicts
    csv_rows = []

    for img_path, true_cls in samples:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        img_tensor = preprocess(bgr, device)
        row = {"image": img_path.name, "folder": img_path.parent.name,
               "true_class": true_cls, "true_label": CLASS_LABELS[true_cls]}

        for name, model in models.items():
            pred_cls, conf, probs = run_model(
                model, name, img_tensor, force_mean, force_std, device
            )
            all_records[name].append({
                "path": str(img_path),
                "true_class": true_cls,
                "pred_class": pred_cls,
                "confidence": conf,
                "probs": probs,
            })
            row[f"{name}_pred"]       = CLASS_LABELS[pred_cls]
            row[f"{name}_confidence"] = f"{conf:.4f}"
        csv_rows.append(row)

    # Save CSV
    if csv_rows:
        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved: {csv_path}\n")

    # Per-model metrics + plots
    for name, records in all_records.items():
        y_true = [r["true_class"] for r in records]
        y_pred = [r["pred_class"] for r in records]

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        print(f"{'='*55}")
        print(f" {name}")
        print(f"{'='*55}")
        print(f"  Accuracy:   {acc*100:.1f}%")
        print(f"  Macro F1:   {macro_f1:.3f}")
        print()
        print(classification_report(
            y_true, y_pred,
            target_names=CLASS_LABELS,
            zero_division=0,
            digits=3,
        ))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        plot_confusion_matrix(cm, name, out_dir / f"confusion_{name}.png")
        plot_confidence_histograms(records, name, out_dir / f"confidence_{name}.png")

        failures = [r for r in records if r["true_class"] != r["pred_class"]]
        plot_failure_grid(failures, name, out_dir / f"failures_{name}.png",
                          k=args.worst_k)
        print()

    print(f"\nAll results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
