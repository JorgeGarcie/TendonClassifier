"""Run spatial inference on a directory of images.

Loads one or more spatial models and runs every image in the input
directory through them, printing predictions + confidence.

Usage:
    cd scripts/classification
    python run_inference.py                          # all models, test_frames/
    python run_inference.py --model spatial_image_only
    python run_inference.py --frames /path/to/dir
    python run_inference.py --save results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from models_v2 import get_model_v2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_LABELS  = ["none", "single", "crossed", "double"]

MODELS = {
    "spatial_image_only": "configs/spatial_image_only.yaml",
    "spatial_combined":   "configs/spatial_combined.yaml",
    "spatial_force_only": "configs/spatial_force_only.yaml",
}
CHECKPOINTS = {
    k: f"checkpoints/{k}/best.pth" for k in MODELS
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="test_frames",
                   help="Directory containing images to classify")
    p.add_argument("--model", default=None,
                   choices=list(MODELS.keys()),
                   help="Which model to use (default: all spatial models)")
    p.add_argument("--force", nargs=6, type=float,
                   default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   metavar=("fx", "fy", "fz", "tx", "ty", "tz"),
                   help="Force vector for combined/force models (default: zeros)")
    p.add_argument("--stats", default="force_stats.json",
                   help="Path to force_stats.json for z-score normalization")
    p.add_argument("--save", default=None,
                   help="Save results to CSV file")
    return p.parse_args()


def load_force_stats(path):
    import json
    with open(path) as f:
        s = json.load(f)
    mean = torch.tensor(s["mean"], dtype=torch.float32)
    std  = torch.tensor(s["std"],  dtype=torch.float32)
    return mean, std


def preprocess_image(bgr, device):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)  # (1, 3, 224, 224)


def load_model(model_name, device):
    cfg = load_config(MODELS[model_name])
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg.get("model", {})
    model = get_model_v2(model_cfg).to(device)
    ckpt = torch.load(CHECKPOINTS[model_name], map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def run_model(model, model_name, img_tensor, force_raw, force_mean, force_std, device):
    with torch.no_grad():
        if model_name == "spatial_image_only":
            out = model(img_tensor)
        elif model_name == "spatial_force_only":
            f = torch.tensor(force_raw, dtype=torch.float32).to(device)
            f = ((f - force_mean.to(device)) / force_std.to(device)).unsqueeze(0)
            out = model(f)
        else:  # spatial_combined
            f = torch.tensor(force_raw, dtype=torch.float32).to(device)
            f = ((f - force_mean.to(device)) / force_std.to(device)).unsqueeze(0)
            out = model(img_tensor, f)

        logits = out[0] if isinstance(out, (tuple, list)) else out

    probs = F.softmax(logits, dim=-1)[0]
    class_id = int(probs.argmax())
    return CLASS_LABELS[class_id], class_id, float(probs[class_id]), probs.cpu().tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        sys.exit(1)

    images = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[ERROR] No images found in {frames_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) in {frames_dir}")

    # Load force stats
    force_mean, force_std = load_force_stats(args.stats)

    # Which models to run
    model_names = [args.model] if args.model else list(MODELS.keys())

    # Load models
    models = {}
    for name in model_names:
        print(f"Loading {name}...")
        models[name] = load_model(name, device)
    print()

    # Run inference
    rows = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Could not read {img_path.name}, skipping")
            continue

        img_tensor = preprocess_image(bgr, device)
        row = {"image": img_path.name}

        print(f"{img_path.name}")
        for name, model in models.items():
            label, class_id, conf, probs = run_model(
                model, name, img_tensor, args.force, force_mean, force_std, device
            )
            print(f"  {name:25s}  {label:8s}  conf={conf:.3f}  "
                  f"probs=[{', '.join(f'{p:.3f}' for p in probs)}]")
            row[f"{name}_label"]      = label
            row[f"{name}_confidence"] = f"{conf:.4f}"
        rows.append(row)
        print()

    # Save CSV
    if args.save and rows:
        out = Path(args.save)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
