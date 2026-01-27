"""Validate tendon classifier + depth regression.

Usage:
    python validate.py --model combined
    python validate.py --model force --checkpoint checkpoints/force/best.pth
    python validate.py --model image --test-phantoms p3 p4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import TendonDataset
from models import get_model


CLASS_NAMES = {0: "none", 1: "single", 2: "crossed"}


def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = get_model(args.model).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(__file__).parent / "checkpoints" / args.model
        best = ckpt_dir / "best.pth"
        if best.exists():
            ckpt_path = best
        else:
            candidates = sorted(ckpt_dir.glob("epoch_*.pth"))
            if not candidates:
                print(f"No checkpoints in {ckpt_dir}")
                return
            ckpt_path = candidates[-1]

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {ckpt_path}  (epoch {ckpt['epoch']})\n")
    model.eval()

    # Dataset
    dataset = TendonDataset(args.manifest,
                            img_size=(args.img_size, args.img_size))

    # Filter
    indices = list(range(len(dataset)))
    if args.test_phantoms:
        indices = [i for i in indices
                   if dataset.df.iloc[i]["phantom_type"] in args.test_phantoms]
        print(f"Filtering to phantoms {args.test_phantoms}: {len(indices)} samples")

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []
    all_depth_pred = []
    all_depth_gt = []

    with torch.no_grad():
        for imgs, forces, labels, depth_gt in loader:
            imgs, forces = imgs.to(device), forces.to(device)

            if args.model == "force":
                cls_logits, depth_pred = model(forces)
            elif args.model == "image":
                cls_logits, depth_pred = model(imgs)
            else:
                cls_logits, depth_pred = model(imgs, forces)

            all_preds.append(cls_logits.argmax(dim=1).cpu())
            all_labels.append(labels)
            all_depth_pred.append(depth_pred.cpu())
            all_depth_gt.append(depth_gt)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    depth_pred = torch.cat(all_depth_pred).numpy()
    depth_gt = torch.cat(all_depth_gt).numpy()

    # Classification metrics
    accuracy = (preds == labels).mean()
    print(f"Overall accuracy: {accuracy:.4f}  ({(preds == labels).sum()}/{len(labels)})\n")

    print(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 46)
    for c in range(3):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = (labels == c).sum()
        print(f"{CLASS_NAMES[c]:<10} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {support:>8d}")

    # Depth metrics (only where tendon is present)
    mask = labels > 0
    if mask.sum() > 0:
        errs = np.abs(depth_pred[mask] - depth_gt[mask])
        print(f"\nDepth regression (tendon-present samples only, n={mask.sum()}):")
        print(f"  MAE:    {errs.mean():.3f} mm")
        print(f"  Std:    {errs.std():.3f} mm")
        print(f"  Median: {np.median(errs):.3f} mm")
        print(f"  Max:    {errs.max():.3f} mm")

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate tendon classifier + depth")
    p.add_argument("--model", choices=["force", "image", "combined"],
                   default="combined")
    p.add_argument("--manifest", type=str,
                   default="../labeling/output/gt_dataset/gt_manifest.csv")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--test-phantoms", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()
    validate(args)
