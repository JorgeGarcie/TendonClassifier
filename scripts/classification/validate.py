"""Validate tendon classifier + depth regression.

Usage:
    # Single-label mode
    python validate.py --model combined
    python validate.py --model force --checkpoint checkpoints/force/best.pth

    # Multi-label mode
    python validate.py --model combined --multi-label
    python validate.py --model combined --multi-label --threshold 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import TendonDataset, PATTERN_NAMES, NUM_PATTERNS
from models import get_model


CLASS_NAMES = {i: name for i, name in enumerate(PATTERN_NAMES)}
NUM_CLASSES = NUM_PATTERNS


def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine multi-label mode from args or checkpoint
    multi_label = args.multi_label
    threshold = args.threshold

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        model_dir = f"{args.model}_multilabel" if multi_label else args.model
        ckpt_dir = Path(__file__).parent / "checkpoints" / model_dir
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

    # Check if checkpoint was trained with multi-label
    ckpt_multi_label = ckpt.get("multi_label", False)
    if ckpt_multi_label != multi_label:
        print(f"Warning: Checkpoint was trained with multi_label={ckpt_multi_label}, "
              f"but running with --multi-label={multi_label}")
        multi_label = ckpt_multi_label
    if "threshold" in ckpt and not args.threshold:
        threshold = ckpt["threshold"]

    mode_str = "multi-label" if multi_label else "single-label"
    print(f"Mode: {mode_str}")
    if multi_label:
        print(f"Threshold: {threshold}")

    model = get_model(args.model, multi_label=multi_label,
                      num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {ckpt_path}  (epoch {ckpt['epoch']})\n")
    model.eval()

    # Dataset
    dataset = TendonDataset(args.manifest,
                            img_size=(args.img_size, args.img_size),
                            multi_label=multi_label)

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
    all_probs = []  # For multi-label: store probabilities
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

            if multi_label:
                probs = torch.sigmoid(cls_logits)
                preds = (probs >= threshold).float()
                all_probs.append(probs.cpu())
            else:
                preds = cls_logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_depth_pred.append(depth_pred.cpu())
            all_depth_gt.append(depth_gt)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    depth_pred = torch.cat(all_depth_pred).numpy()
    depth_gt = torch.cat(all_depth_gt).numpy()

    if multi_label:
        probs = torch.cat(all_probs).numpy()
        validate_multi_label(preds, labels, probs, depth_pred, depth_gt, args, threshold)
    else:
        validate_single_label(preds, labels, depth_pred, depth_gt, args)


def validate_single_label(preds, labels, depth_pred, depth_gt, args):
    """Validate single-label (mutually exclusive) classification."""
    # Classification metrics
    accuracy = (preds == labels).mean()
    print(f"Overall accuracy: {accuracy:.4f}  ({(preds == labels).sum()}/{len(labels)})\n")

    print(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 46)
    for c in range(NUM_CLASSES):
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
    plot_results(preds, labels, depth_pred, depth_gt, args.model,
                 multi_label=False)


def validate_multi_label(preds, labels, probs, depth_pred, depth_gt, args, threshold):
    """Validate multi-label classification."""
    n_samples = len(labels)

    # Overall metrics
    exact_match = (preds == labels).all(axis=1).mean()
    per_pattern_acc = (preds == labels).mean()

    print(f"Multi-label metrics (n={n_samples}):")
    print(f"  Exact match accuracy: {exact_match:.4f}")
    print(f"  Per-pattern accuracy: {per_pattern_acc:.4f}")

    # Hamming loss (fraction of incorrect labels)
    hamming = (preds != labels).mean()
    print(f"  Hamming loss: {hamming:.4f}")

    # Per-pattern metrics
    print(f"\n{'Pattern':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 46)
    for c in range(NUM_CLASSES):
        tp = ((preds[:, c] == 1) & (labels[:, c] == 1)).sum()
        fp = ((preds[:, c] == 1) & (labels[:, c] == 0)).sum()
        fn = ((preds[:, c] == 0) & (labels[:, c] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = (labels[:, c] == 1).sum()
        print(f"{CLASS_NAMES[c]:<10} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {support:>8d}")

    # Co-occurrence analysis: which patterns appear together
    print("\nPattern co-occurrence (in ground truth):")
    for i in range(1, NUM_CLASSES):  # Skip "none"
        for j in range(i + 1, NUM_CLASSES):
            count = ((labels[:, i] == 1) & (labels[:, j] == 1)).sum()
            if count > 0:
                print(f"  {CLASS_NAMES[i]} + {CLASS_NAMES[j]}: {count} samples")

    # Depth metrics (where any pattern except "none" is present)
    mask = labels[:, 1:].sum(axis=1) > 0
    if mask.sum() > 0:
        errs = np.abs(depth_pred[mask] - depth_gt[mask])
        print(f"\nDepth regression (tendon-present samples only, n={mask.sum()}):")
        print(f"  MAE:    {errs.mean():.3f} mm")
        print(f"  Std:    {errs.std():.3f} mm")
        print(f"  Median: {np.median(errs):.3f} mm")
        print(f"  Max:    {errs.max():.3f} mm")

    print()
    plot_results(preds, labels, depth_pred, depth_gt, args.model,
                 multi_label=True, probs=probs, threshold=threshold)


def plot_results(preds, labels, depth_pred, depth_gt, model_name,
                 multi_label=False, probs=None, threshold=0.5):
    """Generate validation result plots.

    For single-label: 2x2 grid with confusion matrix, per-class metrics,
    depth scatter, and depth error histogram.

    For multi-label: 2x2 grid with per-pattern confusion matrices,
    per-pattern metrics, depth scatter, and probability distribution.
    """
    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    model_dir = f"{model_name}_multilabel" if multi_label else model_name

    if multi_label:
        _plot_multi_label_results(preds, labels, depth_pred, depth_gt,
                                  class_labels, model_dir, probs, threshold)
    else:
        _plot_single_label_results(preds, labels, depth_pred, depth_gt,
                                   class_labels, model_dir)


def _plot_single_label_results(preds, labels, depth_pred, depth_gt,
                               class_labels, model_dir):
    """Plot single-label validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Validation Results — {model_dir} (single-label)", fontsize=14)

    # --- (0,0) Confusion matrix ---
    ax = axes[0, 0]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for gt, pr in zip(labels, preds):
        cm[gt, pr] += 1
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=class_labels, yticklabels=class_labels,
           xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    # --- (0,1) Per-class precision / recall / F1 bar chart ---
    ax = axes[0, 1]
    precs, recs, f1s = [], [], []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precs.append(p); recs.append(r); f1s.append(f)
    x = np.arange(NUM_CLASSES)
    w = 0.25
    ax.bar(x - w, precs, w, label="Precision")
    ax.bar(x, recs, w, label="Recall")
    ax.bar(x + w, f1s, w, label="F1")
    ax.set(xticks=x, xticklabels=class_labels, ylim=(0, 1.05),
           ylabel="Score", title="Per-Class Metrics")
    ax.legend(fontsize=8)

    # --- (1,0) Depth scatter (predicted vs GT) ---
    ax = axes[1, 0]
    mask = labels > 0
    if mask.sum() > 0:
        ax.scatter(depth_gt[mask], depth_pred[mask], alpha=0.3, s=10,
                   edgecolors="none")
        lo = min(depth_gt[mask].min(), depth_pred[mask].min())
        hi = max(depth_gt[mask].max(), depth_pred[mask].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
        ax.set(xlabel="GT depth (mm)", ylabel="Predicted depth (mm)",
               title="Depth: Predicted vs GT")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No tendon-present samples", transform=ax.transAxes,
                ha="center")

    # --- (1,1) Depth error histogram ---
    ax = axes[1, 1]
    if mask.sum() > 0:
        errs = depth_pred[mask] - depth_gt[mask]
        ax.hist(errs, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set(xlabel="Depth error (mm)  [pred - GT]", ylabel="Count",
               title=f"Depth Error Distribution  (MAE={np.abs(errs).mean():.2f} mm)")
    else:
        ax.text(0.5, 0.5, "No tendon-present samples", transform=ax.transAxes,
                ha="center")

    plt.tight_layout()
    out_dir = Path(__file__).parent / "checkpoints" / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_results.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


def _plot_multi_label_results(preds, labels, depth_pred, depth_gt,
                              class_labels, model_dir, probs, threshold):
    """Plot multi-label validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Validation Results — {model_dir} (multi-label, threshold={threshold})",
                 fontsize=14)

    # --- (0,0) Per-pattern binary confusion matrices ---
    ax = axes[0, 0]
    # Create a grid showing TP/FP/FN/TN for each pattern
    metrics_grid = []
    for c in range(NUM_CLASSES):
        tp = ((preds[:, c] == 1) & (labels[:, c] == 1)).sum()
        fp = ((preds[:, c] == 1) & (labels[:, c] == 0)).sum()
        fn = ((preds[:, c] == 0) & (labels[:, c] == 1)).sum()
        tn = ((preds[:, c] == 0) & (labels[:, c] == 0)).sum()
        metrics_grid.append([tp, fp, fn, tn])
    metrics_grid = np.array(metrics_grid)
    im = ax.imshow(metrics_grid, cmap="Blues", aspect="auto")
    ax.set(xticks=range(4), yticks=range(NUM_CLASSES),
           xticklabels=["TP", "FP", "FN", "TN"], yticklabels=class_labels,
           title="Per-Pattern Confusion (rows=patterns)")
    for i in range(NUM_CLASSES):
        for j in range(4):
            color = "white" if metrics_grid[i, j] > metrics_grid.max() / 2 else "black"
            ax.text(j, i, str(metrics_grid[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- (0,1) Per-pattern precision / recall / F1 bar chart ---
    ax = axes[0, 1]
    precs_list, recs_list, f1s_list = [], [], []
    for c in range(NUM_CLASSES):
        tp = ((preds[:, c] == 1) & (labels[:, c] == 1)).sum()
        fp = ((preds[:, c] == 1) & (labels[:, c] == 0)).sum()
        fn = ((preds[:, c] == 0) & (labels[:, c] == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precs_list.append(p); recs_list.append(r); f1s_list.append(f)
    x = np.arange(NUM_CLASSES)
    w = 0.25
    ax.bar(x - w, precs_list, w, label="Precision")
    ax.bar(x, recs_list, w, label="Recall")
    ax.bar(x + w, f1s_list, w, label="F1")
    ax.set(xticks=x, xticklabels=class_labels, ylim=(0, 1.05),
           ylabel="Score", title="Per-Pattern Metrics")
    ax.legend(fontsize=8)

    # --- (1,0) Depth scatter (predicted vs GT) ---
    ax = axes[1, 0]
    # Tendon present = any pattern except "none"
    mask = labels[:, 1:].sum(axis=1) > 0
    if mask.sum() > 0:
        ax.scatter(depth_gt[mask], depth_pred[mask], alpha=0.3, s=10,
                   edgecolors="none")
        lo = min(depth_gt[mask].min(), depth_pred[mask].min())
        hi = max(depth_gt[mask].max(), depth_pred[mask].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
        ax.set(xlabel="GT depth (mm)", ylabel="Predicted depth (mm)",
               title="Depth: Predicted vs GT")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No tendon-present samples", transform=ax.transAxes,
                ha="center")

    # --- (1,1) Probability distribution per pattern ---
    ax = axes[1, 1]
    if probs is not None:
        for c in range(1, NUM_CLASSES):  # Skip "none" for clarity
            # Separate positives and negatives
            pos_probs = probs[labels[:, c] == 1, c]
            neg_probs = probs[labels[:, c] == 0, c]
            if len(pos_probs) > 0:
                ax.hist(pos_probs, bins=20, alpha=0.5, label=f"{class_labels[c]}+ (n={len(pos_probs)})")
        ax.axvline(threshold, color="r", linestyle="--", linewidth=2, label=f"threshold={threshold}")
        ax.set(xlabel="Predicted probability", ylabel="Count",
               title="Probability Distribution (positive samples)")
        ax.legend(fontsize=7, loc="upper left")
    else:
        ax.text(0.5, 0.5, "Probabilities not available", transform=ax.transAxes,
                ha="center")

    plt.tight_layout()
    out_dir = Path(__file__).parent / "checkpoints" / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_results.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


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
    # Multi-label options
    p.add_argument("--multi-label", action="store_true",
                   help="Enable multi-label mode (detect multiple patterns per image)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Prediction threshold for multi-label mode (default 0.5)")
    args = p.parse_args()
    validate(args)
