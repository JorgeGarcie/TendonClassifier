"""Train tendon classifier + depth regressor (force-only, image-only, or combined).

Usage:
    # Single-label mode (mutually exclusive classes)
    python train.py --model combined

    # Multi-label mode (multiple patterns per image)
    python train.py --model combined --multi-label
    python train.py --model combined --multi-label --threshold 0.5
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import TendonDataset, NUM_PATTERNS, PATTERN_NAMES
from models import get_model
from train_utils import get_device, save_learning_curve


def parse_args():
    p = argparse.ArgumentParser(description="Train tendon classifier + depth")
    p.add_argument("--model", choices=["force", "image", "combined"],
                   default="combined")
    p.add_argument("--manifest", type=str,
                   default="../labeling/output/gt_dataset/gt_manifest.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--depth-weight", type=float, default=0.1,
                   help="Weight for depth regression loss (lambda)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--exclude-phantoms", nargs="*", default=None)
    p.add_argument("--img-size", type=int, default=224)
    # Multi-label options
    p.add_argument("--multi-label", action="store_true",
                   help="Enable multi-label mode (detect multiple patterns per image)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Prediction threshold for multi-label mode (default 0.5)")
    return p.parse_args()


def split_by_run(dataset, val_ratio=0.2, seed=42):
    """Split dataset by run_id so all frames from a run stay in the same set.

    Runs are randomly shuffled (seeded) to ensure each split sees a mix of
    phantom types rather than grouping by name.
    """
    import random
    df = dataset.df
    run_ids = list(df["run_id"].unique())
    random.Random(seed).shuffle(run_ids)

    n_val = max(1, int(len(run_ids) * val_ratio))
    n_train = len(run_ids) - n_val

    train_runs = set(run_ids[:n_train])
    val_runs = set(run_ids[n_train:])

    train_idx = df.index[df["run_id"].isin(train_runs)].tolist()
    val_idx = df.index[df["run_id"].isin(val_runs)].tolist()

    print(f"Run-level split (seed={seed}):")
    print(f"  Train ({n_train} runs, {len(train_idx)} frames): {sorted(train_runs)}")
    print(f"  Val   ({n_val} runs, {len(val_idx)} frames): {sorted(val_runs)}")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def forward_model(model, model_name, imgs, forces):
    if model_name == "force":
        return model(forces)
    elif model_name == "image":
        return model(imgs)
    else:
        return model(imgs, forces)


def compute_loss(cls_logits, depth_pred, labels, depth_gt, depth_weight,
                 multi_label=False):
    """Compute classification + depth regression loss.

    For single-label: CE loss (mutually exclusive classes)
    For multi-label: BCE loss (independent pattern predictions)

    Depth loss is MSE only where tendon is present.
    """
    if multi_label:
        # Multi-label: BCE with logits for independent pattern predictions
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels)
        # Depth loss on samples where any pattern (except "none") is present
        # labels[:, 0] is "none", so presence means labels[:, 1:].sum(dim=1) > 0
        mask = labels[:, 1:].sum(dim=1) > 0
    else:
        # Single-label: Cross-entropy for mutually exclusive classes
        cls_loss = F.cross_entropy(cls_logits, labels)
        # Depth loss on samples where tendon is present (label > 0)
        mask = labels > 0

    if mask.sum() > 0:
        depth_loss = F.mse_loss(depth_pred[mask], depth_gt[mask])
    else:
        depth_loss = torch.tensor(0.0, device=cls_loss.device)

    return cls_loss + depth_weight * depth_loss, cls_loss, depth_loss


def run_epoch(model, loader, optimizer, device, model_name, depth_weight,
              train=True, multi_label=False, threshold=0.5):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_depth_loss = 0.0
    correct = 0
    total = 0
    depth_abs_err = 0.0
    depth_count = 0

    # Multi-label metrics
    ml_correct_patterns = 0  # Total correct pattern predictions
    ml_total_patterns = 0    # Total pattern predictions

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for imgs, forces, labels, depth_gt in loader:
            imgs = imgs.to(device)
            forces = forces.to(device)
            labels = labels.to(device)
            depth_gt = depth_gt.to(device)

            cls_logits, depth_pred = forward_model(model, model_name,
                                                   imgs, forces)
            loss, cls_loss, depth_loss = compute_loss(
                cls_logits, depth_pred, labels, depth_gt, depth_weight,
                multi_label=multi_label
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_depth_loss += depth_loss.item() * bs

            if multi_label:
                # Multi-label: threshold-based predictions
                preds = (torch.sigmoid(cls_logits) >= threshold).float()
                # Exact match accuracy (all patterns correct for a sample)
                correct += (preds == labels).all(dim=1).sum().item()
                # Per-pattern accuracy
                ml_correct_patterns += (preds == labels).sum().item()
                ml_total_patterns += labels.numel()
                # Depth mask: any pattern except "none" present
                mask = labels[:, 1:].sum(dim=1) > 0
            else:
                # Single-label: argmax predictions
                preds = cls_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                mask = labels > 0

            total += bs

            # Depth MAE on present samples
            if mask.sum() > 0:
                depth_abs_err += (depth_pred[mask] - depth_gt[mask]).abs().sum().item()
                depth_count += mask.sum().item()

    n = total
    depth_mae = depth_abs_err / depth_count if depth_count > 0 else 0.0
    acc = correct / n

    # For multi-label, also compute per-pattern accuracy
    if multi_label and ml_total_patterns > 0:
        pattern_acc = ml_correct_patterns / ml_total_patterns
        # Return pattern_acc as the primary metric (more meaningful than exact match)
        return (total_loss / n, total_cls_loss / n, total_depth_loss / n,
                pattern_acc, depth_mae)

    return (total_loss / n, total_cls_loss / n, total_depth_loss / n,
            acc, depth_mae)


def main():
    args = parse_args()
    device = get_device()

    mode_str = "multi-label" if args.multi_label else "single-label"
    print(f"Training mode: {mode_str}")
    if args.multi_label:
        print(f"  Patterns: {PATTERN_NAMES}")
        print(f"  Threshold: {args.threshold}")

    # Data
    dataset = TendonDataset(
        args.manifest,
        img_size=(args.img_size, args.img_size),
        exclude_phantom_types=args.exclude_phantoms,
        multi_label=args.multi_label,
    )
    print(f"Dataset: {len(dataset)} samples")

    train_ds, val_ds = split_by_run(dataset, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = get_model(args.model, multi_label=args.multi_label,
                      num_classes=NUM_PATTERNS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Checkpoint setup - separate directory for multi-label models
    model_dir = f"{args.model}_multilabel" if args.multi_label else args.model
    ckpt_dir = Path(__file__).parent / "checkpoints" / model_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config = vars(args).copy()
    config["num_patterns"] = NUM_PATTERNS
    config["pattern_names"] = PATTERN_NAMES

    csv_path = ckpt_dir / "training_log.csv"
    acc_col = "pattern_acc" if args.multi_label else "acc"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_cls_loss", "train_depth_loss",
            f"train_{acc_col}", "train_depth_mae",
            "val_loss", "val_cls_loss", "val_depth_loss",
            f"val_{acc_col}", "val_depth_mae",
        ])

    # Resume if checkpoint exists
    start_epoch = 0
    existing = sorted(ckpt_dir.glob("epoch_*.pth"))
    if existing:
        ckpt = torch.load(existing[-1], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Train
    history = {k: [] for k in ["train_loss", "val_loss",
                                "train_acc", "val_acc",
                                "train_depth_mae", "val_depth_mae"]}
    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        t_loss, t_cls, t_dep, t_acc, t_dmae = run_epoch(
            model, train_loader, optimizer, device,
            args.model, args.depth_weight, train=True,
            multi_label=args.multi_label, threshold=args.threshold,
        )
        v_loss, v_cls, v_dep, v_acc, v_dmae = run_epoch(
            model, val_loader, optimizer, device,
            args.model, args.depth_weight, train=False,
            multi_label=args.multi_label, threshold=args.threshold,
        )

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_depth_mae"].append(t_dmae)
        history["val_depth_mae"].append(v_dmae)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={t_loss:.4f}/{v_loss:.4f}  "
              f"{acc_col}={t_acc:.4f}/{v_acc:.4f}  "
              f"depth_mae={t_dmae:.3f}/{v_dmae:.3f} mm")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, t_loss, t_cls, t_dep, t_acc, t_dmae,
                v_loss, v_cls, v_dep, v_acc, v_dmae,
            ])

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "multi_label": args.multi_label,
            "threshold": args.threshold,
            "num_patterns": NUM_PATTERNS,
        }
        torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(state, ckpt_dir / "best.pth")

    curve_prefix = f"{args.model}_multilabel" if args.multi_label else args.model
    save_learning_curve(
        {"train_loss": history["train_loss"],
         "val_loss": history["val_loss"],
         f"train_{acc_col}": history["train_acc"],
         f"val_{acc_col}": history["val_acc"]},
        filename=f"{curve_prefix}_learning_curve.png",
    )
    save_learning_curve(
        {"train_depth_mae": history["train_depth_mae"],
         "val_depth_mae": history["val_depth_mae"]},
        filename=f"{curve_prefix}_depth_curve.png",
    )
    print(f"Done. Best val {acc_col}: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
