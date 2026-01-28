"""Train tendon classifier + depth regressor (force-only, image-only, or combined).

Usage:
    python train.py --model force
    python train.py --model image
    python train.py --model combined
    python train.py --model combined --depth-weight 0.1
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import TendonDataset
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

    print(f"Run-level split: {n_train} train runs ({len(train_idx)} frames), "
          f"{n_val} val runs ({len(val_idx)} frames)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def forward_model(model, model_name, imgs, forces):
    if model_name == "force":
        return model(forces)
    elif model_name == "image":
        return model(imgs)
    else:
        return model(imgs, forces)


def compute_loss(cls_logits, depth_pred, labels, depth_gt, depth_weight):
    """CE for classification + masked MSE for depth (only where presence==1)."""
    cls_loss = F.cross_entropy(cls_logits, labels)

    # Depth loss only on samples that have a tendon present
    mask = labels > 0  # type 1 or 2 means tendon is present
    if mask.sum() > 0:
        depth_loss = F.mse_loss(depth_pred[mask], depth_gt[mask])
    else:
        depth_loss = torch.tensor(0.0, device=cls_loss.device)

    return cls_loss + depth_weight * depth_loss, cls_loss, depth_loss


def run_epoch(model, loader, optimizer, device, model_name, depth_weight,
              train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_depth_loss = 0.0
    correct = 0
    total = 0
    depth_abs_err = 0.0
    depth_count = 0

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
                cls_logits, depth_pred, labels, depth_gt, depth_weight
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_depth_loss += depth_loss.item() * bs
            preds = cls_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            # Depth MAE on present samples
            mask = labels > 0
            if mask.sum() > 0:
                depth_abs_err += (depth_pred[mask] - depth_gt[mask]).abs().sum().item()
                depth_count += mask.sum().item()

    n = total
    depth_mae = depth_abs_err / depth_count if depth_count > 0 else 0.0
    return (total_loss / n, total_cls_loss / n, total_depth_loss / n,
            correct / n, depth_mae)


def main():
    args = parse_args()
    device = get_device()

    # Data
    dataset = TendonDataset(
        args.manifest,
        img_size=(args.img_size, args.img_size),
        exclude_phantom_types=args.exclude_phantoms,
    )
    print(f"Dataset: {len(dataset)} samples")

    train_ds, val_ds = split_by_run(dataset, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Checkpoint setup
    ckpt_dir = Path(__file__).parent / "checkpoints" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ckpt_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_cls_loss", "train_depth_loss",
            "train_acc", "train_depth_mae",
            "val_loss", "val_cls_loss", "val_depth_loss",
            "val_acc", "val_depth_mae",
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
        )
        v_loss, v_cls, v_dep, v_acc, v_dmae = run_epoch(
            model, val_loader, optimizer, device,
            args.model, args.depth_weight, train=False,
        )

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_depth_mae"].append(t_dmae)
        history["val_depth_mae"].append(v_dmae)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={t_loss:.4f}/{v_loss:.4f}  "
              f"acc={t_acc:.4f}/{v_acc:.4f}  "
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
        }
        torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(state, ckpt_dir / "best.pth")

    save_learning_curve(
        {"train_loss": history["train_loss"],
         "val_loss": history["val_loss"],
         "train_acc": history["train_acc"],
         "val_acc": history["val_acc"]},
        filename=f"{args.model}_learning_curve.png",
    )
    save_learning_curve(
        {"train_depth_mae": history["train_depth_mae"],
         "val_depth_mae": history["val_depth_mae"]},
        filename=f"{args.model}_depth_curve.png",
    )
    print(f"Done. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
