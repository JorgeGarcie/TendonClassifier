"""Training script for TendonClassifier v2.

Supports:
- YAML configuration
- Multiple vision encoders (ResNet, DinoV2, CLIP)
- Spatial and temporal models
- Attention-based fusion
- Wandb logging
- Class-weighted loss
- Enhanced checkpointing (best + last + every N epochs)

Usage:
    python train_v2.py --config configs/default.yaml
    python train_v2.py --config configs/spatial_dino.yaml
    python train_v2.py --config configs/temporal_dino.yaml
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

from config import load_config, config_to_dict
from dataset import TendonDatasetV2
from models_v2 import get_model_v2, count_parameters, verify_frozen_encoder
from wandb_logger import create_logger
from train_utils import get_device, save_learning_curve


def parse_args():
    p = argparse.ArgumentParser(description="Train TendonClassifier v2")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--override", nargs="*", default=[],
                   help="Override config values (key=value pairs)")
    return p.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_run(dataset, val_ratio=0.2, seed=42):
    """Split dataset by run_id so all frames from a run stay in the same set."""
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
    print(f"  Train: {n_train} runs, {len(train_idx)} frames")
    print(f"  Val:   {n_val} runs, {len(val_idx)} frames")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_class_weights(dataset, num_classes: int, method: str = "balanced"):
    """Compute class weights for imbalanced dataset.

    Args:
        dataset: TendonDatasetV2 instance
        num_classes: Number of classes
        method: "balanced" (inverse frequency), "none", or list of weights

    Returns:
        Tensor of class weights or None
    """
    if method == "none" or method is None:
        return None

    if isinstance(method, list):
        return torch.tensor(method, dtype=torch.float32)

    # "balanced" - inverse frequency
    return dataset.get_class_weights(num_classes)


def compute_loss(cls_logits, depth_pred, labels, depth_gt, depth_weight,
                 class_weights=None, device=None):
    """Compute combined classification and depth loss.

    Args:
        cls_logits: Classification logits (B, num_classes)
        depth_pred: Depth predictions (B,)
        labels: Ground truth labels (B,)
        depth_gt: Ground truth depths (B,)
        depth_weight: Weight for depth loss
        class_weights: Optional class weights for CE loss
        device: Device for class weights

    Returns:
        (total_loss, cls_loss, depth_loss) tuple
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
        cls_loss = F.cross_entropy(cls_logits, labels, weight=class_weights)
    else:
        cls_loss = F.cross_entropy(cls_logits, labels)

    # Depth loss only on samples with tendon present
    mask = labels > 0
    if mask.sum() > 0:
        depth_loss = F.mse_loss(depth_pred[mask], depth_gt[mask])
    else:
        depth_loss = torch.tensor(0.0, device=device)

    total_loss = cls_loss + depth_weight * depth_loss
    return total_loss, cls_loss, depth_loss


def run_epoch(model, loader, optimizer, device, config, class_weights=None,
              train=True):
    """Run one epoch of training or validation.

    Args:
        model: Model to train/evaluate
        loader: DataLoader
        optimizer: Optimizer (only used if train=True)
        device: Device
        config: Config object
        class_weights: Optional class weights
        train: Whether this is a training epoch

    Returns:
        Dict of metrics
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_depth_loss = 0.0
    correct = 0
    total = 0
    depth_abs_err = 0.0
    depth_count = 0

    model_type = config.model.type
    use_force = config.model.use_force
    depth_weight = config.training.loss.depth_weight

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            # Handle spatial, temporal, and temporal_force modes
            if model_type in ("temporal", "temporal_force"):
                images, forces, labels, depth_gt, mask = batch
                if model_type == "temporal":
                    images = images.to(device)
                mask = mask.to(device)
            else:
                images, forces, labels, depth_gt = batch
                images = images.to(device)
                mask = None

            forces = forces.to(device)
            labels = labels.to(device)
            depth_gt = depth_gt.to(device)

            # Forward pass
            if model_type == "temporal_force":
                # Force-only temporal model: forces is (B, T, 6)
                output = model(forces, mask)
            elif model_type == "temporal":
                if use_force:
                    output = model(images, forces, mask)
                else:
                    output = model(images, mask=mask)
            elif model_type == "spatial_force":
                # Force-only spatial model
                output = model(forces)
            else:
                if use_force:
                    output = model(images, forces)
                else:
                    output = model(images)

            # Handle output format
            if isinstance(output, tuple):
                cls_logits, depth_pred = output
            else:
                cls_logits = output
                depth_pred = torch.zeros(labels.size(0), device=device)

            # Compute loss
            loss, cls_loss, depth_loss = compute_loss(
                cls_logits, depth_pred, labels, depth_gt, depth_weight,
                class_weights, device
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate metrics
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls_loss += cls_loss.item() * bs
            total_depth_loss += depth_loss.item() * bs
            preds = cls_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            # Depth MAE on present samples
            depth_mask = labels > 0
            if depth_mask.sum() > 0:
                depth_abs_err += (depth_pred[depth_mask] - depth_gt[depth_mask]).abs().sum().item()
                depth_count += depth_mask.sum().item()

    n = total
    depth_mae = depth_abs_err / depth_count if depth_count > 0 else 0.0

    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "depth_loss": total_depth_loss / n,
        "acc": correct / n,
        "depth_mae": depth_mae,
    }


def collect_predictions(model, loader, device, config):
    """Collect all predictions and labels from a dataloader.

    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device
        config: Config object

    Returns:
        (all_preds, all_labels) numpy arrays
    """
    model.eval()
    all_preds = []
    all_labels = []

    model_type = config.model.type
    use_force = config.model.use_force

    with torch.no_grad():
        for batch in loader:
            if model_type in ("temporal", "temporal_force"):
                images, forces, labels, depth_gt, mask = batch
                if model_type == "temporal":
                    images = images.to(device)
                mask = mask.to(device)
            else:
                images, forces, labels, depth_gt = batch
                images = images.to(device)
                mask = None

            forces = forces.to(device)
            labels = labels.to(device)

            # Forward pass
            if model_type == "temporal_force":
                output = model(forces, mask)
            elif model_type == "temporal":
                if use_force:
                    output = model(images, forces, mask)
                else:
                    output = model(images, mask=mask)
            elif model_type == "spatial_force":
                output = model(forces)
            else:
                if use_force:
                    output = model(images, forces)
                else:
                    output = model(images)

            if isinstance(output, tuple):
                cls_logits, _ = output
            else:
                cls_logits = output

            preds = cls_logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def save_checkpoint(model, optimizer, epoch, metrics, path, config=None):
    """Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dict of metrics
        path: Path to save checkpoint
        config: Optional config to include
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if config is not None:
        state["config"] = config_to_dict(config)

    torch.save(state, path)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Set seed
    set_seed(config.experiment.seed)

    # Device
    device = get_device()

    # Initialize wandb logger
    logger = create_logger(config)

    # Dataset
    # Determine if we need temporal mode
    is_temporal = config.model.type in ("temporal", "temporal_force")
    temporal_frames = config.model.temporal.num_frames if is_temporal else 1
    return_force_sequence = config.model.type == "temporal_force"

    dataset = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        img_size=(config.data.img_size, config.data.img_size),
        exclude_phantom_types=config.data.exclude_phantoms,
        normalization=config.data.normalization.type,
        norm_mean=config.data.normalization.mean,
        norm_std=config.data.normalization.std,
        temporal_frames=temporal_frames,
        subtraction_enabled=config.data.subtraction.enabled,
        subtraction_reference=config.data.subtraction.reference,
        return_force_sequence=return_force_sequence,
        augmentation={
            "enabled": config.data.augmentation.enabled,
            "horizontal_flip": config.data.augmentation.horizontal_flip,
            "rotation_degrees": config.data.augmentation.rotation_degrees,
            "color_jitter": {
                "brightness": config.data.augmentation.color_jitter.brightness,
                "contrast": config.data.augmentation.color_jitter.contrast,
                "saturation": config.data.augmentation.color_jitter.saturation,
            },
        },
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Split
    train_ds, val_ds = split_by_run(
        dataset, val_ratio=config.training.val_ratio, seed=config.experiment.seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=4
    )

    # Class weights
    class_weights = get_class_weights(
        dataset, config.model.num_classes, config.training.loss.class_weights
    )
    if class_weights is not None:
        print(f"Class weights: {class_weights.tolist()}")

    # Model
    model = get_model_v2(config.model).to(device)
    print(f"Model: {config.model.type} with {config.model.encoder.name} encoder")
    print(f"  Total parameters: {count_parameters(model, trainable_only=False):,}")
    print(f"  Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # Verify encoder is frozen
    if config.model.encoder.freeze:
        if verify_frozen_encoder(model):
            print("  Encoder frozen: Yes")
        else:
            print("  WARNING: Encoder has trainable parameters!")

    # Optimizer
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.training.lr, momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler = None
    scheduler_type = config.training.scheduler.type
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs - config.training.scheduler.warmup_epochs
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )

    # Checkpoint directory
    ckpt_dir = Path(__file__).parent / config.checkpoint.dir / config.experiment.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # CSV logging
    csv_path = ckpt_dir / "training_log.csv"
    if config.logging.csv.get("enabled", True):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_cls_loss", "train_depth_loss",
                "train_acc", "train_depth_mae",
                "val_loss", "val_cls_loss", "val_depth_loss",
                "val_acc", "val_depth_mae", "lr",
            ])

    # Resume from checkpoint
    start_epoch = 0
    if config.checkpoint.resume:
        ckpt_path = Path(config.checkpoint.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc",
                                "train_depth_mae", "val_depth_mae"]}
    best_val_acc = 0.0

    for epoch in range(start_epoch, config.training.epochs):
        # Warmup phase
        if epoch < config.training.scheduler.warmup_epochs:
            warmup_lr = config.training.lr * (epoch + 1) / config.training.scheduler.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Train
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, config, class_weights, train=True
        )

        # Validate
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, config, class_weights, train=False
        )

        # Update scheduler (after warmup)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler and epoch >= config.training.scheduler.warmup_epochs:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_depth_mae"].append(train_metrics["depth_mae"])
        history["val_depth_mae"].append(val_metrics["depth_mae"])

        # Print progress
        if (epoch + 1) % config.logging.print_every == 0:
            print(f"Epoch {epoch+1}/{config.training.epochs}  "
                  f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}  "
                  f"acc={train_metrics['acc']:.4f}/{val_metrics['acc']:.4f}  "
                  f"depth_mae={train_metrics['depth_mae']:.3f}/{val_metrics['depth_mae']:.3f} mm  "
                  f"lr={current_lr:.2e}")

        # Wandb logging
        logger.log_epoch(epoch, train_metrics, val_metrics, lr=current_lr)

        # CSV logging
        if config.logging.csv.get("enabled", True):
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1,
                    train_metrics["loss"], train_metrics["cls_loss"], train_metrics["depth_loss"],
                    train_metrics["acc"], train_metrics["depth_mae"],
                    val_metrics["loss"], val_metrics["cls_loss"], val_metrics["depth_loss"],
                    val_metrics["acc"], val_metrics["depth_mae"], current_lr,
                ])

        # Checkpointing
        metrics = {"train": train_metrics, "val": val_metrics}

        # Save best model
        if config.checkpoint.save_best and val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_dir / "best.pth", config)
            print(f"  New best model saved (val_acc={best_val_acc:.4f})")

        # Save every N epochs
        if config.checkpoint.save_every_n_epochs > 0:
            if (epoch + 1) % config.checkpoint.save_every_n_epochs == 0:
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    ckpt_dir / f"epoch_{epoch+1:03d}.pth", config
                )

        # Save last model
        if config.checkpoint.save_last:
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_dir / "last.pth", config)

    # Save learning curves
    save_learning_curve(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"],
         "train_acc": history["train_acc"], "val_acc": history["val_acc"]},
        filename=f"{config.experiment.name}_learning_curve.png",
    )
    save_learning_curve(
        {"train_depth_mae": history["train_depth_mae"],
         "val_depth_mae": history["val_depth_mae"]},
        filename=f"{config.experiment.name}_depth_curve.png",
    )

    # Final evaluation with best model
    print("\n--- Final Evaluation (Best Model) ---")
    best_ckpt_path = ckpt_dir / "best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    # Collect predictions on validation set
    y_pred, y_true = collect_predictions(model, val_loader, device, config)

    # Class names
    class_names = ["No Tendon", "Tendon 1", "Tendon 2", "Tendon 3"][:config.model.num_classes]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':>12} " + " ".join(f"{name:>10}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>12} " + " ".join(f"{val:>10}" for val in row))

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{report}")

    # Log to wandb
    logger.log_confusion_matrix(y_true, y_pred, class_names, title="Final Confusion Matrix")

    # Log per-class metrics to wandb
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    final_metrics = {
        "final/accuracy": report_dict["accuracy"],
        "final/macro_f1": report_dict["macro avg"]["f1-score"],
        "final/macro_precision": report_dict["macro avg"]["precision"],
        "final/macro_recall": report_dict["macro avg"]["recall"],
    }
    for class_name in class_names:
        final_metrics[f"final/{class_name}/precision"] = report_dict[class_name]["precision"]
        final_metrics[f"final/{class_name}/recall"] = report_dict[class_name]["recall"]
        final_metrics[f"final/{class_name}/f1"] = report_dict[class_name]["f1-score"]
    logger.log(final_metrics)

    # Finish wandb
    logger.finish()

    print(f"\nTraining complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
