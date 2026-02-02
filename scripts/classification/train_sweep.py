"""Training script for wandb sweeps.

This script is called by wandb sweep agent and reads hyperparameters
from wandb.config instead of a YAML file.

Usage:
    # Create sweep
    wandb sweep configs/sweep_spatial.yaml

    # Run agent (use the sweep_id from above)
    wandb agent <entity>/<project>/<sweep_id>
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import wandb
from dataset import TendonDatasetV2
from models_v2 import get_model_v2, count_parameters
from train_utils import get_device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_run(dataset, val_ratio=0.2, seed=42):
    df = dataset.df
    run_ids = list(df["run_id"].unique())
    random.Random(seed).shuffle(run_ids)
    n_val = max(1, int(len(run_ids) * val_ratio))
    n_train = len(run_ids) - n_val
    train_runs = set(run_ids[:n_train])
    val_runs = set(run_ids[n_train:])
    train_idx = df.index[df["run_id"].isin(train_runs)].tolist()
    val_idx = df.index[df["run_id"].isin(val_runs)].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def compute_loss(cls_logits, depth_pred, labels, depth_gt, depth_weight,
                 class_weights=None, device=None):
    if class_weights is not None:
        class_weights = class_weights.to(device)
        cls_loss = F.cross_entropy(cls_logits, labels, weight=class_weights)
    else:
        cls_loss = F.cross_entropy(cls_logits, labels)

    mask = labels > 0
    if mask.sum() > 0:
        depth_loss = F.mse_loss(depth_pred[mask], depth_gt[mask])
    else:
        depth_loss = torch.tensor(0.0, device=device)

    return cls_loss + depth_weight * depth_loss, cls_loss, depth_loss


def run_epoch(model, loader, optimizer, device, model_type, use_force,
              depth_weight, class_weights=None, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    depth_abs_err = 0.0
    depth_count = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            images, forces, labels, depth_gt = batch
            images = images.to(device)
            forces = forces.to(device)
            labels = labels.to(device)
            depth_gt = depth_gt.to(device)

            # Forward pass based on model type
            if model_type == "spatial_force":
                output = model(forces)
            elif use_force:
                output = model(images, forces)
            else:
                output = model(images)

            if isinstance(output, tuple):
                cls_logits, depth_pred = output
            else:
                cls_logits = output
                depth_pred = torch.zeros(labels.size(0), device=device)

            loss, _, _ = compute_loss(
                cls_logits, depth_pred, labels, depth_gt, depth_weight,
                class_weights, device
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = cls_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            mask = labels > 0
            if mask.sum() > 0:
                depth_abs_err += (depth_pred[mask] - depth_gt[mask]).abs().sum().item()
                depth_count += mask.sum().item()

    acc = correct / total
    loss = total_loss / total
    depth_mae = depth_abs_err / depth_count if depth_count > 0 else 0.0

    return {"loss": loss, "acc": acc, "depth_mae": depth_mae}


def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    # Set seed
    set_seed(config.get("seed", 42))
    device = get_device()

    # Get hyperparameters from sweep config
    model_type = config.get("model_type", "spatial")
    encoder_name = config.get("encoder_name", "resnet18")
    use_force = config.get("use_force", True)
    fusion_type = config.get("fusion_type", "attention")
    fusion_hidden_dim = config.get("fusion_hidden_dim", 128)
    lr = config.get("lr", 1e-4)
    batch_size = config.get("batch_size", 32)
    optimizer_name = config.get("optimizer", "adam")
    weight_decay = config.get("weight_decay", 1e-4)
    depth_weight = config.get("depth_weight", 0.1)
    epochs = config.get("epochs", 50)

    # Skip invalid combinations
    if model_type == "spatial_force" and not use_force:
        wandb.log({"val/acc": 0.0})  # Invalid config
        return
    if model_type == "spatial_force":
        use_force = True  # Force-only model always uses force

    # Dataset
    dataset = TendonDatasetV2(
        manifest_csv="../labeling/output/gt_dataset/gt_manifest.csv",
        img_size=(224, 224),
        normalization="imagenet",
    )

    train_ds, val_ds = split_by_run(dataset, val_ratio=0.2, seed=config.get("seed", 42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Class weights
    class_weights = dataset.get_class_weights(4)

    # Build model config dict
    model_config = {
        "type": model_type,
        "encoder": {
            "name": encoder_name,
            "pretrained": True,
            "freeze": True,
        },
        "fusion": {
            "type": fusion_type,
            "hidden_dim": fusion_hidden_dim,
        },
        "num_classes": 4,
        "use_force": use_force,
        "use_depth_head": True,
        "temporal": {"num_frames": 1, "aggregation": "attention"},
    }

    model = get_model_v2(model_config).to(device)

    wandb.log({
        "model/total_params": count_parameters(model, trainable_only=False),
        "model/trainable_params": count_parameters(model, trainable_only=True),
    })

    # Optimizer
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, model_type, use_force,
            depth_weight, class_weights, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, model_type, use_force,
            depth_weight, class_weights, train=False
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "train/depth_mae": train_metrics["depth_mae"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "val/depth_mae": val_metrics["depth_mae"],
        })

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

    wandb.log({"best_val_acc": best_val_acc})


if __name__ == "__main__":
    train()
