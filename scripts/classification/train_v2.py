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
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from config import load_config, load_config_from_dict, config_to_dict
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


def apply_overrides(yaml_dict: dict, overrides: list) -> dict:
    """Apply dotted key=value overrides to a raw YAML dict.

    Args:
        yaml_dict: Nested dict from yaml.safe_load
        overrides: List of "dotted.key=value" strings

    Returns:
        Modified yaml_dict (in-place)
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (missing '='): {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")

        # Auto-coerce types
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string

        # Navigate to parent dict
        d = yaml_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return yaml_dict


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


def split_by_run_stratified(dataset, val_ratio=0.2, seed=42, val_n_override=None,
                            train_n_override=None, frame_split_phantoms=None,
                            forced_val_runs=None):
    """Split by run, stratified by phantom_type so every phantom appears in val.

    Groups runs by phantom_type, then allocates at least 1 run per phantom to
    val. This ensures all tendon-type classes (none/single/crossed/double) are
    represented in the validation set regardless of the random seed.

    Args:
        val_n_override: Optional dict mapping phantom_type -> exact n_val.
            Overrides val_ratio for specific phantoms. E.g. {"p4": 2, "p5": 2}
            to give the rare-class phantoms extra val coverage.
        train_n_override: Optional dict mapping phantom_type -> max n_train.
            Limits training runs per phantom — excess runs are excluded entirely.
            E.g. {"p2": 1, "p3": 1} to keep only 1 training run each.
        frame_split_phantoms: Optional list of phantom types to split at the
            frame level (50/50) instead of by run. Useful for phantoms with a
            single run where all frames are nearly identical (e.g. "none").
        forced_val_runs: Optional list of run_ids that must go to val regardless
            of diversity sort. Counts toward the n_val quota for their phantom.
    """
    df = dataset.df
    rng = random.Random(seed)
    forced_val = set(forced_val_runs or [])
    frame_split_set = set(frame_split_phantoms or [])

    # Map run_id -> phantom_type
    run_phantom = df.groupby("run_id")["phantom_type"].first().to_dict()

    # Group runs by phantom
    phantom_runs = {}
    for run_id, phantom in run_phantom.items():
        phantom_runs.setdefault(phantom, []).append(run_id)

    train_runs, val_runs, excluded_runs = set(), set(), set()
    train_idx_extra, val_idx_extra = [], []  # For frame-level splits

    for phantom, runs in sorted(phantom_runs.items()):
        # --- Frame-level 50/50 split for this phantom ---
        if phantom in frame_split_set:
            phantom_indices = df.index[df["phantom_type"] == phantom].tolist()
            frame_rng = random.Random(seed)
            frame_rng.shuffle(phantom_indices)
            mid = len(phantom_indices) // 2
            train_idx_extra.extend(phantom_indices[:mid])
            val_idx_extra.extend(phantom_indices[mid:])
            print(f"  {phantom}: frame-split {mid} train / {len(phantom_indices)-mid} val")
            continue

        # --- Normal run-level split ---
        if val_n_override and phantom in val_n_override:
            n_val = min(val_n_override[phantom], len(runs) - 1)
        else:
            n_val = max(1, round(len(runs) * val_ratio))

        # Pre-assign forced val runs for this phantom
        forced_in_group = [r for r in runs if r in forced_val]
        remaining = [r for r in runs if r not in forced_val]

        # Sort remaining by class diversity descending (most diverse first)
        def _sort_key(run_id):
            n_unique = df[df["run_id"] == run_id]["tendon_type"].nunique()
            return (-n_unique, rng.random())

        remaining_sorted = sorted(remaining, key=_sort_key)

        # Fill val: forced first, then diversity-sorted
        val_from_group = forced_in_group[:]
        slots_left = max(0, n_val - len(val_from_group))
        val_from_group += remaining_sorted[:slots_left]
        train_candidates = remaining_sorted[slots_left:]

        # Apply train_n_override: keep only N runs in train, exclude the rest
        if train_n_override and phantom in train_n_override:
            n_train = train_n_override[phantom]
            train_from_group = train_candidates[:n_train]
            excluded_from_group = train_candidates[n_train:]
            excluded_runs.update(excluded_from_group)
        else:
            train_from_group = train_candidates

        val_runs.update(val_from_group)
        train_runs.update(train_from_group)

    train_idx = df.index[df["run_id"].isin(train_runs)].tolist() + train_idx_extra
    val_idx   = df.index[df["run_id"].isin(val_runs)].tolist() + val_idx_extra

    label_names = {0: "none", 1: "single", 2: "crossed", 3: "double"}

    n_train_runs = len(train_runs) + (1 if train_idx_extra else 0)
    n_val_runs = len(val_runs) + (1 if val_idx_extra else 0)
    print(f"Stratified split (seed={seed}):")
    print(f"  Train: {n_train_runs} run-groups, {len(train_idx)} frames")
    print(f"  Val:   {n_val_runs} run-groups, {len(val_idx)} frames")
    if excluded_runs:
        n_excl_frames = len(df[df["run_id"].isin(excluded_runs)])
        print(f"  Excluded: {len(excluded_runs)} runs, {n_excl_frames} frames")
    for phantom in sorted(phantom_runs.keys()):
        if phantom in frame_split_set:
            continue
        runs = phantom_runs[phantom]
        t = sum(1 for r in runs if r in train_runs)
        v = sum(1 for r in runs if r in val_runs)
        e = sum(1 for r in runs if r in excluded_runs)
        extra = f", {e} excluded" if e else ""
        print(f"    {phantom}: {t} train, {v} val{extra}")

    # Print class distribution
    for split_name, idx_list in [("Train", train_idx), ("Val", val_idx)]:
        split_df = df.iloc[idx_list]
        counts = split_df["tendon_type"].value_counts().sort_index()
        print(f"  {split_name} class distribution:")
        for cls, cnt in counts.items():
            print(f"    class {cls} ({label_names.get(cls, cls)}): {cnt} ({cnt/len(split_df)*100:.1f}%)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_frame_contiguous(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                           seed=42, purge_frames=0):
    """Contiguous per-run frame split with purge gaps at boundaries.

    Within each run (sorted by frame_idx):
      [train 80%] [purge T-1] [val 10%] [purge T-1] [test 10%]

    Purge gaps ensure that val/test temporal windows don't overlap with train frames.

    Args:
        dataset: TendonDatasetV2 instance
        train_ratio: Fraction of frames for training
        val_ratio: Fraction of frames for validation
        test_ratio: Fraction of frames for test (0 = no test set)
        seed: Random seed (unused — split is deterministic given frame order)
        purge_frames: Number of frames to discard at each boundary

    Returns:
        (train_ds, val_ds, test_ds) — test_ds is None if test_ratio == 0
    """
    df = dataset.df
    train_idx, val_idx, test_idx = [], [], []

    label_names = {0: "none", 1: "single", 2: "crossed", 3: "double"}
    print(f"Contiguous frame split (train={train_ratio}, val={val_ratio}, "
          f"test={test_ratio}, purge={purge_frames}):")

    for run_id, group in sorted(df.groupby("run_id"), key=lambda x: x[0]):
        sorted_group = group.sort_values("frame_idx")
        indices = sorted_group.index.tolist()
        n = len(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_end = n_train
        val_start = train_end + purge_frames
        val_end = val_start + n_val
        test_start = val_end + purge_frames

        run_train = indices[:train_end]
        run_val = indices[val_start:val_end] if val_start < n else []
        run_test = indices[test_start:] if test_ratio > 0 and test_start < n else []

        n_purged = (val_start - train_end) + (test_start - val_end) if test_ratio > 0 else (val_start - train_end)
        n_purged = min(n_purged, n - len(run_train) - len(run_val) - len(run_test))

        train_idx.extend(run_train)
        val_idx.extend(run_val)
        test_idx.extend(run_test)

        print(f"  {run_id}: {n} total → {len(run_train)} train, "
              f"{len(run_val)} val, {len(run_test)} test, {n_purged} purged")

    print(f"  Total: {len(train_idx)} train, {len(val_idx)} val, "
          f"{len(test_idx)} test")

    # Print class distribution per split
    for split_name, idx_list in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        if not idx_list:
            continue
        split_df = df.iloc[idx_list]
        counts = split_df["tendon_type"].value_counts().sort_index()
        print(f"  {split_name} class distribution:")
        for cls, cnt in counts.items():
            print(f"    class {cls} ({label_names.get(cls, cls)}): {cnt} ({cnt/len(split_df)*100:.1f}%)")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx) if test_idx else None
    return train_ds, val_ds, test_ds


def get_balanced_sampler(subset):
    """WeightedRandomSampler that gives each class equal expected frequency.

    Assigns per-sample weights = 1 / class_count so that each class is sampled
    at equal rate regardless of its frequency in the subset. Total samples per
    epoch stays equal to the subset size (with replacement).
    """
    df = subset.dataset.df
    labels = np.array([df.iloc[i]["tendon_type"] for i in subset.indices])

    class_counts = np.bincount(labels, minlength=4).astype(float)
    class_counts = np.maximum(class_counts, 1)  # avoid div/0
    sample_weights = 1.0 / class_counts[labels]

    print(f"  Balanced sampler weights (inverse class freq):")
    label_names = {0: "none", 1: "single", 2: "crossed", 3: "double"}
    for cls in range(4):
        if class_counts[cls] > 1:
            print(f"    class {cls} ({label_names[cls]}): "
                  f"count={int(class_counts[cls])}, weight={1/class_counts[cls]:.5f}")

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


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
    all_preds = []
    all_labels_list = []

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
            all_preds.append(preds.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())

            # Depth MAE on present samples
            depth_mask = labels > 0
            if depth_mask.sum() > 0:
                depth_abs_err += (depth_pred[depth_mask] - depth_gt[depth_mask]).abs().sum().item()
                depth_count += depth_mask.sum().item()

    n = total
    depth_mae = depth_abs_err / depth_count if depth_count > 0 else 0.0
    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels_list)
    macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)

    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "depth_loss": total_depth_loss / n,
        "acc": correct / n,
        "depth_mae": depth_mae,
        "macro_f1": macro_f1,
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


def run_training(config, sweep_mode: bool = False):
    """Run the full training pipeline.

    Args:
        config: Config dataclass with all settings.
        sweep_mode: If True, use sweep-friendly defaults:
            - Checkpoint dir under checkpoints/sweep/{wandb.run.id}
            - Skip CSV logging, learning curve PNGs, periodic checkpoints
    """
    # Set seed
    set_seed(config.experiment.seed)

    # Device
    device = get_device()

    # Initialize wandb logger
    logger = create_logger(config, sweep_mode=sweep_mode)

    # Dataset
    # Determine if we need temporal mode
    is_temporal = config.model.type in ("temporal", "temporal_force")
    temporal_frames = config.model.temporal.num_frames if is_temporal else 1
    return_force_sequence = config.model.type == "temporal_force" or (
        config.model.type == "temporal" and config.model.use_force
    )

    dataset = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        img_size=(config.data.img_size, config.data.img_size),
        exclude_phantom_types=config.data.exclude_phantoms,
        exclude_run_regex=config.data.exclude_run_regex,
        include_run_regex=config.data.include_run_regex,
        normalization=config.data.normalization.type,
        norm_mean=config.data.normalization.mean if config.data.normalization.type != "none" else None,
        norm_std=config.data.normalization.std if config.data.normalization.type != "none" else None,
        temporal_frames=temporal_frames,
        subtraction_enabled=config.data.subtraction.enabled,
        subtraction_reference=config.data.subtraction.reference,
        subtraction_type=config.data.subtraction.type,
        return_force_sequence=return_force_sequence,
        sparsh_temporal_stride=config.data.sparsh_temporal_stride,
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

    # Split dataset into train / val (/ test).
    split_cfg = config.training.split
    test_ds = None

    if config.training.split_by == "frame":
        # Contiguous per-run frame split with temporal purge gaps
        test_ratio = split_cfg.test_ratio
        val_ratio = config.training.val_ratio
        train_ratio = 1.0 - val_ratio - test_ratio
        purge_frames = (temporal_frames - 1) if temporal_frames > 1 else 0
        train_ds, val_ds, test_ds = split_frame_contiguous(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=config.experiment.seed,
            purge_frames=purge_frames,
        )
    else:
        # Stratified run-level split (default)
        train_ds, val_ds = split_by_run_stratified(
            dataset, val_ratio=config.training.val_ratio, seed=config.experiment.seed,
            val_n_override=split_cfg.val_n_override,
            train_n_override=split_cfg.train_n_override,
            frame_split_phantoms=split_cfg.frame_split_phantoms,
        )

    # Compute force normalization stats from training split only
    dataset.compute_force_stats(train_ds.indices)

    # Balanced sampler: oversample minority classes so each class is seen equally
    use_balanced = getattr(config.training, "balanced_sampling", True)
    if use_balanced:
        sampler = get_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size,
            sampler=sampler, num_workers=4, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size,
            shuffle=True, num_workers=4, drop_last=True
        )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=4
    )

    # Class weights — disable when balanced_sampling is on to avoid double-correction
    if use_balanced:
        class_weights = None
        print("Balanced sampler active → class-weighted loss disabled (would double-correct)")
    else:
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
    if sweep_mode:
        import wandb
        ckpt_dir = Path(__file__).parent / config.checkpoint.dir / "sweep" / wandb.run.id
    else:
        ckpt_dir = Path(__file__).parent / config.checkpoint.dir / config.experiment.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # CSV logging (skip in sweep mode)
    csv_enabled = not sweep_mode and config.logging.csv.get("enabled", True)
    csv_path = ckpt_dir / "training_log.csv"
    if csv_enabled:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_cls_loss", "train_depth_loss",
                "train_acc", "train_macro_f1", "train_depth_mae",
                "val_loss", "val_cls_loss", "val_depth_loss",
                "val_acc", "val_macro_f1", "val_depth_mae", "lr",
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
    best_val_macro_f1 = 0.0

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
                  f"macro_f1={train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}  "
                  f"depth_mae={train_metrics['depth_mae']:.3f}/{val_metrics['depth_mae']:.3f} mm  "
                  f"lr={current_lr:.2e}")

        # Wandb logging
        logger.log_epoch(epoch, train_metrics, val_metrics, lr=current_lr)

        # CSV logging
        if csv_enabled:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1,
                    train_metrics["loss"], train_metrics["cls_loss"], train_metrics["depth_loss"],
                    train_metrics["acc"], train_metrics["macro_f1"], train_metrics["depth_mae"],
                    val_metrics["loss"], val_metrics["cls_loss"], val_metrics["depth_loss"],
                    val_metrics["acc"], val_metrics["macro_f1"], val_metrics["depth_mae"], current_lr,
                ])

        # Checkpointing
        metrics = {"train": train_metrics, "val": val_metrics}

        # Save best model (tracked by val macro-F1, not accuracy)
        if config.checkpoint.save_best and val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_dir / "best.pth", config)
            print(f"  New best model saved (val_macro_f1={best_val_macro_f1:.4f})")

        # Save every N epochs (skip in sweep mode)
        if not sweep_mode and config.checkpoint.save_every_n_epochs > 0:
            if (epoch + 1) % config.checkpoint.save_every_n_epochs == 0:
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    ckpt_dir / f"epoch_{epoch+1:03d}.pth", config
                )

        # Save last model
        if config.checkpoint.save_last:
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_dir / "last.pth", config)

    # Save learning curves (skip in sweep mode)
    if not sweep_mode:
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

    # Test set evaluation (if available)
    if test_ds is not None:
        print("\n--- Test Set Evaluation (Best Model) ---")
        test_loader = DataLoader(
            test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=4
        )
        test_pred, test_true = collect_predictions(model, test_loader, device, config)

        test_cm = confusion_matrix(test_true, test_pred)
        print("\nTest Confusion Matrix:")
        print(f"{'':>12} " + " ".join(f"{name:>10}" for name in class_names))
        for i, row_vals in enumerate(test_cm):
            print(f"{class_names[i]:>12} " + " ".join(f"{val:>10}" for val in row_vals))

        test_report = classification_report(
            test_true, test_pred, target_names=class_names, digits=4
        )
        print(f"\nTest Classification Report:\n{test_report}")

        test_report_dict = classification_report(
            test_true, test_pred, target_names=class_names, output_dict=True
        )
        test_metrics = {
            "test/accuracy": test_report_dict["accuracy"],
            "test/macro_f1": test_report_dict["macro avg"]["f1-score"],
            "test/macro_precision": test_report_dict["macro avg"]["precision"],
            "test/macro_recall": test_report_dict["macro avg"]["recall"],
        }
        for class_name in class_names:
            test_metrics[f"test/{class_name}/precision"] = test_report_dict[class_name]["precision"]
            test_metrics[f"test/{class_name}/recall"] = test_report_dict[class_name]["recall"]
            test_metrics[f"test/{class_name}/f1"] = test_report_dict[class_name]["f1-score"]
        logger.log(test_metrics)
        logger.log_confusion_matrix(test_true, test_pred, class_names, title="Test Confusion Matrix")

    # Finish wandb
    logger.finish()

    print(f"\nTraining complete!")
    print(f"  Best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"  Checkpoints saved to: {ckpt_dir}")


def main():
    args = parse_args()

    # Load config as raw dict, apply overrides, then convert to dataclass
    with open(args.config) as f:
        yaml_dict = yaml.safe_load(f)

    if args.override:
        apply_overrides(yaml_dict, args.override)
        print(f"Applied {len(args.override)} override(s)")

    config = load_config_from_dict(yaml_dict)
    print(f"Loaded config from: {args.config}")

    run_training(config)


if __name__ == "__main__":
    main()
