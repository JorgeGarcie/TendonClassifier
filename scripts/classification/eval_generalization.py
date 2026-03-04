"""Evaluate a trained model on held-out p1-p5 phantom runs.

Loads a checkpoint (with embedded config), reconstructs the model,
computes force normalization stats from the training distribution (traverse),
then runs inference on all p1-p5 phantom runs the model has never seen.

Usage:
    # Evaluate on all p1-p5 phantoms
    python eval_generalization.py --checkpoint checkpoints/sweep/<run_id>/best.pth

    # Evaluate on specific phantoms only
    python eval_generalization.py --checkpoint checkpoints/sweep/<run_id>/best.pth --phantoms p1 p3

    # Override include regex
    python eval_generalization.py --checkpoint checkpoints/sweep/<run_id>/best.pth --include-regex "^p1_"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import load_config_from_dict
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_utils import get_device
from train_v2 import collect_predictions


LABEL_NAMES = {0: "none", 1: "single", 2: "crossed", 3: "double"}
CLASS_NAMES = ["none", "single", "crossed", "double"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate generalization on p1-p5 phantoms")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint (must contain 'config' and 'model_state_dict')")
    p.add_argument("--phantoms", nargs="*", default=None,
                   help="Specific phantoms to evaluate (e.g. p1 p3). Default: all p1-p5")
    p.add_argument("--include-regex", type=str, default=None,
                   help="Override include_run_regex for eval dataset (default: ^p[1-5]_)")
    p.add_argument("--subtraction-ref", type=str, default=None,
                   help="Path to a global subtraction reference image (e.g. a 'none' frame). "
                        "Overrides the checkpoint's first_frame reference.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-dir", type=str, default=None,
                   help="Directory to save results JSON (default: next to checkpoint)")
    return p.parse_args()


def build_eval_regex(phantoms):
    """Build regex to match specific phantom runs.

    Args:
        phantoms: List like ["p1", "p3"] or None for all p1-p5.

    Returns:
        Regex string like "^(p1|p3)_" or "^p[1-5]_".
    """
    if phantoms is None:
        return r"^p[1-5]_"
    return r"^(" + "|".join(phantoms) + r")_"


def evaluate_subset(model, dataset, indices, device, config, batch_size):
    """Run collect_predictions on a subset of the dataset.

    Args:
        model: Loaded model in eval mode.
        dataset: TendonDatasetV2 instance.
        indices: List of DataFrame indices for this subset.
        device: torch device.
        config: Config object (for model type dispatch).
        batch_size: Batch size for DataLoader.

    Returns:
        (preds, labels) numpy arrays.
    """
    if not indices:
        return np.array([], dtype=int), np.array([], dtype=int)

    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return collect_predictions(model, loader, device, config)


def print_metrics(preds, labels, title=""):
    """Print classification metrics for a set of predictions."""
    if len(preds) == 0:
        print(f"  {title}: no samples")
        return {}

    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    present_classes = sorted(set(labels) | set(preds))
    target_names = [CLASS_NAMES[c] for c in present_classes]

    print(f"\n{'='*60}")
    print(f"  {title}  ({len(preds)} samples)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=present_classes)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10} " + " ".join(f"{n:>8}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>10} " + " ".join(f"{v:>8}" for v in row))

    # Per-class report
    report = classification_report(
        labels, preds, labels=present_classes,
        target_names=target_names, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    # Build results dict
    report_dict = classification_report(
        labels, preds, labels=present_classes,
        target_names=target_names, output_dict=True, zero_division=0,
    )
    return {
        "n_samples": int(len(preds)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }
            for name in target_names if name in report_dict
        },
    }


def main():
    args = parse_args()

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "config" not in ckpt:
        raise ValueError("Checkpoint missing 'config' dict — was it saved with train_v2.py?")

    config = load_config_from_dict(ckpt["config"])
    print(f"  Model type: {config.model.type}")
    print(f"  Fusion: {config.model.fusion.type}")
    print(f"  Subtraction: {config.data.subtraction.enabled}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    if "metrics" in ckpt:
        val_m = ckpt["metrics"].get("val", {})
        print(f"  Val macro-F1 at save: {val_m.get('macro_f1', '?')}")

    # --- Reconstruct model ---
    model = get_model_v2(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully")

    # --- Determine dataset params from config ---
    is_temporal = config.model.type in ("temporal", "temporal_force")
    temporal_frames = config.model.temporal.num_frames if is_temporal else 1
    return_force_sequence = config.model.type == "temporal_force" or (
        config.model.type == "temporal" and config.model.use_force
    )

    # Sparsh-specific params
    sparsh_stride = config.data.sparsh_temporal_stride
    sub_type = getattr(config.data.subtraction, "type", "simple")
    norm_mean = config.data.normalization.mean if config.data.normalization.type != "none" else None
    norm_std = config.data.normalization.std if config.data.normalization.type != "none" else None

    # --- Compute force stats from training distribution ---
    train_regex = config.data.include_run_regex or "traverse"
    print(f"\nComputing force normalization stats from '{train_regex}' data...")
    train_ds = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        img_size=(config.data.img_size, config.data.img_size),
        include_run_regex=train_regex,
        normalization=config.data.normalization.type,
        norm_mean=norm_mean,
        norm_std=norm_std,
        temporal_frames=temporal_frames,
        subtraction_enabled=config.data.subtraction.enabled,
        subtraction_reference=config.data.subtraction.reference,
        subtraction_type=sub_type,
        return_force_sequence=return_force_sequence,
        sparsh_temporal_stride=sparsh_stride,
    )
    train_ds.compute_force_stats(range(len(train_ds)))

    # --- Create eval dataset for p1-p5 ---
    if args.include_regex is not None:
        eval_regex = args.include_regex
    else:
        eval_regex = build_eval_regex(args.phantoms)
    print(f"\nCreating eval dataset with include_run_regex='{eval_regex}'")

    # Use global subtraction reference if provided, otherwise fall back to config
    sub_ref = args.subtraction_ref if args.subtraction_ref else config.data.subtraction.reference
    if args.subtraction_ref:
        print(f"  Subtraction reference override: {args.subtraction_ref}")

    eval_ds = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        img_size=(config.data.img_size, config.data.img_size),
        include_run_regex=eval_regex,
        normalization=config.data.normalization.type,
        norm_mean=norm_mean,
        norm_std=norm_std,
        temporal_frames=temporal_frames,
        subtraction_enabled=config.data.subtraction.enabled,
        subtraction_reference=sub_ref,
        subtraction_type=sub_type,
        return_force_sequence=return_force_sequence,
        sparsh_temporal_stride=sparsh_stride,
    )
    # Apply training force stats to eval dataset
    eval_ds.force_mean = train_ds.force_mean
    eval_ds.force_std = train_ds.force_std
    print(f"Eval dataset: {len(eval_ds)} samples")

    if len(eval_ds) == 0:
        print("ERROR: No samples found. Check --phantoms or --include-regex.")
        return

    # --- Overall evaluation ---
    all_indices = list(range(len(eval_ds)))
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    preds, labels = collect_predictions(model, loader, device, config)
    overall_results = print_metrics(preds, labels, title="Overall (all phantoms)")

    # --- Per-phantom breakdown ---
    phantom_results = {}
    phantoms_in_data = sorted(eval_ds.df["phantom_type"].unique())
    print(f"\n{'#'*60}")
    print(f"  Per-Phantom Breakdown")
    print(f"{'#'*60}")

    for phantom in phantoms_in_data:
        phantom_indices = eval_ds.df.index[eval_ds.df["phantom_type"] == phantom].tolist()
        p_preds, p_labels = evaluate_subset(
            model, eval_ds, phantom_indices, device, config, args.batch_size
        )
        result = print_metrics(p_preds, p_labels, title=f"Phantom: {phantom}")
        phantom_results[phantom] = result

    # --- Per-run breakdown (compact table) ---
    print(f"\n{'#'*60}")
    print(f"  Per-Run Summary")
    print(f"{'#'*60}")
    print(f"  {'Run ID':<35} {'N':>5} {'Acc':>7} {'F1':>7}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7}")

    run_results = {}
    for run_id in sorted(eval_ds.df["run_id"].unique()):
        run_indices = eval_ds.df.index[eval_ds.df["run_id"] == run_id].tolist()
        r_preds, r_labels = evaluate_subset(
            model, eval_ds, run_indices, device, config, args.batch_size
        )
        if len(r_preds) == 0:
            continue
        acc = float((r_preds == r_labels).mean())
        mf1 = float(f1_score(r_labels, r_preds, average="macro", zero_division=0))
        print(f"  {run_id:<35} {len(r_preds):>5} {acc:>7.3f} {mf1:>7.3f}")
        run_results[run_id] = {"n_samples": len(r_preds), "accuracy": acc, "macro_f1": mf1}

    # --- Save results ---
    save_dir = Path(args.save_dir) if args.save_dir else ckpt_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(ckpt_path),
        "eval_regex": eval_regex,
        "overall": overall_results,
        "per_phantom": phantom_results,
        "per_run": run_results,
    }
    out_path = save_dir / "generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
