"""Analyze per-class and per-phantom failure modes for each experiment."""
import sys, random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score

sys.path.insert(0, ".")
from config import load_config
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_v2 import split_by_run_stratified

CLASS_NAMES = ["none", "single", "crossed", "double"]
EXPERIMENTS = [
    ("spatial_force_only",  "configs/spatial_force_only.yaml"),
    ("spatial_image_only",  "configs/spatial_image_only.yaml"),
    ("spatial_combined",    "configs/spatial_combined.yaml"),
    ("temporal_combined",   "configs/temporal_combined.yaml"),
]


def get_val_loader_and_meta(config):
    is_temporal = config.model.type in ("temporal", "temporal_force")
    temporal_frames = config.model.temporal.num_frames if is_temporal else 1
    return_force_seq = config.model.type == "temporal_force"
    dataset = TendonDatasetV2(
        manifest_csv=config.data.manifest,
        img_size=(config.data.img_size, config.data.img_size),
        exclude_phantom_types=config.data.exclude_phantoms,
        exclude_run_regex=config.data.exclude_run_regex,
        normalization=config.data.normalization.type,
        temporal_frames=temporal_frames,
        subtraction_enabled=config.data.subtraction.enabled,
        subtraction_reference=config.data.subtraction.reference,
        return_force_sequence=return_force_seq,
        augmentation={"enabled": False, "horizontal_flip": False,
                      "rotation_degrees": 0,
                      "color_jitter": {"brightness": 0, "contrast": 0, "saturation": 0}},
    )
    _, val_ds = split_by_run_stratified(
        dataset, val_ratio=config.training.val_ratio,
        seed=config.experiment.seed, val_n_override={"p4": 2, "p5": 2}
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    return loader, dataset, val_ds


def run_inference(model, loader, config, device):
    model.eval()
    model_type = config.model.type
    use_force = config.model.use_force
    all_preds, all_labels = [], []
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

            if model_type == "temporal_force":
                out = model(forces, mask)
            elif model_type == "temporal":
                out = model(images, forces, mask) if use_force else model(images, mask=mask)
            elif model_type == "spatial_force":
                out = model(forces)
            else:
                out = model(images, forces) if use_force else model(images)

            logits = out[0] if isinstance(out, tuple) else out
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp_name, cfg_path in EXPERIMENTS:
    ckpt_path = Path(f"checkpoints/{exp_name}/best.pth")
    if not ckpt_path.exists():
        print(f"\n{exp_name}: checkpoint not found")
        continue

    config = load_config(cfg_path)
    loader, dataset, val_ds = get_val_loader_and_meta(config)
    model = get_model_v2(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    preds, labels = run_inference(model, loader, config, device)

    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>10} " + "  ".join(f"{n:>8}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        recall = row[i] / row.sum() if row.sum() > 0 else 0
        print(f"  {CLASS_NAMES[i]:>10} " + "  ".join(f"{v:>8}" for v in row)
              + f"   recall={recall:.2f}")

    # Per-class report
    print()
    rpt = classification_report(labels, preds, target_names=CLASS_NAMES,
                                labels=[0, 1, 2, 3], digits=3, zero_division=0)
    for line in rpt.split("\n"):
        print(f"  {line}")

    # Per-phantom breakdown
    df = dataset.df
    val_rows = df.iloc[val_ds.indices].copy().reset_index(drop=True)
    val_rows["pred"] = preds
    val_rows["label"] = labels

    print(f"  Per-phantom:")
    for ph, g in val_rows.groupby("phantom_type"):
        f1 = f1_score(g["label"], g["pred"], average="macro", zero_division=0)
        wrong = g[g["label"] != g["pred"]]
        if len(wrong) > 0:
            err_counts = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False)
            top = err_counts.index[0]
            top_str = (f"true={CLASS_NAMES[top[0]]}→pred={CLASS_NAMES[top[1]]} "
                       f"({err_counts.iloc[0]}x, {err_counts.iloc[0]/len(g)*100:.0f}% of {ph})")
        else:
            top_str = "no errors"
        print(f"    {ph}: macro-F1={f1:.3f}  top error: {top_str}")
