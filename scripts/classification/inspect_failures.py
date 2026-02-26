"""Visualize model failure cases to identify boundary / labeling issues.

For each failure type (true crossed→pred single, true double→pred single),
saves:
  1. A grid of the most confidently wrong images
  2. A per-run label sequence plot showing where in the run failures occur
     (to check if they cluster at bounding-box boundaries)

Usage:
    python inspect_failures.py --config configs/temporal_combined.yaml \
                               --save /tmp/failures
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from config import load_config
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_v2 import split_by_run_stratified

CLASS_NAMES = ["none", "single", "crossed", "double"]
CLASS_COLORS = ["#888888", "#4c72b0", "#dd8452", "#55a868"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/temporal_combined.yaml")
    p.add_argument("--save", required=True, help="Directory to save figures")
    p.add_argument("--n_images", type=int, default=24,
                   help="Number of failure images to show per grid")
    return p.parse_args()


def load_raw_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load image as uint8 RGB for display (no normalization)."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img


def run_inference_with_probs(model, loader, config, device):
    """Return (probs, preds, labels, indices) for the val set."""
    model.eval()
    model_type = config.model.type
    use_force = config.model.use_force
    all_probs, all_preds, all_labels, all_idx = [], [], [], []
    sample_idx = 0

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
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1)

            bs = labels.size(0)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_idx.extend(range(sample_idx, sample_idx + bs))
            sample_idx += bs

    return (np.concatenate(all_probs), np.concatenate(all_preds),
            np.concatenate(all_labels), all_idx)


def save_failure_grid(rows, title, save_path, n_cols=6, img_size=224):
    """Save a grid of failure images.

    rows: list of dicts with keys:
        image_path, true_label, pred_label, confidence, run_id, frame_idx,
        neighbor_labels (list of ints around this frame in the run)
    """
    n = len(rows)
    if n == 0:
        print(f"  No failures for: {title}")
        return

    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 3.4))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, row in enumerate(rows):
        ax = axes[i]
        img = load_raw_image(row["image_path"], img_size)
        ax.imshow(img)
        ax.axis("off")

        true_name  = CLASS_NAMES[row["true_label"]]
        pred_name  = CLASS_NAMES[row["pred_label"]]
        conf       = row["confidence"]
        run_short  = row["run_id"].split("-")[0]  # drop timestamp
        fidx       = row["frame_idx"]

        # Neighbour label strip (colour bar below image)
        nbrs = row["neighbor_labels"]
        if nbrs:
            strip_ax = ax.inset_axes([0, -0.07, 1, 0.05])
            # Convert class indices to RGB array for imshow
            cmap = {0: [0.53,0.53,0.53], 1: [0.30,0.45,0.69],
                    2: [0.87,0.52,0.32], 3: [0.33,0.66,0.41]}
            rgb = np.array([[cmap[l] for l in nbrs]], dtype=np.float32)
            strip_ax.imshow(rgb, aspect="auto")
            pos = row["neighbor_pos"]
            strip_ax.axvline(pos, color="red", linewidth=1.5)
            strip_ax.axis("off")

        ax.set_title(
            f"TRUE: {true_name}\nPRED: {pred_name} ({conf:.2f})\n"
            f"{run_short}\nframe {fidx}",
            fontsize=7, color="red", pad=2
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_run_sequence_plot(run_data, title, save_path):
    """For each val run plot the per-frame true label, pred label, and errors.

    run_data: dict of run_id -> dict with keys:
        frame_idxs, true_labels, pred_labels
    """
    runs = sorted(run_data.keys())
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), squeeze=False)

    for ax, run_id in zip(axes[:, 0], runs):
        d = run_data[run_id]
        fidx = np.array(d["frame_idxs"])
        true = np.array(d["true_labels"])
        pred = np.array(d["pred_labels"])
        errors = true != pred

        # True label band
        for cls in range(4):
            mask = true == cls
            if mask.any():
                ax.fill_between(fidx, cls - 0.4, cls + 0.4,
                                where=mask, color=CLASS_COLORS[cls],
                                alpha=0.4, label=CLASS_NAMES[cls])

        # Predicted label dots
        ax.scatter(fidx, pred, c=[CLASS_COLORS[p] for p in pred],
                   s=8, zorder=3, alpha=0.7)

        # Error markers
        if errors.any():
            ax.scatter(fidx[errors], true[errors],
                       marker="x", c="red", s=25, linewidths=1,
                       zorder=4, label="error")

        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlim(fidx.min(), fidx.max())
        ax.set_ylim(-0.6, 3.6)
        short = run_id.split("-")[0]
        ax.set_title(short, fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(4)]
    handles.append(plt.Line2D([0], [0], marker="x", color="red",
                               linestyle="None", markersize=6, label="error"))
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    # Dataset (no augmentation, no normalization for display purposes)
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

    split_cfg = config.training.split
    _, val_ds = split_by_run_stratified(
        dataset, val_ratio=config.training.val_ratio,
        seed=config.experiment.seed,
        val_n_override=split_cfg.val_n_override,
        train_n_override=split_cfg.train_n_override,
        frame_split_phantoms=split_cfg.frame_split_phantoms,
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Load model
    exp_name = config.experiment.name
    ckpt_path = Path(f"checkpoints/{exp_name}/best.pth")
    model = get_model_v2(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {ckpt_path}")

    # Inference
    probs, preds, labels, sample_idxs = run_inference_with_probs(
        model, loader, config, device
    )
    print(f"Val samples: {len(labels)}")

    # Build per-frame metadata table
    df = dataset.df
    val_df = df.iloc[val_ds.indices].copy().reset_index(drop=True)
    val_df["pred"]       = preds
    val_df["true_label"] = labels
    val_df["conf_pred"]  = probs[np.arange(len(preds)), preds]  # confidence for predicted class
    val_df["conf_true"]  = probs[np.arange(len(preds)), labels] # confidence for true class

    # Absolute image paths
    manifest_dir = Path(config.data.manifest).parent
    val_df["abs_image_path"] = val_df["image_path"].apply(
        lambda p: str(manifest_dir / p)
    )

    # Build neighbour label lookup: run_id -> sorted (frame_idx, label, pred)
    run_sequences = {}
    for run_id, g in val_df.groupby("run_id"):
        g_sorted = g.sort_values("frame_idx")
        run_sequences[run_id] = {
            "frame_idxs":  g_sorted["frame_idx"].tolist(),
            "true_labels": g_sorted["true_label"].tolist(),
            "pred_labels": g_sorted["pred"].tolist(),
        }

    # ── Sequence plot for all val runs ──────────────────────────────────────
    print("\nGenerating sequence plots...")
    # p4 val runs (crossed failures)
    p4_runs = {k: v for k, v in run_sequences.items() if "p4" in k}
    p5_runs = {k: v for k, v in run_sequences.items() if "p5" in k}
    all_runs = {k: v for k, v in run_sequences.items()}

    save_run_sequence_plot(all_runs, "Val runs — true label (band) vs predicted (dots), errors=red ×",
                           save_dir / "sequence_all_runs.png")

    # ── Failure grids ────────────────────────────────────────────────────────
    WINDOW = 20  # neighbour frames to show in strip

    def get_neighbor_labels(run_id, frame_idx, window=WINDOW):
        seq = run_sequences.get(run_id)
        if seq is None:
            return [], 0
        fidxs = seq["frame_idxs"]
        tlbls = seq["true_labels"]
        pos = min(range(len(fidxs)), key=lambda i: abs(fidxs[i] - frame_idx))
        lo = max(0, pos - window // 2)
        hi = min(len(fidxs), lo + window)
        lo = max(0, hi - window)
        return tlbls[lo:hi], pos - lo

    failure_cases = [
        # (true_class, pred_class, label)
        (2, 1, "crossed_as_single"),
        (3, 1, "double_as_single"),
        (2, 3, "crossed_as_double"),
        (3, 2, "double_as_crossed"),
        (1, 2, "single_as_crossed"),
        (1, 3, "single_as_double"),
    ]

    print("\nGenerating failure grids...")
    for true_cls, pred_cls, tag in failure_cases:
        mask = (val_df["true_label"] == true_cls) & (val_df["pred"] == pred_cls)
        subset = val_df[mask].copy()
        if len(subset) == 0:
            continue
        # Sort by confidence for wrong prediction (most confident first)
        subset = subset.sort_values("conf_pred", ascending=False)
        top = subset.head(args.n_images)

        rows = []
        for _, row in top.iterrows():
            nbr_labels, nbr_pos = get_neighbor_labels(row["run_id"], row["frame_idx"])
            rows.append({
                "image_path":    row["abs_image_path"],
                "true_label":    int(row["true_label"]),
                "pred_label":    int(row["pred"]),
                "confidence":    float(row["conf_pred"]),
                "run_id":        row["run_id"],
                "frame_idx":     int(row["frame_idx"]),
                "neighbor_labels": nbr_labels,
                "neighbor_pos":  nbr_pos,
            })

        true_name = CLASS_NAMES[true_cls]
        pred_name = CLASS_NAMES[pred_cls]
        title = (f"TRUE={true_name} → PREDICTED={pred_name}  "
                 f"(n={len(subset)}, showing top {len(rows)} by confidence)\n"
                 f"Red bar = current frame in sequence strip below each image")
        save_failure_grid(rows, title,
                          save_dir / f"failures_{tag}.png")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n=== Failure summary ===")
    errors = val_df[val_df["true_label"] != val_df["pred"]]
    tbl = errors.groupby(["true_label", "pred"]).size().reset_index(name="count")
    tbl["true_name"] = tbl["true_label"].map(lambda x: CLASS_NAMES[x])
    tbl["pred_name"] = tbl["pred"].map(lambda x: CLASS_NAMES[x])
    tbl = tbl.sort_values("count", ascending=False)
    print(f"  {'True':>10} → {'Pred':>10}   count")
    for _, r in tbl.iterrows():
        print(f"  {r['true_name']:>10} → {r['pred_name']:>10}   {int(r['count'])}")

    # ── Boundary proximity analysis ──────────────────────────────────────────
    print("\n=== Where in the run do failures occur? ===")
    for run_id, seq in run_sequences.items():
        fidxs = np.array(seq["frame_idxs"])
        true  = np.array(seq["true_labels"])
        pred  = np.array(seq["pred_labels"])
        n_total = len(fidxs)
        n_err   = (true != pred).sum()

        # Find label transition points (where true label changes)
        transitions = np.where(np.diff(true) != 0)[0]  # indices just before change

        # For each error, find distance to nearest transition
        err_idxs = np.where(true != pred)[0]
        if len(err_idxs) == 0 or len(transitions) == 0:
            continue

        dists = np.abs(err_idxs[:, None] - transitions[None, :]).min(axis=1)
        near_boundary = (dists <= 10).sum()  # within 10 frames of a transition
        short = run_id.split("-")[0]
        print(f"  {short}: {n_err}/{n_total} errors, "
              f"{near_boundary}/{n_err} within 10 frames of label transition "
              f"({near_boundary/n_err*100:.0f}% boundary)" if n_err > 0 else
              f"  {short}: 0 errors")

    print(f"\nAll figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
