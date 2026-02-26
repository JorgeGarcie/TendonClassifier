"""Visualize specific runs spatially: plot each frame's TCP (x,y) position,
color by GT label and mark model predictions, then show the actual images
for frames where the model is wrong.

Usage:
    python inspect_run_spatial.py \
        --runs p4_s2m_0_str-2026-02-04_16.19.45 p5_s2m_90_str-2026-02-04_16.35.21 \
        --config configs/temporal_combined.yaml \
        --save /tmp/spatial_inspect
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
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from config import load_config
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_v2 import split_by_run_stratified

RAWDATA_DIR = Path("../labeling/rawdata")
CLASS_NAMES  = ["none", "single", "crossed", "double"]
CLASS_COLORS = ["#888888", "#4c72b0", "#dd8452", "#55a868"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=["p4_s2m_0_str-2026-02-04_16.19.45",
                            "p5_s2m_90_str-2026-02-04_16.35.21"])
    p.add_argument("--config", default="configs/temporal_combined.yaml")
    p.add_argument("--save", required=True)
    p.add_argument("--n_failure_images", type=int, default=30,
                   help="Max failure images to show per run")
    return p.parse_args()


def load_tcp_for_run(run_id: str) -> pd.DataFrame:
    """Load tcp_pose and camera_frames, return df with frame_idx, x, y, z."""
    run_dir = RAWDATA_DIR / run_id
    tcp   = pd.read_csv(run_dir / "tcp_pose.csv")
    cams  = pd.read_csv(run_dir / "camera_frames.csv")

    # For each camera frame time, find nearest TCP pose
    tcp_times = tcp["time"].values

    def nearest_tcp(t):
        idx = np.argmin(np.abs(tcp_times - t))
        return tcp.iloc[idx][["x", "y", "z"]].values

    records = []
    for _, row in cams.iterrows():
        xyz = nearest_tcp(row["time"])
        records.append({
            "frame_idx": int(row["frame_number"]),
            "tcp_x": xyz[0], "tcp_y": xyz[1], "tcp_z": xyz[2],
        })
    return pd.DataFrame(records)


def run_inference_for_run(run_id, val_df_full, model, config, device, dataset):
    """Return predictions and probabilities for a specific run."""
    run_rows = val_df_full[val_df_full["run_id"] == run_id]
    if len(run_rows) == 0:
        return None
    return run_rows


def load_raw_image(image_path, size=224):
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))


def save_spatial_plot(run_df, tcp_df, run_id, save_path):
    """Plot TCP (x,y) positions colored by GT label and mark prediction errors."""
    merged = run_df.merge(tcp_df, on="frame_idx", how="inner")
    merged = merged.sort_values("frame_idx")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, color_by, title_suffix in zip(
        axes,
        ["true_label", "pred"],
        ["GT label", "Model prediction"]
    ):
        for cls in range(4):
            mask = merged[color_by] == cls
            if not mask.any():
                continue
            ax.scatter(
                merged.loc[mask, "tcp_x"] * 1000,  # m → mm
                merged.loc[mask, "tcp_y"] * 1000,
                c=CLASS_COLORS[cls], s=18, alpha=0.7,
                label=CLASS_NAMES[cls], zorder=2
            )

        # Mark errors
        errors = merged["true_label"] != merged["pred"]
        if errors.any():
            ax.scatter(
                merged.loc[errors, "tcp_x"] * 1000,
                merged.loc[errors, "tcp_y"] * 1000,
                facecolors="none", edgecolors="red",
                s=40, linewidths=0.8, zorder=3, label="error"
            )

        # Arrow showing trajectory direction
        xs = merged["tcp_x"].values * 1000
        ys = merged["tcp_y"].values * 1000
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        ax.set_xlabel("TCP x (mm)")
        ax.set_ylabel("TCP y (mm)")
        ax.set_title(f"{run_id.split('-')[0]}\n{title_suffix}", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Error rate annotation
    n_err = (merged["true_label"] != merged["pred"]).sum()
    fig.suptitle(
        f"{run_id}\n{n_err}/{len(merged)} frames wrong ({n_err/len(merged)*100:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return merged


def save_failure_images(merged, run_id, manifest_dir, save_path, n_max=30):
    """Show actual images for wrong predictions, sorted by model confidence."""
    errors = merged[merged["true_label"] != merged["pred"]].copy()
    errors = errors.sort_values("conf_pred", ascending=False).head(n_max)

    if len(errors) == 0:
        print(f"  No errors to show for {run_id}")
        return

    n = len(errors)
    n_cols = min(6, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i, (_, row) in enumerate(errors.iterrows()):
        ax = axes[i]
        img_path = manifest_dir / row["image_path"]
        img = load_raw_image(img_path)
        ax.imshow(img)
        ax.axis("off")

        true_name = CLASS_NAMES[int(row["true_label"])]
        pred_name = CLASS_NAMES[int(row["pred"])]
        ax.set_title(
            f"TRUE: {true_name}\nPRED: {pred_name} ({row['conf_pred']:.2f})\n"
            f"frame {int(row['frame_idx'])}  "
            f"({row['tcp_x']*1000:.1f}, {row['tcp_y']*1000:.1f}) mm",
            fontsize=7, color="red", pad=2
        )

    fig.suptitle(
        f"{run_id.split('-')[0]} — failure images\n"
        f"sorted by model confidence (most wrong first)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_correct_images(merged, run_id, manifest_dir, save_path, cls, n_max=12):
    """Show correct predictions for a class for comparison."""
    correct = merged[
        (merged["true_label"] == cls) & (merged["pred"] == cls)
    ].copy().head(n_max)

    if len(correct) == 0:
        return

    n = len(correct)
    n_cols = min(6, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 3.0))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i, (_, row) in enumerate(correct.iterrows()):
        ax = axes[i]
        img = load_raw_image(manifest_dir / row["image_path"])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"TRUE: {CLASS_NAMES[cls]}  ✓\n"
            f"frame {int(row['frame_idx'])}  "
            f"({row['tcp_x']*1000:.1f}, {row['tcp_y']*1000:.1f}) mm",
            fontsize=7, color="green", pad=2
        )

    fig.suptitle(
        f"{run_id.split('-')[0]} — correct {CLASS_NAMES[cls]} predictions (for comparison)",
        fontsize=10, fontweight="bold"
    )
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

    is_temporal = config.model.type in ("temporal", "temporal_force")
    temporal_frames = config.model.temporal.num_frames if is_temporal else 1
    return_force_seq = config.model.type == "temporal_force" or (
        config.model.type == "temporal" and config.model.use_force
    )

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

    # Load model
    model = get_model_v2(config.model).to(device)
    ckpt_path = Path(f"checkpoints/{config.experiment.name}/best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Run inference over full val set
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if config.model.type in ("temporal", "temporal_force"):
                images, forces, labels, depth_gt, mask = batch
                if config.model.type == "temporal":
                    images = images.to(device)
                mask = mask.to(device)
            else:
                images, forces, labels, depth_gt = batch
                images = images.to(device)
                mask = None
            forces = forces.to(device)

            if config.model.type == "temporal_force":
                out = model(forces, mask)
            elif config.model.type == "temporal":
                out = model(images, forces, mask) if config.model.use_force else model(images, mask=mask)
            elif config.model.type == "spatial_force":
                out = model(forces)
            else:
                out = model(images, forces) if config.model.use_force else model(images)

            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())

    probs_np = np.concatenate(all_probs)
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    # Build annotated val dataframe
    df = dataset.df
    val_df = df.iloc[val_ds.indices].copy().reset_index(drop=True)
    val_df["pred"]       = preds_np
    val_df["true_label"] = labels_np
    val_df["conf_pred"]  = probs_np[np.arange(len(preds_np)), preds_np]

    manifest_dir = Path(config.data.manifest).parent

    # Process each requested run
    for run_id in args.runs:
        print(f"\n{'='*60}")
        print(f"  {run_id}")
        print(f"{'='*60}")

        run_df = val_df[val_df["run_id"] == run_id].copy()
        if len(run_df) == 0:
            print(f"  WARNING: {run_id} not found in val set")
            continue

        # Load TCP positions
        tcp_df = load_tcp_for_run(run_id)

        tag = run_id.split("-")[0]

        # Spatial plot (GT vs predictions)
        merged = save_spatial_plot(
            run_df, tcp_df, run_id,
            save_dir / f"{tag}_spatial.png"
        )

        n_err = (merged["true_label"] != merged["pred"]).sum()
        print(f"  Frames: {len(merged)}, Errors: {n_err} ({n_err/len(merged)*100:.1f}%)")
        print(f"  Error breakdown:")
        for (tl, pl), cnt in (merged[merged["true_label"] != merged["pred"]]
                               .groupby(["true_label", "pred"]).size().items()):
            print(f"    {CLASS_NAMES[int(tl)]} → {CLASS_NAMES[int(pl)]}: {cnt}")

        # Failure images
        save_failure_images(
            merged, run_id, manifest_dir,
            save_dir / f"{tag}_failure_images.png",
            n_max=args.n_failure_images
        )

        # Correct predictions for main tendon class (for comparison)
        # Figure out the non-single class in this run
        tendon_cls = merged["true_label"].value_counts().index
        tendon_cls = [c for c in tendon_cls if c != 1]  # exclude single
        for cls in tendon_cls:
            save_correct_images(
                merged, run_id, manifest_dir,
                save_dir / f"{tag}_correct_{CLASS_NAMES[int(cls)]}.png",
                cls=int(cls)
            )

    print(f"\nAll saved to: {save_dir}")


if __name__ == "__main__":
    main()
