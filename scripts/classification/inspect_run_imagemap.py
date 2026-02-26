"""Spatial image map: show every frame as a thumbnail at its (x,y) position
on the phantom, side-by-side GT label vs model prediction, with the tendon
bounding box overlaid.

This lets you see directly whether labeling is correct and where the model
is confused relative to the physical tendon location.

Usage (run from scripts/classification/):
    python inspect_run_imagemap.py \
        --runs p4_s2m_0_str-2026-02-04_16.19.45 p5_s2m_90_str-2026-02-04_16.35.21 \
        --config configs/temporal_combined.yaml \
        --save /tmp/imagemap
"""

import argparse
import json
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from config import load_config       # classification config
from dataset import TendonDatasetV2
from models_v2 import get_model_v2
from train_v2 import split_by_run_stratified

# Load labeling config by path to avoid name collision with classification config
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lab_cfg", "../labeling/config.py")
lab_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(lab_cfg)

sys.path.insert(0, "../labeling")
from generate_gt import world_to_grid_coords

CLASS_NAMES  = ["none", "single", "crossed", "double"]
CLASS_COLORS = {
    0: (0.53, 0.53, 0.53),   # gray
    1: (0.30, 0.45, 0.69),   # blue
    2: (0.87, 0.52, 0.32),   # orange
    3: (0.33, 0.66, 0.41),   # green
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=["p4_s2m_0_str-2026-02-04_16.19.45",
                            "p5_s2m_90_str-2026-02-04_16.35.21"])
    p.add_argument("--config", default="configs/temporal_combined.yaml")
    p.add_argument("--save",   required=True)
    p.add_argument("--zoom",   type=float, default=0.13,
                   help="Thumbnail zoom (0.13 ≈ 29px per image)")
    return p.parse_args()


def load_run_metadata(run_id):
    """Load rotation_deg for a run from run_manifest.json."""
    manifest_path = Path("../labeling/configs/run_manifest.json")
    with open(manifest_path) as f:
        data = json.load(f)
    for entry in data.get("runs", []):
        if entry["run_id"] == run_id:
            return entry.get("rotation_deg", 0)
    return 0


def load_phantom_bounds(phantom_type):
    """Return (t_stl_to_world, tendon_bounds, x_bounds, y_bounds)."""
    cfg_path = Path("../labeling/configs/phantom_configs.json")
    with open(cfg_path) as f:
        cfgs = json.load(f)
    ph = cfgs[phantom_type]["gt_params"]
    t  = np.array(ph["t_stl_to_world"])
    bounds = ph["tendon_bounds"]
    return t, bounds["x"], bounds["y"]


def get_grid_positions(run_id, rotation_deg, t_stl_to_world):
    """Load tcp_pose + camera_frames, return df with frame_idx, gx, gy."""
    run_dir = Path("../labeling/rawdata") / run_id
    tcp  = pd.read_csv(run_dir / "tcp_pose.csv",
                       names=lab_cfg.TCP_POSE_COLS, header=0)
    cams = pd.read_csv(run_dir / "camera_frames.csv",
                       names=lab_cfg.CAMERA_FRAMES_COLS, header=0)

    tcp_times = tcp["time"].values
    records = []
    for _, row in cams.iterrows():
        i = np.argmin(np.abs(tcp_times - row["time"]))
        r = tcp.iloc[i]
        gx, gy = world_to_grid_coords(
            r["x"], r["y"], r["z"], t_stl_to_world, rotation_deg
        )
        records.append({"frame_idx": int(row["frame_number"]),
                         "gx": gx, "gy": gy})
    return pd.DataFrame(records)


def build_image_map(ax, merged, manifest_dir, x_bounds, y_bounds,
                    color_col, title, zoom, error_mask=None):
    """Place thumbnails at (gx, gy) positions on ax, bordered by color_col class."""

    ax.set_facecolor("#1a1a1a")

    for _, row in merged.iterrows():
        img_path = manifest_dir / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))

        cls = int(row[color_col])
        color = CLASS_COLORS[cls]

        # Colored border: 4px border in class color
        border = 4
        bordered = np.ones((64 + 2*border, 64 + 2*border, 3), dtype=np.uint8)
        bordered[:] = (np.array(color) * 255).astype(np.uint8)
        bordered[border:-border, border:-border] = img

        # Extra red ring for errors
        is_err = (error_mask is not None) and bool(error_mask.loc[row.name])
        if is_err:
            bordered[0:2, :] = [220, 30, 30]
            bordered[-2:, :] = [220, 30, 30]
            bordered[:, 0:2] = [220, 30, 30]
            bordered[:, -2:] = [220, 30, 30]

        oi = OffsetImage(bordered, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(oi, (row["gx"], row["gy"]),
                            frameon=False, pad=0)
        ax.add_artist(ab)

    # Bounding box
    rect = mpatches.Rectangle(
        (x_bounds[0], y_bounds[0]),
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        linewidth=2, edgecolor="white", facecolor="none",
        linestyle="--", label="tendon_bounds", zorder=10
    )
    ax.add_patch(rect)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(4)]
    if error_mask is not None:
        handles.append(mpatches.Patch(color=(0.86, 0.12, 0.12),
                                       label="error (red ring)"))
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              facecolor="#333333", labelcolor="white")

    # Axis
    pad_x = (x_bounds[1] - x_bounds[0]) * 2.5
    pad_y = (y_bounds[1] - y_bounds[0]) * 0.15
    ax.set_xlim(x_bounds[0] - pad_x, x_bounds[1] + pad_x)
    ax.set_ylim(y_bounds[0] - pad_y, y_bounds[1] + pad_y)
    ax.set_xlabel("Grid X (m)", color="white")
    ax.set_ylabel("Grid Y (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")


def main():
    args = parse_args()
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

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

    # Load model
    model = get_model_v2(config.model).to(device)
    ckpt  = torch.load(f"checkpoints/{config.experiment.name}/best.pth",
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {config.experiment.name}")

    # Run inference
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if config.model.type in ("temporal", "temporal_force"):
                images, forces, labels, _, mask = batch
                if config.model.type == "temporal":
                    images = images.to(device)
                mask = mask.to(device)
            else:
                images, forces, labels, _ = batch
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
            probs  = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())

    probs_np  = np.concatenate(all_probs)
    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    df = dataset.df
    val_df = df.iloc[val_ds.indices].copy().reset_index(drop=True)
    val_df["pred"]       = preds_np
    val_df["true_label"] = labels_np
    val_df["conf_pred"]  = probs_np[np.arange(len(preds_np)), preds_np]

    manifest_dir = Path(config.data.manifest).parent

    for run_id in args.runs:
        print(f"\n=== {run_id} ===")

        run_df = val_df[val_df["run_id"] == run_id].copy()
        if len(run_df) == 0:
            print(f"  Not in val set — skipping")
            continue

        phantom_type = run_id.split("_")[0]
        rotation_deg = load_run_metadata(run_id)
        t_stl, x_bounds, y_bounds = load_phantom_bounds(phantom_type)

        pos_df = get_grid_positions(run_id, rotation_deg, t_stl)
        merged = run_df.merge(pos_df, on="frame_idx", how="inner")

        n_err = (merged["true_label"] != merged["pred"]).sum()
        print(f"  {len(merged)} frames, {n_err} errors ({n_err/len(merged)*100:.1f}%)")

        error_mask = merged["true_label"] != merged["pred"]

        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        fig.patch.set_facecolor("#111111")

        # Left: GT labels
        build_image_map(
            axes[0], merged, manifest_dir,
            x_bounds, y_bounds,
            color_col="true_label",
            title=f"Ground Truth labels\n{run_id.split('-')[0]}",
            zoom=args.zoom,
            error_mask=None,
        )

        # Right: Model predictions (errors marked with red ring)
        build_image_map(
            axes[1], merged, manifest_dir,
            x_bounds, y_bounds,
            color_col="pred",
            title=f"Model predictions  ({n_err}/{len(merged)} wrong)\n{run_id.split('-')[0]}",
            zoom=args.zoom,
            error_mask=error_mask,
        )

        fig.suptitle(
            f"{run_id}   |   {config.experiment.name}",
            color="white", fontsize=11, fontweight="bold"
        )
        plt.tight_layout()

        out = save_dir / f"{run_id.split('-')[0]}_imagemap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
        plt.close(fig)
        print(f"  Saved: {out}")

    print(f"\nDone. All figures in: {save_dir}")


if __name__ == "__main__":
    main()
