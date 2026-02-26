"""Dump all frames from a run as a single grid JPEG.

Each cell shows the actual image with:
  - Top bar:    GT label color + class name
  - Bottom bar: Prediction color + class name  (red border if wrong)
  - Text row:   frame_idx or gy position (mm) depending on --sort-by

Frames are ordered by frame_idx (default) or by gy grid position (--sort-by gy),
so you can see the spatial transition along the phantom.

Usage (from scripts/classification/):
    python dump_run_frames.py \
        --runs p4_s2m_0_str-2026-02-04_16.19.45 p5_s2m_90_str-2026-02-04_16.35.21 \
        --config configs/temporal_combined.yaml \
        --sort-by gy \
        --save /tmp/frames

    # Dump all val runs:
    python dump_run_frames.py \
        --all-val \
        --config configs/temporal_combined.yaml \
        --sort-by gy \
        --save /tmp/frames
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
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

# Load labeling modules without name-colliding with classification config
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lab_cfg", "../labeling/config.py")
_lab_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lab_cfg)
sys.path.insert(0, "../labeling")
from generate_gt import world_to_grid_coords

CLASS_NAMES = ["none", "single", "crossed", "double"]
CLASS_BGR = {
    0: (160, 160, 160),   # gray   — none
    1: (180, 120,  50),   # blue   — single
    2: ( 80, 130, 220),   # orange — crossed
    3: (100, 170,  80),   # green  — double
}
ERROR_BGR  = (30,  30, 220)   # red border — wrong prediction
CELL_W     = 180
CELL_H     = 180
BAR_H      = 18
TEXT_H     = 14
N_COLS     = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",    nargs="+", default=None)
    p.add_argument("--all-val", action="store_true",
                   help="Dump all runs in the val set")
    p.add_argument("--config",  default="configs/temporal_combined.yaml")
    p.add_argument("--sort-by", choices=["frame_idx", "gy"], default="gy",
                   help="Sort frames by frame_idx or gy grid position (default: gy)")
    p.add_argument("--save",    required=True)
    return p.parse_args()


def load_gy_positions(run_id, phantom_type, rotation_deg):
    """Return df with frame_idx → gy (m) for every camera frame in the run."""
    cfg_path = Path("../labeling/configs/phantom_configs.json")
    with open(cfg_path) as f:
        cfgs = json.load(f)
    t_stl = np.array(cfgs[phantom_type]["gt_params"]["t_stl_to_world"])

    run_dir = Path("../labeling/rawdata") / run_id
    tcp  = pd.read_csv(run_dir / "tcp_pose.csv",
                       names=_lab_cfg.TCP_POSE_COLS, header=0)
    cams = pd.read_csv(run_dir / "camera_frames.csv",
                       names=_lab_cfg.CAMERA_FRAMES_COLS, header=0)

    tcp_times = tcp[_lab_cfg.TIME_COL].values
    records = []
    for _, row in cams.iterrows():
        i = np.argmin(np.abs(tcp_times - row["time"]))
        r = tcp.iloc[i]
        _, gy = world_to_grid_coords(r["x"], r["y"], r["z"], t_stl, rotation_deg)
        records.append({"frame_idx": int(row["frame_number"]), "gy": gy})
    return pd.DataFrame(records)


def load_run_rotation(run_id):
    with open("../labeling/configs/run_manifest.json") as f:
        data = json.load(f)
    for entry in data.get("runs", []):
        if entry["run_id"] == run_id:
            return entry.get("rotation_deg", 0)
    return 0


def make_cell(img_path, true_cls, pred_cls, label_text):
    """Build a CELL_W × (CELL_H + 2*BAR_H + TEXT_H) BGR cell."""
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    img = cv2.resize(img, (CELL_W, CELL_H))

    wrong = (true_cls != pred_cls)

    top_bar = np.full((BAR_H, CELL_W, 3), CLASS_BGR[true_cls], dtype=np.uint8)
    cv2.putText(top_bar, f"GT:{CLASS_NAMES[true_cls][:3].upper()}",
                (3, BAR_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)

    bot_bar = np.full((BAR_H, CELL_W, 3), CLASS_BGR[pred_cls], dtype=np.uint8)
    cv2.putText(bot_bar, f"PR:{CLASS_NAMES[pred_cls][:3].upper()}",
                (3, BAR_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)

    txt_row = np.zeros((TEXT_H, CELL_W, 3), dtype=np.uint8)
    cv2.putText(txt_row, str(label_text),
                (3, TEXT_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (200, 200, 200), 1, cv2.LINE_AA)

    cell = np.vstack([top_bar, img, bot_bar, txt_row])

    if wrong:
        cv2.rectangle(cell, (0, 0),
                      (cell.shape[1] - 1, cell.shape[0] - 1),
                      ERROR_BGR, 3)
    return cell


def build_grid(rows_data, title):
    n          = len(rows_data)
    n_cols     = N_COLS
    n_rows     = (n + n_cols - 1) // n_cols
    cell_full_h = CELL_H + 2 * BAR_H + TEXT_H

    title_h  = 36
    canvas_w = n_cols * CELL_W
    canvas_h = n_rows * cell_full_h + title_h
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    cv2.putText(canvas, title,
                (8, title_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    for i, d in enumerate(rows_data):
        cell = make_cell(d["img_path"], d["true_cls"], d["pred_cls"], d["label_text"])
        row = i // n_cols
        col = i  % n_cols
        y0  = title_h + row * cell_full_h
        x0  = col * CELL_W
        canvas[y0:y0 + cell_full_h, x0:x0 + CELL_W] = cell

    return canvas


def main():
    args     = parse_args()
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    is_temporal      = config.model.type in ("temporal", "temporal_force")
    temporal_frames  = config.model.temporal.num_frames if is_temporal else 1
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
        seed=config.experiment.seed,
        val_n_override={"p4": 2, "p5": 2, "none": 0},
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = get_model_v2(config.model).to(device)
    ckpt  = torch.load(f"checkpoints/{config.experiment.name}/best.pth",
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference over full val set
    all_preds, all_labels = [], []
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
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())

    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    df     = dataset.df
    val_df = df.iloc[val_ds.indices].copy().reset_index(drop=True)
    val_df["pred"]       = preds_np
    val_df["true_label"] = labels_np
    manifest_dir = Path(config.data.manifest).parent

    run_ids = args.runs if args.runs else val_df["run_id"].unique().tolist()

    for run_id in run_ids:
        run_df = val_df[val_df["run_id"] == run_id].copy()
        if len(run_df) == 0:
            print(f"  {run_id}: not in val set"); continue

        # Sort and compute label text
        if args.sort_by == "gy":
            if "gy" in run_df.columns:
                # gy already in manifest — use directly
                run_df = run_df.sort_values("gy")
            else:
                # fallback: compute from raw TCP data
                phantom_type = run_df["phantom_type"].iloc[0]
                rotation_deg = load_run_rotation(run_id)
                pos_df = load_gy_positions(run_id, phantom_type, rotation_deg)
                run_df = run_df.merge(pos_df, on="frame_idx", how="inner")
                run_df = run_df.sort_values("gy")
            run_df["label_text"] = run_df["gy"].apply(lambda v: f"{v*1000:+.0f}mm")
        else:
            run_df = run_df.sort_values("frame_idx")
            run_df["label_text"] = run_df["frame_idx"].apply(lambda v: f"f{v}")

        n_err = (run_df["true_label"] != run_df["pred"]).sum()
        print(f"{run_id}: {len(run_df)} frames, {n_err} errors ({n_err/len(run_df)*100:.1f}%)")

        rows_data = []
        for _, r in run_df.iterrows():
            rows_data.append({
                "img_path":   manifest_dir / r["image_path"],
                "true_cls":   int(r["true_label"]),
                "pred_cls":   int(r["pred"]),
                "label_text": r["label_text"],
            })

        sort_label = "sorted by gy" if args.sort_by == "gy" else "sorted by frame"
        title = (f"{run_id}  |  {n_err}/{len(run_df)} wrong  |  {sort_label}  |  "
                 f"top=GT  bot=PRED  red=error  "
                 f"gray=none  blue=single  orange=crossed  green=double")
        grid = build_grid(rows_data, title)

        out = save_dir / f"{run_id}.jpg"
        cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved: {out}  ({grid.shape[1]}x{grid.shape[0]}px)")


if __name__ == "__main__":
    main()
