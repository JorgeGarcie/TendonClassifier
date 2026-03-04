"""Visualize GT labels spatially: frames sorted by their gy grid position
(along-tendon axis), laid out as a flat grid left→right top→bottom.
Each thumbnail has a colored border showing its GT class.
No spatial overlap — every frame is clearly visible.

Pure OpenCV output — one JPEG per run.

Usage (from scripts/labeling/):
    python visualize_gt_spatial.py --run-id p4_s2m_0_str-2026-02-04_16.19.45 --save /tmp/gt_vis
    python visualize_gt_spatial.py --all --save /tmp/gt_vis
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from generate_gt import world_to_grid_coords

CLASS_BGR = {
    0: (160, 160, 160),   # gray   — none
    1: (180, 120,  50),   # blue   — single
    2: ( 80, 130, 220),   # orange — crossed
    3: (100, 170,  80),   # green  — double
}
CLASS_NAMES = {0: "none", 1: "single", 2: "crossed", 3: "double"}

THUMB   = 96    # thumbnail side (px)
BORDER  = 5     # colored border width (px)
TEXT_H  = 14    # gy label row at bottom of each cell
CELL_W  = THUMB + 2 * BORDER
CELL_H  = THUMB + 2 * BORDER + TEXT_H
N_COLS  = 15


def load_phantom_info(phantom_type):
    cfg_path = Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE
    with open(cfg_path) as f:
        cfgs = json.load(f)
    ph = cfgs[phantom_type]["gt_params"]
    t  = np.array(ph["t_stl_to_world"])
    bounds = ph["tendon_bounds"]
    return t, bounds["x"], bounds["y"]


def get_frame_positions(run_id, t_stl_to_world, rotation_deg):
    run_dir = Path(config.DATA_ROOT) / run_id
    tcp  = pd.read_csv(run_dir / "tcp_pose.csv",
                       names=config.TCP_POSE_COLS, header=0)
    cams = pd.read_csv(run_dir / "camera_frames.csv",
                       names=config.CAMERA_FRAMES_COLS, header=0)
    tcp_times = tcp[config.TIME_COL].values
    records = []
    for _, row in cams.iterrows():
        i = np.argmin(np.abs(tcp_times - row["time"]))
        r = tcp.iloc[i]
        gx, gy = world_to_grid_coords(
            r["x"], r["y"], r["z"], t_stl_to_world, rotation_deg
        )
        records.append({"frame_idx": int(row["frame_number"]), "gx": gx, "gy": gy})
    return pd.DataFrame(records)


def make_cell(img_path, cls, gy_mm):
    """Build one CELL_W × CELL_H BGR cell with colored border and gy label."""
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    img = cv2.resize(img, (THUMB, THUMB))

    color = CLASS_BGR[cls]
    cell = np.full((THUMB + 2 * BORDER, CELL_W, 3), color, dtype=np.uint8)
    cell[BORDER:BORDER + THUMB, BORDER:BORDER + THUMB] = img

    txt_row = np.zeros((TEXT_H, CELL_W, 3), dtype=np.uint8)
    cv2.putText(txt_row, f"{gy_mm:+.1f}mm",
                (2, TEXT_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([cell, txt_row])


def build_grid(merged, manifest_dir):
    """Sort by gy, render as flat grid."""
    merged = merged.sort_values("gy").reset_index(drop=True)
    n       = len(merged)
    n_cols  = N_COLS
    n_rows  = (n + n_cols - 1) // n_cols

    canvas_w = n_cols * CELL_W
    canvas_h = n_rows * CELL_H

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, (_, row) in enumerate(merged.iterrows()):
        cell = make_cell(
            manifest_dir / row["image_path"],
            int(row["tendon_type"]),
            row["gy"] * 1000,   # m → mm
        )
        r  = i // n_cols
        c  = i  % n_cols
        y0 = r * CELL_H
        x0 = c * CELL_W
        canvas[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell

    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--all",    action="store_true")
    p.add_argument("--save",   required=True)
    args = p.parse_args()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(config.OUTPUT_ROOT) / "gt_dataset" / "gt_manifest.csv"
    df = pd.read_csv(manifest_path)
    manifest_dir = manifest_path.parent

    with open(Path(config.CONFIGS_ROOT) / "run_manifest.json") as f:
        run_meta = {r["run_id"]: r for r in json.load(f)["runs"]}

    run_ids = df["run_id"].unique().tolist()
    if args.run_id:
        run_ids = [args.run_id]
    elif not args.all:
        p.error("Specify --run-id or --all")

    for run_id in sorted(run_ids):
        run_df = df[df["run_id"] == run_id].copy()
        if len(run_df) == 0:
            continue

        meta         = run_meta.get(run_id, {})
        phantom_type = run_df["phantom_type"].iloc[0]
        rotation_deg = meta.get("rotation_deg", 0)

        t_stl, x_bounds, y_bounds = load_phantom_info(phantom_type)
        pos_df = get_frame_positions(run_id, t_stl, rotation_deg)
        merged = run_df.merge(pos_df, on="frame_idx", how="inner")
        if len(merged) == 0:
            print(f"  {run_id}: no frames matched")
            continue

        grid = build_grid(merged, manifest_dir)

        # Legend bar
        legend_h = 24
        legend_w = grid.shape[1]
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        lx = 8
        for cls_id, name in CLASS_NAMES.items():
            cv2.rectangle(legend, (lx, 6), (lx + 14, legend_h - 4),
                          CLASS_BGR[cls_id], -1)
            cv2.putText(legend, name, (lx + 17, legend_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
            lx += 80

        # Title bar
        title_h = 26
        title = np.zeros((title_h, legend_w, 3), dtype=np.uint8)
        cv2.putText(title,
                    f"{run_id}   rot={rotation_deg}   {len(merged)} frames  "
                    f"(sorted by gy position)",
                    (6, title_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out_img  = np.vstack([title, legend, grid])
        out_path = save_dir / f"{run_id}.jpg"
        cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Saved: {out_path}  ({out_img.shape[1]}×{out_img.shape[0]}px)")


if __name__ == "__main__":
    main()
