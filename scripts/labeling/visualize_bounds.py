"""Visualize TCP trajectory and tendon bounds for a run.

Shows a 2D scatter plot of TCP (x, y) positions visited during a run,
colored by tendon presence, overlaid with the current tendon_bounds box
from phantom_configs.json.

Helps manually calibrate the tendon bounds — you can see where the probe
was when the GT says "no tendon" but you might expect one (boundary cases).

NOTE: All positions are in STL grid coordinates (after center crop is applied
to images). Use this AFTER running generate_gt.py so image crops match.

Usage:
    python visualize_bounds.py --run-id p4_m2s_35N-2025-11-25_22.46.29
    python visualize_bounds.py --run-id p4_m2s_35N-...  --show-frames
    python visualize_bounds.py --run-id p4_m2s_35N-...  --save /tmp/plots
    python visualize_bounds.py --all --save /tmp/plots
    python visualize_bounds.py --list
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from generate_gt import center_crop, world_to_grid_coords


def load_run_manifest():
    path = Path(config.CONFIGS_ROOT) / "run_manifest.json"
    with open(path) as f:
        return json.load(f)


def load_phantom_configs():
    path = Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE
    with open(path) as f:
        return json.load(f)


def load_tcp_for_run(run_path: Path) -> pd.DataFrame:
    tcp_csv = run_path / "tcp_pose.csv"
    return pd.read_csv(tcp_csv, names=config.TCP_POSE_COLS, header=0)


def load_wrench_for_run(run_path: Path) -> pd.DataFrame:
    wrench_csv = run_path / "wrench_data.csv"
    df = pd.read_csv(wrench_csv, names=config.WRENCH_DATA_COLS, header=0)
    return df[df["sensor"] == config.WRENCH_SENSOR].copy()


def load_camera_frames(run_path: Path) -> pd.DataFrame:
    cam_csv = run_path / "camera_frames.csv"
    return pd.read_csv(cam_csv, names=config.CAMERA_FRAMES_COLS, header=0)


def get_run_path(run_id: str) -> Path:
    return Path(config.DATA_ROOT) / run_id


def visualize_run(run_id: str, run_metadata: dict, phantom_cfgs: dict,
                  show_frames: bool = False, save_dir=None):
    run_path = get_run_path(run_id)
    if not run_path.exists():
        print(f"Run path not found: {run_path}")
        return

    # Get phantom config
    meta = run_metadata.get(run_id, {})
    phantom_type = run_id.split("_")[0]
    motion_type = meta.get("motion_type", "default")
    rotation_deg = meta.get("rotation_deg", 0)

    ph_cfg = phantom_cfgs.get(phantom_type, {})
    gt_params = ph_cfg.get("gt_params", {})
    tendon_bounds = gt_params.get("tendon_bounds", {})
    t_stl_to_world = np.array(gt_params.get("t_stl_to_world", np.eye(4)))

    # Load data
    tcp_df = load_tcp_for_run(run_path)
    wrench_df = load_wrench_for_run(run_path)
    cam_df = load_camera_frames(run_path)

    # Compute force magnitude per frame via nearest-neighbor timestamp
    force_mag = []
    gx_list, gy_list = [], []

    for _, cam_row in cam_df.iterrows():
        t = cam_row[config.TIME_COL]

        # Nearest TCP
        idx = (tcp_df["time"] - t).abs().idxmin()
        tcp_row = tcp_df.loc[idx]
        tcp_x, tcp_y, tcp_z = tcp_row["x"], tcp_row["y"], tcp_row["z"]

        # Grid coords
        gx, gy = world_to_grid_coords(tcp_x, tcp_y, tcp_z,
                                       t_stl_to_world, rotation_deg)
        gx_list.append(gx)
        gy_list.append(gy)

        # Nearest force
        if len(wrench_df) > 0:
            fidx = (wrench_df["time"] - t).abs().idxmin()
            frow = wrench_df.loc[fidx]
            fm = np.sqrt(frow["fx"]**2 + frow["fy"]**2 + frow["fz"]**2)
            force_mag.append(fm)
        else:
            force_mag.append(0.0)

    gx_arr = np.array(gx_list)
    gy_arr = np.array(gy_list)
    force_arr = np.array(force_mag)

    # Contact mask
    in_contact = force_arr >= config.FORCE_THRESHOLD_N

    # Tendon bounds box
    x_bounds = tendon_bounds.get("x", [-0.01, 0.01])
    y_bounds = tendon_bounds.get("y", [-0.075, 0.075])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2 if show_frames else 1,
                             figsize=(14 if show_frames else 7, 7))
    ax = axes[0] if show_frames else axes

    # Scatter: not in contact (gray), in contact (colored by x position)
    ax.scatter(gx_arr[~in_contact], gy_arr[~in_contact],
               c="lightgray", s=4, alpha=0.4, label="No contact")
    sc = ax.scatter(gx_arr[in_contact], gy_arr[in_contact],
                    c=force_arr[in_contact], cmap="plasma",
                    s=6, alpha=0.7, label="In contact")
    plt.colorbar(sc, ax=ax, label="Force magnitude (N)")

    # Current tendon bounds box
    rect = mpatches.Rectangle(
        (x_bounds[0], y_bounds[0]),
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        linewidth=2, edgecolor="red", facecolor="none",
        label=f"tendon_bounds x={x_bounds} y={y_bounds}"
    )
    ax.add_patch(rect)

    ax.set_xlabel("Grid X (m)")
    ax.set_ylabel("Grid Y (m)")
    ax.set_title(f"TCP trajectory — {run_id}\n"
                 f"phantom={phantom_type}, motion={motion_type}, "
                 f"rotation={rotation_deg}°")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Print stats
    print(f"\nRun: {run_id}")
    print(f"Phantom: {phantom_type}, motion: {motion_type}")
    print(f"Total frames: {len(cam_df)}")
    print(f"Frames in contact (force >= {config.FORCE_THRESHOLD_N}N): "
          f"{in_contact.sum()}")
    print(f"TCP X range (in contact): "
          f"[{gx_arr[in_contact].min():.4f}, {gx_arr[in_contact].max():.4f}] m")
    print(f"TCP Y range (in contact): "
          f"[{gy_arr[in_contact].min():.4f}, {gy_arr[in_contact].max():.4f}] m")
    print(f"Current tendon_bounds: x={x_bounds}, y={y_bounds}")

    # --- Sample center-cropped frames ---
    if show_frames:
        ax2 = axes[1]
        frames_dir = run_path / "frames"
        frame_files = sorted(frames_dir.glob("*.jpg"))

        # Pick a frame near the middle of the contact window
        contact_indices = np.where(in_contact)[0]
        if len(contact_indices) > 0:
            mid_idx = contact_indices[len(contact_indices) // 2]
            frame_row = cam_df.iloc[mid_idx]
            frame_num = int(frame_row[config.FRAME_IDX_COL])
            frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"

            if frame_path.exists():
                img = cv2.imread(str(frame_path))
                img_crop = center_crop(img, config.CROP_SIZE)
                img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                ax2.imshow(img_rgb)
                ax2.set_title(f"Center crop ({config.CROP_SIZE}x{config.CROP_SIZE})\n"
                              f"Frame {frame_num} (mid-contact)")
                ax2.axis("off")
            else:
                ax2.text(0.5, 0.5, "Frame not found", ha="center", va="center")
        else:
            ax2.text(0.5, 0.5, "No contact frames found",
                     ha="center", va="center")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{run_id}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TCP trajectory and tendon bounds for a run"
    )
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID to visualize")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all runs (requires --save)")
    parser.add_argument("--list", action="store_true",
                        help="List all available run IDs")
    parser.add_argument("--show-frames", action="store_true",
                        help="Show a sample center-cropped frame alongside the plot")
    parser.add_argument("--save", type=str, default=None, metavar="DIR",
                        help="Save plots as PNGs to this directory instead of showing")
    args = parser.parse_args()

    run_manifest_data = load_run_manifest()
    phantom_cfgs = load_phantom_configs()

    run_metadata = {}
    for run_info in run_manifest_data.get("runs", []):
        rid = run_info["run_id"]
        run_metadata[rid] = {
            "motion_type": run_info.get("motion_type"),
            "rotation_deg": run_info.get("rotation_deg", 0),
        }

    if args.list:
        print("Available runs:")
        for rid in sorted(run_metadata.keys()):
            print(f"  {rid}")
        return

    if args.all:
        if args.save is None:
            parser.error("--all requires --save DIR")
        for rid in sorted(run_metadata.keys()):
            visualize_run(rid, run_metadata, phantom_cfgs,
                          show_frames=args.show_frames, save_dir=args.save)
        return

    if not args.run_id:
        parser.print_help()
        sys.exit(1)

    visualize_run(args.run_id, run_metadata, phantom_cfgs,
                  show_frames=args.show_frames, save_dir=args.save)


if __name__ == "__main__":
    main()
