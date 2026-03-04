"""Per-frame inspection tool: trajectory position + camera image side by side.

For each frame in a run, saves a PNG with:
  Left  — 2D TCP trajectory scatter (colored by force) + tendon bounds box +
           current frame position highlighted as a star.
  Right — center-cropped camera image from gt_dataset/images/.

The GT label (detected / none) and force are shown in the title so you can
verify that the bounding box correctly captures tendon-contact frames.

Usage:
    python inspect_frames.py --run-id p1_t_0_nat-2026-02-04_15.32.47 \\
                             --save-dir /tmp/inspect/p1_t_nat
    python inspect_frames.py --run-id p4_m2s_35N-2025-11-25_22.46.29 \\
                             --save-dir /tmp/inspect/p4_m2s
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import config
from generate_gt import world_to_grid_coords


LABEL_NAMES = {0: "none", 1: "single", 2: "crossed", 3: "double"}
LABEL_COLORS = {0: "red", 1: "limegreen", 2: "dodgerblue", 3: "orange"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_manifest(run_id):
    """Return list of manifest rows for this run_id."""
    manifest_path = Path(config.OUTPUT_ROOT) / "gt_dataset" / "gt_manifest.csv"
    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run_id:
                rows.append(row)
    return rows


def load_tcp_grid(run_path, t_stl_to_world, rotation_deg):
    """Return arrays (times, gx, gy) for all TCP rows."""
    times, gx_list, gy_list = [], [], []
    with open(run_path / "tcp_pose.csv") as f:
        for row in csv.DictReader(f):
            t = float(row["time"])
            gx, gy = world_to_grid_coords(
                float(row["x"]), float(row["y"]), float(row["z"]),
                t_stl_to_world, rotation_deg
            )
            times.append(t)
            gx_list.append(gx)
            gy_list.append(gy)
    return np.array(times), np.array(gx_list), np.array(gy_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inspector(run_id, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- configs ---
    with open(Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE) as f:
        phantom_cfgs = json.load(f)
    with open(Path(config.CONFIGS_ROOT) / "run_manifest.json") as f:
        run_manifest = json.load(f)

    run_meta = next((r for r in run_manifest["runs"] if r["run_id"] == run_id), None)
    if run_meta is None:
        print(f"Run not found in manifest: {run_id}")
        sys.exit(1)

    phantom_type = run_meta["phantom_type"]
    motion_type  = run_meta["motion_type"]
    rotation_deg = run_meta.get("rotation_deg", 0)

    gt_params       = phantom_cfgs[phantom_type]["gt_params"]
    t_stl_to_world  = np.array(gt_params["t_stl_to_world"])
    cfg_entry       = phantom_cfgs[phantom_type]["configs"].get(motion_type, {})

    # Bounds (per-config override or phantom-level)
    x_bounds = cfg_entry.get("tendon_x_bounds", gt_params["tendon_bounds"]["x"])
    y_bounds = cfg_entry.get("tendon_y_bounds", gt_params["tendon_bounds"]["y"])

    run_path = Path(config.DATA_ROOT) / run_id
    gt_images_root = Path(config.OUTPUT_ROOT) / "gt_dataset"

    # Load manifest rows for this run
    manifest_rows = load_manifest(run_id)
    if not manifest_rows:
        print(f"No manifest rows for {run_id} — run generate_gt.py first.")
        sys.exit(1)
    print(f"Found {len(manifest_rows)} manifest frames for {run_id}")

    # Load full TCP trajectory for background scatter
    tcp_times, tcp_gx, tcp_gy = load_tcp_grid(run_path, t_stl_to_world, rotation_deg)

    # Load wrench for coloring
    wrench_force = {}
    with open(run_path / "wrench_data.csv") as f:
        for row in csv.DictReader(f):
            if row["sensor"] == config.WRENCH_SENSOR:
                t = float(row["time"])
                fx, fy, fz = float(row["fx"]), float(row["fy"]), float(row["fz"])
                wrench_force[t] = np.sqrt(fx**2 + fy**2 + fz**2)
    w_times = np.array(sorted(wrench_force.keys()))
    w_force = np.array([wrench_force[t] for t in w_times])

    # Compute force for each TCP sample
    tcp_force = np.interp(tcp_times, w_times, w_force)

    # --- Build reusable figure ---
    fig, (ax_traj, ax_img) = plt.subplots(1, 2, figsize=(13, 6))

    # Background scatter (all TCP trajectory)
    sc = ax_traj.scatter(
        tcp_gx, tcp_gy, c=tcp_force, cmap="plasma",
        s=3, alpha=0.4, vmin=0, vmax=tcp_force.max(), zorder=1
    )
    plt.colorbar(sc, ax=ax_traj, label="Force (N)", shrink=0.8)

    # Bounds box
    rect = mpatches.Rectangle(
        (x_bounds[0], y_bounds[0]),
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        linewidth=2, edgecolor="white", facecolor="none",
        label=f"bounds x={x_bounds} y={y_bounds}", zorder=2
    )
    ax_traj.add_patch(rect)

    # Placeholders updated each frame
    current_marker, = ax_traj.plot([], [], marker="*", color="cyan",
                                   markersize=14, zorder=5, linestyle="none")
    ax_traj.set_xlabel("Grid X (m)")
    ax_traj.set_ylabel("Grid Y (m)")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.2)
    ax_traj.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#111111")
    ax_traj.tick_params(colors="white")
    ax_traj.xaxis.label.set_color("white")
    ax_traj.yaxis.label.set_color("white")

    # Blank image placeholder
    blank = np.zeros((800, 800, 3), dtype=np.uint8)
    img_display = ax_img.imshow(blank)
    ax_img.axis("off")

    title_obj = fig.suptitle("", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # --- Per-frame loop ---
    for i, row in enumerate(manifest_rows):
        frame_idx   = int(row["frame_idx"])
        timestamp   = float(row["timestamp"])
        tendon_type = int(row["tendon_type"])
        presence    = int(row["presence"])
        force       = float(row["force_magnitude"])
        img_rel     = row["image_path"]

        # Grid position at this frame (nearest TCP)
        ti = np.searchsorted(tcp_times, timestamp)
        ti = np.clip(ti, 0, len(tcp_times) - 1)
        gx = float(tcp_gx[ti])
        gy = float(tcp_gy[ti])

        # Update marker
        current_marker.set_data([gx], [gy])

        # Load image
        img_path = gt_images_root / img_rel
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = blank

        img_display.set_data(img_rgb)
        img_display.set_extent([0, img_rgb.shape[1], img_rgb.shape[0], 0])

        # Label color
        label_str = LABEL_NAMES[tendon_type]
        color = LABEL_COLORS[tendon_type]
        title_obj.set_text(
            f"{run_id}  |  Frame {frame_idx:05d}  |  "
            f"GT: [{label_str.upper()}]  |  Force: {force:.1f} N  |  "
            f"pos ({gx*1000:.1f}, {gy*1000:.1f}) mm"
        )
        title_obj.set_color(color)

        out_path = save_dir / f"frame_{frame_idx:05d}_{label_str}.jpg"
        fig.savefig(out_path, dpi=80, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(manifest_rows)} frames saved")

    plt.close(fig)
    print(f"Done — {len(manifest_rows)} frames saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Save per-frame trajectory+image inspection plots"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to inspect")
    parser.add_argument("--save-dir", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    run_inspector(args.run_id, args.save_dir)


if __name__ == "__main__":
    main()
