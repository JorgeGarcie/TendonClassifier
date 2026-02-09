"""Generate GT dataset from precomputed GT grids and valid frames.

For each run/frame, looks up the precomputed GT grid using the TCP position,
reads the full 6-axis wrench, and produces images + a manifest CSV.

Usage:
    python generate_gt.py                                  # all runs
    python generate_gt.py --run-id p1_n2t_25N-...          # specific run
    python generate_gt.py --stride 5                       # every 5th frame
    python generate_gt.py --image-size 540x960             # resize images
"""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GT grid lookup
# ---------------------------------------------------------------------------

class GTGridLookup:
    """O(1) lookup into a precomputed .npz GT grid."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.hit = data["hit"]       # (ny, nx) uint8
        self.depth_mm = data["depth_mm"]  # (ny, nx) float32
        self.type_id = data["type_id"]    # (ny, nx) uint8
        self.x_min = float(data["x_min"])
        self.x_max = float(data["x_max"])
        self.y_min = float(data["y_min"])
        self.y_max = float(data["y_max"])
        self.dx = float(data["dx"])
        self.dy = float(data["dy"])
        self.ny, self.nx = self.hit.shape

    def lookup(self, x, y):
        """Return (presence, depth_mm, type_id) for a point in grid coords.

        Returns (0, nan, 0) if out of bounds or no hit.
        """
        ix = round((x - self.x_min) / self.dx)
        iy = round((y - self.y_min) / self.dy)

        # Clamp to bounds
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))

        presence = int(self.hit[iy, ix])
        depth = float(self.depth_mm[iy, ix]) if presence else float("nan")
        tid = int(self.type_id[iy, ix])
        return presence, depth, tid


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def world_to_grid_coords(tcp_x, tcp_y, tcp_z, t_stl_to_world, rotation_deg):
    """Convert a TCP world position to STL grid coordinates.

    1. Apply inv(t_stl_to_world) to get STL-frame position.
    2. If rotation_deg != 0, apply R_z(rotation_deg) to match the rotated grid.
    3. Return (x, y) in grid coordinates.
    """
    t_world_to_stl = np.linalg.inv(t_stl_to_world)
    world_pt = np.array([tcp_x, tcp_y, tcp_z, 1.0])
    stl_pt = t_world_to_stl @ world_pt

    x, y = stl_pt[0], stl_pt[1]

    if rotation_deg != 0:
        angle = np.radians(rotation_deg)
        c, s = np.cos(angle), np.sin(angle)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        x, y = x_rot, y_rot

    return x, y


# ---------------------------------------------------------------------------
# Nearest-timestamp helpers
# ---------------------------------------------------------------------------

def find_nearest_tcp(frame_time, tcp_times, tcp_data):
    """Find nearest TCP pose for given frame timestamp."""
    idx = np.searchsorted(tcp_times, frame_time)
    idx = np.clip(idx, 0, len(tcp_times) - 1)
    if idx > 0 and abs(tcp_times[idx - 1] - frame_time) < abs(tcp_times[idx] - frame_time):
        idx = idx - 1
    return tcp_data.iloc[idx]


def interpolate_wrench(frame_time, wrench_times, wrench_arrays):
    """Linearly interpolate wrench components at frame_time."""
    return {col: float(np.interp(frame_time, wrench_times, arr))
            for col, arr in wrench_arrays.items()}


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def load_phantom_configs():
    """Load phantom_configs.json."""
    cfg_path = Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE
    with open(cfg_path) as f:
        return json.load(f)


def process_run(run, run_metadata, phantom_cfgs, gt_grids,
                output_root, stride, image_size):
    """Process all frames for one run, return manifest rows."""
    run_id = run["run_id"]
    phantom_type = run["phantom_type"]
    run_path = Path(run.get("path", config.DATA_ROOT) or config.DATA_ROOT)

    logger.info(f"Processing run: {run_id}")

    # Get motion type and rotation from run_manifest metadata
    meta = run_metadata.get(run_id)
    if meta is None:
        logger.warning(f"No metadata for run {run_id}, skipping")
        return []

    motion_type = meta["motion_type"]
    rotation_deg = meta.get("rotation_deg", 0)

    # Look up GT grid
    grid_key = f"{phantom_type}_{motion_type}"
    grid = gt_grids.get(grid_key)
    if grid is None:
        # Try "default" config
        grid_key = f"{phantom_type}_default"
        grid = gt_grids.get(grid_key)
    if grid is None:
        logger.warning(f"No GT grid for {phantom_type}/{motion_type}, skipping run")
        return []

    # Get gt_params for coordinate transform
    phantom_block = phantom_cfgs.get(phantom_type)
    if not phantom_block or "gt_params" not in phantom_block:
        logger.warning(f"No gt_params for {phantom_type}, skipping run")
        return []

    gt_params = phantom_block["gt_params"]
    t_stl_to_world = np.array(gt_params["t_stl_to_world"])

    # Load raw TCP and wrench CSVs
    tcp_path = run_path / "tcp_pose.csv"
    wrench_path = run_path / "wrench_data.csv"

    if not tcp_path.exists() or not wrench_path.exists():
        logger.warning(f"Missing tcp_pose.csv or wrench_data.csv for {run_id}")
        return []

    tcp_data = pd.read_csv(tcp_path)
    wrench_data = pd.read_csv(wrench_path)

    tcp_times = tcp_data[config.TIME_COL].values

    # Filter wrench to configured sensor and prepare interpolation arrays
    wrench_sensor = wrench_data[wrench_data["sensor"] == config.WRENCH_SENSOR].sort_values(config.TIME_COL)
    wrench_times = wrench_sensor[config.TIME_COL].values
    wrench_arrays = {col: wrench_sensor[col].values for col in ["fx", "fy", "fz", "tx", "ty", "tz"]}

    # Output directories
    images_dir = Path(output_root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Apply stride
    frames = run["frames"][::stride]

    manifest_rows = []

    for frame in frames:
        frame_idx = frame["frame_idx"]
        timestamp = frame["timestamp"]

        # Load source image
        image_path = run_path / frame["image_path"]
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        # Find nearest TCP pose (full 3D position)
        tcp_row = find_nearest_tcp(timestamp, tcp_times, tcp_data)
        tcp_x = float(tcp_row["x"])
        tcp_y_val = float(tcp_row["y"])
        tcp_z = float(tcp_row["z"])

        # Interpolate wrench at camera timestamp (resampling to camera rate)
        w = interpolate_wrench(timestamp, wrench_times, wrench_arrays)
        fx, fy, fz = w["fx"], w["fy"], w["fz"]
        tx, ty, tz = w["tx"], w["ty"], w["tz"]
        force_magnitude = float(np.sqrt(fx**2 + fy**2 + fz**2))

        # Convert TCP world position to grid coordinates
        gx, gy = world_to_grid_coords(
            tcp_x, tcp_y_val, tcp_z, t_stl_to_world, rotation_deg
        )

        # Look up GT
        presence, depth_mm, tendon_type = grid.lookup(gx, gy)

        # Copy/resize image
        img_filename = f"{run_id}_frame_{frame_idx:05d}.jpg"
        img_output_path = images_dir / img_filename

        if image_size is not None:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(img_output_path), img_resized)
        else:
            shutil.copy2(str(image_path), str(img_output_path))

        manifest_rows.append({
            "run_id": run_id,
            "phantom_type": phantom_type,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_path": f"images/{img_filename}",
            "presence": presence,
            "depth_mm": depth_mm if presence else "",
            "tendon_type": tendon_type,
            "force_magnitude": force_magnitude,
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "tx": tx,
            "ty": ty,
            "tz": tz,
        })

    logger.info(f"  Processed {len(manifest_rows)} frames for {run_id}")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate GT dataset from precomputed grids and valid frames"
    )
    parser.add_argument("--run-id", type=str, default=None,
                        help="Process only this run_id")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth frame (default: 1)")
    parser.add_argument("--image-size", type=str, default=None,
                        help="Resize images to WxH (e.g. 540x960)")
    args = parser.parse_args()

    # Parse image size
    image_size = None
    if args.image_size:
        parts = args.image_size.split("x")
        if len(parts) != 2:
            parser.error("--image-size must be WxH (e.g. 540x960)")
        image_size = (int(parts[0]), int(parts[1]))

    # Load configs
    valid_frames_path = Path(config.CONFIGS_ROOT) / "valid_frames.json"
    run_manifest_path = Path(config.CONFIGS_ROOT) / "run_manifest.json"

    if not valid_frames_path.exists():
        logger.error(f"valid_frames.json not found at {valid_frames_path}")
        logger.info("Run extract_valid_windows.py first.")
        return

    if not run_manifest_path.exists():
        logger.error(f"run_manifest.json not found at {run_manifest_path}")
        logger.info("Run discover_and_index.py first.")
        return

    with open(valid_frames_path) as f:
        valid_frames_data = json.load(f)

    with open(run_manifest_path) as f:
        run_manifest_data = json.load(f)

    # Build run metadata lookup
    run_metadata = {}
    for run_info in run_manifest_data.get("runs", []):
        rid = run_info["run_id"]
        run_metadata[rid] = {
            "motion_type": run_info.get("motion_type"),
            "stl_file": run_info.get("stl_file"),
            "rotation_deg": run_info.get("rotation_deg", 0),
        }

    # Load phantom configs
    phantom_cfgs = load_phantom_configs()

    # Load all available GT grids
    gt_grid_dir = Path(config.GT_GRID_DIR)
    gt_grids = {}
    if gt_grid_dir.exists():
        for npz_file in gt_grid_dir.glob("*.npz"):
            key = npz_file.stem  # e.g. "p1_n2t"
            gt_grids[key] = GTGridLookup(npz_file)
            logger.info(f"Loaded GT grid: {key}")

    if not gt_grids:
        logger.error(f"No GT grids found in {gt_grid_dir}")
        logger.info("Run gt_labeler.py first.")
        return

    # Output
    output_root = Path(config.OUTPUT_ROOT) / "gt_dataset"
    output_root.mkdir(parents=True, exist_ok=True)

    all_manifest_rows = []

    for run in valid_frames_data["runs"]:
        run_id = run["run_id"]
        if args.run_id and run_id != args.run_id:
            continue

        rows = process_run(
            run, run_metadata, phantom_cfgs, gt_grids,
            str(output_root), args.stride, image_size
        )
        all_manifest_rows.extend(rows)

    # Write manifest CSV
    if all_manifest_rows:
        manifest_csv_path = output_root / "gt_manifest.csv"
        fieldnames = [
            "run_id", "phantom_type", "frame_idx", "timestamp",
            "image_path", "presence", "depth_mm", "tendon_type",
            "force_magnitude", "fx", "fy", "fz", "tx", "ty", "tz",
        ]

        with open(manifest_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest_rows)

        logger.info(f"GT manifest saved to {manifest_csv_path}")
        logger.info(f"Total frames: {len(all_manifest_rows)}")
    else:
        logger.warning("No frames were processed")


if __name__ == "__main__":
    main()
