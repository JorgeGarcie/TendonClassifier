"""Precompute 32-step force windows aligned to gt_manifest.csv rows.

For each manifest row, extracts the last 32 raw wrench readings (300Hz)
ending at or before the frame timestamp. Pads with zeros if fewer than
32 readings are available (early frames).

Output: force_windows.npy shape (N, 32, 6) in gt_dataset/, row-aligned
with gt_manifest.csv.

Usage:
    python precompute_force_windows.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

WINDOW_SIZE = 32
FORCE_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]
SENSOR_NAME = "coinft"

# Paths relative to this script (scripts/labeling/)
SCRIPT_DIR = Path(__file__).parent
MANIFEST_PATH = SCRIPT_DIR / "output" / "gt_dataset" / "gt_manifest.csv"
RAWDATA_DIR = SCRIPT_DIR / "rawdata"
OUTPUT_PATH = SCRIPT_DIR / "output" / "gt_dataset" / "force_windows.npy"


def load_wrench_data(run_id: str) -> tuple:
    """Load and filter wrench data for a run.

    Returns:
        (timestamps, forces) where timestamps is (M,) and forces is (M, 6)
    """
    wrench_path = RAWDATA_DIR / run_id / "wrench_data.csv"
    if not wrench_path.exists():
        raise FileNotFoundError(f"Wrench data not found: {wrench_path}")

    df = pd.read_csv(wrench_path)
    df = df[df["sensor"] == SENSOR_NAME].sort_values("time")

    timestamps = df["time"].values.astype(np.float64)
    forces = df[FORCE_COLS].values.astype(np.float32)
    return timestamps, forces


def extract_window(wrench_times: np.ndarray, wrench_forces: np.ndarray,
                   frame_time: float) -> np.ndarray:
    """Extract the last WINDOW_SIZE readings ending at/before frame_time.

    Args:
        wrench_times: Sorted wrench timestamps (M,)
        wrench_forces: Wrench force values (M, 6)
        frame_time: Target frame timestamp

    Returns:
        (WINDOW_SIZE, 6) array, zero-padded if fewer readings available
    """
    # Find index of last reading at or before frame_time
    idx = np.searchsorted(wrench_times, frame_time, side="right")  # first > frame_time
    # idx is count of readings <= frame_time

    start = max(0, idx - WINDOW_SIZE)
    window = wrench_forces[start:idx]  # up to WINDOW_SIZE rows

    # Pad with zeros if not enough readings
    if len(window) < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - len(window), 6), dtype=np.float32)
        window = np.vstack([pad, window])

    return window


def main():
    print(f"Loading manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    N = len(manifest)
    print(f"Manifest rows: {N}")

    # Pre-load wrench data per run (avoid re-reading)
    run_ids = manifest["run_id"].unique()
    print(f"Unique runs: {len(run_ids)}")

    wrench_cache = {}
    for run_id in run_ids:
        try:
            wrench_cache[run_id] = load_wrench_data(run_id)
            n_readings = len(wrench_cache[run_id][0])
            print(f"  {run_id}: {n_readings} wrench readings")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Extract windows
    force_windows = np.zeros((N, WINDOW_SIZE, 6), dtype=np.float32)
    missing_runs = set()

    for i, row in manifest.iterrows():
        run_id = row["run_id"]
        if run_id not in wrench_cache:
            missing_runs.add(run_id)
            continue

        wrench_times, wrench_forces = wrench_cache[run_id]
        force_windows[i] = extract_window(wrench_times, wrench_forces, row["timestamp"])

    if missing_runs:
        print(f"\nWARNING: {len(missing_runs)} runs had no wrench data: {missing_runs}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), force_windows)
    size_mb = force_windows.nbytes / 1e6
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  Shape: {force_windows.shape}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  dtype: {force_windows.dtype}")


if __name__ == "__main__":
    main()
