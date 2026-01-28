"""Quick visualizer for precomputed GT grid .npz files."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_grid(npz_path):
    data = np.load(npz_path)
    hit = data["hit"]
    depth = np.where(hit, data["depth_mm"], np.nan)
    type_id = np.where(hit, data["type_id"].astype(float), np.nan)

    extent = [float(data["x_min"]) * 1000, float(data["x_max"]) * 1000,
              float(data["y_min"]) * 1000, float(data["y_max"]) * 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(npz_path.stem)

    im1 = ax1.imshow(depth, origin="lower", extent=extent, aspect="equal")
    ax1.set(title="Depth below surface (mm)", xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im1, ax=ax1, label="mm")

    im2 = ax2.imshow(type_id, origin="lower", extent=extent, aspect="equal", vmin=0, vmax=3)
    ax2.set(title="Type ID (1=single, 2=crossed, 3=double)", xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()


def main():
    grid_dir = Path(config.GT_GRID_DIR)
    files = sorted(grid_dir.glob("*.npz"))

    if sys.argv[1:]:
        files = [grid_dir / f"{name}.npz" for name in sys.argv[1:]]

    if not files:
        print(f"No .npz files found in {grid_dir}")
        return

    for f in files:
        if f.exists():
            plot_grid(f)
        else:
            print(f"Not found: {f}")

    plt.show()


if __name__ == "__main__":
    main()
