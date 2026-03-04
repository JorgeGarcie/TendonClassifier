"""Config-driven GT grid generation via bounding box labeling.

For each phantom+config pair in phantom_configs.json, labels each grid cell
as tendon-present if its (x, y) position falls within the configured bounds.
No STL raycasting — the tendon region is defined purely by 2D bounding boxes.

Bounds come from phantom_configs.json:
  - phantom-level  gt_params.tendon_bounds.x / .y  (default for all configs)
  - config-level   tendon_x_bounds / tendon_y_bounds  (override per config)

Use visualize_bounds.py after generating grids to visually verify and
adjust bounds per run before re-running.

Usage:
    python gt_labeler.py                            # all phantoms, all configs
    python gt_labeler.py --phantom p1               # all configs for p1
    python gt_labeler.py --phantom p1 --config n2t  # specific pair
    python gt_labeler.py --plot                     # show debug plots
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def create_ray_grid(x_half, y_half, dx, dy):
    """Create meshgrid for the GT grid."""
    xs = np.arange(-x_half, x_half + 1e-9, dx)
    ys = np.arange(-y_half, y_half + 1e-9, dy)
    return np.meshgrid(xs, ys, indexing="xy")


def generate_bbox_hits(XX, YY, x_bounds, y_bounds):
    """Return flattened hit mask: True where (x, y) is within the bounds box.

    All coordinates are in the grid frame (post-rotation if any).
    x_bounds and y_bounds must already be expressed in that same frame.
    """
    x_flat = XX.ravel()
    y_flat = YY.ravel()
    return (
        (x_flat >= x_bounds[0]) & (x_flat <= x_bounds[1]) &
        (y_flat >= y_bounds[0]) & (y_flat <= y_bounds[1])
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_phantom_configs():
    """Load phantom_configs.json and return the full dict."""
    cfg_path = Path(config.CONFIGS_ROOT) / config.PHANTOM_CONFIGS_FILE
    with open(cfg_path) as f:
        return json.load(f)


def load_phantom_gt_config(phantom, all_configs):
    """Return gt_params for a phantom, or None if missing."""
    phantom_block = all_configs.get(phantom)
    if phantom_block is None:
        return None
    return phantom_block.get("gt_params")


# ---------------------------------------------------------------------------
# Type determination
# ---------------------------------------------------------------------------

def determine_tendon_type(yy_flat, hit_mask, pattern, cross_y_threshold=0.0,
                          flip_y=False):
    """Return type_id array: 0=none, 1=single, 2=crossed, 3=double.

    For pattern "crossed" (p4): grid points on one side of the threshold are
    crossed, the other side single.
    For pattern "double" (p5): grid points on one side of the threshold are
    double, the other side single.
    flip_y inverts the comparison (used when the phantom is rotated 180°).
    For all other patterns every hit is single.
    """
    type_id = np.zeros(len(hit_mask), dtype=np.uint8)

    if pattern == "crossed":
        if flip_y:
            type_id[hit_mask & (yy_flat >= cross_y_threshold)] = 2
            type_id[hit_mask & (yy_flat < cross_y_threshold)] = 1
        else:
            type_id[hit_mask & (yy_flat <= cross_y_threshold)] = 2
            type_id[hit_mask & (yy_flat > cross_y_threshold)] = 1
    elif pattern == "double":
        if flip_y:
            type_id[hit_mask & (yy_flat >= cross_y_threshold)] = 3
            type_id[hit_mask & (yy_flat < cross_y_threshold)] = 1
        else:
            type_id[hit_mask & (yy_flat <= cross_y_threshold)] = 3
            type_id[hit_mask & (yy_flat > cross_y_threshold)] = 1
    else:
        type_id[hit_mask] = 1

    return type_id


# ---------------------------------------------------------------------------
# Main per-pair function
# ---------------------------------------------------------------------------

def generate_gt_grid(phantom, config_name, output_dir, all_configs,
                     show_plot=False):
    """Generate and save a GT grid .npz for one phantom+config pair."""
    phantom_block = all_configs[phantom]
    configs = phantom_block.get("configs", phantom_block)
    cfg = configs[config_name]
    gt = phantom_block["gt_params"]

    rotation_deg = cfg.get("rotation_deg", 0)
    pattern = cfg.get("pattern", "single")
    cross_y_threshold = cfg.get("cross_y_threshold", 0.0)

    # Per-config bounds override phantom-level bounds
    x_bounds = tuple(cfg.get("tendon_x_bounds", gt["tendon_bounds"]["x"]))
    y_bounds = tuple(cfg.get("tendon_y_bounds", gt["tendon_bounds"]["y"]))

    logger.info(
        f"Processing {phantom}/{config_name}  rot={rotation_deg}  "
        f"x={[round(v*1000,1) for v in x_bounds]}mm  "
        f"y={[round(v*1000,1) for v in y_bounds]}mm"
    )

    # Create grid
    gp = config.GT_GRID_PARAMS
    XX, YY = create_ray_grid(gp["x_half"], gp["y_half"], gp["dx"], gp["dy"])

    # Bounding-box hit detection (grid frame = post-rotation frame)
    hit_mask = generate_bbox_hits(XX, YY, x_bounds, y_bounds)
    logger.info(f"  Hits: {hit_mask.sum()} / {hit_mask.size}")

    # depth_mm is not available in bbox mode; set to 0 for all hits
    depth_mm = np.where(hit_mask, 0.0, np.nan)

    # 180° rotation flips the phantom, so invert the y-comparison for type
    flip_y = (rotation_deg == 180)

    type_id = determine_tendon_type(
        YY.ravel(), hit_mask, pattern, cross_y_threshold, flip_y
    )

    # Save .npz
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    npz_file = out_path / f"{phantom}_{config_name}.npz"

    np.savez(
        npz_file,
        xx=XX,
        yy=YY,
        hit=hit_mask.reshape(XX.shape).astype(np.uint8),
        depth_mm=depth_mm.reshape(XX.shape).astype(np.float32),
        type_id=type_id.reshape(XX.shape).astype(np.uint8),
        x_min=XX.min(),
        x_max=XX.max(),
        y_min=YY.min(),
        y_max=YY.max(),
        dx=gp["dx"],
        dy=gp["dy"],
    )
    logger.info(f"  Saved: {npz_file}")

    if show_plot:
        _plot_gt_grid(XX, YY, hit_mask.reshape(XX.shape),
                      depth_mm.reshape(XX.shape),
                      type_id.reshape(XX.shape),
                      f"{phantom}/{config_name}")


def _plot_gt_grid(XX, YY, hit_grid, depth_grid, type_grid, title):
    """Show debug hit-map and type-id plots."""
    import matplotlib.pyplot as plt

    extent_mm = [XX.min() * 1000, XX.max() * 1000,
                 YY.min() * 1000, YY.max() * 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    hit_vis = hit_grid.astype(float)
    im1 = ax1.imshow(hit_vis, origin="lower", extent=extent_mm, aspect="equal",
                     vmin=0, vmax=1, cmap="Blues")
    ax1.set(title=f"Hit mask — {title}", xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im1, ax=ax1, label="hit (0/1)")

    type_vis = np.where(hit_grid, type_grid.astype(float), np.nan)
    im2 = ax2.imshow(type_vis, origin="lower", extent=extent_mm, aspect="equal",
                     vmin=0, vmax=3)
    ax2.set(title=f"Type ID — {title}", xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im2, ax=ax2, label="type_id (0=none,1=single,2=crossed,3=double)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GT grids from bounding box config"
    )
    parser.add_argument("--phantom", type=str, default=None,
                        help="Process only this phantom (e.g. p1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Process only this config (e.g. n2t). Requires --phantom.")
    parser.add_argument("--plot", action="store_true",
                        help="Show debug plots")
    args = parser.parse_args()

    if args.config and not args.phantom:
        parser.error("--config requires --phantom")

    all_configs = load_phantom_configs()
    output_dir = config.GT_GRID_DIR

    phantoms = [args.phantom] if args.phantom else list(all_configs.keys())

    for phantom in phantoms:
        gt = load_phantom_gt_config(phantom, all_configs)
        if gt is None:
            logger.warning(f"Skipping {phantom}: no gt_params in config")
            continue

        phantom_block = all_configs[phantom]
        configs_block = phantom_block.get("configs", phantom_block)

        config_names = ([args.config] if args.config
                        else list(configs_block.keys()))

        for cfg_name in config_names:
            if cfg_name not in configs_block:
                logger.warning(f"Config '{cfg_name}' not found for {phantom}")
                continue
            try:
                generate_gt_grid(phantom, cfg_name, output_dir,
                                 all_configs, show_plot=args.plot)
            except Exception as e:
                logger.error(f"Error processing {phantom}/{cfg_name}: {e}")


if __name__ == "__main__":
    main()
