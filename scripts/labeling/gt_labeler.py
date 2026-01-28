"""Config-driven GT grid generation via tendon mesh raycasting.

For each phantom+config pair in phantom_configs.json, loads the STL,
raycasts downward, and saves a .npz grid with hit/depth/type data.

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
import trimesh

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions (kept from original)
# ---------------------------------------------------------------------------

def load_and_extract_tendon(stl_path, bounds):
    """Load mesh and extract tendon region within specified bounds."""
    mesh = trimesh.load_mesh(stl_path)
    mesh.update_faces(trimesh.triangles.nondegenerate(mesh.triangles))
    mesh.remove_unreferenced_vertices()

    verts = mesh.vertices
    mask = (
        (verts[:, 0] >= bounds["x"][0]) & (verts[:, 0] <= bounds["x"][1]) &
        (verts[:, 1] >= bounds["y"][0]) & (verts[:, 1] <= bounds["y"][1]) &
        (verts[:, 2] >= bounds["z"][0]) & (verts[:, 2] <= bounds["z"][1])
    )

    valid_verts = np.where(mask)[0]
    valid_faces = np.where(np.all(np.isin(mesh.faces, valid_verts), axis=1))[0]

    return mesh.submesh([valid_faces], append=True)


def create_ray_grid(x_half, y_half, dx, dy):
    """Create meshgrid for raycasting."""
    xs = np.arange(-x_half, x_half + 1e-9, dx)
    ys = np.arange(-y_half, y_half + 1e-9, dy)
    return np.meshgrid(xs, ys, indexing="xy")


def raycast_mesh(mesh, XX, YY, z0):
    """Cast rays downward and return hit results."""
    n_rays = XX.size
    origins = np.column_stack([XX.ravel(), YY.ravel(), np.full(n_rays, z0)])
    directions = np.tile([0.0, 0.0, -1.0], (n_rays, 1))

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, indices, _ = intersector.intersects_location(
        origins, directions, multiple_hits=False
    )

    return locations, indices, n_rays


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

def determine_tendon_type(yy_flat, hit_mask, pattern, cross_y_threshold=0.0):
    """Return type_id array: 0=none, 1=single, 2=crossed, 3=double.

    For pattern "crossed" (p4): grid points with y <= threshold are crossed,
    y > threshold are single.
    For pattern "double" (p5): grid points with y <= threshold are double,
    y > threshold are single.
    For all other patterns every hit is single.
    """
    type_id = np.zeros(len(hit_mask), dtype=np.uint8)

    if pattern == "crossed":
        type_id[hit_mask & (yy_flat <= cross_y_threshold)] = 2
        type_id[hit_mask & (yy_flat > cross_y_threshold)] = 1
    elif pattern == "double":
        type_id[hit_mask & (yy_flat <= cross_y_threshold)] = 3
        type_id[hit_mask & (yy_flat > cross_y_threshold)] = 1
    else:
        type_id[hit_mask] = 1

    return type_id


# ---------------------------------------------------------------------------
# Main per-pair function
# ---------------------------------------------------------------------------

def generate_gt_grid(phantom, config_name, stl_dir, output_dir, all_configs,
                     show_plot=False):
    """Generate and save a GT grid .npz for one phantom+config pair."""
    phantom_block = all_configs[phantom]
    configs = phantom_block.get("configs", phantom_block)
    cfg = configs[config_name]
    gt = phantom_block["gt_params"]

    stl_file = cfg["stl_file"]
    rotation_deg = cfg.get("rotation_deg", 0)
    pattern = cfg.get("pattern", "single")
    cross_y_threshold = cfg.get("cross_y_threshold", 0.0)

    stl_path = Path(stl_dir) / stl_file
    if not stl_path.exists():
        logger.error(f"STL not found: {stl_path}")
        return

    bounds = {
        "x": tuple(gt["tendon_bounds"]["x"]),
        "y": tuple(gt["tendon_bounds"]["y"]),
        "z": tuple(gt["tendon_bounds"]["z"]),
    }
    z_surface = gt["z_surface"]

    logger.info(f"Processing {phantom}/{config_name}  STL={stl_file}  rot={rotation_deg}")

    # Load and extract tendon submesh
    mesh = load_and_extract_tendon(str(stl_path), bounds)

    # Apply rotation around Z axis if needed
    if rotation_deg != 0:
        angle_rad = np.radians(rotation_deg)
        rot = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh.apply_transform(rot)

    logger.info(f"  Mesh bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")

    # Create grid and raycast
    gp = config.GT_GRID_PARAMS
    XX, YY = create_ray_grid(gp["x_half"], gp["y_half"], gp["dx"], gp["dy"])
    hits, hit_idx, n_rays = raycast_mesh(mesh, XX, YY, gp["ray_z0"])
    logger.info(f"  Hits: {len(hit_idx)} / {n_rays}")

    # Build result arrays
    hit_mask = np.zeros(n_rays, dtype=bool)
    depth_mm = np.full(n_rays, np.nan)

    if len(hits) > 0:
        hit_mask[hit_idx] = True
        depth_mm[hit_idx] = (z_surface - hits[:, 2]) * 1000.0  # positive = below surface

    type_id = determine_tendon_type(
        YY.ravel(), hit_mask, pattern, cross_y_threshold
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

    # Optional debug plot
    if show_plot:
        _plot_gt_grid(XX, YY, hit_mask.reshape(XX.shape),
                      depth_mm.reshape(XX.shape),
                      type_id.reshape(XX.shape),
                      f"{phantom}/{config_name}")


def _plot_gt_grid(XX, YY, hit_grid, depth_grid, type_grid, title):
    """Show debug depth-map and type-id plots."""
    import matplotlib.pyplot as plt

    extent_mm = [XX.min() * 1000, XX.max() * 1000,
                 YY.min() * 1000, YY.max() * 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    depth_vis = np.where(hit_grid, depth_grid, np.nan)
    im1 = ax1.imshow(depth_vis, origin="lower", extent=extent_mm, aspect="equal")
    ax1.set(title=f"Depth below surface (mm) — {title}",
            xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im1, ax=ax1, label="depth (mm)")

    type_vis = np.where(hit_grid, type_grid.astype(float), np.nan)
    im2 = ax2.imshow(type_vis, origin="lower", extent=extent_mm, aspect="equal",
                     vmin=0, vmax=2)
    ax2.set(title=f"Type ID — {title}",
            xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im2, ax=ax2, label="type_id (0=none,1=single,2=crossed,3=double)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GT grids from phantom STL files"
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
    stl_dir = config.STL_DIR
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
                generate_gt_grid(phantom, cfg_name, stl_dir, output_dir,
                                 all_configs, show_plot=args.plot)
            except Exception as e:
                logger.error(f"Error processing {phantom}/{cfg_name}: {e}")


if __name__ == "__main__":
    main()
