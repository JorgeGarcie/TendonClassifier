"""Tendon mesh raycasting and coordinate transformation pipeline."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh

# Configuration
CONFIG = {
    "stl_path": r"C:/Users/georg/Documents/Stanford/TactileSensing/TendonClassifier/scripts/labeling/rawdata/p3.stl",
    "grid": {"x_half": 0.0381, "y_half": 0.0635, "dx": 0.00015, "dy": 0.00015},
    "tendon_bounds": {
        "x": (-0.01, 0.01),
        "y": (-0.075, 0.075),
        "z": (0.001, 0.021),
    },
    "ray_z0": 0.025,
}

T_WORLD_TO_PHANTOM = np.array([
    [0.999837, -0.018044, 0.0, 0.5116879957],
    [0.018044,  0.999837, 0.0, 0.3225671259],
    [0.0,       0.0,      1.0, 0.00],
    [0.0,       0.0,      0.0, 1.0], 
])

Z_SURFACE = 0.014 # This is the value that is the surface of the phantom

def load_and_extract_tendon(stl_path: str, bounds: dict) -> trimesh.Trimesh:
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


def create_ray_grid(x_half: float, y_half: float, dx: float, dy: float):
    """Create meshgrid for raycasting."""
    xs = np.arange(-x_half, x_half + 1e-9, dx)
    ys = np.arange(-y_half, y_half + 1e-9, dy)
    return np.meshgrid(xs, ys, indexing="xy")


def raycast_mesh(mesh: trimesh.Trimesh, XX: np.ndarray, YY: np.ndarray, z0: float):
    """Cast rays downward and return hit results."""
    n_rays = XX.size
    origins = np.column_stack([XX.ravel(), YY.ravel(), np.full(n_rays, z0)])
    directions = np.tile([0.0, 0.0, -1.0], (n_rays, 1))

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, indices, _ = intersector.intersects_location(origins, directions, multiple_hits=False)

    return locations, indices, n_rays


def transform_to_phantom(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform points using homogeneous transformation matrix."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (transform @ homogeneous.T).T[:, :3]


def main():
    cfg = CONFIG
    
    # Load and process mesh
    mesh = load_and_extract_tendon(cfg["stl_path"], cfg["tendon_bounds"])
    print(f"Mesh bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")

    # Create grid and raycast
    XX, YY = create_ray_grid(**cfg["grid"])
    hits, hit_idx, n_rays = raycast_mesh(mesh, XX, YY, cfg["ray_z0"])
    print(f"Hits: {len(hit_idx)} / {n_rays}")

    # Initialize result arrays
    results = {
        "hit": np.zeros(n_rays, dtype=np.uint8),
        "z_original": np.full(n_rays, np.nan),
        "x_phantom": np.full(n_rays, np.nan),
        "y_phantom": np.full(n_rays, np.nan),
        "z_phantom": np.full(n_rays, np.nan),
    }

    if len(hits) > 0:
        results["hit"][hit_idx] = 1
        results["z_original"][hit_idx] = hits[:, 2] - Z_SURFACE
        
        phantom_pts = transform_to_phantom(hits, T_WORLD_TO_PHANTOM)
        results["x_phantom"][hit_idx] = phantom_pts[:, 0]
        results["y_phantom"][hit_idx] = phantom_pts[:, 1]
        results["z_phantom"][hit_idx] = phantom_pts[:, 2] - Z_SURFACE

    # Save results
    df = pd.DataFrame({
        "x_mm": XX.ravel() * 1000,
        "y_mm": YY.ravel() * 1000,
        "hit": results["hit"],
        "z_hit_original_mm": results["z_original"] * 1000,
        "z_hit_phantom_mm": results["z_phantom"] * 1000,
    })
    df.to_csv("p1_hits_list.csv", index=False)
    print("Saved: p1_hits_list.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    extent_mm = [XX.min() * 1000, XX.max() * 1000, YY.min() * 1000, YY.max() * 1000]

    # Original frame
    z_grid = np.where(results["hit"].reshape(YY.shape), 
                      results["z_original"].reshape(YY.shape) * 1000, np.nan)
    im1 = ax1.imshow(z_grid, origin="lower", extent=extent_mm, aspect="equal")
    ax1.set(title="Z in original frame (mm)", xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im1, ax=ax1, label="z_hit (mm)")

    # Phantom frame
    mask = results["hit"] == 1
    sc = ax2.scatter(results["x_phantom"][mask] * 1000, results["y_phantom"][mask] * 1000,
                     c=results["z_phantom"][mask] * 1000, s=1, cmap="viridis")
    ax2.set(title="Z in phantom frame (mm)", xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    plt.colorbar(sc, ax=ax2, label="z_hit (mm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()