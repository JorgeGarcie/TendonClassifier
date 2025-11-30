import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuration constants
TENDON_RADIUS_PIXELS = 50  
NORMAL_MAP_OUTPUT_DIR = "normal_maps"


def ensure_output_dir():
    """Create output directory for normal maps if it doesn't exist"""
    os.makedirs(NORMAL_MAP_OUTPUT_DIR, exist_ok=True)


def get_click(ax):
    """
    Get a single click from the user on the displayed image.

    Args:
        ax: matplotlib axis to listen for clicks on

    Returns:
        np.array: (u, v) coordinates of the click

    Raises:
        KeyboardInterrupt: If user presses Ctrl+C
    """
    clicks = []

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is not None and event.ydata is not None:
            u, v = event.xdata, event.ydata
            clicks.append(np.array([u, v]))
            # Plot point to show where user clicked
            ax.plot(u, v, 'r*', markersize=15)
            plt.draw()

    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Wait for click (with Ctrl+C support)
    try:
        while len(clicks) == 0:
            plt.pause(0.1)
    except KeyboardInterrupt:
        plt.gcf().canvas.mpl_disconnect(cid)
        plt.close('all')
        raise KeyboardInterrupt("User skipped annotation")

    plt.gcf().canvas.mpl_disconnect(cid)
    return clicks[0]


def normalize_vector(v):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def save_normal_map(image_id, normal_map, metadata=None):
    """
    Save normal map as an image file and metadata.

    Args:
        image_id: Identifier for the image (e.g., filename without extension)
        normal_map: (H, W, 3) array with unit normals in range [-1, 1]
        metadata: Optional dict with tendon information (length, pixel width, etc.)
    """
    import json

    ensure_output_dir()

    # Convert from [-1, 1] to [0, 255] for storage
    normal_map_uint8 = ((normal_map + 1) * 127.5).astype(np.uint8)

    # Save as image
    output_path = os.path.join(NORMAL_MAP_OUTPUT_DIR, f"{image_id}_normals.png")
    img = Image.fromarray(normal_map_uint8)
    img.save(output_path)

    print(f"Normal map saved to {output_path}")

    # Also save as numpy array for precise recovery
    npy_path = os.path.join(NORMAL_MAP_OUTPUT_DIR, f"{image_id}_normals.npy")
    np.save(npy_path, normal_map)
    print(f"Normal map (numpy) saved to {npy_path}")

    # Save metadata if provided
    if metadata is not None:
        metadata_path = os.path.join(NORMAL_MAP_OUTPUT_DIR, f"{image_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")


def load_normal_map(image_id):
    """
    Load normal map from file.

    Args:
        image_id: Identifier for the image

    Returns:
        (H, W, 3) array with unit normals
    """
    npy_path = os.path.join(NORMAL_MAP_OUTPUT_DIR, f"{image_id}_normals.npy")
    return np.load(npy_path)


def label_tendon_normals(image_id, delta_I_clean, tendon_radius=None, tilt_factor=None, visualize=False):
    """
    Interactively label tendon surface normals based on user-defined axis.

    Creates a synthetic normal map by:
    1. Having the user click two points defining the tendon axis
    2. Having the user define pixel width by clicking reference points
    3. Computing normals for all pixels based on cylindrical geometry
    4. Saving the normal map with metadata (tendon length)

    Args:
        image_id: Identifier for the image (e.g., "frame_000585")
        delta_I_clean: (H, W, 3) cleaned difference image to display
        tendon_radius: Override default tendon radius in pixels
        tilt_factor: Override default cylinder tilt factor (α)
        visualize: If False (default), skip all visualization plots for speed

    Returns:
        normal_map: (H, W, 3) array of unit surface normals
        metadata: dict with tendon_length_px, pixels_per_unit, tendon_length_physical
    """

    # Use provided parameters or defaults
    R_px = tendon_radius if tendon_radius is not None else TENDON_RADIUS_PIXELS

    print(f"\n{'='*60}")
    print(f"Labeling tendon normals for: {image_id}")
    print(f"Tendon radius: {R_px} pixels")
    print(f"{'='*60}\n")

    ############################################
    # 1. USER CLICKS TENDON AXIS
    ############################################
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the cleaned difference image
    ax.imshow(delta_I_clean * 0.5 + 0.5, cmap='viridis')
    ax.set_title(f"Step 1: Click two points along the tendon axis\n{image_id}", fontsize=14)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    plt.tight_layout()

    print("Click point A (start of tendon axis)...")
    A = get_click(ax)
    print(f"Point A: {A}")

    print("Click point B (end of tendon axis)...")
    B = get_click(ax)
    print(f"Point B: {B}")

    plt.close(fig)

    ############################################
    # 1.5. USER DEFINES PIXEL WIDTH
    ############################################
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(delta_I_clean * 0.5 + 0.5, cmap='viridis')
    ax.plot([A[0], B[0]], [A[1], B[1]], 'r-', linewidth=2, label='Tendon axis')
    ax.plot(A[0], A[1], 'go', markersize=10, label='Point A')
    ax.plot(B[0], B[1], 'ro', markersize=10, label='Point B')
    ax.set_title(f"Step 2: Define pixel width - Click two reference points\n{image_id}", fontsize=14)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.legend()
    plt.tight_layout()

    print("\nDefining pixel width...")
    print("Click reference point C...")
    C = get_click(ax)
    print(f"Point C: {C}")

    print("Click reference point D (known physical distance from C)...")
    D = get_click(ax)
    print(f"Point D: {D}")

    plt.close(fig)

    # Calculate pixel distance between C and D
    CD_pixel_distance = np.linalg.norm(D - C)
    print(f"Pixel distance C-D: {CD_pixel_distance:.2f} pixels")

    # Use default physical distance (4.5 mm)
    physical_distance_CD = 4.5  # mm
    print(f"Physical distance C-D: {physical_distance_CD} mm (default)")

    # Calculate pixels per unit physical distance
    pixels_per_unit = CD_pixel_distance / physical_distance_CD
    print(f"Pixel width: {pixels_per_unit:.4f} pixels per unit physical distance")

    # Calculate tendon length in pixels and physical units
    tendon_length_px = np.linalg.norm(B - A)
    tendon_length_physical = tendon_length_px / pixels_per_unit

    print(f"\nTendon axis length: {tendon_length_px:.2f} pixels")
    print(f"Tendon axis length (physical): {tendon_length_physical:.4f} units")

    # Store metadata
    metadata = {
        'tendon_length_px': float(tendon_length_px),
        'tendon_length_physical': float(tendon_length_physical),
        'pixels_per_unit': float(pixels_per_unit),
        'A': A.tolist(),
        'B': B.tolist(),
        'C': C.tolist(),
        'D': D.tolist()
    }

    ############################################
    # 2. PREP GEOMETRY
    ############################################
    # Axis direction (unit 2D vector)
    axis_vec = B - A
    d = normalize_vector(axis_vec)

    print(f"Axis vector: {axis_vec}")
    print(f"Axis direction (normalized): {d}")

    # Create perpendicular vector to the axis (in 2D image plane)
    # If d = (dx, dy), then perp = (-dy, dx)
    d_perp = np.array([-d[1], d[0]], dtype=np.float32)

    print(f"Perpendicular direction: {d_perp}")

    H, W = delta_I_clean.shape[:2]
    N = np.zeros((H, W, 3), dtype=np.float32)

    ############################################
    # 3. VECTORIZED NORMAL COMPUTATION
    # Using 3D-consistent circular arc model:
    # z(s) = R - sqrt(R^2 - s^2)
    # where s is distance from axis perpendicular to direction d
    ############################################
    print(f"\nComputing normals for {H}x{W} image using circular arc model (vectorized)...")

    # 1) Build coordinate grid
    u_coords, v_coords = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy'
    )

    # 2) Vector (p - A) for all pixels
    pu = u_coords - A[0]
    pv = v_coords - A[1]

    # 3) Signed distance s along perpendicular direction
    s = pu * d_perp[0] + pv * d_perp[1]           # shape (H, W)
    r_dist = np.abs(s)

    # 4) Mask for pixels inside tendon cross-section
    inside = r_dist <= R_px

    # 5) Discriminant and z_s (with safe clipping)
    discriminant = R_px**2 - s**2
    discriminant = np.clip(discriminant, 1e-6, None) # Clip for the pixels that are at the border of the cylinder. 
    z_s = s / np.sqrt(discriminant)               # dz/ds

    # 6) Normal components in (perp, z)
    norm_factor = np.sqrt(1.0 + z_s**2)
    Nx_perp = -z_s / norm_factor                  # along d_perp
    Nz = 1.0 / norm_factor

    # 7) Project back to image-plane (u, v)
    Nx = Nx_perp * d_perp[0]
    Ny = Nx_perp * d_perp[1]

    # 8) Fill N with normals (inside cylinder)
    N[inside, 0] = Nx[inside]
    N[inside, 1] = Ny[inside]
    N[inside, 2] = Nz[inside]

    # 9) Background: set outside-cylinder pixels to [0, 0, 1]
    N[~inside] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 10) Vectorized normalization (renormalize all non-background normals)
    norms = np.linalg.norm(N[inside], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)  # Avoid division by zero
    N[inside] = N[inside] / norms

    print("Normal map computation complete!")

    ############################################
    # 4. SAVE (AND OPTIONAL VISUALIZATION)
    ############################################
    # Save normal map with metadata
    save_normal_map(image_id, N, metadata=metadata)

    # Optional visualization (disabled by default for speed)
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original difference image
        axes[0].imshow(delta_I_clean * 0.5 + 0.5)
        axes[0].plot([A[0], B[0]], [A[1], B[1]], 'r-', linewidth=2, label='Tendon axis')
        axes[0].plot(A[0], A[1], 'go', markersize=10, label='Point A')
        axes[0].plot(B[0], B[1], 'ro', markersize=10, label='Point B')
        axes[0].set_title('Input (ΔI_clean)')
        axes[0].legend()
        axes[0].set_xlabel("u")
        axes[0].set_ylabel("v")

        # Normal map visualization (X component)
        axes[1].imshow(N[:, :, 0], cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Normal X component')
        plt.colorbar(axes[1].images[0], ax=axes[1])

        # Normal map visualization (Y component)
        axes[2].imshow(N[:, :, 1], cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Normal Y component')
        plt.colorbar(axes[2].images[0], ax=axes[2])

        plt.tight_layout()
        plt.show()

    return N, metadata


if __name__ == "__main__":
    # Example usage
    from calculate_difference import visualize_delta, resize_to_target
    from PIL import Image

    # Load an example image
    contact_image_path = "images/frame_000585.png"
    ref_path = "I_ref.png"

    # Load and resize images
    I_ref = np.array(Image.open(ref_path)).astype(np.float32) / 255.0
    I_contact = np.array(Image.open(contact_image_path)).astype(np.float32) / 255.0

    # Resize to target
    from calculate_difference import resize_to_target
    I_ref = resize_to_target(I_ref, target_shape=(1080, 1920, 3))
    I_contact = resize_to_target(I_contact, target_shape=(1080, 1920, 3))

    # Compute difference
    delta_I = I_contact - I_ref

    # Apply Gaussian filter
    from scipy.ndimage import gaussian_filter
    delta_I_smooth = gaussian_filter(delta_I, sigma=100)
    delta_I_clean = delta_I - delta_I_smooth

    # Label normals
    normals, metadata = label_tendon_normals("frame_000585", delta_I_clean)

    print("\nDone! Normal map and metadata saved.")
    print(f"Tendon length (physical): {metadata['tendon_length_physical']:.4f} units")
    print(f"Pixel width: {metadata['pixels_per_unit']:.4f} pixels per unit")