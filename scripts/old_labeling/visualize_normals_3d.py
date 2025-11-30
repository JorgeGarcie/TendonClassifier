import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings


def load_normal_map(image_id, normal_maps_dir="normal_maps"):
    """
    Load normal map from numpy file.

    Args:
        image_id: Identifier for the image
        normal_maps_dir: Directory containing normal map files

    Returns:
        (H, W, 3) array of unit surface normals
    """
    npy_path = os.path.join(normal_maps_dir, f"{image_id}_normals.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Normal map not found: {npy_path}")
    return np.load(npy_path)


def poisson_reconstruct(p, q):
    """
    Solve Poisson equation for z from gradients p=dz/dx and q=dz/dy
    using the FFT-based method.

    Args:
        p: (H, W) array of dz/dx gradients
        q: (H, W) array of dz/dy gradients

    Returns:
        z: (H, W) reconstructed height field
    """
    H, W = p.shape

    # Compute divergence
    div = np.zeros((H, W), dtype=np.float32)
    div[:, :-1] += p[:, :-1] - p[:, 1:]
    div[:-1, :] += q[:-1, :] - q[1:, :]

    # FFT of divergence
    div_fft = np.fft.fft2(div)

    # Frequencies for Laplacian in Fourier domain
    u = np.fft.fftfreq(W).reshape(1, W)
    v = np.fft.fftfreq(H).reshape(H, 1)

    # Laplacian denominator: 2*cos(2*pi*freq) - 2
    denom = (2*np.cos(2*np.pi*u) - 2) + (2*np.cos(2*np.pi*v) - 2)
    denom[0, 0] = 1.0  # Better than np.inf

    # Solve in frequency domain
    Z = div_fft / denom
    Z[0, 0] = 0  # set integration constant

    # Inverse FFT to get height field
    z = np.real(np.fft.ifft2(Z))

    return z


def reconstruct_surface_from_normals(normals, cmap='viridis', mask_background=True):
    """
    Reconstruct 3D surface from normal map using Poisson reconstruction.

    Args:
        normals: (H, W, 3) array of unit surface normals
        cmap: Colormap for visualization
        mask_background: If True, mask pixels with Nz ≈ 1 (flat background)

    Returns:
        z: (H, W) reconstructed height field
    """
    H, W = normals.shape[:2]

    # Extract normal components
    Nx = normals[:, :, 0]
    Ny = normals[:, :, 1]
    Nz = normals[:, :, 2]

    # Identify valid surface regions (not background)
    if mask_background:
        background_mask = (np.abs(Nx) < 0.01) & (np.abs(Ny) < 0.01) & (Nz > 0.99)
        valid_mask = ~background_mask
        print(f"Valid surface pixels: {valid_mask.sum()} / {H*W} ({100*valid_mask.sum()/(H*W):.1f}%)")
    else:
        valid_mask = np.ones((H, W), dtype=bool)

    # Check for grazing angles
    low_nz_mask = (Nz < 0.1) & valid_mask
    low_nz_count = low_nz_mask.sum()
    if low_nz_count > 0:
        warnings.warn(
            f"Warning: {low_nz_count} pixels have Nz < 0.1 (grazing angles)"
        )

    # Compute gradients from normals
    Nz_safe = np.maximum(np.abs(Nz), 1e-6)
    p = -Nx / Nz_safe  # dz/dx
    q = -Ny / Nz_safe  # dz/dy

    # Zero out gradients in background
    if mask_background:
        p = p * valid_mask
        q = q * valid_mask

    print(f"\nComputing Poisson reconstruction for {H}x{W} image...")
    print(f"Gradient range: p in [{p.min():.3f}, {p.max():.3f}], q in [{q.min():.3f}, {q.max():.3f}]")

    # Reconstruct height field
    z = poisson_reconstruct(p, q)

    # Get stats from valid pixels only
    z_valid = z[valid_mask] if mask_background else z.ravel()
    z_min, z_max = z_valid.min(), z_valid.max()
    
    if z_max > z_min:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z)

    print(f"Reconstructed surface: z in [{z_min:.3f}, {z_max:.3f}]")

    # Visualization (keep same 2-plot layout)
    fig = plt.figure(figsize=(16, 6))

    # 3D surface plot with masking
    ax1 = fig.add_subplot(121, projection='3d')
    u = np.arange(W)
    v = np.arange(H)
    U, V = np.meshgrid(u, v)

    # Use masked array for cleaner visualization
    if mask_background:
        z_plot = np.ma.masked_where(~valid_mask, z)
    else:
        z_plot = z

    surf = ax1.plot_surface(U, V, z_plot, cmap=cmap, alpha=0.8, 
                           antialiased=True, edgecolor='none')
    ax1.set_xlabel('u (pixels)')
    ax1.set_ylabel('v (pixels)')
    ax1.set_zlabel('Height')
    ax1.set_title('Reconstructed Surface (Poisson)')
    plt.colorbar(surf, ax=ax1, shrink=0.5)

    # 2D height map
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(z_norm, cmap=cmap)
    ax2.set_xlabel('u (pixels)')
    ax2.set_ylabel('v (pixels)')
    ax2.set_title('Height Map (normalized)')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()

    return z


def visualize_normal_components(normals):
    """
    Visualize individual normal components as heatmaps.

    Args:
        normals: (H, W, 3) array of unit surface normals
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # X component
    im0 = axes[0].imshow(normals[:, :, 0], cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Normal X component')
    plt.colorbar(im0, ax=axes[0])

    # Y component
    im1 = axes[1].imshow(normals[:, :, 1], cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Normal Y component')
    plt.colorbar(im1, ax=axes[1])

    # Z component
    im2 = axes[2].imshow(normals[:, :, 2], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Normal Z component')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the normal map
    print("Loading normal map...")
    normals = load_normal_map("frame_000585")
    print(f"Normal map shape: {normals.shape}")
    print(f"Normal magnitude stats:")
    magnitude = np.linalg.norm(normals, axis=2)
    print(f"  Min: {magnitude.min():.4f}, Max: {magnitude.max():.4f}, Mean: {magnitude.mean():.4f}")

    # Visualize normal components
    print("\nVisualizing normal components...")
    visualize_normal_components(normals)

    # Reconstruct surface using Poisson reconstruction with background masking
    print("\nReconstructing 3D surface from normals using Poisson reconstruction...")
    surface = reconstruct_surface_from_normals(normals, cmap='viridis', mask_background=True)

    print("\nDone!")