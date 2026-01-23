"""
TendonLabeler class for generating masks of tendon in phantom images.

Based on original implementation from test1.py.
"""

import math
import trimesh
import numpy as np
import cv2
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tendon_labeler import TendonLabeler as BaseTendonLabeler


class TendonLabeler:
    def __init__(self, phantom_path, rotation_deg=0):
        # Load and extract tendon from phantom
        phantom = trimesh.load(phantom_path, process=True)


        # Apply rotation if specified (around Z-axis) depends on the experiment run.
        if rotation_deg != 0:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                math.radians(rotation_deg),
                [0, 0, -1]  # Z-axis
            )
            phantom.apply_transform(rotation_matrix)

        # Test here (good)            

        # Tendon bounding box (We just select the tendon from the STL)
        X_low, X_high = -0.01, 0.01
        Y_low, Y_high = -0.075, 0.075
        Z_low, Z_high = 0.008, 0.021

        verts = phantom.vertices
        mask = (verts[:, 0] >= X_low) & (verts[:, 0] <= X_high) & \
               (verts[:, 1] >= Y_low) & (verts[:, 1] <= Y_high) & \
               (verts[:, 2] >= Z_low) & (verts[:, 2] <= Z_high)

        vertex_indices = np.where(mask)[0]
        face_mask = np.all(np.isin(phantom.faces, vertex_indices), axis=1)
        tendon_face_indices = np.where(face_mask)[0]

        # Extract Tendon Mesh
        self.tendon_mesh = phantom.submesh([tendon_face_indices], append=True)

        # Manual center (per frame)
        self.tendon_center_px = None

        # Camera params
        self.cam_z = 0.03724

    def set_center(self, center_px):
        """Manually set tendon center in pixels."""
        self.tendon_center_px = center_px

    def visualize_setup(self):
        """Visualize tendon mesh and camera position."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot tendon mesh vertices
        verts = self.tendon_mesh.vertices
        ax.scatter(verts[:, 0] * 1000, verts[:, 1] * 1000, verts[:, 2] * 1000,
                   s=1, alpha=0.3, label='Tendon mesh')

        # Plot camera position
        ax.scatter([0], [0], [self.cam_z * 1000],
                   s=100, c='red', marker='^', label='Camera')

        # Draw camera viewing direction
        ax.quiver(0, 0, self.cam_z * 1000, 0, 0, -10, color='red', arrow_length_ratio=0.3)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend()
        ax.set_title('Tendon + Camera Setup')

        # Equal aspect ratio
        max_range = max(verts.max(axis=0) - verts.min(axis=0)) * 1000 / 2
        mid = verts.mean(axis=0) * 1000
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(0, self.cam_z * 1000 + 5)

        plt.show()

    def _fill_depth_in_mask(self, sparse_depth, mask):
        """Interpolate depth within mask region."""
        valid = sparse_depth > 0
        if not np.any(valid):
            return sparse_depth

        coords = np.array(np.nonzero(valid)).T
        values = sparse_depth[valid]
        interp = NearestNDInterpolator(coords, values)

        mask_coords = np.array(np.nonzero(mask > 0)).T

        depth_map = np.zeros_like(sparse_depth)
        depth_map[mask > 0] = interp(mask_coords)

        return depth_map


if __name__ == "__main__":
    phantom_path = "rawdata/p1n2t.stl"  # update this path
    
    labeler = TendonLabeler(phantom_path, rotation_deg=0)
    comp = BaseTendonLabeler(phantom_path, rotation_deg=0)
    labeler.visualize_setup()