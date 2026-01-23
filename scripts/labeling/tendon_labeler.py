import math
import numpy as np
import trimesh
import pyrender
import cv2
import os
import matplotlib.pyplot as plt

# If running on a server/headless machine, uncomment this:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

class TendonLabeler:
    def __init__(self, phantom_path, rotation_deg=0):
        # 1. Load the original phantom mesh
        phantom = trimesh.load(phantom_path, process=True)

        # 2. Apply Rotation (around Z-axis)
        if rotation_deg != 0:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                math.radians(rotation_deg), [0, 0, 1]
            )
            phantom.apply_transform(rotation_matrix)

        # 3. Extract the Tendon (Bounding Box logic)
        # We keep your exact logic here to slice the mesh
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

        # Create the submesh for the tendon
        self.tendon_trimesh = phantom.submesh([tendon_face_indices], append=True)
        
        # 4. Convert to PyRender Mesh (Crucial for Rasterization)
        # We create a simple material so it renders solidly
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, 
            alphaMode='OPAQUE', 
            baseColorFactor=(1.0, 1.0, 1.0, 1.0)
        )
        self.tendon_mesh = pyrender.Mesh.from_trimesh(
            self.tendon_trimesh, 
            material=material
        )

        self.tendon_center_px = None
        
        # Camera Z offset (from your original code)
        deformation = 0.013
        self.cam_z_offset = 0.03724 - deformation 

    def set_center(self, center_px):
        self.tendon_center_px = center_px

    def generate_mask(self, width_px, height_px, cam_y=0.32):
        """
        Render the mesh to generate a perfect mask and depth map.
        """
        
        # --- 1. Camera Intrinsics (Pinhole) ---
        FOV_deg = 100
        fx = (0.5 * width_px) / math.tan(math.radians(FOV_deg) / 2)
        fy = fx 
        cx, cy = width_px / 2.0, height_px / 2.0
        
        # Apply your manual pixel offset if set
        if self.tendon_center_px is not None:
            offset_px = self.tendon_center_px - (width_px // 2)
            cx += offset_px

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=1.0)

        # --- 2. Camera Extrinsics (Pose) ---
        # PyRender/OpenGL coordinates: -Z is forward, +Y is up.
        # Your data seems to have Z as depth. We need to construct the pose matrix carefully.
        
        # We want the camera to look at the tendon.
        # Based on your previous code logic:
        # Camera is at: [0, cam_y - 0.32, self.cam_z_offset]
        # Tendon is at [0, 0, 0] (roughly)
        
        y_cam_relative = cam_y - 0.32
        
        # Create a transformation matrix (4x4)
        # This places the camera at the correct spot relative to the mesh
        camera_pose = np.eye(4)
        
        # Translation
        camera_pose[0, 3] = 0.0               # X
        camera_pose[1, 3] = y_cam_relative    # Y
        camera_pose[2, 3] = self.cam_z_offset # Z
        
        # Rotation: We need to align the camera to look down -Z (or however your mesh is oriented)
        # Assuming the mesh is flat on XY plane and Z is up:
        # We rotate 180 deg around X so camera looks "down" if Z is up
        # (You may need to tweak this rotation depending on your specific mesh orientation)
        rot_x_180 = trimesh.transformations.rotation_matrix(math.radians(180), [0,0,1])
        camera_pose = camera_pose @ rot_x_180

        # --- 3. Scene Setup ---
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        scene.add(self.tendon_mesh)
        scene.add(camera, pose=camera_pose)

        # --- 4. Render ---
        r = pyrender.OffscreenRenderer(width_px, height_px)
        # We only need depth for the mask, but color is useful for debugging
        color, depth = r.render(scene)
        
        # Clean up renderer to free GPU context
        r.delete()

        # --- 5. Post-Process ---
        # Mask: Anywhere depth > 0 is the object
        mask = (depth > 0).astype(np.uint8) * 255
        
        # Depth Map: Convert to mm
        depth_map_mm = depth.astype(np.float32) * 1000.0

        return mask, depth_map_mm

    def apply_fisheye_distortion(self, image, K, D):
        """
        Optional: Warp the pinhole render to match the real fisheye camera.
        K: Camera Matrix (3x3)
        D: Distortion coefficients
        """
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w,h), 5)
        
        # Note: Usually we undistort real images. 
        # To DISTORT synthetic images to match fisheye, you need the inverse mapping,
        # which is complex. A simple approx is to use a barrel distortion shader 
        # or standard OpenCV remap if you have the specific mappings.
        return image
    
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