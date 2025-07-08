# data_transforms.py
"""
Data transformation utils for LMO dataset preprocessing - FoundationPose Compatible

Handles:
- Depth denoising (bilateral filtering + erosion)
- Pose-conditioned cropping to 160x160
- RGB-D to RGBXYZ conversion with FoundationPose normalization
- Mesh rendering with correct coordinate transforms
- Pose hypothesis generation from icosphere viewpoints

Coordinate system: +X right, +Y down, +Z forward (OpenCV/camera convention)
"""

import numpy as np
import cv2
import trimesh
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available. Install with: pip install pyrender")

class DataTransforms:
    """Data preprocessing for pose estimation models - FoundationPose compatible"""

    CROP_SIZE = 160
    CROP_RATIO = 1.2  # Padding factor for crops

    def __init__(self, camera_intrinsics: np.ndarray, image_size: Tuple[int, int] = (640, 480)):
        """
        Args:
            camera_intrinsics: (3,3) camera K matrix
            image_size: (width, height) of images
        """
        self.K = camera_intrinsics
        self.width, self.height = image_size

        if not PYRENDER_AVAILABLE:
            raise ImportError("pyrender required for mesh rendering. Install with: pip install pyrender")

        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)

    def bilateral_filter_depth(self, depth: np.ndarray, radius: int = 2, sigmaD: float = 2.0,
                              sigmaR: float = 100000.0, zfar: float = 100.0) -> np.ndarray:
        """
        Apply bilateral filtering to depth - FoundationPose implementation

        Args:
            depth: (H,W) depth in meters
            radius: filter radius
            sigmaD: spatial sigma
            sigmaR: range sigma
            zfar: far clipping distance

        Returns:
            filtered_depth: (H,W) filtered depth
        """
        h, w = depth.shape
        filtered = np.zeros_like(depth)

        # Create spatial kernel
        spatial_kernel = np.zeros((2*radius+1, 2*radius+1))
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                spatial_kernel[dy+radius, dx+radius] = np.exp(-(dx*dx + dy*dy) / (2 * sigmaD * sigmaD))

        # Apply bilateral filter
        for y in range(h):
            for x in range(w):
                center_depth = depth[y, x]

                # Skip invalid depths
                if center_depth < 0.001 or center_depth >= zfar:
                    filtered[y, x] = 0.0
                    continue

                # Collect valid neighbors for mean computation
                valid_neighbors = []
                for dy in range(max(0, y-radius), min(h, y+radius+1)):
                    for dx in range(max(0, x-radius), min(w, x+radius+1)):
                        neighbor_depth = depth[dy, dx]
                        if 0.001 <= neighbor_depth < zfar:
                            valid_neighbors.append(neighbor_depth)

                if len(valid_neighbors) == 0:
                    filtered[y, x] = 0.0
                    continue

                mean_depth = np.mean(valid_neighbors)

                # Weighted sum
                sum_weights = 0.0
                sum_values = 0.0

                for dy in range(max(0, y-radius), min(h, y+radius+1)):
                    for dx in range(max(0, x-radius), min(w, x+radius+1)):
                        neighbor_depth = depth[dy, dx]

                        if (0.001 <= neighbor_depth < zfar and
                            abs(neighbor_depth - mean_depth) < 0.01):

                            # Spatial weight
                            spatial_dist = (x-dx)**2 + (y-dy)**2
                            spatial_w = np.exp(-spatial_dist / (2 * sigmaD**2))

                            # Range weight
                            range_dist = (center_depth - neighbor_depth)**2
                            range_w = np.exp(-range_dist / (2 * sigmaR**2))

                            weight = spatial_w * range_w
                            sum_weights += weight
                            sum_values += weight * neighbor_depth

                if sum_weights > 0:
                    filtered[y, x] = sum_values / sum_weights
                else:
                    filtered[y, x] = 0.0

        return filtered

    def erode_depth(self, depth: np.ndarray, radius: int = 2, depth_diff_thres: float = 0.001,
                    ratio_thres: float = 0.8, zfar: float = 100.0) -> np.ndarray:
        """
        Apply erosion to remove isolated depth pixels - FoundationPose implementation

        Args:
            depth: (H,W) depth in meters
            radius: erosion radius
            depth_diff_thres: threshold for depth difference
            ratio_thres: ratio threshold for bad neighbors
            zfar: far clipping distance

        Returns:
            eroded_depth: (H,W) eroded depth
        """
        h, w = depth.shape
        eroded = depth.copy()

        for y in range(h):
            for x in range(w):
                center_depth = depth[y, x]

                if center_depth < 0.001 or center_depth >= zfar:
                    eroded[y, x] = 0.0
                    continue

                # Count bad neighbors
                bad_count = 0
                total_count = 0

                for dy in range(max(0, y-radius), min(h, y+radius+1)):
                    for dx in range(max(0, x-radius), min(w, x+radius+1)):
                        total_count += 1
                        neighbor_depth = depth[dy, dx]

                        if (neighbor_depth < 0.001 or neighbor_depth >= zfar or
                            abs(neighbor_depth - center_depth) > depth_diff_thres):
                            bad_count += 1

                if total_count > 0 and bad_count / total_count > ratio_thres:
                    eroded[y, x] = 0.0

        return eroded

    def denoise_depth(self, depth: np.ndarray) -> np.ndarray:
        """Apply FoundationPose depth denoising pipeline"""
        depth_filtered = self.bilateral_filter_depth(depth)
        depth_eroded = self.erode_depth(depth_filtered)
        return depth_eroded

    def pose_conditioned_crop(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray,
                             mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        FoundationPose pose-conditioned cropping strategy

        Args:
            rgb: (H,W,3) uint8 image
            depth: (H,W) float32 depth in meters
            pose: (4,4) camera_T_object pose
            mesh: Object mesh for diameter calculation

        Returns:
            cropped_rgb: (160,160,3) uint8
            cropped_depth: (160,160) float32 meters
            crop_info: dict with crop parameters
        """
        # Apply depth denoising first
        depth_clean = self.denoise_depth(depth)

        # Project object center to image
        object_center_3d = pose[:3, 3]
        center_proj = self.K @ object_center_3d
        center_u = center_proj[0] / center_proj[2]
        center_v = center_proj[1] / center_proj[2]

        # Calculate object diameter in pixels
        mesh_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

        # Project diameter to pixel size at object depth
        depth_at_center = object_center_3d[2]
        diameter_pixels = (self.K[0,0] * mesh_diameter) / depth_at_center

        # Calculate crop size with padding
        crop_size = diameter_pixels * self.CROP_RATIO
        half_size = crop_size / 2

        # Calculate crop bounds
        crop_x_min = int(center_u - half_size)
        crop_x_max = int(center_u + half_size)
        crop_y_min = int(center_v - half_size)
        crop_y_max = int(center_v + half_size)

        # Handle padding if crop goes outside image bounds
        pad_left = max(0, -crop_x_min)
        pad_right = max(0, crop_x_max - self.width)
        pad_top = max(0, -crop_y_min)
        pad_bottom = max(0, crop_y_max - self.height)

        if any([pad_left, pad_right, pad_top, pad_bottom]):
            rgb_padded = np.pad(rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
            depth_padded = np.pad(depth_clean, ((pad_top, pad_bottom), (pad_left, pad_right)))

            crop_x_min += pad_left
            crop_x_max += pad_left
            crop_y_min += pad_top
            crop_y_max += pad_top
        else:
            rgb_padded = rgb
            depth_padded = depth_clean

        # Extract crop
        cropped_rgb = rgb_padded[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        cropped_depth = depth_padded[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # Resize to target size
        cropped_rgb = cv2.resize(cropped_rgb, (self.CROP_SIZE, self.CROP_SIZE))
        cropped_depth = cv2.resize(cropped_depth, (self.CROP_SIZE, self.CROP_SIZE),
                                  interpolation=cv2.INTER_NEAREST)

        crop_info = {
            'center_uv': (center_u, center_v),
            'crop_size': crop_size,
            'mesh_diameter': mesh_diameter,
            'padding': (pad_left, pad_right, pad_top, pad_bottom)
        }

        return cropped_rgb, cropped_depth, crop_info

    def mask_crop(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simple mask-based cropping for compatibility

        Args:
            rgb: (H,W,3) uint8 image
            depth: (H,W) float32 depth in meters
            mask: (H,W) bool object mask

        Returns:
            cropped_rgb: (160,160,3) uint8
            cropped_depth: (160,160) float32 meters
            crop_info: dict with crop parameters
        """
        # Apply depth denoising
        depth_clean = self.denoise_depth(depth)

        # Get mask bounding box
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            raise ValueError("Empty mask provided")

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Calculate center and size
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        crop_size = max(bbox_w, bbox_h) * self.CROP_RATIO

        # Calculate crop bounds
        half_size = crop_size / 2
        crop_x_min = int(center_x - half_size)
        crop_x_max = int(center_x + half_size)
        crop_y_min = int(center_y - half_size)
        crop_y_max = int(center_y + half_size)

        # Handle padding
        pad_left = max(0, -crop_x_min)
        pad_right = max(0, crop_x_max - self.width)
        pad_top = max(0, -crop_y_min)
        pad_bottom = max(0, crop_y_max - self.height)

        if any([pad_left, pad_right, pad_top, pad_bottom]):
            rgb_padded = np.pad(rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
            depth_padded = np.pad(depth_clean, ((pad_top, pad_bottom), (pad_left, pad_right)))

            crop_x_min += pad_left
            crop_x_max += pad_left
            crop_y_min += pad_top
            crop_y_max += pad_top
        else:
            rgb_padded = rgb
            depth_padded = depth_clean

        # Extract and resize crop
        cropped_rgb = rgb_padded[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        cropped_depth = depth_padded[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        cropped_rgb = cv2.resize(cropped_rgb, (self.CROP_SIZE, self.CROP_SIZE))
        cropped_depth = cv2.resize(cropped_depth, (self.CROP_SIZE, self.CROP_SIZE),
                                  interpolation=cv2.INTER_NEAREST)

        crop_info = {
            'center': (center_x, center_y),
            'crop_size': crop_size,
            'bbox': (x_min, y_min, x_max, y_max),
            'padding': (pad_left, pad_right, pad_top, pad_bottom)
        }

        return cropped_rgb, cropped_depth, crop_info

    def rgbd_to_rgbxyz(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray,
                       mesh: trimesh.Trimesh, crop_info: Optional[Dict] = None) -> np.ndarray:
        """
        Convert RGB-D to RGBXYZ representation for model input - FoundationPose style

        Args:
            rgb: (160,160,3) uint8 cropped RGB
            depth: (160,160) float32 cropped depth in meters
            pose: (4,4) object pose for normalization center
            mesh: Object mesh for diameter calculation
            crop_info: Crop metadata for coordinate transformation

        Returns:
            rgbxyz: (160,160,6) float32 normalized RGBXYZ
                   Channels: [R,G,B,X,Y,Z] where RGB in [0,1] range,
                   XYZ relative to object center, normalized by mesh radius
        """
        h, w = depth.shape

        # Create pixel coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Transform cropped coords back to original image coords
        if crop_info and 'crop_size' in crop_info:
            if 'center_uv' in crop_info:  # Pose-conditioned crop
                center_u, center_v = crop_info['center_uv']
            else:  # Mask crop
                center_u, center_v = crop_info['center']
            crop_size = crop_info['crop_size']
            scale = crop_size / self.CROP_SIZE

            u_orig = (u_coords - self.CROP_SIZE/2) * scale + center_u
            v_orig = (v_coords - self.CROP_SIZE/2) * scale + center_v
        else:
            # Fallback if no crop info
            scale_x = self.width / self.CROP_SIZE
            scale_y = self.height / self.CROP_SIZE
            u_orig = u_coords * scale_x
            v_orig = v_coords * scale_y

        # Back-project to 3D using camera intrinsics
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]

        # Valid depth mask
        valid_mask = depth >= 0.001

        # Initialize XYZ maps
        X = np.zeros_like(depth)
        Y = np.zeros_like(depth)
        Z = np.zeros_like(depth)

        # Compute 3D points for valid depths
        X[valid_mask] = (u_orig[valid_mask] - cx) * depth[valid_mask] / fx
        Y[valid_mask] = (v_orig[valid_mask] - cy) * depth[valid_mask] / fy
        Z[valid_mask] = depth[valid_mask]

        # Transform to object-centered coordinates (FoundationPose approach)
        object_center = pose[:3, 3]  # Object center in camera coords
        X_rel = X - object_center[0]
        Y_rel = Y - object_center[1]
        Z_rel = Z - object_center[2]

        # Normalize by mesh radius (critical for FoundationPose)
        mesh_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        mesh_radius = mesh_diameter / 2

        X_norm = X_rel / mesh_radius
        Y_norm = Y_rel / mesh_radius
        Z_norm = Z_rel / mesh_radius

        # Set invalid regions to 0 (FoundationPose behavior)
        # Invalid if: invalid depth OR normalized coords too large
        invalid_mask = (~valid_mask |
                       (np.abs(X_norm) >= 2) |
                       (np.abs(Y_norm) >= 2) |
                       (np.abs(Z_norm) >= 2))

        X_norm[invalid_mask] = 0
        Y_norm[invalid_mask] = 0
        Z_norm[invalid_mask] = 0

        # Normalize RGB to [0,1] range (FoundationPose approach - NO ImageNet normalization)
        rgb_norm = rgb.astype(np.float32) / 255.0

        # Stack into RGBXYZ
        rgbxyz = np.stack([
            rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2],
            X_norm, Y_norm, Z_norm
        ], axis=2)

        return rgbxyz.astype(np.float32)

    def render_mesh_at_pose(self, mesh: trimesh.Trimesh, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Render mesh at specified pose to RGB, depth, and mask

        Args:
            mesh: Object mesh
            pose: (4,4) camera_T_object pose

        Returns:
            rendered_rgb: (H,W,3) uint8
            rendered_depth: (H,W) float32 meters
            rendered_mask: (H,W) bool
        """
        # FoundationPose coordinate transform: OpenCV to OpenGL
        cvcam_in_glcam = np.array([[1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]], dtype=np.float32)

        # Create pyrender scene
        scene = pyrender.Scene(ambient_light=np.ones(3) * 0.4, bg_color=[0,0,0])

        # Transform mesh: apply pose then convert to OpenGL coords
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(cvcam_in_glcam @ pose)

        # Create pyrender mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_copy, smooth=False)
        scene.add(mesh_pyrender, pose=np.eye(4))

        # Add camera at origin (in OpenGL coords)
        camera = pyrender.IntrinsicsCamera(
            fx=self.K[0,0], fy=self.K[1,1],
            cx=self.K[0,2], cy=self.K[1,2],
            znear=0.001, zfar=100.0
        )
        scene.add(camera, pose=np.eye(4))

        # Add directional light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))

        # Render
        color, depth = self.renderer.render(scene)

        # Convert depth and create mask
        rendered_depth = depth.astype(np.float32)
        rendered_mask = rendered_depth > 0
        rendered_depth[~rendered_mask] = 0

        return color, rendered_depth, rendered_mask

    def sample_viewpoints_icosphere(self, n_views: int = 40) -> List[np.ndarray]:
        """
        Sample viewpoints uniformly on icosphere - FoundationPose approach

        Args:
            n_views: Minimum number of viewpoints

        Returns:
            cam_in_obs: List of (4,4) camera poses looking at origin
        """
        # Create icosphere
        ico = trimesh.creation.icosphere(subdivisions=2)
        vertices = ico.vertices
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Select evenly distributed vertices
        if len(vertices) > n_views:
            # Use farthest point sampling
            selected_indices = [0]
            distances = np.linalg.norm(vertices - vertices[0], axis=1)

            for _ in range(1, n_views):
                farthest_idx = np.argmax(distances)
                selected_indices.append(farthest_idx)

                # Update distances
                new_distances = np.linalg.norm(vertices - vertices[farthest_idx], axis=1)
                distances = np.minimum(distances, new_distances)

            vertices = vertices[selected_indices]

        # Convert to camera poses
        cam_in_obs = []
        for vertex in vertices:
            # Camera position
            cam_pos = vertex * 0.6  # 60cm from origin

            # Look-at matrix (camera looking at origin)
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0, 1, 0])

            # Handle case when forward is parallel to up
            if abs(np.dot(forward, up)) > 0.99:
                up = np.array([0, 0, 1])

            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            # Build camera pose
            cam_pose = np.eye(4)
            cam_pose[:3, 0] = right
            cam_pose[:3, 1] = up
            cam_pose[:3, 2] = forward
            cam_pose[:3, 3] = cam_pos

            cam_in_obs.append(cam_pose)

        return cam_in_obs

    def generate_pose_hypotheses(self, mesh: trimesh.Trimesh, depth: np.ndarray,
                                mask: np.ndarray, K: np.ndarray,
                                n_views: int = 40, n_inplane: int = 6) -> List[np.ndarray]:
        """
        Generate pose hypotheses using FoundationPose strategy

        Args:
            mesh: Object mesh
            depth: (H,W) depth image
            mask: (H,W) object mask
            K: (3,3) camera intrinsics
            n_views: Number of viewpoints on icosphere
            n_inplane: Number of in-plane rotations per viewpoint

        Returns:
            poses: List of (4,4) pose hypotheses
        """
        # Sample viewpoints on icosphere
        cam_in_obs = self.sample_viewpoints_icosphere(n_views)

        # Initialize translation using median depth
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            raise ValueError("Empty mask")

        # Get center of 2D bounding box
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0

        # Get median depth in mask
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            raise ValueError("No valid depth in mask")

        zc = np.median(depth[valid])

        # Back-project to get 3D center
        center_3d = np.linalg.inv(K) @ np.array([uc, vc, 1]) * zc

        # Generate pose hypotheses
        poses = []
        for cam_in_ob in cam_in_obs:
            # Add in-plane rotations
            for angle in np.linspace(0, 2*np.pi, n_inplane, endpoint=False):
                # In-plane rotation matrix
                R_inplane = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0],
                                     [0, 0, 1]])

                # Apply in-plane rotation
                cam_in_ob_rot = cam_in_ob.copy()
                cam_in_ob_rot[:3, :3] = cam_in_ob[:3, :3] @ R_inplane

                # Object-to-camera transform
                ob_in_cam = np.linalg.inv(cam_in_ob_rot)

                # Set translation
                ob_in_cam[:3, 3] = center_3d

                poses.append(ob_in_cam)

        return poses
