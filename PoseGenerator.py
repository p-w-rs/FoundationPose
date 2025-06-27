import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict
import cv2
from LMODataLoader import LMODataLoader
from DepthProcessor import DepthProcessor

class PoseProposalGenerator:
    """
    Generates initial pose proposals for FoundationPose.

    UNITS: Everything in millimeters (mm)
    - Mesh vertices: mm
    - Pose translations: mm
    - Depth rendering: meters (for visualization compatibility)

    Key methods from the paper:
    1. Uniform SO(3) rotation sampling on sphere
    2. Translation from mask center and median depth
    3. Fast GPU-based rendering for hypothesis generation
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Args:
            mesh: Object mesh with vertices in millimeters
        """
        self.mesh = mesh

        # Compute mesh properties (in mm)
        self.diameter = self._compute_diameter()
        self.centroid = mesh.centroid

        # Pre-generate rotation grid for fast sampling
        self.rotation_grid = self._generate_rotation_grid()

    def _compute_diameter(self) -> float:
        """Compute mesh diameter as max distance between vertices"""
        vertices = self.mesh.vertices
        # Sample subset for efficiency
        n_sample = min(1000, len(vertices))
        idx = np.random.choice(len(vertices), n_sample, replace=False)
        sampled = vertices[idx]

        # Compute pairwise distances
        dists = np.linalg.norm(sampled[:, None] - sampled[None, :], axis=2)
        diameter = dists.max()

        return diameter

    def _generate_rotation_grid(self, min_n_views: int = 42) -> List[np.ndarray]:
        """
        Generate uniformly distributed rotations on SO(3).
        Uses spherical Fibonacci sampling for uniform coverage.

        Args:
            min_n_views: Minimum number of viewpoints (will be rounded up)

        Returns:
            rotations: List of 3x3 rotation matrices
        """
        # Spherical fibonacci for uniform sampling
        n_views = int(np.ceil(min_n_views))

        rotations = []

        # Generate viewpoints on sphere
        for i in range(n_views):
            # Fibonacci sphere sampling
            theta = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)
            phi = np.arccos(1 - 2 * (i + 0.5) / n_views)

            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            # Create rotation matrix (camera looking at origin)
            forward = -np.array([x, y, z])  # Camera looks down -Z
            up_guess = np.array([0, 1, 0]) if abs(z) < 0.9 else np.array([1, 0, 0])
            right = np.cross(forward, up_guess)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)

            R_mat = np.stack([right, up, -forward], axis=0).T

            # Add in-plane rotations
            for in_plane_angle in [0, 60, 120, 180, 240, 300]:
                R_inplane = self._rotation_matrix_z(np.radians(in_plane_angle))
                R_total = R_mat @ R_inplane
                rotations.append(R_total)

        return rotations

    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Rotation matrix around Z axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def estimate_translation(self, depth: np.ndarray, mask: np.ndarray,
                           K: np.ndarray) -> np.ndarray:
        """
        Estimate initial translation from mask center and median depth.

        Args:
            depth: Depth map in meters
            mask: Binary object mask
            K: Camera intrinsic matrix

        Returns:
            translation: 3D translation in millimeters
        """
        # Find mask center
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3)

        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0

        # Get median depth of masked region
        valid = mask & (depth > 0.001)
        if not valid.any():
            return np.zeros(3)

        zc = np.median(depth[valid]) * 1000  # Convert to mm

        # Back-project center point
        xc = (uc - K[0, 2]) * zc / K[0, 0]
        yc = (vc - K[1, 2]) * zc / K[1, 1]

        return np.array([xc, yc, zc])

    def generate_poses(self, depth: np.ndarray, mask: np.ndarray,
                      K: np.ndarray, n_proposals: int = 256) -> np.ndarray:
        """
        Generate pose proposals combining rotation grid and estimated translation.

        Args:
            depth: Depth map in meters
            mask: Binary object mask
            K: Camera intrinsic matrix
            n_proposals: Number of proposals to generate

        Returns:
            poses: Array of 4x4 transformation matrices (object to camera) in mm
        """
        # Estimate translation
        translation = self.estimate_translation(depth, mask, K)

        # Sample rotations
        n_rotations = min(n_proposals, len(self.rotation_grid))
        if n_rotations < len(self.rotation_grid):
            indices = np.random.choice(len(self.rotation_grid), n_rotations, replace=False)
            rotations = [self.rotation_grid[i] for i in indices]
        else:
            rotations = self.rotation_grid[:n_rotations]

        # Build 4x4 matrices
        poses = []
        for R_mat in rotations:
            pose = np.eye(4)
            pose[:3, :3] = R_mat
            pose[:3, 3] = translation
            poses.append(pose)

        return np.array(poses)

    def visualize_proposals(self, rgb: np.ndarray, depth: np.ndarray,
                          poses: np.ndarray, K: np.ndarray, n_show: int = 6):
        """Visualize pose proposals by rendering object"""
        fig, axes = plt.subplots(2, n_show//2, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(n_show, len(poses))):
            # Simple projection visualization
            pose = poses[i]

            # Transform mesh vertices
            vertices_cam = (pose[:3, :3] @ self.mesh.vertices.T).T + pose[:3, 3]

            # Project to image
            vertices_proj = (K @ vertices_cam.T).T
            vertices_2d = vertices_proj[:, :2] / vertices_proj[:, 2:3]

            # Plot
            ax = axes[i]
            ax.imshow(rgb)

            # Draw projected vertices (subsample for visibility)
            idx = np.random.choice(len(vertices_2d), min(500, len(vertices_2d)), replace=False)
            ax.scatter(vertices_2d[idx, 0], vertices_2d[idx, 1], c='r', s=1, alpha=0.5)

            ax.set_title(f'Proposal {i}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def render_pose(self, pose: np.ndarray, K: np.ndarray,
                   H: int = 480, W: int = 640) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple rendering of mesh at given pose.

        Returns:
            rendered_rgb: Rendered color image
            rendered_depth: Rendered depth in meters
        """
        # Transform vertices to camera frame
        vertices_cam = (pose[:3, :3] @ self.mesh.vertices.T).T + pose[:3, 3]

        # Create depth buffer
        depth_buffer = np.full((H, W), np.inf)
        rendered_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Simple rasterization (for visualization only)
        # Project all vertices
        vertices_proj = (K @ vertices_cam.T).T
        vertices_2d = vertices_proj[:, :2] / vertices_proj[:, 2:3]
        depths = vertices_proj[:, 2]

        # Draw each face
        for face in self.mesh.faces:
            # Get face vertices
            v0, v1, v2 = vertices_2d[face]
            z0, z1, z2 = depths[face]

            # Skip if behind camera
            if z0 <= 0 or z1 <= 0 or z2 <= 0:
                continue

            # Get bounding box
            min_x = int(max(0, min(v0[0], v1[0], v2[0])))
            max_x = int(min(W-1, max(v0[0], v1[0], v2[0])))
            min_y = int(max(0, min(v0[1], v1[1], v2[1])))
            max_y = int(min(H-1, max(v0[1], v1[1], v2[1])))

            # Simple fill with average depth (for visualization)
            avg_depth = (z0 + z1 + z2) / 3.0 / 1000.0  # Convert to meters
            depth_buffer[min_y:max_y+1, min_x:max_x+1] = np.minimum(
                depth_buffer[min_y:max_y+1, min_x:max_x+1], avg_depth
            )
            rendered_rgb[min_y:max_y+1, min_x:max_x+1] = [200, 200, 200]  # Gray

        # Convert inf to 0
        rendered_depth = np.where(depth_buffer == np.inf, 0, depth_buffer)

        return rendered_rgb, rendered_depth

    def crop_and_resize_proposal(self, rgb: np.ndarray, depth: np.ndarray,
                                pose: np.ndarray, K: np.ndarray,
                                target_size: int = 160, padding: float = 1.4) -> Dict:
        """
        Crop and resize RGBD around pose proposal for model input.

        Args:
            rgb: Color image (H, W, 3)
            depth: Depth map in meters (H, W)
            pose: 4x4 pose matrix (object to camera)
            K: Camera intrinsics
            target_size: Model input size (160x160)
            padding: Padding factor for bounding box

        Returns:
            Dict with cropped/resized rgb, depth, K_crop
        """
        # Project mesh to get 2D bounding box
        vertices_cam = (pose[:3, :3] @ self.mesh.vertices.T).T + pose[:3, 3]
        vertices_proj = (K @ vertices_cam.T).T
        vertices_2d = vertices_proj[:, :2] / vertices_proj[:, 2:3]

        # Get 2D bounding box
        u_min, v_min = vertices_2d.min(axis=0)
        u_max, v_max = vertices_2d.max(axis=0)

        # Apply padding
        cx = (u_min + u_max) / 2
        cy = (v_min + v_max) / 2
        size = max(u_max - u_min, v_max - v_min) * padding
        half_size = size / 2

        # Ensure within image bounds
        H, W = rgb.shape[:2]
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(W, cx + half_size))
        y2 = int(min(H, cy + half_size))

        # Crop
        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]

        # Resize to target size
        scale = target_size / max(rgb_crop.shape[:2])
        new_H = int(rgb_crop.shape[0] * scale)
        new_W = int(rgb_crop.shape[1] * scale)

        rgb_resized = cv2.resize(rgb_crop, (new_W, new_H))
        depth_resized = cv2.resize(depth_crop, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        # Pad to square
        pad_h = target_size - new_H
        pad_w = target_size - new_W
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        rgb_final = np.pad(rgb_resized,
                          ((pad_top, pad_h - pad_top),
                           (pad_left, pad_w - pad_left),
                           (0, 0)), mode='constant')
        depth_final = np.pad(depth_resized,
                           ((pad_top, pad_h - pad_top),
                            (pad_left, pad_w - pad_left)),
                           mode='constant')

        # Adjust intrinsics
        K_crop = K.copy()
        K_crop[0, 2] -= x1
        K_crop[1, 2] -= y1
        K_crop[:2] *= scale
        K_crop[0, 2] += pad_left
        K_crop[1, 2] += pad_top

        return {
            'rgb': rgb_final,
            'depth': depth_final,
            'K': K_crop,
            'bbox': [x1, y1, x2-x1, y2-y1]
        }

    def visualize_model_inputs(self, rgb: np.ndarray, depth: np.ndarray,
                             poses: np.ndarray, K: np.ndarray, n_show: int = 6):
        """Visualize what the model actually sees - both real and rendered patches"""
        fig, axes = plt.subplots(5, n_show, figsize=(18, 15))

        for i in range(min(n_show, len(poses))):
            # Render the pose
            rendered_rgb, rendered_depth = self.render_pose(poses[i], K,
                                                           rgb.shape[0], rgb.shape[1])

            # Get cropped patches for both real and rendered
            real_patch = self.crop_and_resize_proposal(rgb, depth, poses[i], K)
            rendered_patch = self.crop_and_resize_proposal(rendered_rgb, rendered_depth,
                                                         poses[i], K)

            # Original view with bbox
            ax = axes[0, i]
            ax.imshow(rgb)
            x, y, w, h = real_patch['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f'Proposal {i}')
            ax.axis('off')

            # Real RGB patch
            ax = axes[1, i]
            ax.imshow(real_patch['rgb'])
            ax.set_title('Real RGB')
            ax.axis('off')

            # Real Depth patch
            ax = axes[2, i]
            ax.imshow(real_patch['depth'], cmap='jet', vmin=0,
                     vmax=real_patch['depth'][real_patch['depth']>0].max() if real_patch['depth'].max() > 0 else 1)
            ax.set_title('Real Depth')
            ax.axis('off')

            # Rendered RGB patch
            ax = axes[3, i]
            ax.imshow(rendered_patch['rgb'])
            ax.set_title('Rendered RGB')
            ax.axis('off')

            # Rendered Depth patch
            ax = axes[4, i]
            ax.imshow(rendered_patch['depth'], cmap='jet', vmin=0,
                     vmax=rendered_patch['depth'].max() if rendered_patch['depth'].max() > 0 else 1)
            ax.set_title('Rendered Depth')
            ax.axis('off')

        plt.suptitle('Model Inputs: Real vs Rendered (160x160)', fontsize=14)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    import sys
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."

    # Load data
    loader = LMODataLoader(base_path)
    scenes = loader.get_available_scenes()

    if scenes:
        # Load object and scene
        object_id = 1
        mesh = loader.load_object_model(object_id)
        scene_data = loader.load_scene_data(scenes[0])

        # Initialize generator
        generator = PoseProposalGenerator(mesh)
        print(f"Mesh diameter: {generator.diameter:.1f} mm")
        print(f"Rotation grid size: {len(generator.rotation_grid)} rotations")

        # Generate proposals
        if scene_data['masks'][0] is not None:
            mask = scene_data['masks'][0]
            depth = scene_data['depth']

            # Generate pose proposals
            print("\nGenerating pose proposals...")
            poses = generator.generate_poses(depth, mask, loader.K, n_proposals=256)
            print(f"Generated {len(poses)} pose proposals")

            # Analyze translations
            translations = poses[:, :3, 3]
            print(f"\nTranslation statistics (mm):")
            print(f"  X: mean={translations[:, 0].mean():.1f}, std={translations[:, 0].std():.1f}")
            print(f"  Y: mean={translations[:, 1].mean():.1f}, std={translations[:, 1].std():.1f}")
            print(f"  Z: mean={translations[:, 2].mean():.1f}, std={translations[:, 2].std():.1f}")

            # Compare with ground truth
            gt_pose = scene_data['poses'][0]
            gt_trans = gt_pose[:3, 3]
            init_trans = generator.estimate_translation(depth, mask, loader.K)
            print(f"\nGround truth translation: {gt_trans}")
            print(f"Estimated translation: {init_trans}")
            print(f"Translation error: {np.linalg.norm(gt_trans - init_trans):.1f} mm")

            # Visualize proposals
            generator.visualize_proposals(scene_data['rgb'], depth, poses, loader.K)

            # Visualize model input patches
            print("\nVisualizing model input patches (160x160)...")
            generator.visualize_model_inputs(scene_data['rgb'], depth, poses[:6], loader.K)
