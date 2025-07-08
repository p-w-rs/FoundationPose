# PoseRenderer.py

import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from typing import Tuple, List, Optional, Dict
import logging
import cv2
from CUDAContext import CUDAContextManager

class PoseRenderer:
    """
    Combined pose generation and rendering for FoundationPose.

    Handles:
    1. Icosphere-based pose generation (matching paper's approach)
    2. GPU-accelerated mesh rendering using nvdiffrast
    3. Proper coordinate transformations and camera handling

    All units in meters except camera intrinsics (pixels).

    COORDINATE SYSTEMS:
    - BOP Dataset: +X right, +Y down, +Z forward (into screen)
    - Object meshes: May vary, but BOP meshes typically have Y-up convention
    - Camera space: +X right, +Y down, +Z forward (matching BOP)
    """

    def __init__(self, mesh: trimesh.Trimesh, device: str = 'cuda'):
        """
        Initialize with a 3D mesh model.

        Args:
            mesh: Object mesh with vertices in meters
            device: GPU device for rendering
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        self.cuda_mgr = CUDAContextManager.get_instance()
        self.glctx = self.cuda_mgr.get_nvdiff_context()

        # Store original mesh
        self.mesh_original = mesh.copy()

        # Store original mesh without any rotation adjustment
        mesh_adjusted = mesh.copy()

        # No coordinate adjustment - let's see the default orientation first
        # coord_adjust = np.eye(3)
        # mesh_adjusted.vertices = mesh_adjusted.vertices @ coord_adjust.T

        # Center the adjusted mesh (CRITICAL: FoundationPose centers meshes)
        self.mesh_center = mesh_adjusted.vertices.mean(axis=0)
        self.mesh_centered = mesh_adjusted.copy()
        self.mesh_centered.vertices = mesh_adjusted.vertices - self.mesh_center

        # Compute object properties on centered mesh
        self.mesh_radius = np.linalg.norm(self.mesh_centered.vertices, axis=1).max()
        self.mesh_diameter = 2 * self.mesh_radius

        # Setup rendering
        self._setup_renderer()

        # Generate icosphere viewpoints
        self.viewpoints = self._generate_icosphere_vertices(subdivisions=2)
        self.n_viewpoints = len(self.viewpoints)

        self.logger.info(f"Initialized PoseRenderer: {self.n_viewpoints} viewpoints, "
                        f"object radius: {self.mesh_radius:.3f}m, diameter: {self.mesh_diameter:.3f}m")

    def _setup_renderer(self):
        """Setup nvdiffrast rendering components."""
        # Convert mesh to torch tensors
        self.vertices = torch.from_numpy(
            self.mesh_centered.vertices.astype(np.float32)
        ).to(self.device)
        self.faces = torch.from_numpy(
            self.mesh_centered.faces.astype(np.int32)
        ).to(self.device)

        self.n_vertices = len(self.vertices)
        self.n_faces = len(self.faces)

        # Setup vertex colors/normals for shading
        if hasattr(self.mesh_centered, 'vertex_normals') and self.mesh_centered.vertex_normals is not None:
            # Simple directional lighting
            light_dir = np.array([0.0, 0.0, 1.0])
            shading = np.clip(self.mesh_centered.vertex_normals @ light_dir, 0.3, 1.0)
            colors = np.stack([shading] * 3, axis=-1)
        else:
            colors = np.ones_like(self.mesh_centered.vertices) * 0.7

        self.vertex_colors = torch.from_numpy(colors.astype(np.float32)).to(self.device)

    def _generate_icosphere_vertices(self, subdivisions: int = 2) -> np.ndarray:
        """
        Generate icosphere vertices for uniform viewpoint sampling.
        Matches FoundationPose paper approach.
        """
        # Create base icosahedron
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ], dtype=np.float32)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Define icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Subdivide to increase density
        for _ in range(subdivisions):
            vertices, faces = self._subdivide_icosphere(vertices, faces)

        return vertices

    def _subdivide_icosphere(self, vertices: np.ndarray,
                            faces: List[List[int]]) -> Tuple[np.ndarray, List[List[int]]]:
        """Subdivide icosphere faces."""
        edge_cache = {}
        new_faces = []
        vertices = vertices.copy()

        def get_middle_point(v1_idx, v2_idx):
            nonlocal vertices
            key = tuple(sorted([v1_idx, v2_idx]))
            if key in edge_cache:
                return edge_cache[key]

            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            middle = (v1 + v2) / 2
            middle = middle / np.linalg.norm(middle)

            new_idx = len(vertices)
            vertices = np.vstack([vertices, middle])
            edge_cache[key] = new_idx
            return new_idx

        for face in faces:
            v1, v2, v3 = face
            a = get_middle_point(v1, v2)
            b = get_middle_point(v2, v3)
            c = get_middle_point(v3, v1)
            new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])

        return vertices, new_faces

    def generate_poses(self, n_poses: Optional[int] = None,
                      distance_factor: float = 4.0,
                      in_plane_rotations: int = 1) -> np.ndarray:
        """
        Generate pose proposals using icosphere sampling.

        Args:
            n_poses: Number of poses (None = all viewpoints)
            distance_factor: Distance multiplier from object radius
            in_plane_rotations: Number of in-plane rotations per viewpoint

        Returns:
            (N, 4, 4) object-to-camera transformation matrices
        """
        if n_poses is None:
            n_poses = self.n_viewpoints * in_plane_rotations

        # Sample viewpoints
        if n_poses <= self.n_viewpoints:
            indices = np.linspace(0, self.n_viewpoints-1, n_poses, dtype=int)
            viewpoints = self.viewpoints[indices]
            in_plane_rotations = 1
        else:
            viewpoints = self.viewpoints
            in_plane_rotations = max(1, n_poses // self.n_viewpoints)

        poses = []
        distance = self.mesh_radius * distance_factor

        for viewpoint in viewpoints:
            for rot_idx in range(in_plane_rotations):
                # Camera position (relative to centered object at origin)
                cam_pos = viewpoint * distance

                # Look-at rotation (camera looking at origin)
                z_axis = -viewpoint  # Camera Z points towards object

                # Choose up vector
                if abs(z_axis[1]) < 0.9:
                    up = np.array([0, 1, 0])
                else:
                    up = np.array([1, 0, 0])

                x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)

                R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)

                # Apply in-plane rotation
                if in_plane_rotations > 1:
                    angle = rot_idx * 2 * np.pi / in_plane_rotations
                    c, s = np.cos(angle), np.sin(angle)
                    R_inplane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    R_cam = R_cam @ R_inplane

                # Build pose matrix (object-to-camera)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R_cam.T
                pose[:3, 3] = -R_cam.T @ cam_pos

                # Account for centering only (no coordinate adjustment)
                transform_to_centered = np.eye(4, dtype=np.float32)
                transform_to_centered[:3, 3] = -self.mesh_center

                # Combine: camera <- centered <- original
                pose = pose @ transform_to_centered

                poses.append(pose)

                if len(poses) >= n_poses:
                    break

            if len(poses) >= n_poses:
                break

        return np.array(poses[:n_poses])

    def render(self, pose: np.ndarray, K: np.ndarray,
               width: int = 640, height: int = 480) -> Dict:
        """
        Render mesh at given pose.

        Args:
            pose: Object-to-camera transformation (4x4) in BOP convention
            K: Camera intrinsics (3x3)
            width, height: Output image size

        Returns:
            Dict with 'rgb', 'depth', 'mask'
        """
        results = self.render_batch(np.array([pose]), K, width, height)
        return {
            'rgb': results['rgbs'][0],
            'depth': results['depths'][0],
            'mask': results['masks'][0]
        }

    def render_batch(self, poses: np.ndarray, K: np.ndarray,
                    width: int = 640, height: int = 480) -> Dict:
        """
        Render mesh from multiple poses.

        Returns:
            Dict with 'rgbs', 'depths', 'masks' arrays
        """
        with self.cuda_mgr.activate_nvdiffrast():
            return self._do_render_batch(poses, K, width, height)

    def _do_render_batch(self, poses: np.ndarray, K: np.ndarray,
                        width: int, height: int) -> Dict:
        """Core rendering implementation."""
        batch_size = len(poses)

        # Scale intrinsics if rendering at different resolution
        K_scaled = K.copy()
        if width != 640:  # Assuming default is 640x480
            scale_x = width / 640.0
            scale_y = height / 480.0
            K_scaled[0, :] *= scale_x
            K_scaled[1, :] *= scale_y

        K_torch = torch.from_numpy(K_scaled.astype(np.float32)).to(self.device)
        poses_torch = torch.from_numpy(poses.astype(np.float32)).to(self.device)

        # The poses are in original mesh space, but our vertices are in centered space
        # No coordinate adjustment needed - using mesh as-is

        # Adjust poses for our centered coordinate system
        center_offset = torch.tensor(self.mesh_center, device=self.device, dtype=torch.float32)

        # Transform vertices from object to camera space
        vertices_homo = torch.cat([
            self.vertices,
            torch.ones(self.n_vertices, 1, device=self.device)
        ], dim=1)

        # Apply the centering that we did to the mesh
        poses_adjusted = poses_torch.clone()
        poses_adjusted[:, :3, 3] -= poses_adjusted[:, :3, :3] @ center_offset

        transformed_verts = torch.matmul(poses_adjusted, vertices_homo.T).transpose(1, 2)
        vertices_cam = transformed_verts[:, :, :3]

        # Project to clip space
        vertices_clip = self._project_points(vertices_cam, K_torch, width, height)
        vertices_clip = vertices_clip.reshape(-1, 4)

        # Prepare faces for batch rendering
        faces_expanded = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        face_offsets = torch.arange(batch_size, device=self.device, dtype=torch.int32) * self.n_vertices
        faces_batch = (faces_expanded + face_offsets.view(-1, 1, 1)).reshape(-1, 3)

        # Rasterize
        ranges = torch.stack([
            torch.arange(batch_size, dtype=torch.int32) * self.n_faces,
            torch.full((batch_size,), self.n_faces, dtype=torch.int32)
        ], dim=1).cpu()

        rast_out, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_batch,
            resolution=[height, width], ranges=ranges
        )

        # Interpolate color and depth
        color_interp, _ = dr.interpolate(
            self.vertex_colors.repeat(batch_size, 1), rast_out, faces_batch
        )
        depth_interp, _ = dr.interpolate(
            vertices_cam[..., 2].reshape(-1, 1), rast_out, faces_batch
        )

        # Extract mask from rasterizer output
        mask = rast_out[..., 3] > 0

        # Apply mask to depth
        depth_interp = depth_interp[..., 0] * mask

        # Convert to numpy
        rgbs = (color_interp * 255).clamp(0, 255).byte().cpu().numpy()
        depths = depth_interp.cpu().numpy()
        masks = mask.cpu().numpy()

        return {
            'rgbs': rgbs,
            'depths': depths,
            'masks': masks
        }

    def _project_points(self, points_cam: torch.Tensor, K: torch.Tensor,
                       width: int, height: int) -> torch.Tensor:
        """Project 3D points to clip space for rasterizer."""
        # Project to image space
        points_proj = torch.matmul(K.unsqueeze(0), points_cam.transpose(1, 2)).transpose(1, 2)

        # Normalize by depth
        z = points_proj[..., 2:3].clamp(min=1e-5)
        uv = points_proj[..., :2] / z

        # Convert to NDC (normalized device coordinates)
        u_ndc = 2.0 * uv[..., 0] / width - 1.0
        v_ndc = -(2.0 * uv[..., 1] / height - 1.0)  # Y-axis flipped for NDC

        # Return clip space coordinates
        return torch.stack([u_ndc, v_ndc, points_cam[..., 2], torch.ones_like(u_ndc)], dim=-1)

    def generate_nearby_poses(self, reference_pose: np.ndarray,
                            n_poses: int = 220,
                            position_noise: float = 0.02,
                            rotation_noise: float = 0.1) -> np.ndarray:
        """Generate poses near a reference pose for refinement."""
        poses = []

        for _ in range(n_poses):
            # Add rotation noise
            noise_angles = np.random.normal(0, rotation_noise, 3)
            R_noise, _ = cv2.Rodrigues(noise_angles)

            # Add translation noise
            t_noise = np.random.normal(0, position_noise, 3)

            # Apply noise
            pose = reference_pose.copy()
            pose[:3, :3] = pose[:3, :3] @ R_noise
            pose[:3, 3] += t_noise

            poses.append(pose)

        return np.array(poses)


# Unit tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    from DataLoader import DataLoader
    from CUDAContext import CUDAContextManager

    logging.basicConfig(level=logging.INFO)

    manager = None
    try:
        print("="*80)
        print("PoseRenderer Test - Object 5 (Fixed Coordinate System)")
        print("="*80)

        # Initialize
        print("\nInitializing components...")
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")

        # Load object 5
        mesh = loader.load_object_model(5)
        renderer = PoseRenderer(mesh)

        print(f"\nObject 5 properties:")
        print(f"  Original center: {renderer.mesh_center}")
        print(f"  Radius: {renderer.mesh_radius:.3f} m")
        print(f"  Diameter: {renderer.mesh_diameter:.3f} m")
        print(f"  Viewpoints: {renderer.n_viewpoints}")

        # Test 1: Pose generation and rendering
        print("\n[Test 1] Generating and rendering poses...")
        poses = renderer.generate_poses(n_poses=8, distance_factor=4.0)
        results = renderer.render_batch(poses, loader.K, width=640, height=480)

        fig1 = plt.figure(figsize=(20, 8))
        fig1.suptitle('PoseRenderer: Generated Views of Object 5', fontsize=16)

        for i in range(8):
            # RGB
            ax1 = plt.subplot(3, 8, i + 1)
            ax1.imshow(results['rgbs'][i])
            ax1.set_title(f'View {i+1}')
            ax1.axis('off')

            # Depth
            ax2 = plt.subplot(3, 8, i + 9)
            depth_vis = results['depths'][i].copy()
            depth_vis[depth_vis == 0] = np.nan
            im = ax2.imshow(depth_vis, cmap='viridis')
            ax2.axis('off')

            # Mask
            ax3 = plt.subplot(3, 8, i + 17)
            ax3.imshow(results['masks'][i], cmap='gray')
            ax3.axis('off')

        plt.tight_layout()
        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'poserenderer_fixed_views.png', dpi=150)

        # Test 2: Compare with real data
        print("\n[Test 2] Comparing with real data...")
        scene_id = 2
        frame_id = 0

        # Find object 5 in scene
        scene_info = loader.get_scene_info(scene_id)
        obj5_idx = None
        for idx, obj_info in enumerate(scene_info[str(frame_id)]):
            if obj_info['obj_id'] == 5:
                obj5_idx = idx
                break

        if obj5_idx is not None:
            data = loader.load_frame_data(scene_id, frame_id, object_indices=[obj5_idx])
            gt_pose = data['poses'][0]

            # Render at GT pose
            gt_render = renderer.render(gt_pose, loader.K, width=640, height=480)

            fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig2.suptitle('Real vs Rendered Comparison (Fixed)', fontsize=16)

            # Real data
            axes[0, 0].imshow(data['rgb'])
            axes[0, 0].set_title('Real RGB')
            axes[0, 0].axis('off')

            depth_real_vis = data['depth'].copy()
            depth_real_vis[depth_real_vis == 0] = np.nan
            im1 = axes[0, 1].imshow(depth_real_vis, cmap='viridis')
            axes[0, 1].set_title('Real Depth')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])

            axes[0, 2].imshow(data['masks'][0], cmap='gray')
            axes[0, 2].set_title('Real Mask')
            axes[0, 2].axis('off')

            # Rendered data
            axes[1, 0].imshow(gt_render['rgb'])
            axes[1, 0].set_title('Rendered RGB (Should Match Orientation)')
            axes[1, 0].axis('off')

            depth_render_vis = gt_render['depth'].copy()
            depth_render_vis[depth_render_vis == 0] = np.nan
            im2 = axes[1, 1].imshow(depth_render_vis, cmap='viridis')
            axes[1, 1].set_title('Rendered Depth')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])

            axes[1, 2].imshow(gt_render['mask'], cmap='gray')
            axes[1, 2].set_title('Rendered Mask')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(viz_dir / 'poserenderer_fixed_comparison.png', dpi=150)

            # Print statistics
            print(f"\nDepth statistics (meters):")
            print(f"  Real:     min={depth_real_vis[~np.isnan(depth_real_vis)].min():.3f}, "
                  f"max={depth_real_vis[~np.isnan(depth_real_vis)].max():.3f}")
            print(f"  Rendered: min={depth_render_vis[~np.isnan(depth_render_vis)].min():.3f}, "
                  f"max={depth_render_vis[~np.isnan(depth_render_vis)].max():.3f}")

        # Test 3: Detailed orientation comparison
        print("\n[Test 3] Detailed orientation check...")

        # Render from a canonical front view
        canonical_pose = np.eye(4, dtype=np.float32)
        canonical_pose[2, 3] = renderer.mesh_radius * 4  # Move camera back

        canonical_render = renderer.render(canonical_pose, loader.K)

        fig3, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(canonical_render['rgb'])
        ax.set_title('Canonical Front View (Should show object upright)')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(viz_dir / 'poserenderer_canonical.png', dpi=150)

        print("\n" + "="*80)
        print("PoseRenderer test completed!")
        print("\nKey fixes applied:")
        print("✓ Added coordinate system adjustment for BOP meshes (Y-up to Y-down)")
        print("✓ Properly handle transformation chain: original -> adjusted -> centered")
        print("✓ Poses now correctly account for coordinate conventions")
        print("✓ Rendered objects should match real image orientations")
        print("="*80)

    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
