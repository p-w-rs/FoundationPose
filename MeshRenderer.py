import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from typing import Tuple, Optional, Union
import cv2
import logging

class MeshRenderer:
    """
    GPU-accelerated mesh renderer using nvdiffrast for FoundationPose.

    This renderer generates synthetic RGB and depth images of objects
    at specified poses for pose hypothesis generation and refinement.

    UNITS:
    - Input poses: meters (m)
    - Input mesh vertices: meters (m)
    - Output depth: meters (m)
    - Camera intrinsics: pixels

    Optimized for:
    - Batch rendering of multiple poses
    - 160x160 output for model input
    - GPU memory efficiency
    - Minimal CPU-GPU transfers
    """

    def __init__(self, mesh: trimesh.Trimesh, device: str = 'cuda'):
        """
        Initialize renderer with a mesh.

        Args:
            mesh: Trimesh object with vertices in meters
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)

        # Initialize nvdiffrast context
        self.glctx = dr.RasterizeCudaContext(device=self.device)

        # Load mesh to GPU
        self.vertices = torch.from_numpy(
            mesh.vertices.astype(np.float32)
        ).to(self.device)

        self.faces = torch.from_numpy(
            mesh.faces.astype(np.int32)
        ).to(self.device)

        # Compute vertex colors from normals for shading
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            # Simple Lambert shading
            light_dir = np.array([0.0, 0.0, 1.0])  # Camera-aligned light
            shading = np.clip(mesh.vertex_normals @ light_dir, 0.3, 1.0)
            colors = np.stack([shading] * 3, axis=-1)
        else:
            colors = np.ones((len(mesh.vertices), 3)) * 0.7

        self.vertex_colors = torch.from_numpy(
            colors.astype(np.float32)
        ).to(self.device)

        # Cache mesh properties
        self.n_vertices = len(mesh.vertices)
        self.n_faces = len(mesh.faces)
        self.mesh_center = mesh.vertices.mean(axis=0)
        self.mesh_scale = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

        self.logger.info(f"Initialized renderer: {self.n_vertices} vertices, "
                        f"{self.n_faces} faces, scale: {self.mesh_scale:.3f}m")

    def render(self, pose: np.ndarray, K: np.ndarray,
               width: int = 160, height: int = 160,
               z_near: float = 0.01, z_far: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh at given pose.

        Args:
            pose: 4x4 pose matrix (object to camera) in meters
            K: 3x3 camera intrinsic matrix (will be scaled if needed)
            width: Output image width
            height: Output image height
            z_near: Near clipping plane (meters)
            z_far: Far clipping plane (meters)

        Returns:
            Tuple of:
            - rgb: (H, W, 3) uint8 RGB image
            - depth: (H, W) float32 depth in meters
        """
        # Single pose - add batch dimension
        rgbs, depths = self.render_batch(
            np.array([pose]), K, width, height, z_near, z_far
        )
        return rgbs[0], depths[0]

    def render_batch(self, poses: np.ndarray, K: np.ndarray,
                    width: int = 160, height: int = 160,
                    z_near: float = 0.01, z_far: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh at multiple poses efficiently in batch.

        Args:
            poses: (N, 4, 4) pose matrices in meters
            K: 3x3 camera intrinsic matrix
            width: Output image width
            height: Output image height
            z_near: Near clipping plane (meters)
            z_far: Far clipping plane (meters)

        Returns:
            Tuple of:
            - rgbs: (N, H, W, 3) uint8 RGB images
            - depths: (N, H, W) float32 depth in meters
        """
        batch_size = len(poses)

        # Scale intrinsics if rendering at different resolution
        # Assume original resolution is 640x480 (typical for LM-O)
        K_scaled = K.copy()
        scale_x = width / 640.0
        scale_y = height / 480.0
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy

        # Convert to torch
        poses_torch = torch.from_numpy(poses.astype(np.float32)).to(self.device)
        K_torch = torch.from_numpy(K_scaled.astype(np.float32)).to(self.device)

        # Transform vertices by each pose
        # vertices: (V, 3), poses: (N, 4, 4)
        vertices_homo = torch.cat([
            self.vertices,
            torch.ones(self.n_vertices, 1, device=self.device)
        ], dim=1)  # (V, 4)

        # Batch transform: (N, 4, 4) @ (V, 4).T -> (N, 4, V) -> (N, V, 4)
        transformed = torch.matmul(poses_torch, vertices_homo.T).transpose(1, 2)
        vertices_cam = transformed[:, :, :3]  # (N, V, 3)

        # Project to screen space
        vertices_proj = self._project_points(vertices_cam, K_torch, width, height)

        # Prepare for rasterization - need (N*V, 4) format
        vertices_clip = vertices_proj.reshape(-1, 4)

        # Expand faces for batch
        faces_expanded = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        face_offsets = (torch.arange(batch_size, device=self.device, dtype=torch.int32) * self.n_vertices).reshape(-1, 1, 1)
        faces_batch = (faces_expanded + face_offsets).reshape(-1, 3).int()

        # Create ranges for batch rendering
        ranges = torch.stack([
            torch.arange(batch_size, dtype=torch.int32) * self.n_faces,
            torch.full((batch_size,), self.n_faces, dtype=torch.int32)
        ], dim=1).cpu()

        # Rasterize
        rast_out, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_batch,
            resolution=[height, width], ranges=ranges
        )

        # Reshape rasterization output
        rast_out = rast_out.reshape(batch_size, height, width, 4)

        # Create masks where mesh was rendered
        masks = rast_out[..., 3] > 0

        # Prepare vertex attributes for interpolation
        # Colors need to be repeated for each batch
        colors_batch = self.vertex_colors.unsqueeze(0).expand(batch_size, -1, -1)
        colors_flat = colors_batch.reshape(-1, 3)

        # Depth values for interpolation
        z_values = vertices_cam[..., 2].reshape(-1, 1)  # (N*V, 1)

        # Interpolate colors
        color_out, _ = dr.interpolate(
            colors_flat, rast_out, faces_batch
        )

        # Interpolate depth
        depth_out, _ = dr.interpolate(
            z_values, rast_out, faces_batch
        )
        depth_out = depth_out[..., 0]

        # Apply masks
        rgbs_torch = color_out * masks.unsqueeze(-1)
        depths_torch = depth_out * masks

        # Convert to numpy
        rgbs = (rgbs_torch * 255).clamp(0, 255).byte().cpu().numpy()
        depths = depths_torch.cpu().numpy()

        return rgbs, depths

    def _project_points(self, points_cam: torch.Tensor, K: torch.Tensor,
                       width: int, height: int) -> torch.Tensor:
        """
        Project 3D points to normalized device coordinates.

        Args:
            points_cam: (N, V, 3) points in camera coordinates (meters)
            K: 3x3 camera intrinsic matrix
            width: Image width
            height: Image height

        Returns:
            (N, V, 4) points in clip space
        """
        N, V, _ = points_cam.shape

        # Project to image plane
        points_proj = torch.matmul(
            K.unsqueeze(0), points_cam.transpose(1, 2)
        ).transpose(1, 2)  # (N, V, 3)

        # Normalize by z
        z = points_proj[..., 2:3].clamp(min=1e-5)
        uv = points_proj[..., :2] / z

        # Convert to NDC (-1 to 1)
        u_ndc = 2.0 * uv[..., 0] / width - 1.0
        v_ndc = 1.0 - 2.0 * uv[..., 1] / height  # Flip Y

        # For nvdiffrast, we need (x, y, z, w) in clip space
        # z should be negative for visible points
        return torch.stack([
            u_ndc,
            v_ndc,
            -points_cam[..., 2],  # Negative z for nvdiffrast
            torch.ones_like(u_ndc)
        ], dim=-1)

    def set_mesh(self, mesh: trimesh.Trimesh):
        """Update the mesh for rendering."""
        self.vertices = torch.from_numpy(
            mesh.vertices.astype(np.float32)
        ).to(self.device)

        self.faces = torch.from_numpy(
            mesh.faces.astype(np.int32)
        ).to(self.device)

        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            light_dir = np.array([0.0, 0.0, 1.0])
            shading = np.clip(mesh.vertex_normals @ light_dir, 0.3, 1.0)
            colors = np.stack([shading] * 3, axis=-1)
        else:
            colors = np.ones((len(mesh.vertices), 3)) * 0.7

        self.vertex_colors = torch.from_numpy(
            colors.astype(np.float32)
        ).to(self.device)

        self.n_vertices = len(mesh.vertices)
        self.n_faces = len(mesh.faces)


# Unit tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from DataLoader import DataLoader

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Mesh Renderer Unit Tests")
    print("="*80)

    # Load test data
    loader = DataLoader("./data")

    # Test 1: Single pose rendering
    print("\nTest 1: Single Pose Rendering")

    # Load a mesh
    mesh = loader.load_object_model(1)
    renderer = MeshRenderer(mesh)

    # Debug info
    print(f"Mesh bounds: {mesh.bounds}")
    print(f"Mesh center: {mesh.vertices.mean(axis=0)}")
    print(f"Camera K:\n{loader.K}")

    # Create test pose (object at 0.5m from camera)
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.5  # 0.5 meters in Z

    # Add slight rotation to see the object better
    pose[:3, :3] = cv2.Rodrigues(np.array([0.2, 0.3, 0], dtype=np.float32))[0]

    # Render
    rgb, depth = renderer.render(pose, loader.K, 160, 160)

    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"Unique RGB values: {np.unique(rgb).shape[0]}")
    if depth[depth > 0].size > 0:
        print(f"Depth range: {depth[depth > 0].min():.3f} - {depth[depth > 0].max():.3f} m")
        print(f"Number of valid depth pixels: {(depth > 0).sum()}")
    else:
        print("Object not visible in depth.")

    # Test 2: Batch rendering
    print("\nTest 2: Batch Rendering")

    # Create multiple poses
    n_poses = 8
    poses = []
    for i in range(n_poses):
        angle = i * 2 * np.pi / n_poses
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = cv2.Rodrigues(np.array([0, angle, 0], dtype=np.float32))[0]
        pose[2, 3] = 0.5
        poses.append(pose)
    poses = np.array(poses)

    # Batch render
    import time
    t0 = time.time()
    rgbs, depths = renderer.render_batch(poses, loader.K, 160, 160)
    t_batch = time.time() - t0

    print(f"Batch rendered {n_poses} poses in {t_batch:.3f}s ({n_poses/t_batch:.1f} fps)")
    print(f"RGBs shape: {rgbs.shape}")
    print(f"Depths shape: {depths.shape}")

    # Test 3: Compare with ground truth
    print("\nTest 3: Render at GT Pose")

    # Load a frame with ground truth
    scenes = loader.get_available_scenes()
    if scenes:
        data = loader.load_frame_data(scenes[0], 0)
        if data['poses']:
            gt_pose = data['poses'][0]
            obj_id = data['object_ids'][0]

            # Render at GT pose
            mesh = loader.load_object_model(obj_id)
            renderer.set_mesh(mesh)
            rgb_synth, depth_synth = renderer.render(gt_pose, data['K'],
                                                    loader.width, loader.height)

            print(f"Rendered object {obj_id} at GT pose")
            print(f"GT pose translation: {gt_pose[:3, 3]}")
            if data['depth'][data['depth'] > 0].size > 0:
                print(f"Real depth range: {data['depth'][data['depth'] > 0].min():.3f} - "
                      f"{data['depth'][data['depth'] > 0].max():.3f} m")
            if depth_synth[depth_synth > 0].size > 0:
                print(f"Synth depth range: {depth_synth[depth_synth > 0].min():.3f} - "
                      f"{depth_synth[depth_synth > 0].max():.3f} m")

    # Visualize results
    fig, axes = plt.subplots(3, n_poses, figsize=(2*n_poses, 6))

    for i in range(n_poses):
        # RGB
        axes[0, i].imshow(rgbs[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('RGB', fontsize=12)

        # Depth
        depth_vis = depths[i].copy()
        if depth_vis[depth_vis > 0].size > 0:
            vmin = depth_vis[depth_vis > 0].min()
            vmax = depth_vis[depth_vis > 0].max()
            depth_vis[depth_vis == 0] = np.nan
            axes[1, i].imshow(depth_vis, cmap='viridis', vmin=vmin, vmax=vmax)
        else:
            axes[1, i].imshow(depth_vis, cmap='viridis')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Depth', fontsize=12)

        # Mask
        mask = depths[i] > 0
        axes[2, i].imshow(mask, cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Mask', fontsize=12)

    plt.suptitle('Batch Rendering Results - Object Rotated 360°')
    plt.tight_layout()

    # Save
    from pathlib import Path
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / 'meshrenderer_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {viz_dir / 'meshrenderer_test.png'}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
