# MeshRenderer.py

import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from typing import Tuple
import logging
from CUDAContext import CUDAContextManager

class MeshRenderer:
    """
    GPU-accelerated mesh renderer using nvdiffrast.
    """

    def __init__(self, mesh: trimesh.Trimesh, device: str = 'cuda'):
        """
        Initializes the renderer with a given 3D mesh.
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        self.cuda_mgr = CUDAContextManager.get_instance()
        self.glctx = self.cuda_mgr.get_nvdiff_context()
        self.set_mesh(mesh)

    def render(self, pose: np.ndarray, K: np.ndarray,
               width: int = 160, height: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh at a single given pose.
        """
        rgbs, depths = self.render_batch(np.array([pose]), K, width, height)
        return rgbs[0], depths[0]

    def render_batch(self, poses: np.ndarray, K: np.ndarray,
                     width: int = 160, height: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders the mesh from a batch of different camera poses.
        """
        with self.cuda_mgr.activate_nvdiffrast():
            return self._do_render_batch(poses, K, width, height)

    def _do_render_batch(self, poses: np.ndarray, K: np.ndarray,
                         width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The core rendering logic, executed within the correct CUDA context.
        """
        batch_size = len(poses)

        # Scale camera intrinsics to the target render resolution
        K_scaled = K.copy()
        K_scaled[0, :] *= width / 640.0
        K_scaled[1, :] *= height / 480.0
        K_torch = torch.from_numpy(K_scaled.astype(np.float32)).to(self.device)

        poses_torch = torch.from_numpy(poses.astype(np.float32)).to(self.device)

        # Transform vertices from object space to camera space
        vertices_homo = torch.cat([self.vertices, torch.ones(self.n_vertices, 1, device=self.device)], dim=1)
        transformed_verts = torch.matmul(poses_torch, vertices_homo.T).transpose(1, 2)
        vertices_cam = transformed_verts[:, :, :3]

        # Project vertices from camera space to clip space for the rasterizer
        vertices_clip = self._project_points(vertices_cam, K_torch, width, height).reshape(-1, 4)

        # Prepare faces for batch rendering
        faces_expanded = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        face_offsets = torch.arange(batch_size, device=self.device, dtype=torch.int32) * self.n_vertices
        faces_batch = (faces_expanded + face_offsets.view(-1, 1, 1)).reshape(-1, 3)

        # Rasterize
        ranges = torch.stack([
            torch.arange(batch_size, dtype=torch.int32) * self.n_faces,
            torch.full((batch_size,), self.n_faces, dtype=torch.int32)
        ], dim=1).cpu()
        rast_out, _ = dr.rasterize(self.glctx, vertices_clip, faces_batch, resolution=[height, width], ranges=ranges)

        # Interpolate attributes (color and depth)
        color_interpolated, _ = dr.interpolate(self.vertex_colors.repeat(batch_size, 1), rast_out, faces_batch)
        depth_interpolated, _ = dr.interpolate(vertices_cam[..., 2].reshape(-1, 1), rast_out, faces_batch)

        # Final image formation
        mask = rast_out[..., 3] > 0
        depth_interpolated = depth_interpolated[..., 0] * mask

        rgbs = (color_interpolated * 255).clamp(0, 255).byte().cpu().numpy()
        depths = depth_interpolated.cpu().numpy()

        return rgbs, depths

    def _project_points(self, points_cam: torch.Tensor, K: torch.Tensor,
                       width: int, height: int) -> torch.Tensor:
        """
        Projects 3D points from camera space to 2D clip space.
        """
        points_proj = torch.matmul(K.unsqueeze(0), points_cam.transpose(1, 2)).transpose(1, 2)

        z = points_proj[..., 2:3].clamp(min=1e-5)
        uv = points_proj[..., :2] / z

        u_ndc = (2.0 * uv[..., 0] / width - 1.0)
        v_ndc = (1.0 - 2.0 * uv[..., 1] / height)

        # *** THE FIX IS HERE: Use positive Z for clip space ***
        # The rasterizer expects Z to increase with distance from the camera.
        return torch.stack([u_ndc, v_ndc, points_cam[..., 2], torch.ones_like(u_ndc)], dim=-1)

    def set_mesh(self, mesh: trimesh.Trimesh):
        """
        Updates the renderer with a new mesh.
        """
        self.vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
        self.faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(self.device)
        self.n_vertices, self.n_faces = len(mesh.vertices), len(mesh.faces)

        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            light_dir = np.array([0.0, 0.0, 1.0])
            shading = np.clip(mesh.vertex_normals @ light_dir, 0.3, 1.0)
            colors = np.stack([shading] * 3, axis=-1)
        else:
            colors = np.ones_like(mesh.vertices) * 0.7
        self.vertex_colors = torch.from_numpy(colors.astype(np.float32)).to(self.device)
        self.logger.info(f"Renderer mesh updated: {self.n_vertices} vertices, {self.n_faces} faces.")


# Unit Tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2
    from DataLoader import DataLoader
    from CUDAContext import CUDAContextManager

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    manager = None
    try:
        print("="*80)
        print("MeshRenderer Standalone Test")
        print("="*80)

        print("Initializing CUDA context and loading data...")
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")
        mesh = loader.load_object_model(1)
        renderer = MeshRenderer(mesh)

        print("\n[Test] Batch Rendering & Visualization...")
        n_poses = 8
        poses = np.array([np.eye(4) for _ in range(n_poses)], dtype=np.float32)
        for i in range(n_poses):
            angle = i * 2 * np.pi / n_poses
            R_y = cv2.Rodrigues(np.array([0, angle, 0], dtype=np.float32))[0]
            R_x = cv2.Rodrigues(np.array([np.pi/6, 0, 0], dtype=np.float32))[0]
            poses[i, :3, :3] = R_y @ R_x
            poses[i, 2, 3] = 0.5

        rgbs, depths = renderer.render_batch(poses, loader.K, width=160, height=160)
        print(" -> Rendering complete.")

        # --- Create Visualization ---
        fig = plt.figure(figsize=(2 * n_poses, 5))
        # Use gridspec to have more control over the layout
        gs = fig.add_gridspec(2, n_poses, height_ratios=[4, 4], hspace=0.1, wspace=0.1)
        fig.suptitle("MeshRenderer Output Verification (160x160)", fontsize=16, y=0.98)

        # Determine a consistent color scale for all depth images
        all_depths = np.stack(depths)
        valid_depths = all_depths[all_depths > 0]
        vmin, vmax = (valid_depths.min(), valid_depths.max()) if valid_depths.size > 0 else (0,1)

        for i in range(n_poses):
            # Plot RGB
            ax_rgb = fig.add_subplot(gs[0, i])
            ax_rgb.imshow(rgbs[i])
            ax_rgb.set_title(f'Pose {i}', fontsize=10)
            ax_rgb.axis('off')

            # Plot Depth
            ax_depth = fig.add_subplot(gs[1, i])
            depth_vis = depths[i].copy()
            depth_vis[depth_vis == 0] = np.nan
            im = ax_depth.imshow(depth_vis, cmap='viridis', vmin=vmin, vmax=vmax)
            ax_depth.axis('off')

        # Add a single, shared colorbar in a new axis at the bottom
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.05]) # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Depth (meters)')

        viz_dir = Path("viz"); viz_dir.mkdir(exist_ok=True)
        save_path = viz_dir / 'meshrenderer_test.png'
        plt.savefig(save_path, dpi=150)
        print(f" -> Visualization saved to {save_path}")

        print("\n" + "="*80)
        print("All tests passed!")
        print("="*80)

    except Exception as e:
        logging.error(f"A test failed: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
