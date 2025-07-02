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

    This renderer is responsible for creating synthetic images from a 3D mesh
    given a set of camera poses. It uses the nvdiffrast library for high-speed
    rasterization on the GPU. This class is designed to integrate into a larger
    pipeline where GPU resources must be shared with other libraries like
    TensorRT.

    Key Features:
    - Renders RGB and depth images from a trimesh object.
    - Uses a shared CUDA context via CUDAContextManager to ensure compatibility.
    - Transforms vertices and projects them to screen space for rendering.
    - Supports batch rendering for efficient processing of many poses at once.
    """

    def __init__(self, mesh: trimesh.Trimesh, device: str = 'cuda'):
        """
        Initializes the renderer with a given 3D mesh.

        Args:
            mesh (trimesh.Trimesh): A trimesh object containing the model's vertices and faces.
                                  Vertices are expected to be in meters.
            device (str): The PyTorch device to use for rendering ('cuda').
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)

        # Obtain the singleton instance of the CUDA context manager
        self.cuda_mgr = CUDAContextManager.get_instance()
        # Get the shared nvdiffrast context from the manager
        self.glctx = self.cuda_mgr.get_nvdiff_context()

        # Load mesh data onto the GPU
        self.set_mesh(mesh)

    def render(self, pose: np.ndarray, K: np.ndarray,
               width: int = 160, height: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh at a single given pose. This is a convenience wrapper
        for render_batch.
        """
        # Add a batch dimension to the single pose and call the batch renderer
        rgbs, depths = self.render_batch(np.array([pose]), K, width, height)
        return rgbs[0], depths[0]

    def render_batch(self, poses: np.ndarray, K: np.ndarray,
                    width: int = 160, height: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders the mesh from a batch of different camera poses.

        This is the primary rendering method, designed for efficiency. It wraps
        the core rendering logic in the nvdiffrast activation context.

        Args:
            poses (np.ndarray): The poses of the object relative to the camera.
                - Shape: (N, 4, 4), where N is the number of poses in the batch.
                - Represents the object-to-camera transformation matrix.
                - The rotation part (3x3) and translation vector (3x1) are in meters.
            K (np.ndarray): The 3x3 camera intrinsic matrix.
                - Defines the camera's focal length and principal point.
                - Assumes a standard 640x480 resolution and is scaled internally.
            width (int): The desired output image width.
            height (int): The desired output image height.

        Returns:
            A tuple containing:
            - rgbs (np.ndarray): A batch of rendered RGB images.
                - Shape: (N, H, W, 3), with H=height, W=width.
                - Values are uint8, from 0 to 255.
            - depths (np.ndarray): A batch of rendered depth maps.
                - Shape: (N, H, W).
                - Values are float32, representing the distance from the camera
                  plane to the object surface in meters.
        """
        # Activate the nvdiffrast context for the duration of the rendering
        with self.cuda_mgr.activate_nvdiffrast():
            return self._do_render_batch(poses, K, width, height)

    def _do_render_batch(self, poses: np.ndarray, K: np.ndarray,
                         width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The core rendering logic, executed within the correct CUDA context.
        This function orchestrates the transformation, projection, and rasterization pipeline.
        """
        batch_size = len(poses)

        # --- Data Transformation: Scale Camera Intrinsics ---
        # The provided intrinsic matrix K is for a reference resolution (e.g., 640x480).
        # We must scale it to match the target rendering resolution (width, height).
        K_scaled = K.copy()
        K_scaled[0, 0] *= width / 640.0   # Scale fx
        K_scaled[0, 2] *= width / 640.0   # Scale cx
        K_scaled[1, 1] *= height / 480.0  # Scale fy
        K_scaled[1, 2] *= height / 480.0  # Scale cy
        K_torch = torch.from_numpy(K_scaled.astype(np.float32)).to(self.device)

        poses_torch = torch.from_numpy(poses.astype(np.float32)).to(self.device)

        # --- Data Transformation: Vertex Transformation ---
        # 1. Convert vertices to homogeneous coordinates by adding a 'w' component of 1.
        #    This allows us to perform rotation and translation with a single matrix multiplication.
        #    Shape: (num_vertices, 3) -> (num_vertices, 4)
        vertices_homo = torch.cat([self.vertices, torch.ones(self.n_vertices, 1, device=self.device)], dim=1)

        # 2. Apply the batch of pose transformations.
        #    Coordinate System: The vertices are transformed from Object Space to Camera Space.
        #    Matrix multiplication: (N, 4, 4) @ (V, 4).T -> (N, 4, V) -> (N, V, 4)
        transformed_verts = torch.matmul(poses_torch, vertices_homo.T).transpose(1, 2)

        # 3. Discard the 'w' component to get 3D points in camera space.
        #    Shape: (N, V, 3)
        vertices_cam = transformed_verts[:, :, :3]

        # --- Data Transformation: Projection ---
        # Project the 3D points from Camera Space to Clip Space. Clip space is a
        # normalized cube where x,y,z are typically in [-1, 1]. This is the format
        # the rasterizer expects.
        vertices_clip = self._project_points(vertices_cam, K_torch, width, height).reshape(-1, 4)

        # --- Rasterization Setup ---
        # The rasterizer needs a flat list of faces with vertex indices adjusted
        # for the batch. Example: For batch item `i`, a vertex index `v` becomes `v + i * num_vertices`.
        faces_expanded = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        face_offsets = torch.arange(batch_size, device=self.device, dtype=torch.int32) * self.n_vertices
        faces_batch = (faces_expanded + face_offsets.view(-1, 1, 1)).reshape(-1, 3)

        # Define ranges for nvdiffrast to know which faces belong to which image in the batch.
        ranges = torch.stack([
            torch.arange(batch_size, dtype=torch.int32) * self.n_faces,
            torch.full((batch_size,), self.n_faces, dtype=torch.int32)
        ], dim=1).cpu()

        # --- GPU Rasterization & Interpolation ---
        # 1. Rasterize the mesh to get pixel-to-face mappings and barycentric coordinates.
        rast_out, _ = dr.rasterize(self.glctx, vertices_clip, faces_batch, resolution=[height, width], ranges=ranges)

        # 2. Interpolate vertex attributes (color and depth) across the mesh surface for each pixel.
        color_interpolated, _ = dr.interpolate(self.vertex_colors.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3), rast_out, faces_batch)
        depth_interpolated, _ = dr.interpolate(vertices_cam[..., 2].reshape(-1, 1), rast_out, faces_batch)

        # --- Final Image Formation ---
        # Create a mask of valid pixels (where the object was rendered).
        mask = rast_out[..., 3] > 0
        depth_interpolated = depth_interpolated[..., 0] * mask

        # Convert final images to numpy arrays for CPU use.
        rgbs = (color_interpolated * 255).clamp(0, 255).byte().cpu().numpy()
        depths = depth_interpolated.cpu().numpy()

        return rgbs, depths

    def _project_points(self, points_cam: torch.Tensor, K: torch.Tensor,
                       width: int, height: int) -> torch.Tensor:
        """
        Projects 3D points from camera space to 2D clip space required by nvdiffrast.

        Args:
            points_cam (torch.Tensor): 3D points in the camera's coordinate system.
                - Shape: (N, V, 3) where N is batch size, V is number of vertices.
            K (torch.Tensor): The scaled 3x3 camera intrinsic matrix.
            width (int), height (int): The dimensions of the output image.

        Returns:
            torch.Tensor: Points in clip space.
                - Shape: (N, V, 4), where the components are (x_ndc, y_ndc, z_cam, w=1).
        """
        # Project 3D points to 2D image plane using the intrinsic matrix K.
        points_proj = torch.matmul(K.unsqueeze(0), points_cam.transpose(1, 2)).transpose(1, 2)

        # Perform perspective divide (normalize u,v by z) to get pixel coordinates.
        z = points_proj[..., 2:3].clamp(min=1e-5)
        uv = points_proj[..., :2] / z

        # Convert pixel coordinates to Normalized Device Coordinates (NDC) [-1, 1].
        # The rasterizer expects this coordinate system.
        u_ndc = (2.0 * uv[..., 0] / width - 1.0)
        v_ndc = (1.0 - 2.0 * uv[..., 1] / height) # Y is flipped in NDC vs. image coordinates.

        # Return points in clip space (x, y, z, w). nvdiffrast uses z from camera space.
        return torch.stack([u_ndc, v_ndc, -points_cam[..., 2], torch.ones_like(u_ndc)], dim=-1)

    def set_mesh(self, mesh: trimesh.Trimesh):
        """
        Updates the renderer with a new mesh, loading its data to the GPU.
        """
        self.vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
        self.faces = torch.from_numpy(mesh.faces.astype(np.int32)).to(self.device)
        self.n_vertices, self.n_faces = len(mesh.vertices), len(mesh.faces)
        self.mesh_center = mesh.vertices.mean(axis=0)

        # Calculate simple shading based on vertex normals facing a camera-aligned light.
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            light_dir = np.array([0.0, 0.0, 1.0])
            shading = np.clip(mesh.vertex_normals @ light_dir, 0.3, 1.0)
            colors = np.stack([shading] * 3, axis=-1)
        else:
            colors = np.ones((len(mesh.vertices), 3)) * 0.7

        self.vertex_colors = torch.from_numpy(colors.astype(np.float32)).to(self.device)
        self.logger.info(f"Renderer mesh updated: {self.n_vertices} vertices, {self.n_faces} faces.")

# Standalone Unit Tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2
    import time
    from DataLoader import DataLoader

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("MeshRenderer Standalone Test")
    print("="*80)

    manager = None
    try:
        print("Initializing CUDA context...")
        manager = CUDAContextManager.get_instance()

        print("Loading test data...")
        loader = DataLoader("./data")
        mesh = loader.load_object_model(1)
        renderer = MeshRenderer(mesh)

        print("\n[Test 1] Rendering a single pose...")
        pose = np.eye(4); pose[2, 3] = 0.5
        renderer.render(pose, loader.K)
        print(" -> Single pose render successful.")

        print("\n[Test 2] Batch Rendering & Visualization...")
        n_poses = 8
        poses = np.array([np.eye(4) for _ in range(n_poses)], dtype=np.float32)
        for i in range(n_poses):
            angle = i * 2 * np.pi / n_poses
            poses[i, :3, :3] = cv2.Rodrigues(np.array([0, angle, 0], dtype=np.float32))[0]
            poses[i, 2, 3] = 0.4

        rgbs, depths = renderer.render_batch(poses, loader.K)

        viz_dir = Path("viz"); viz_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, n_poses, figsize=(2 * n_poses, 4))
        for i in range(n_poses):
            axes[0, i].imshow(rgbs[i]); axes[0, i].axis('off')
            axes[1, i].imshow(depths[i], cmap='viridis'); axes[1, i].axis('off')
        save_path = viz_dir / 'meshrenderer_standalone_test.png'
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
