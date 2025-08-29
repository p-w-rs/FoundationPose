import torch
import kornia
import trimesh
import cupy as cp
import numpy as np
import nvdiffrast.torch as dr
from contextlib import contextmanager
from functools import wraps

from utils.model import OptimizedModelTRT
from utils.pose import generate_uniform_rotations, guess_object_translation, poses_to_transforms, egocentric_delta_pose_to_pose
from utils.filters import erode_depth, bilateral_filter_depth
from utils.xyz import depth_to_xyz
from utils.mesh import compute_mesh_diameter, make_mesh_tensors, compute_crop_window
from utils.render import nvdiffrast_render, transform_pts, prewarm_caches
from utils.gpu import cupy_to_torch, torch_to_cupy

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

class PoseState:
    """Helper class to manage pose state and conversions efficiently."""

    def __init__(self, poses_cupy):
        self._poses_cupy = poses_cupy
        self._poses_torch = None
        self._is_torch_valid = False

    @property
    def cupy(self):
        """Get poses as CuPy array."""
        return self._poses_cupy

    @property
    def torch(self):
        """Get poses as PyTorch tensor, converting if necessary."""
        if not self._is_torch_valid:
            self._poses_torch = cupy_to_torch(self._poses_cupy)
            self._is_torch_valid = True
        return self._poses_torch

    def update_cupy(self, poses_cupy):
        """Update with new CuPy array."""
        self._poses_cupy = poses_cupy
        self._is_torch_valid = False

    def update_torch(self, poses_torch):
        """Update with new PyTorch tensor."""
        self._poses_torch = poses_torch
        self._poses_cupy = torch_to_cupy(poses_torch)
        self._is_torch_valid = True

    def invalidate_torch(self):
        """Mark torch version as invalid after CuPy update."""
        self._is_torch_valid = False

class FoundationPose:
    def __init__(self, K, mesh_file):
        self.glctx = dr.RasterizeCudaContext()

        self.K = K
        self.K_tensor = torch.as_tensor(K, device='cuda', dtype=torch.float32)

        # Load and process mesh
        self.mesh = trimesh.load(mesh_file)
        self.process_mesh()

        # Pre-warm render caches
        prewarm_caches(K, H=480, W=640, batch_sizes=[252, 1])  # Include batch size 1 for tracking

        # Generate rotations
        self.rotations = generate_uniform_rotations()
        self.pose_last = None

        # Constants
        self.rot_normalizer = 0.3490658503988659
        self.threshold_z = 0.001
        self.threshold_xyz = 2.0

        # Initialize models
        self._init_models()

        # Pre-allocate frequently used tensors
        self._preallocate_tensors()

        # Cache for batch creation
        self._batch_cache = {}

    def _init_models(self):
        """Initialize and warm up TensorRT models."""
        self.refine_model = OptimizedModelTRT(
            model_name="refine_model",
            output_shapes=[
                lambda b: (b, 3),
                lambda b: (b, 3)
            ]
        )

        self.score_model = OptimizedModelTRT(
            model_name="score_model",
            output_shapes=[
                lambda b: (1, b)
            ]
        )

        # Minimal warmup
        self._warmup_models()

    def _warmup_models(self):
        """Minimal model warmup to initialize CUDA kernels."""
        # Single small batch warmup
        input1 = cp.ones((1, 160, 160, 6), dtype=cp.float32)
        input2 = cp.ones((1, 160, 160, 6), dtype=cp.float32)

        self.refine_model.exec_optimized(input1, input2)
        self.score_model.exec_optimized(input1, input2)

        del input1, input2

    def _preallocate_tensors(self):
        """Pre-allocate frequently used tensors."""
        # Standard batch sizes
        self.batch_sizes = {252, 1}

        # Pre-allocate for each batch size
        self.preallocated = {}
        for B in self.batch_sizes:
            self.preallocated[B] = {
                'mesh_radii': torch.full((B, 1, 1, 1), self.diameter / 2,
                                        dtype=torch.float32, device='cuda'),
                'translation_shape': (B, 1, 1, 3)
            }

        # Crop window tensor
        self.bbox2d_crop = torch.tensor([[0, 0, 159, 159]],
                                       device='cuda', dtype=torch.float32).reshape(2, 2)

        # Default mesh radius for non-standard batch sizes
        self.default_mesh_radius = self.diameter / 2

    def process_mesh(self):
        """Process mesh with optimizations."""
        vertices = self.mesh.vertices
        max_xyz = vertices.max(axis=0)
        min_xyz = vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        # Center vertices
        self.mesh.vertices = vertices - self.model_center.reshape(1, 3)

        # Compute properties
        self.diameter = compute_mesh_diameter(self.mesh.vertices)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        # Compute bounds
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

        # Transformation matrices
        self.to_center = cp.eye(4, dtype=cp.float32)
        self.to_center[:3, 3] = -cp.array(self.model_center, dtype=cp.float32)
        self.to_origin = to_origin
        self.extents = extents

    def process_depth(self, depth):
        """Process depth map to XYZ."""
        depth = erode_depth(depth)
        depth = bilateral_filter_depth(depth)
        xyz = depth_to_xyz(depth, self.K)
        return xyz, depth

    @torch.inference_mode()
    def _render_batch(self, poses_torch):
        """Internal rendering function that works with torch tensors."""
        # Compute crop windows
        self.to_crops = compute_crop_window(self.K_tensor, poses_torch, self.diameter)

        # Transform bounding box
        bbox2d_ori = transform_pts(self.bbox2d_crop, self.to_crops.inverse()).reshape(-1, 4)

        # Render
        return nvdiffrast_render(
            self.glctx, self.K, poses_torch, self.mesh_tensors, bbox2d_ori
        )

    @torch.inference_mode()
    def _warp_originals(self, rgb, xyz, batch_size):
        """Warp original RGB and XYZ images to crop windows."""
        # Convert RGB (already normalized from reader)
        if not isinstance(rgb, torch.Tensor):
            rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32, device='cuda')
        else:
            rgb_tensor = rgb

        # Convert XYZ
        if not isinstance(xyz, torch.Tensor):
            xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32, device='cuda')
        else:
            xyz_tensor = xyz

        # Prepare for warping
        rgb_expanded = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        xyz_expanded = xyz_tensor.permute(2, 0, 1).unsqueeze(0)

        # Expand if batch size > 1
        if batch_size > 1:
            rgb_expanded = rgb_expanded.expand(batch_size, -1, -1, -1)
            xyz_expanded = xyz_expanded.expand(batch_size, -1, -1, -1)

        # Warp
        og_rgb = kornia.geometry.transform.warp_perspective(
            rgb_expanded, self.to_crops,
            dsize=(160, 160), mode='bilinear', align_corners=False
        )

        og_xyz = kornia.geometry.transform.warp_perspective(
            xyz_expanded, self.to_crops,
            dsize=(160, 160), mode='nearest', align_corners=False
        )

        return og_rgb.permute(0, 2, 3, 1), og_xyz.permute(0, 2, 3, 1)

    @torch.inference_mode()
    def _process_xyz_batch(self, xyz_tensor, poses_torch, mesh_radii):
        """Process XYZ coordinates for batch creation."""
        B = xyz_tensor.shape[0]
        translation = poses_torch[:, :3, 3].reshape(B, 1, 1, 3)

        # Normalize XYZ
        xyz_normalized = (xyz_tensor - translation) / mesh_radii

        # Create invalid mask
        invalid = (xyz_tensor[:, :, :, 2] < self.threshold_z).unsqueeze(-1)
        invalid = invalid | (xyz_normalized.abs() >= self.threshold_xyz)

        # Apply mask
        xyz_normalized[invalid.expand(-1, -1, -1, 3)] = 0

        return xyz_normalized

    @torch.inference_mode()
    def make_batch_torch(self, rgb, xyz, poses_torch):
        """Create batch using torch tensors throughout - cleaner version."""
        B = poses_torch.shape[0]

        # Get pre-allocated tensors if available
        if B in self.preallocated:
            mesh_radii = self.preallocated[B]['mesh_radii']
        else:
            mesh_radii = self.default_mesh_radius

        # Render
        r_rgb, r_xyz = self._render_batch(poses_torch)

        # Warp originals
        og_rgb, og_xyz = self._warp_originals(rgb, xyz, B)

        # Process XYZ coordinates
        r_xyz = self._process_xyz_batch(r_xyz, poses_torch, mesh_radii)
        og_xyz = self._process_xyz_batch(og_xyz, poses_torch, mesh_radii)

        # Concatenate
        A = torch.cat([r_rgb, r_xyz], dim=3).contiguous()
        B = torch.cat([og_rgb, og_xyz], dim=3).contiguous()

        return A, B

    def make_batch(self, rgb, xyz):
        """Main batch creation function - handles conversions cleanly."""
        # Ensure poses is in torch format
        if isinstance(self.poses, cp.ndarray):
            poses_torch = cupy_to_torch(self.poses)
        else:
            poses_torch = self.poses

        # Create batch in torch
        A, B = self.make_batch_torch(rgb, xyz, poses_torch)

        # Convert to CuPy for model
        return torch_to_cupy(A), torch_to_cupy(B)

    def refine(self, rgb, xyz, iterations):
        """Optimized refinement loop with cleaner conversions."""
        # Use PoseState for cleaner state management
        if not isinstance(self.poses, cp.ndarray):
            self.poses = torch_to_cupy(self.poses)

        pose_state = PoseState(self.poses)

        for _ in range(iterations):
            # Create batch (handles conversions internally)
            A, B = self.make_batch_torch(rgb, xyz, pose_state.torch)

            # Convert to CuPy and run model
            A_cp = torch_to_cupy(A)
            B_cp = torch_to_cupy(B)

            outputs = self.refine_model.exec_optimized(A_cp, B_cp)

            # Compute deltas
            trans_delta = outputs[0] * (self.diameter / 2)
            rot_delta = cp.tanh(outputs[1]) * self.rot_normalizer

            # Update poses in CuPy
            new_poses = egocentric_delta_pose_to_pose(pose_state.cupy, trans_delta, rot_delta)
            pose_state.update_cupy(new_poses)

        # Store final poses
        self.poses = pose_state.cupy

    def score(self, rgb, xyz):
        """Optimized scoring with cleaner conversions."""
        # Create batch
        if isinstance(self.poses, cp.ndarray):
            poses_torch = cupy_to_torch(self.poses)
        else:
            poses_torch = self.poses

        A, B = self.make_batch_torch(rgb, xyz, poses_torch)

        # Convert to CuPy and score
        A_cp = torch_to_cupy(A)
        B_cp = torch_to_cupy(B)

        output = self.score_model.exec_optimized(A_cp, B_cp)
        return output[0][0, :]

    def register(self, rgb, depth, mask, iterations=5):
        """Register object in first frame."""
        # Process depth
        xyz, depth = self.process_depth(depth)

        # Estimate initial translation
        translation = guess_object_translation(depth, mask, cp.array(self.K))

        # Generate initial poses
        self.poses = poses_to_transforms(self.rotations, translation)

        # Refine
        self.refine(rgb, xyz, iterations)

        # Score and sort
        scores = self.score(rgb, xyz)
        ids = cp.flip(cp.argsort(scores))

        # Keep best pose
        self.poses = self.poses[ids]
        self.pose_last = self.poses[0:1]
        self.best_id = ids[0]

        return cp.asnumpy(self.pose_last).reshape(4, 4)

    def track_one(self, rgb, depth, iterations=2):
        """Track object in subsequent frames."""
        # Process depth
        xyz, depth = self.process_depth(depth)

        # Start from last pose
        self.poses = self.pose_last

        # Refine
        self.refine(rgb, xyz, iterations)

        # Update last pose
        self.pose_last = self.poses

        # Return transformed pose
        return cp.asnumpy(self.pose_last[0] @ self.to_center).reshape(4, 4)

    def clear_caches(self):
        """Clear internal caches to free memory."""
        self._batch_cache.clear()
        cp.get_default_memory_pool().free_all_blocks()
