# utils/mesh.py
import trimesh
import torch
import cupy as cp
import numpy as np
import torch.nn.functional as F
import nvdiffrast.torch as dr
from functools import lru_cache

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

# Pre-allocated constants
EYE_3 = torch.eye(3, dtype=torch.float32, device='cuda')
DEFAULT_VERTEX_COLOR = torch.tensor([0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
                                    dtype=torch.float32, device='cuda')  # 128/255

# Cache for crop window offsets
CROP_OFFSETS_CACHE = {}

def compute_mesh_diameter(model_pts, n_sample=10000):
    """Optimized mesh diameter computation using PyTorch on GPU."""
    n_pts = len(model_pts)
    sample_size = min(n_sample, n_pts)

    # Convert to torch tensor if not already
    if not isinstance(model_pts, torch.Tensor):
        pts_gpu = torch.as_tensor(model_pts, device='cuda', dtype=torch.float32)
    else:
        pts_gpu = model_pts

    # Use GPU for sampling if we need to subsample
    if sample_size < n_pts:
        # Generate random indices on GPU
        ids = torch.randperm(n_pts, device='cuda')[:sample_size]
        pts = pts_gpu[ids]
    else:
        pts = pts_gpu

    # Compute pairwise distances on GPU using broadcasting
    # More memory efficient for large samples: compute in chunks if needed
    if sample_size > 5000:
        # Chunk computation to avoid memory issues
        chunk_size = 2500
        max_dist = 0.0
        for i in range(0, sample_size, chunk_size):
            end_i = min(i + chunk_size, sample_size)
            chunk = pts[i:end_i]
            # Compute distances for this chunk against all points
            dists = torch.cdist(chunk, pts, p=2)
            chunk_max = dists.max().item()
            if chunk_max > max_dist:
                max_dist = chunk_max
        diameter = max_dist
    else:
        # Small enough to compute all at once
        dists = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0), p=2).squeeze()
        diameter = dists.max().item()

    return diameter

@torch.inference_mode()
@torch.no_grad()
def make_mesh_tensors(mesh):
    """Optimized mesh tensor creation with better memory management."""
    mesh_tensors = {}

    # Handle texture or vertex colors
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        # Texture path - optimize image conversion
        img = np.array(mesh.visual.material.image.convert('RGB'), dtype=np.float32)
        # Normalize and convert to tensor in one operation
        mesh_tensors['tex'] = torch.from_numpy(img[..., :3] / 255.0).to(device='cuda', dtype=torch.float32).unsqueeze(0)

        # Use torch.from_numpy for efficiency when possible
        mesh_tensors['uv_idx'] = torch.from_numpy(mesh.faces.astype(np.int32)).to(device='cuda')

        # UV coordinates
        uv = torch.from_numpy(mesh.visual.uv.astype(np.float32)).to(device='cuda')
        uv[:, 1] = 1.0 - uv[:, 1]  # Flip V coordinate
        mesh_tensors['uv'] = uv
    else:
        # Vertex colors path
        if mesh.visual.vertex_colors is None:
            # Use pre-allocated default color
            n_vertices = len(mesh.vertices)
            mesh_tensors['vertex_color'] = DEFAULT_VERTEX_COLOR.unsqueeze(0).expand(n_vertices, -1)
        else:
            # Convert vertex colors efficiently
            vertex_colors = mesh.visual.vertex_colors[..., :3].astype(np.float32) / 255.0
            mesh_tensors['vertex_color'] = torch.from_numpy(vertex_colors).to(device='cuda', dtype=torch.float32)

    # Convert mesh geometry - use from_numpy when possible for efficiency
    mesh_tensors.update({
        'pos': torch.from_numpy(mesh.vertices.astype(np.float32)).to(device='cuda'),
        'faces': torch.from_numpy(mesh.faces.astype(np.int32)).to(device='cuda'),
        'vnormals': torch.from_numpy(mesh.vertex_normals.astype(np.float32)).to(device='cuda'),
    })

    return mesh_tensors

@lru_cache(maxsize=32)
def get_crop_offsets(radius):
    """Cache crop offsets for different radii."""
    return torch.tensor([0, 0, 0,
                         radius, 0, 0,
                         -radius, 0, 0,
                         0, radius, 0,
                         0, -radius, 0], device='cuda', dtype=torch.float32).reshape(-1, 3)

@torch.inference_mode()
@torch.no_grad()
def compute_crop_window(K, poses, mesh_diameter, crop_ratio=1.2, out_size=(160, 160)):
    """Optimized crop window computation with better tensor operations."""
    B = len(poses)
    radius = mesh_diameter * crop_ratio / 2

    # Get cached offsets
    offsets = get_crop_offsets(radius)

    # Ensure K is on GPU
    if not isinstance(K, torch.Tensor):
        K_gpu = torch.as_tensor(K, device='cuda', dtype=torch.float32)
    else:
        K_gpu = K.to(device='cuda', dtype=torch.float32) if K.device != torch.device('cuda') else K

    # Extract translation from poses and compute offset points
    translations = poses[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    pts = translations + offsets.unsqueeze(0)  # (B, 5, 3)

    # Project points efficiently using batched matrix multiplication
    pts_flat = pts.reshape(-1, 3, 1)  # (B*5, 3, 1)
    projected = torch.bmm(K_gpu.unsqueeze(0).expand(B * 5, -1, -1), pts_flat)  # (B*5, 3, 1)
    projected = projected.squeeze(-1)  # (B*5, 3)

    # Compute UV coordinates
    uvs = projected[:, :2] / projected[:, 2:3]  # (B*5, 2)
    uvs = uvs.reshape(B, -1, 2)  # (B, 5, 2)

    # Extract center and compute radius
    center = uvs[:, 0]  # (B, 2)
    # Compute maximum distance from center more efficiently
    diffs = uvs - center.unsqueeze(1)  # (B, 5, 2)
    radius_per_batch = torch.abs(diffs).reshape(B, -1).max(dim=-1)[0]  # (B,)

    # Compute bounding box coordinates
    left = center[:, 0] - radius_per_batch
    right = center[:, 0] + radius_per_batch
    top = center[:, 1] - radius_per_batch
    bottom = center[:, 1] + radius_per_batch

    # Round coordinates
    left = left.round()
    right = right.round()
    top = top.round()
    bottom = bottom.round()

    # Compute transformation matrices efficiently
    # Pre-allocate result tensor
    tf = torch.empty((B, 3, 3), device='cuda', dtype=torch.float32)

    # Compute scale factors
    width_scale = out_size[0] / (right - left)
    height_scale = out_size[1] / (bottom - top)

    # Fill transformation matrix using vectorized operations
    tf[:, 0, 0] = width_scale
    tf[:, 0, 1] = 0
    tf[:, 0, 2] = -left * width_scale
    tf[:, 1, 0] = 0
    tf[:, 1, 1] = height_scale
    tf[:, 1, 2] = -top * height_scale
    tf[:, 2, 0] = 0
    tf[:, 2, 1] = 0
    tf[:, 2, 2] = 1

    return tf

# Optional: Optimized batch version for multiple objects
@torch.inference_mode()
@torch.no_grad()
def compute_crop_window_batch(K_batch, poses_batch, mesh_diameters, crop_ratio=1.2, out_size=(160, 160)):
    """
    Batch version of compute_crop_window for processing multiple objects at once.

    Args:
        K_batch: Either single K matrix or batch of K matrices (N, 3, 3)
        poses_batch: Batch of poses (N, M, 4, 4) where N is number of objects, M is poses per object
        mesh_diameters: List or tensor of mesh diameters (N,)
        crop_ratio: Crop ratio (default 1.2)
        out_size: Output size tuple (default (160, 160))

    Returns:
        Batch of transformation matrices (N, M, 3, 3)
    """
    if len(poses_batch.shape) == 3:
        # Single object, multiple poses
        return compute_crop_window(K_batch, poses_batch, mesh_diameters, crop_ratio, out_size)

    # Multiple objects
    results = []
    for i, (poses, diameter) in enumerate(zip(poses_batch, mesh_diameters)):
        K = K_batch[i] if len(K_batch.shape) == 3 else K_batch
        tf = compute_crop_window(K, poses, diameter, crop_ratio, out_size)
        results.append(tf)

    return torch.stack(results)

# Cache clearing functions
def clear_crop_offsets_cache():
    """Clear the crop offsets cache."""
    get_crop_offsets.cache_clear()

def get_cache_info():
    """Get cache statistics."""
    return {
        'crop_offsets': get_crop_offsets.cache_info()
    }
