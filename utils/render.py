# utils/render.py
import torch
import numpy as np
import torch.nn.functional as F
import nvdiffrast.torch as dr
from functools import lru_cache

# Enable TensorFloat32 for better performance on Ampere/newer GPUs
torch.set_float32_matmul_precision('high')

# Pre-convert to torch tensor to avoid repeated CPU->GPU transfers
GLCAM_IN_CVCAM = torch.tensor([[1,0,0,0],
                                [0,-1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]], dtype=torch.float32, device='cuda')

# Pre-allocated constants
EYE_4 = torch.eye(4, dtype=torch.float32, device='cuda')
ZERO_LIGHT = torch.zeros(1, 1, 3, dtype=torch.float32, device='cuda')

# Cache for projection matrices to avoid recomputation
_projection_cache = {}
# Cache for resolution arrays
_resolution_cache = {}
# Cache for viewport transform matrices by batch size
_viewport_cache = {}
# Cache for light position tensors
_light_cache = {}

@lru_cache(maxsize=32)
def _get_resolution_array(output_size):
    """Cache resolution arrays."""
    if isinstance(output_size, tuple):
        return np.asarray(output_size, dtype=np.int32)
    return output_size

def to_homo_torch(pts):
    """Optimized homogeneous coordinate conversion."""
    # Use F.pad which is more efficient than cat for adding a single dimension
    return F.pad(pts, (0, 1), mode='constant', value=1.0)

def transform_pts(pts, tf):
    """Optimized point transformation."""
    # Keep original implementation for compatibility with nvdiffrast
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

def transform_dirs(dirs, tf):
    """Optimized direction transformation."""
    # Keep original implementation for compatibility
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]

def get_projection_matrix(K, height, width, znear=0.001, zfar=100, window_coords='y_down'):
    """Cached projection matrix computation - only compute once per unique K, H, W combination."""
    # Create cache key from K matrix values and dimensions
    if isinstance(K, np.ndarray):
        cache_key = (tuple(K.flatten().tolist()), height, width, znear, zfar, window_coords)
    else:
        cache_key = (tuple(K.flatten().tolist()) if hasattr(K, 'flatten') else tuple(K),
                     height, width, znear, zfar, window_coords)

    if cache_key not in _projection_cache:
        w, h = float(width), float(height)
        depth = zfar - znear
        q = -(zfar + znear) / depth
        qn = -2 * (zfar * znear) / depth

        # Pre-compute all divisions
        fx_factor = 2 * K[0, 0] / w
        fy_factor = 2 * K[1, 1] / h
        skew_factor = -2 * K[0, 1] / w
        cx_factor = (-2 * K[0, 2] + w) / w
        cy_factor = (2 * K[1, 2] - h) / h

        # Directly create as torch tensor to avoid conversion later
        if window_coords == 'y_down':
            proj = torch.tensor([
                [fx_factor, skew_factor, cx_factor, 0],
                [0, fy_factor, cy_factor, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0]
            ], dtype=torch.float32, device='cuda')
        else:  # y_up
            proj = torch.tensor([
                [fx_factor, skew_factor, cx_factor, 0],
                [0, -fy_factor, (-2 * K[1, 2] + h) / h, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0]
            ], dtype=torch.float32, device='cuda')

        _projection_cache[cache_key] = proj

    return _projection_cache[cache_key]

def get_viewport_transform(bbox2d, H, W, batch_size):
    """Optimized viewport transform with caching for common batch sizes."""
    # For common batch sizes, try to reuse cached base matrix
    cache_key = (batch_size, H, W)

    if cache_key not in _viewport_cache:
        # Create base matrix for this batch size
        base_tf = EYE_4.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        _viewport_cache[cache_key] = base_tf

    # Clone cached base matrix
    tf = _viewport_cache[cache_key].clone()

    # Compute scales and offsets with minimal operations
    l = bbox2d[:, 0]
    r = bbox2d[:, 2]
    t = H - bbox2d[:, 1]
    b = H - bbox2d[:, 3]

    # Vectorized computation of all parameters
    rl_diff = r - l
    tb_diff = t - b

    # Direct assignment with pre-computed values
    tf[:, 0, 0] = W / rl_diff
    tf[:, 1, 1] = H / tb_diff
    tf[:, 3, 0] = (W - r - l) / rl_diff
    tf[:, 3, 1] = (H - t - b) / tb_diff

    return tf

@torch.inference_mode()
@torch.no_grad()
def nvdiffrast_render(glctx, K, ob_in_cams, mesh_tensors, bbox2d, H=480, W=640,
                     output_size=(160, 160), w_ambient=0.8, w_diffuse=0.5):
    """Highly optimized nvdiffrast rendering function."""

    # Ensure ob_in_cams is a torch tensor (handle CuPy arrays efficiently)
    if not isinstance(ob_in_cams, torch.Tensor):
        ob_in_cams = torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float32)

    # Cache frequently used values
    pos = mesh_tensors['pos']
    vnormals = mesh_tensors['vnormals']
    pos_idx = mesh_tensors['faces']
    has_tex = 'tex' in mesh_tensors
    batch_size = len(ob_in_cams)

    # Pre-compute combined transformation matrix
    projection_mat = get_projection_matrix(K, height=H, width=W, znear=0.001, zfar=100)
    # Fuse operations: projection @ glcam @ ob_in_cams
    combined_mtx = projection_mat.unsqueeze(0) @ (GLCAM_IN_CVCAM.unsqueeze(0) @ ob_in_cams)

    # Transform points to camera space
    pts_cam = transform_pts(pos, ob_in_cams)

    # Homogeneous conversion and clip space transformation
    pos_homo = to_homo_torch(pos)
    # Keep original method for correctness, but with combined matrix
    pos_clip = (combined_mtx[:, None] @ pos_homo[None, ..., None])[..., 0]

    # Get optimized viewport transformation
    tf = get_viewport_transform(bbox2d, H, W, batch_size)

    # Apply viewport transformation
    pos_clip = pos_clip @ tf

    # Get cached resolution array
    resolution = _get_resolution_array(output_size)

    # Rasterization - pts_cam from transform_pts is always contiguous
    # pos_clip might not be after matrix multiplication
    if not pos_clip.is_contiguous():
        pos_clip = pos_clip.contiguous()
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution)

    # Interpolate XYZ - pts_cam is already contiguous from transform_pts
    xyz, _ = dr.interpolate(pts_cam, rast_out, pos_idx)

    # Color interpolation
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
        rgb = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
    else:
        rgb, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)

    # Optimized lighting computation with caching
    vnormals_cam = transform_dirs(vnormals, ob_in_cams)
    light_dir = ZERO_LIGHT - pts_cam

    # vnormals_cam from transform_dirs is already contiguous
    # Fused normalize and dot product
    vnormals_norm = F.normalize(vnormals_cam, dim=-1)
    light_dir_norm = F.normalize(light_dir, dim=-1)

    # Compute diffuse with in-place operations
    diffuse_intensity = torch.sum(vnormals_norm * light_dir_norm, dim=-1, keepdim=True).clamp_(min=0.0, max=1.0)

    # diffuse_intensity is contiguous after sum operation
    diffuse_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)

    # Optimized final color computation
    # Fuse ambient and diffuse in single operation
    rgb *= (w_ambient + diffuse_map * w_diffuse)
    rgb.clamp_(min=0.0, max=1.0)

    # Apply alpha mask with in-place operation
    alpha = rast_out[..., -1:].clamp_(min=0.0, max=1.0)
    rgb *= alpha

    # Flip Y coordinates (single flip operation for both)
    rgb = torch.flip(rgb, dims=[1])
    xyz = torch.flip(xyz, dims=[1])

    return rgb, xyz

def clear_all_caches():
    """Clear all caches if camera parameters change or to free memory."""
    global _projection_cache, _resolution_cache, _viewport_cache, _light_cache
    _projection_cache.clear()
    _resolution_cache.clear()
    _viewport_cache.clear()
    _light_cache.clear()
    _get_resolution_array.cache_clear()

def clear_projection_cache():
    """Clear the projection matrix cache if camera parameters change."""
    global _projection_cache
    _projection_cache.clear()

# Optional: Pre-warm caches for common configurations
def prewarm_caches(K, H=480, W=640, output_sizes=[(160, 160)], batch_sizes=[252]):
    """Pre-warm caches with common configurations for faster first frame."""
    # Pre-compute projection matrices
    for output_size in output_sizes:
        _ = get_projection_matrix(K, H, W)
        _ = _get_resolution_array(output_size)

    # Pre-allocate viewport transforms for common batch sizes
    for batch_size in batch_sizes:
        cache_key = (batch_size, H, W)
        if cache_key not in _viewport_cache:
            base_tf = EYE_4.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            _viewport_cache[cache_key] = base_tf
