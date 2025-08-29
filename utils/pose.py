# utils/pose.py
import cupy as cp
from functools import lru_cache

# Pre-allocated constants for better performance
EYE_3 = cp.eye(3, dtype=cp.float32)
EYE_4 = cp.eye(4, dtype=cp.float32)
ZERO_3x3 = cp.zeros((3, 3), dtype=cp.float32)

# Pre-computed constants for rotation generation
PHI = float((1 + cp.sqrt(5)) / 2)  # Golden ratio
GOLDEN_ANGLE = cp.pi * (3 - cp.sqrt(5))

# Cache for frequently used rotation matrices
_rotation_cache = {}
_inplane_rotation_cache = {}

def poses_to_transforms(rotations: cp.ndarray, translation: cp.ndarray) -> cp.ndarray:
    """
    Convert rotation matrices and translation to 4x4 transformation matrices.

    Optimized version with better memory patterns.
    """
    n = len(rotations)
    # More efficient: allocate once and fill
    transforms = cp.empty((n, 4, 4), dtype=cp.float32)
    transforms[:] = EYE_4  # Broadcast assignment
    transforms[:, :3, :3] = rotations

    if translation.ndim == 1:
        transforms[:, :3, 3] = translation
    else:
        transforms[:, :3, 3] = translation

    return transforms

def guess_object_translation(depth: cp.ndarray, mask: cp.ndarray, K: cp.ndarray):
    """Optimized translation estimation - matches original implementation."""
    vs, us = cp.where(mask > 0)

    # Convert to Python floats for center calculation (required by original code)
    uc = float((us.min() + us.max()) / 2.0)
    vc = float((vs.min() + vs.max()) / 2.0)

    valid = mask & (depth >= 0.001)
    zc = cp.median(depth[valid])

    center = (cp.linalg.inv(K) @ cp.asarray([uc, vc, 1]).reshape(3, 1)) * zc

    return center.reshape(3)

@lru_cache(maxsize=10)
def _get_inplane_rotations(inplane_steps: int) -> cp.ndarray:
    """Cache in-plane rotation matrices for common step counts."""
    angles = cp.linspace(0, 2*cp.pi, inplane_steps + 1, dtype=cp.float32)[:-1]

    # Vectorized computation of all rotation matrices
    cos_angles = cp.cos(angles)
    sin_angles = cp.sin(angles)

    R_inplane = cp.zeros((inplane_steps, 3, 3), dtype=cp.float32)
    R_inplane[:, 0, 0] = cos_angles
    R_inplane[:, 0, 1] = -sin_angles
    R_inplane[:, 1, 0] = sin_angles
    R_inplane[:, 1, 1] = cos_angles
    R_inplane[:, 2, 2] = 1.0

    return R_inplane

def generate_uniform_rotations(n_views: int = 40, inplane_steps: int = 6) -> cp.ndarray:
    """Optimized rotation generation with caching."""
    cache_key = (n_views, inplane_steps)
    if cache_key in _rotation_cache:
        return _rotation_cache[cache_key].copy()

    # Generate base rotations
    base_rotations = sample_rotations_icosphere(n_views)

    # Get cached in-plane rotations
    R_inplane_all = _get_inplane_rotations(inplane_steps)

    # Vectorized computation of all rotations
    n_total = n_views * inplane_steps
    rotations = cp.empty((n_total, 3, 3), dtype=cp.float32)

    # Batch matrix multiplication for all combinations
    for i in range(n_views):
        start_idx = i * inplane_steps
        end_idx = start_idx + inplane_steps
        # Broadcast base rotation and multiply with all in-plane rotations
        rotations[start_idx:end_idx] = base_rotations[i] @ R_inplane_all

    # Cache for future use
    _rotation_cache[cache_key] = rotations.copy()

    return rotations

def sample_rotations_icosphere(n_views: int) -> cp.ndarray:
    """Optimized icosphere rotation sampling."""
    vertices = generate_icosphere_vertices(n_views)
    n = len(vertices)

    # Vectorized computation of rotation matrices
    positions = vertices
    z_axes = -positions / cp.linalg.norm(positions, axis=1, keepdims=True)

    # Vectorized up vector selection
    z_component_abs = cp.abs(z_axes[:, 2])
    up_vectors = cp.where((z_component_abs > 0.9).reshape(-1, 1),
                          cp.array([0, 1, 0], dtype=cp.float32),
                          cp.array([0, 0, 1], dtype=cp.float32))

    # Batch cross product and normalization
    x_axes = cp.cross(up_vectors, z_axes)
    x_axes /= cp.linalg.norm(x_axes, axis=1, keepdims=True)

    y_axes = cp.cross(z_axes, x_axes)
    y_axes /= cp.linalg.norm(y_axes, axis=1, keepdims=True)

    # Efficient matrix assembly
    rotations = cp.stack([x_axes, y_axes, z_axes], axis=-1)

    return rotations

def generate_icosphere_vertices(n_target: int) -> cp.ndarray:
    """Optimized icosphere vertex generation."""
    # Initial icosahedron vertices (pre-computed)
    vertices = cp.array([
        [-1,  PHI,  0], [ 1,  PHI,  0], [-1, -PHI,  0], [ 1, -PHI,  0],
        [ 0, -1,  PHI], [ 0,  1,  PHI], [ 0, -1, -PHI], [ 0,  1, -PHI],
        [ PHI,  0, -1], [ PHI,  0,  1], [-PHI,  0, -1], [-PHI,  0,  1]
    ], dtype=cp.float32)

    # Normalize in-place
    vertices /= cp.linalg.norm(vertices, axis=1, keepdims=True)

    if len(vertices) < n_target:
        vertices = add_uniform_points(vertices, n_target)

    return vertices[:n_target]

def add_uniform_points(vertices: cp.ndarray, n_target: int) -> cp.ndarray:
    """Optimized uniform point addition."""
    n_current = len(vertices)
    n_needed = n_target - n_current

    if n_needed <= 0:
        return vertices

    # Vectorized fibonacci spiral generation
    i = cp.arange(1, n_needed + 1, dtype=cp.float32)
    y = 1 - (i / (n_needed + 1)) * 2
    radius = cp.sqrt(1 - y * y)
    theta = GOLDEN_ANGLE * i

    # Direct assignment is more efficient
    new_points = cp.stack([
        cp.cos(theta) * radius,
        cp.sin(theta) * radius,
        y
    ], axis=-1)

    return cp.vstack([vertices, new_points])

def so3_exp_map(log_rot: cp.ndarray, eps: float = 1e-8) -> cp.ndarray:
    N = log_rot.shape[0]

    # Compute angle (magnitude) - avoid repeated computation
    theta = cp.linalg.norm(log_rot, axis=1, keepdims=True)
    theta_safe = theta + eps  # Avoid division by zero

    # Normalize to get axis
    axis = log_rot / theta_safe

    # Compute trig functions once
    cos_theta = cp.cos(theta)
    sin_theta = cp.sin(theta)
    one_minus_cos = 1 - cos_theta

    # Optimized skew-symmetric matrix computation using broadcasting
    # Extract axis components for vectorized operations
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]

    # Pre-compute products that appear multiple times
    ax_ay = ax * ay * one_minus_cos.squeeze()
    ax_az = ax * az * one_minus_cos.squeeze()
    ay_az = ay * az * one_minus_cos.squeeze()

    sin_theta_sq = sin_theta.squeeze()
    cos_theta_sq = cos_theta.squeeze()
    one_minus_cos_sq = one_minus_cos.squeeze()

    # Direct construction of rotation matrix using Rodrigues' formula
    # This avoids creating intermediate K and K^2 matrices
    R = cp.empty((N, 3, 3), dtype=cp.float32)

    # Row 0
    R[:, 0, 0] = cos_theta_sq + ax * ax * one_minus_cos_sq
    R[:, 0, 1] = ax_ay - az * sin_theta_sq
    R[:, 0, 2] = ax_az + ay * sin_theta_sq

    # Row 1
    R[:, 1, 0] = ax_ay + az * sin_theta_sq
    R[:, 1, 1] = cos_theta_sq + ay * ay * one_minus_cos_sq
    R[:, 1, 2] = ay_az - ax * sin_theta_sq

    # Row 2
    R[:, 2, 0] = ax_az - ay * sin_theta_sq
    R[:, 2, 1] = ay_az + ax * sin_theta_sq
    R[:, 2, 2] = cos_theta_sq + az * az * one_minus_cos_sq

    return R

def egocentric_delta_pose_to_pose(poses: cp.ndarray, trans_delta: cp.ndarray,
                                  rot_delta: cp.ndarray) -> cp.ndarray:
    N = poses.shape[0]

    # Compute rotation update matrices efficiently
    rot_mat_delta = so3_exp_map(rot_delta)

    # Transpose for PyTorch3D compatibility - use swapaxes which is faster
    rot_mat_delta = cp.swapaxes(rot_mat_delta, 1, 2)

    # Reuse input poses array if possible to save memory
    # Create output array
    new_poses = cp.empty_like(poses)

    # Update rotation: new_rot = delta @ old_rot
    # Use batched matrix multiplication
    new_poses[:, :3, :3] = cp.matmul(rot_mat_delta, poses[:, :3, :3])

    # Update translation: new_trans = old_trans + delta
    new_poses[:, :3, 3] = poses[:, :3, 3] + trans_delta

    # Copy bottom row (always [0, 0, 0, 1])
    new_poses[:, 3, :] = poses[:, 3, :]

    return new_poses

# Optional optimized batch version for multiple objects
def egocentric_delta_pose_to_pose_batch(poses_batch: cp.ndarray,
                                        trans_delta_batch: cp.ndarray,
                                        rot_delta_batch: cp.ndarray) -> cp.ndarray:
    """
    Batch version for processing multiple objects at once.
    Shape: poses_batch (M, N, 4, 4), trans_delta_batch (M, N, 3), rot_delta_batch (M, N, 3)
    Where M is number of objects, N is number of poses per object.
    """
    M, N = poses_batch.shape[:2]

    # Flatten batch dimension for processing
    poses_flat = poses_batch.reshape(-1, 4, 4)
    trans_delta_flat = trans_delta_batch.reshape(-1, 3)
    rot_delta_flat = rot_delta_batch.reshape(-1, 3)

    # Process all at once
    new_poses_flat = egocentric_delta_pose_to_pose(poses_flat, trans_delta_flat, rot_delta_flat)

    # Reshape back
    return new_poses_flat.reshape(M, N, 4, 4)

# Cache management functions
def clear_rotation_cache():
    """Clear the rotation cache to free memory."""
    global _rotation_cache, _inplane_rotation_cache
    _rotation_cache.clear()
    _inplane_rotation_cache.clear()
    _get_inplane_rotations.cache_clear()

def get_cache_info():
    """Get cache statistics."""
    return {
        'rotation_cache_size': len(_rotation_cache),
        'inplane_rotations': _get_inplane_rotations.cache_info()
    }
