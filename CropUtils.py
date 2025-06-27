import numpy as np
import cv2
import torch

def compute_crop_window_tf_batch(pts, H, W, poses, K, crop_ratio=1.2, out_size=(160, 160),
                                 method='box_3d', mesh_diameter=None):
    """
    Compute homography transformations for cropping around projected 3D bounding box.

    This follows the original FoundationPose implementation which projects a 3D bounding
    box around the object and computes a square crop that encloses it.

    Args:
        pts: Mesh vertices (N, 3) - not used in box_3d method
        H, W: Original image dimensions
        poses: Object poses (B, 4, 4) as numpy arrays
        K: Camera intrinsics (3, 3)
        crop_ratio: Padding ratio for crop
        out_size: Output size (height, width)
        method: 'box_3d' (default)
        mesh_diameter: Diameter of mesh in mm

    Returns:
        tf_to_crops: (B, 3, 3) homography matrices from original to crop
    """
    def compute_tf_batch(left, right, top, bottom):
        """Compute transformation matrices for batch"""
        B = len(left)
        left = np.round(left)
        right = np.round(right)
        top = np.round(top)
        bottom = np.round(bottom)

        # Translation to move crop origin to (0,0)
        tf = np.tile(np.eye(3), (B, 1, 1))
        tf[:, 0, 2] = -left
        tf[:, 1, 2] = -top

        # Scale to resize crop to output size
        scale_tf = np.tile(np.eye(3), (B, 1, 1))
        scale_tf[:, 0, 0] = out_size[1] / (right - left)
        scale_tf[:, 1, 1] = out_size[0] / (bottom - top)

        # Combined transformation
        tf = scale_tf @ tf
        return tf

    B = len(poses)
    poses = np.array(poses)  # Ensure numpy array

    if method == 'box_3d':
        # Create 3D bounding box around object center
        radius = mesh_diameter * crop_ratio / 2

        # Offsets for 3D bounding box corners
        offsets = np.array([
            [0, 0, 0],           # Center
            [radius, 0, 0],      # +X
            [-radius, 0, 0],     # -X
            [0, radius, 0],      # +Y
            [0, -radius, 0],     # -Y
        ])

        # Apply to all poses
        centers = poses[:, :3, 3]  # (B, 3)
        pts_3d = centers[:, None, :] + offsets[None, :, :]  # (B, 5, 3)

        # Project to 2D
        pts_3d_flat = pts_3d.reshape(-1, 3)  # (B*5, 3)
        projected = (K @ pts_3d_flat.T).T  # (B*5, 3)
        uvs = projected[:, :2] / projected[:, 2:3]  # (B*5, 2)
        uvs = uvs.reshape(B, -1, 2)  # (B, 5, 2)

        # Get center and radius in image space
        center_2d = uvs[:, 0]  # (B, 2)

        # Find maximum distance from center to any point
        distances = np.linalg.norm(uvs - center_2d[:, None, :], axis=2)  # (B, 5)
        radius_2d = distances.max(axis=1)  # (B,)

        # Compute square bounding box
        left = center_2d[:, 0] - radius_2d
        right = center_2d[:, 0] + radius_2d
        top = center_2d[:, 1] - radius_2d
        bottom = center_2d[:, 1] + radius_2d

        # Ensure within image bounds
        left = np.maximum(left, 0)
        right = np.minimum(right, W - 1)
        top = np.maximum(top, 0)
        bottom = np.minimum(bottom, H - 1)

        # Compute transformation matrices
        tfs = compute_tf_batch(left, right, top, bottom)

        return tfs
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def warp_perspective_batch(images, M, dsize, mode='bilinear'):
    """
    Warp images using homography matrices.

    Args:
        images: (B, H, W, C) numpy arrays
        M: (B, 3, 3) homography matrices
        dsize: (width, height) output size
        mode: 'bilinear' or 'nearest'

    Returns:
        warped: (B, dsize[1], dsize[0], C) warped images
    """
    B = len(images)
    warped = []

    flags = cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST

    for i in range(B):
        if len(images[i].shape) == 3:
            # Color image
            w = cv2.warpPerspective(images[i], M[i], dsize, flags=flags)
        else:
            # Grayscale/depth
            w = cv2.warpPerspective(images[i], M[i], dsize, flags=flags)
            w = w[..., None]  # Add channel dimension
        warped.append(w)

    return np.array(warped)


def transform_pts(pts, tf):
    """
    Transform 2D points using homography.

    Args:
        pts: (..., N, 2) points
        tf: (..., 3, 3) homography matrices

    Returns:
        transformed: (..., N, 2) transformed points
    """
    # Convert to homogeneous
    ones = np.ones((*pts.shape[:-1], 1))
    pts_homo = np.concatenate([pts, ones], axis=-1)  # (..., N, 3)

    # Transform
    transformed = (tf @ pts_homo[..., None])[..., 0]  # (..., N, 3)

    # Convert back to 2D
    transformed_2d = transformed[..., :2] / transformed[..., 2:3]

    return transformed_2d
