# utils/xyz.py
import cupy as cp
import numpy as np

# Unified batched CUDA kernel for depth to XYZ conversion
depth_to_xyz_kernel = cp.RawKernel(r'''
extern "C" __global__
void depth_to_xyz_batch(float* xyz, const float* depth, int N, int H, int W,
                       float fx, float fy, float cx, float cy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // width
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // height
    int n = blockIdx.z * blockDim.z + threadIdx.z;  // batch

    if (n >= N || i >= H || j >= W) return;

    int depth_idx = n * H * W + i * W + j;
    float z = depth[depth_idx];

    int xyz_idx = n * H * W * 3 + i * W * 3 + j * 3;

    if (z > 0.001f) {
        float x = (j - cx) * z / fx;
        float y = (i - cy) * z / fy;

        xyz[xyz_idx + 0] = x;
        xyz[xyz_idx + 1] = y;
        xyz[xyz_idx + 2] = z;
    } else {
        xyz[xyz_idx + 0] = 0.0f;
        xyz[xyz_idx + 1] = 0.0f;
        xyz[xyz_idx + 2] = 0.0f;
    }
}
''', 'depth_to_xyz_batch')

def depth_to_xyz(depth: cp.ndarray, K: np.ndarray) -> cp.ndarray:
    """
    Convert depth image(s) to XYZ coordinates.

    Args:
        depth: (N, H, W) or (H, W) depth array
        K: Camera intrinsic matrix (3, 3)

    Returns:
        XYZ coordinates: (N, H, W, 3) or (H, W, 3) matching input shape
    """
    # Extract camera parameters
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Handle 2D input by adding batch dimension
    if depth.ndim == 2:
        depth = depth[cp.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False

    N, H, W = depth.shape
    xyz = cp.zeros((N, H, W, 3), dtype=cp.float32)

    # Launch kernel with 3D grid for batch processing
    threads_per_block = (8, 8, 4)
    blocks_per_grid = (
        (W + threads_per_block[0] - 1) // threads_per_block[0],
        (H + threads_per_block[1] - 1) // threads_per_block[1],
        (N + threads_per_block[2] - 1) // threads_per_block[2]
    )

    depth_to_xyz_kernel(blocks_per_grid, threads_per_block,
                       (xyz, depth, cp.int32(N), cp.int32(H), cp.int32(W),
                        cp.float32(fx), cp.float32(fy),
                        cp.float32(cx), cp.float32(cy)))

    return xyz.squeeze(0) if squeeze_output else xyz
