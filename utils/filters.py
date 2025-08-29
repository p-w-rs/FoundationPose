# utils/filters.py
import cupy as cp

# Batched erode depth kernel
erode_depth_kernel = cp.RawKernel(r'''
extern "C" __global__
void erode_depth_batch(const float* depth, float* out, int N, int H, int W,
                      int radius, float depth_diff_thres, float ratio_thres, float zfar) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (w >= W || h >= H || n >= N) return;

    int batch_offset = n * H * W;
    int idx = batch_offset + h * W + w;
    float d_ori = depth[idx];

    if (d_ori < 0.001f || d_ori >= zfar) {
        out[idx] = 0.0f;
        return;
    }

    float bad_cnt = 0.0f;
    float total = 0.0f;

    for (int u = max(0, w - radius); u <= min(W - 1, w + radius); u++) {
        for (int v = max(0, h - radius); v <= min(H - 1, h + radius); v++) {
            float cur_depth = depth[batch_offset + v * W + u];
            total += 1.0f;
            if (cur_depth < 0.001f || cur_depth >= zfar || fabsf(cur_depth - d_ori) > depth_diff_thres) {
                bad_cnt += 1.0f;
            }
        }
    }

    if (bad_cnt / total > ratio_thres) {
        out[idx] = 0.0f;
    } else {
        out[idx] = d_ori;
    }
}
''', 'erode_depth_batch')

def erode_depth(depth: cp.ndarray, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100):
    """
    CuPy implementation of depth erosion for batched inputs

    Args:
        depth: (N, H, W) or (H, W) depth array
        radius: erosion radius
        depth_diff_thres: threshold for depth difference
        ratio_thres: threshold for bad pixel ratio
        zfar: far clipping distance

    Returns:
        Eroded depth map with same shape as input
    """
    # Handle 2D input by adding batch dimension
    if depth.ndim == 2:
        depth = depth[cp.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False

    N, H, W = depth.shape
    out = cp.zeros_like(depth, dtype=cp.float32)

    # Launch kernel with 3D grid for batch processing
    threads_per_block = (8, 8, 4)
    blocks_per_grid = (
        (W + threads_per_block[0] - 1) // threads_per_block[0],
        (H + threads_per_block[1] - 1) // threads_per_block[1],
        (N + threads_per_block[2] - 1) // threads_per_block[2]
    )

    erode_depth_kernel(blocks_per_grid, threads_per_block,
                      (depth, out, cp.int32(N), cp.int32(H), cp.int32(W),
                       cp.int32(radius), cp.float32(depth_diff_thres),
                       cp.float32(ratio_thres), cp.float32(zfar)))

    return out.squeeze(0) if squeeze_output else out

# Batched bilateral filter kernel
bilateral_filter_kernel = cp.RawKernel(r'''
extern "C" __global__
void bilateral_filter_depth_batch(const float* depth, float* out, int N, int H, int W,
                                 int radius, float zfar, float sigmaD, float sigmaR) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (w >= W || h >= H || n >= N) return;

    int batch_offset = n * H * W;
    int idx = batch_offset + h * W + w;
    out[idx] = 0.0f;

    // First pass: compute mean depth
    float mean_depth = 0.0f;
    int num_valid = 0;

    for (int u = max(0, w - radius); u <= min(W - 1, w + radius); u++) {
        for (int v = max(0, h - radius); v <= min(H - 1, h + radius); v++) {
            float cur_depth = depth[batch_offset + v * W + u];
            if (cur_depth >= 0.001f && cur_depth < zfar) {
                num_valid++;
                mean_depth += cur_depth;
            }
        }
    }

    if (num_valid == 0) return;
    mean_depth /= (float)num_valid;

    // Second pass: bilateral filtering
    float depthCenter = depth[idx];
    float sum_weight = 0.0f;
    float sum = 0.0f;

    for (int u = max(0, w - radius); u <= min(W - 1, w + radius); u++) {
        for (int v = max(0, h - radius); v <= min(H - 1, h + radius); v++) {
            float cur_depth = depth[batch_offset + v * W + u];
            if (cur_depth >= 0.001f && cur_depth < zfar && fabsf(cur_depth - mean_depth) < 0.01f) {
                float spatial_dist_sq = (float)((u - w) * (u - w) + (h - v) * (h - v));
                float depth_diff_sq = (depthCenter - cur_depth) * (depthCenter - cur_depth);
                float weight = expf(-spatial_dist_sq / (2.0f * sigmaD * sigmaD) -
                                   depth_diff_sq / (2.0f * sigmaR * sigmaR));
                sum_weight += weight;
                sum += weight * cur_depth;
            }
        }
    }

    if (sum_weight > 0 && num_valid > 0) {
        out[idx] = sum / sum_weight;
    }
}
''', 'bilateral_filter_depth_batch')

def bilateral_filter_depth(depth: cp.ndarray, radius=2, zfar=100, sigmaD=2, sigmaR=100000):
    """
    CuPy implementation of bilateral filter for batched depth maps

    Args:
        depth: (N, H, W) or (H, W) depth array
        radius: filter radius
        zfar: far clipping distance
        sigmaD: spatial sigma
        sigmaR: range sigma

    Returns:
        Filtered depth map with same shape as input
    """
    # Handle 2D input by adding batch dimension
    if depth.ndim == 2:
        depth = depth[cp.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False

    N, H, W = depth.shape
    out = cp.zeros_like(depth, dtype=cp.float32)

    # Launch kernel with 3D grid for batch processing
    threads_per_block = (8, 8, 4)
    blocks_per_grid = (
        (W + threads_per_block[0] - 1) // threads_per_block[0],
        (H + threads_per_block[1] - 1) // threads_per_block[1],
        (N + threads_per_block[2] - 1) // threads_per_block[2]
    )

    bilateral_filter_kernel(blocks_per_grid, threads_per_block,
                          (depth, out, cp.int32(N), cp.int32(H), cp.int32(W),
                           cp.int32(radius), cp.float32(zfar),
                           cp.float32(sigmaD), cp.float32(sigmaR)))

    return out.squeeze(0) if squeeze_output else out
