# utils/gpu.py
import torch
import cupy as cp

@torch.inference_mode()
def cupy_to_torch(cp_array: cp.ndarray):
    """Convert CuPy array to PyTorch tensor without copying using DLPack."""
    return torch.from_dlpack(cp_array)

@torch.inference_mode()
def torch_to_cupy(torch_tensor: torch.Tensor):
    """Convert PyTorch tensor to CuPy array without copying using DLPack."""
    return cp.from_dlpack(torch_tensor.detach())
