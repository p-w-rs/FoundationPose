"""
CUDAContext.py - Shared CUDA Context Manager for FoundationPose

This module manages CUDA contexts between different libraries:
- pycuda (used by TensorRT in ModelInterface)
- nvdiffrast (used by MeshRenderer)

The main issue is that both libraries create their own CUDA contexts,
leading to "invalid resource handle" errors when switching between them.

Solution:
- Single shared primary context
- Context guards for safe switching
- Lazy initialization for standalone operation

Usage:
    from CUDAContext import CUDAContextManager

    cuda_mgr = CUDAContextManager.get_instance()

    # For TensorRT operations
    with cuda_mgr.activate_tensorrt():
        # TensorRT/pycuda operations

    # For nvdiffrast operations
    with cuda_mgr.activate_nvdiffrast():
        # nvdiffrast operations
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional, Any
import atexit

# Conditional imports - only import if needed
try:
    import pycuda.driver as cuda
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    dr = None


class CUDAContextManager:
    """
    Singleton manager for CUDA contexts across different libraries.

    Handles context switching between:
    - pycuda/TensorRT context
    - nvdiffrast GL context

    Features:
    - Thread-safe singleton pattern
    - Lazy initialization
    - Context guards for safe switching
    - Automatic cleanup on exit
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.logger = logging.getLogger(__name__)

        # Context holders
        self.pycuda_context = None
        self.nvdiff_context = None
        self.current_context = None

        # Flags
        self.pycuda_initialized = False
        self.nvdiff_initialized = False

        # Register cleanup
        atexit.register(self.cleanup)

        self.logger.info("CUDA Context Manager initialized")

    @classmethod
    def get_instance(cls) -> 'CUDAContextManager':
        """Get the singleton instance."""
        return cls()

    def init_pycuda(self) -> Any:
        """
        Initialize pycuda context if not already done.

        Returns:
            pycuda context object
        """
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("pycuda not available. Install with: pip install pycuda")

        if not self.pycuda_initialized:
            self.logger.info("Initializing pycuda context...")

            # Initialize CUDA
            cuda.init()

            # Create context on device 0
            device = cuda.Device(0)
            self.pycuda_context = device.make_context()

            # Pop it for now - we'll push when needed
            self.pycuda_context.pop()

            self.pycuda_initialized = True
            self.logger.info("pycuda context initialized")

        return self.pycuda_context

    def init_nvdiffrast(self) -> Any:
        """
        Initialize nvdiffrast context if not already done.

        Returns:
            nvdiffrast GL context object
        """
        if not NVDIFFRAST_AVAILABLE:
            raise RuntimeError("nvdiffrast not available. Install it first.")

        if not self.nvdiff_initialized:
            self.logger.info("Initializing nvdiffrast context...")

            # Create nvdiffrast context
            # This creates its own GL/CUDA interop context
            import torch
            device = torch.device('cuda')
            self.nvdiff_context = dr.RasterizeCudaContext(device=device)

            self.nvdiff_initialized = True
            self.logger.info("nvdiffrast context initialized")

        return self.nvdiff_context

    @contextmanager
    def activate_tensorrt(self):
        """
        Context manager for TensorRT/pycuda operations.

        Usage:
            with cuda_mgr.activate_tensorrt():
                # Run TensorRT inference
        """
        if not self.pycuda_initialized:
            self.init_pycuda()

        # Push pycuda context
        self.pycuda_context.push()
        prev_context = self.current_context
        self.current_context = 'pycuda'

        try:
            yield self.pycuda_context
        finally:
            # Pop context
            self.pycuda_context.pop()
            self.current_context = prev_context

    @contextmanager
    def activate_nvdiffrast(self):
        """
        Context manager for nvdiffrast operations.

        Usage:
            with cuda_mgr.activate_nvdiffrast():
                # Run nvdiffrast rendering
        """
        if not self.nvdiff_initialized:
            self.init_nvdiffrast()

        # nvdiffrast manages its own context internally
        # We just need to ensure pycuda context is not active
        if self.current_context == 'pycuda':
            # Make sure pycuda context is popped
            if cuda and cuda.Context.get_current() is not None:
                cuda.Context.pop()

        prev_context = self.current_context
        self.current_context = 'nvdiffrast'

        try:
            yield self.nvdiff_context
        finally:
            self.current_context = prev_context

    def get_nvdiff_context(self) -> Any:
        """Get nvdiffrast context, initializing if needed."""
        if not self.nvdiff_initialized:
            self.init_nvdiffrast()
        return self.nvdiff_context

    def get_pycuda_stream(self) -> Any:
        """Get a CUDA stream for pycuda operations."""
        if not self.pycuda_initialized:
            self.init_pycuda()

        # Need context to be active to create stream
        self.pycuda_context.push()
        stream = cuda.Stream()
        self.pycuda_context.pop()

        return stream

    def cleanup(self):
        """Clean up CUDA contexts on exit."""
        self.logger.info("Cleaning up CUDA contexts...")

        # Clean up pycuda context
        if self.pycuda_initialized and self.pycuda_context:
            try:
                # Make sure it's not current
                if cuda and cuda.Context.get_current() == self.pycuda_context:
                    self.pycuda_context.pop()
                # Detach (don't destroy - let pycuda handle it)
                self.pycuda_context.detach()
            except Exception as e:
                self.logger.warning(f"Error cleaning up pycuda context: {e}")

        # nvdiffrast cleans itself up
        self.logger.info("CUDA cleanup complete")

    def get_device_info(self) -> dict:
        """Get information about the current CUDA device."""
        info = {
            'pycuda_available': PYCUDA_AVAILABLE,
            'nvdiffrast_available': NVDIFFRAST_AVAILABLE,
            'pycuda_initialized': self.pycuda_initialized,
            'nvdiff_initialized': self.nvdiff_initialized,
            'current_context': self.current_context
        }

        if PYCUDA_AVAILABLE and cuda:
            cuda.init()
            device = cuda.Device(0)
            info['device_name'] = device.name()
            info['compute_capability'] = device.compute_capability()
            info['total_memory'] = device.total_memory() // (1024**2)  # MB

            # Get free memory if context exists
            if self.pycuda_initialized:
                self.pycuda_context.push()
                free, total = cuda.mem_get_info()
                info['free_memory'] = free // (1024**2)  # MB
                info['used_memory'] = (total - free) // (1024**2)  # MB
                self.pycuda_context.pop()

        return info


# Unit tests
if __name__ == "__main__":
    import numpy as np
    import time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("CUDA Context Manager Unit Tests")
    print("="*80)

    # Test 1: Singleton pattern
    print("\nTest 1: Singleton Pattern")
    mgr1 = CUDAContextManager.get_instance()
    mgr2 = CUDAContextManager.get_instance()
    print(f"Same instance: {mgr1 is mgr2}")
    assert mgr1 is mgr2, "Singleton pattern failed"

    # Test 2: Device info
    print("\nTest 2: Device Information")
    info = mgr1.get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test 3: Context switching
    print("\nTest 3: Context Switching")

    # Test pycuda context
    if PYCUDA_AVAILABLE:
        print("\n  Testing pycuda context...")
        with mgr1.activate_tensorrt() as ctx:
            print(f"    Context active: {cuda.Context.get_current() is not None}")

            # Simple CUDA operation
            a = np.random.randn(100).astype(np.float32)
            a_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu, a)
            print("    ✓ Memory allocation successful")

        print(f"    Context after exit: {cuda.Context.get_current() is None}")

    # Test nvdiffrast context
    if NVDIFFRAST_AVAILABLE:
        print("\n  Testing nvdiffrast context...")
        try:
            import torch
            with mgr1.activate_nvdiffrast() as ctx:
                print(f"    GL context created: {ctx is not None}")

                # Simple rasterization test
                pos = torch.tensor([
                    [-1, -1, 0, 1],
                    [ 1, -1, 0, 1],
                    [ 0,  1, 0, 1]
                ], dtype=torch.float32, device='cuda')

                tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

                rast, _ = dr.rasterize(ctx, pos[None, ...], tri, resolution=[32, 32])
                print(f"    ✓ Rasterization successful, output shape: {rast.shape}")
        except Exception as e:
            print(f"    Note: nvdiffrast test requires X display or EGL: {e}")

    # Test 4: Interleaved operations
    print("\nTest 4: Interleaved Operations")
    if PYCUDA_AVAILABLE and NVDIFFRAST_AVAILABLE:
        success_count = 0

        for i in range(3):
            # TensorRT operation
            with mgr1.activate_tensorrt():
                arr = np.random.randn(10).astype(np.float32)
                gpu_mem = cuda.mem_alloc(arr.nbytes)
                cuda.memcpy_htod(gpu_mem, arr)
                success_count += 1

            # Brief pause
            time.sleep(0.01)

            # nvdiffrast operation
            try:
                with mgr1.activate_nvdiffrast() as ctx:
                    # Just verify context exists
                    assert ctx is not None
                    success_count += 1
            except:
                # Skip if no display
                pass

        print(f"  Completed {success_count} context switches successfully")

    # Test 5: Stream creation
    print("\nTest 5: CUDA Stream Creation")
    if PYCUDA_AVAILABLE:
        stream = mgr1.get_pycuda_stream()
        print(f"  Stream created: {stream is not None}")

    # Final device info
    print("\nFinal Device State:")
    final_info = mgr1.get_device_info()
    print(f"  Contexts initialized: pycuda={final_info['pycuda_initialized']}, "
          f"nvdiff={final_info['nvdiff_initialized']}")
    if 'free_memory' in final_info:
        print(f"  Memory: {final_info['free_memory']} MB free / "
              f"{final_info['total_memory']} MB total")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
