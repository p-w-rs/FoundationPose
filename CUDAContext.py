# CUDAContext.py

import torch
import pycuda.driver as cuda
import nvdiffrast.torch as dr
from contextlib import contextmanager
import logging

class CUDAContextManager:
    """
    A singleton class to manage a unified CUDA context for PyTorch, PyCUDA,
    and nvdiffrast.

    This manager initializes CUDA and creates a primary context that is shared
    across different parts of the application. It prevents conflicts that can
    arise when multiple libraries (like PyTorch and PyCUDA) try to manage their
    own CUDA contexts, leading to errors like "invalid device context".

    The singleton pattern ensures that only one instance of this manager exists,
    providing a single source of truth for CUDA operations.
    """
    _instance = None

    @staticmethod
    def get_instance():
        """
        Provides global access to the singleton instance of the manager.
        If an instance does not exist, it is created.
        """
        if CUDAContextManager._instance is None:
            CUDAContextManager()
        return CUDAContextManager._instance

    def __init__(self):
        """
        Initializes the CUDA context. This is called only once.
        """
        if CUDAContextManager._instance is not None:
            raise Exception("This class is a singleton! Use get_instance() to access it.")
        else:
            CUDAContextManager._instance = self
            self.device = torch.device("cuda")

            cuda.init()
            self.cu_device = cuda.Device(0)
            self.cu_context = self.cu_device.make_context()
            self.glctx = dr.RasterizeCudaContext()
            self.stream = cuda.Stream()

            logging.info("CUDAContextManager initialized successfully.")

    def cleanup(self):
        """
        Cleans up the CUDA context. This must be called before the program exits
        to avoid resource leaks and errors.
        """
        self.cu_context.pop()
        self.cu_context.detach()
        logging.info("CUDA context cleaned up successfully.")

    @contextmanager
    def activate_tensorrt(self):
        """
        A context manager to safely scope PyCUDA/TensorRT operations.
        """
        self.cu_context.push()
        try:
            yield
        finally:
            self.cu_context.pop()

    @contextmanager
    def activate_nvdiffrast(self):
        """
        A context manager for nvdiffrast operations.
        """
        yield

    def get_pycuda_stream(self):
        """Returns the shared PyCUDA stream for asynchronous execution."""
        return self.stream

    def get_nvdiff_context(self):
        """Returns the shared nvdiffrast rasterization context."""
        return self.glctx

# Standalone Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("CUDAContextManager Standalone Test")
    print("="*80)

    manager = None
    try:
        # 1. Get the singleton instance
        print("\nAttempting to get CUDAContextManager instance...")
        manager = CUDAContextManager.get_instance()
        print("Successfully obtained manager instance.")

        # 2. Verify that its components are available
        print("\nVerifying manager components:")
        print(f"  -> Nvdiffrast Context: {'OK' if manager.get_nvdiff_context() else 'FAIL'}")
        print(f"  -> PyCUDA Stream: {'OK' if manager.get_pycuda_stream() else 'FAIL'}")

        # 3. Test the context activation methods
        print("\nTesting context activation:")
        with manager.activate_tensorrt():
            print("  -> Successfully entered and exited `activate_tensorrt` context.")

        with manager.activate_nvdiffrast():
            print("  -> Successfully entered and exited `activate_nvdiffrast` context.")

        print("\n" + "="*80)
        print("All tests passed!")
        print("="*80)

    except Exception as e:
        logging.error(f"A test failed: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
