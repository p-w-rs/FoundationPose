# ModelInterface.py

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import Dict
import logging
from pathlib import Path
import cv2
import time
from CUDAContext import CUDAContextManager

# Use a non-verbose TensorRT logger to avoid unnecessary console spam.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelInterface:
    """
    Manages the TensorRT engines for FoundationPose inference.

    This class encapsulates the complexity of working with TensorRT, including:
    - Building engines from ONNX files or loading them from a cache.
    - Managing GPU memory (buffers) for model inputs and outputs.
    - Executing the `refine` and `score` models asynchronously.
    - Applying the predicted pose corrections.

    It operates within the shared CUDA context provided by CUDAContextManager
    to ensure compatibility with other GPU libraries like PyTorch/nvdiffrast.
    """

    def __init__(self, model_dir: str = "models", max_batch_size: int = 252, use_fp16: bool = True):
        """
        Initializes the ModelInterface.
        """
        self.model_dir = Path(model_dir)
        self.max_batch_size = max_batch_size
        self.use_fp16 = use_fp16
        self.logger = logging.getLogger(__name__)
        self.cuda_mgr = CUDAContextManager.get_instance()
        self.stream = self.cuda_mgr.get_pycuda_stream()
        self.refiner_engine = self.scorer_engine = None
        self.refiner_context = self.scorer_context = None
        self.refiner_buffers = {}
        self.scorer_buffers = {}

    def __del__(self):
        """
        Destructor to ensure clean release of TensorRT and PyCUDA resources.

        This is a CRITICAL step. Python's garbage collector does not guarantee the
        order of object destruction. It's possible for the CUDA context to be
        destroyed before the TensorRT engines/contexts are. Explicitly deleting
        them here ensures they are released while the context is still valid,
        preventing the "invalid device context" errors on exit.
        """
        self.logger.info("Releasing TensorRT resources...")
        # Explicitly delete contexts and engines to manage destruction order
        del self.refiner_context
        del self.scorer_context
        del self.refiner_engine
        del self.scorer_engine
        # Buffers are managed by PyCUDA and will be released with the context
        self.logger.info("TensorRT resources released.")

    def load_models(self):
        """Loads ONNX models and builds or deserializes TensorRT engines."""
        trt_cache_dir = self.model_dir / 'trt_cache'
        trt_cache_dir.mkdir(exist_ok=True)

        self.logger.info("Loading refiner model...")
        refiner_engine_path = trt_cache_dir / f"refiner_fp16_{self.max_batch_size}.engine"
        self.refiner_engine = self._build_or_load_engine(self.model_dir / "refine_model.onnx", refiner_engine_path)
        self.refiner_context = self.refiner_engine.create_execution_context()
        self._allocate_buffers(self.refiner_engine, self.refiner_buffers)

        self.logger.info("Loading scorer model...")
        scorer_engine_path = trt_cache_dir / f"scorer_fp16_{self.max_batch_size}.engine"
        self.scorer_engine = self._build_or_load_engine(self.model_dir / "score_model.onnx", scorer_engine_path)
        self.scorer_context = self.scorer_engine.create_execution_context()
        self._allocate_buffers(self.scorer_engine, self.scorer_buffers)
        self.logger.info("TensorRT models loaded successfully.")

    def _build_or_load_engine(self, onnx_path: Path, engine_path: Path) -> trt.ICudaEngine:
        """Builds a TensorRT engine from an ONNX file or loads a pre-built one."""
        if engine_path.exists():
            self.logger.info(f"Loading cached engine: {engine_path}")
            with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        self.logger.info(f"Building new engine from {onnx_path} (this may take a few minutes)...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors): self.logger.error(parser.get_error(error))
                raise RuntimeError(f"ONNX parsing failed for {onnx_path}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB
        if self.use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            min_shape = tuple([1 if dim == -1 else dim for dim in tensor.shape])
            opt_shape = tuple([self.max_batch_size // 2 if dim == -1 else dim for dim in tensor.shape])
            max_shape = tuple([self.max_batch_size if dim == -1 else dim for dim in tensor.shape])
            profile.set_shape(tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None: raise RuntimeError("Failed to build the TensorRT engine.")

        with open(engine_path, 'wb') as f: f.write(serialized_engine)
        with trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(serialized_engine)

    def _allocate_buffers(self, engine: trt.ICudaEngine, buffers: Dict):
        """Allocates GPU memory for the engine's I/O tensors."""
        with self.cuda_mgr.activate_tensorrt():
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                alloc_shape = tuple([self.max_batch_size if dim == -1 else dim for dim in shape])
                size = trt.volume(alloc_shape) * np.dtype(dtype).itemsize
                if size < 0: raise ValueError(f"Calculated negative buffer size for tensor '{name}'")
                buffers[name] = {'device': cuda.mem_alloc(int(size))}

    def refine_poses_batch(self, poses: np.ndarray, real_rgbddd: np.ndarray, rendered_rgbddd: np.ndarray) -> np.ndarray:
        """Refines a batch of poses by predicting a corrective transformation."""
        with self.cuda_mgr.activate_tensorrt():
            n_poses = len(poses)
            ctx, buffers = self.refiner_context, self.refiner_buffers
            ctx.set_input_shape('input1', real_rgbddd.shape); ctx.set_input_shape('input2', rendered_rgbddd.shape)
            cuda.memcpy_htod_async(buffers['input1']['device'], real_rgbddd, self.stream)
            cuda.memcpy_htod_async(buffers['input2']['device'], rendered_rgbddd, self.stream)
            for name, buf in buffers.items(): ctx.set_tensor_address(name, int(buf['device']))
            ctx.execute_async_v3(stream_handle=self.stream.handle)
            rot = np.empty((n_poses, 3), np.float32); trans = np.empty((n_poses, 3), np.float32)
            cuda.memcpy_dtoh_async(rot, buffers['output1']['device'], self.stream)
            cuda.memcpy_dtoh_async(trans, buffers['output2']['device'], self.stream)
            self.stream.synchronize()
            for i in range(n_poses): poses[i] = self._apply_pose_delta(poses[i], rot[i], trans[i])
            return poses

    def score_poses_batch(self, real_rgbddd: np.ndarray, rendered_rgbddd: np.ndarray) -> np.ndarray:
        """Scores how well each rendered image matches the real image."""
        with self.cuda_mgr.activate_tensorrt():
            n_poses = len(rendered_rgbddd)
            ctx, buffers = self.scorer_context, self.scorer_buffers
            ctx.set_input_shape('input1', real_rgbddd.shape); ctx.set_input_shape('input2', rendered_rgbddd.shape)
            cuda.memcpy_htod_async(buffers['input1']['device'], real_rgbddd, self.stream)
            cuda.memcpy_htod_async(buffers['input2']['device'], rendered_rgbddd, self.stream)
            for name, buf in buffers.items(): ctx.set_tensor_address(name, int(buf['device']))
            ctx.execute_async_v3(stream_handle=self.stream.handle)
            scores_output = np.empty((1, n_poses), np.float32)
            cuda.memcpy_dtoh_async(scores_output, buffers['output1']['device'], self.stream)
            self.stream.synchronize()
            return scores_output.flatten()

    def _apply_pose_delta(self, pose: np.ndarray, rot_delta: np.ndarray, trans_delta: np.ndarray) -> np.ndarray:
        """Applies the predicted pose correction delta to a pose matrix."""
        delta_transform = np.eye(4, dtype=np.float32)
        R_delta, _ = cv2.Rodrigues(rot_delta)
        delta_transform[:3, :3] = R_delta
        delta_transform[:3, 3] = trans_delta
        return pose @ delta_transform

    def warmup(self):
        """Runs dummy inferences to compile CUDA kernels."""
        self.logger.info(f"Warming up models with batch size {self.max_batch_size // 2}...")
        dummy_batch_size = self.max_batch_size // 2
        dummy_data = np.random.rand(dummy_batch_size, 160, 160, 6).astype(np.float32)
        dummy_poses = np.tile(np.eye(4), (dummy_batch_size, 1, 1)).astype(np.float32)
        for _ in range(3):
            self.refine_poses_batch(dummy_poses.copy(), dummy_data, dummy_data)
            self.score_poses_batch(dummy_data, dummy_data)
        self.logger.info("Warmup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("="*80); print("ModelInterface Standalone Performance Test"); print("="*80)
    manager = None
    try:
        print("Initializing ModelInterface...")
        interface = ModelInterface(max_batch_size=252, use_fp16=True)
        manager = interface.cuda_mgr
        print("\n[Test 1] Loading and building TensorRT engines...")
        interface.load_models()
        print("\n[Test 2] Warming up models...")
        interface.warmup()
        print("\n[Test 3] Running batch performance tests...")
        for n_poses in [64, 128, 252]:
            print(f"\n--- Testing with batch size: {n_poses} ---")
            poses = np.tile(np.eye(4), (n_poses, 1, 1)).astype(np.float32)
            real = np.random.rand(n_poses, 160, 160, 6).astype(np.float32)
            rendered = np.random.rand(n_poses, 160, 160, 6).astype(np.float32)
            t0 = time.time(); interface.refine_poses_batch(poses.copy(), real, rendered); t_refine = (time.time() - t0) * 1000
            print(f"  -> Refine Network Time: {t_refine:.2f} ms")
            t0 = time.time(); interface.score_poses_batch(real, rendered); t_score = (time.time() - t0) * 1000
            print(f"  -> Score Network Time:  {t_score:.2f} ms")
        print("\n" + "="*80); print("ModelInterface test completed successfully!"); print("="*80)
    except Exception as e:
        logging.error(f"An error occurred during the standalone test: {e}", exc_info=True)
    finally:
        # Explicitly delete the interface object before cleaning up the context
        del interface
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
