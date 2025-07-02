# ModelInterface.py

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import Dict
import logging
from pathlib import Path
import cv2
from CUDAContext import CUDAContextManager

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelInterface:
    """
    An interface for executing FoundationPose models using TensorRT.

    Manages loading TensorRT engines, memory allocations, and execution of
    the pose refinement and scoring models, using a shared CUDA context to
    coexist peacefully with other GPU libraries like PyTorch/nvdiffrast.
    """

    def __init__(self, model_dir: str = "models", max_batch_size: int = 252,
                 opt_batch_size: int = 64, use_fp16: bool = True):
        self.model_dir = Path(model_dir)
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size
        self.use_fp16 = use_fp16
        self.logger = logging.getLogger(__name__)
        self.cuda_mgr = CUDAContextManager.get_instance()
        self.stream = self.cuda_mgr.get_pycuda_stream()
        self.refiner_engine = self.scorer_engine = None
        self.refiner_context = self.scorer_context = None
        self.refiner_buffers = {}
        self.scorer_buffers = {}

    def load_models(self):
        """
        Loads ONNX models and builds or deserializes TensorRT engines.
        This process can be time-consuming on the first run as TensorRT
        optimizes the model for the specific GPU. Subsequent runs will be
        fast as they load from the cached engine file.
        """
        trt_cache_dir = self.model_dir / 'trt_cache'
        trt_cache_dir.mkdir(exist_ok=True)

        self.logger.info("Loading refiner model...")
        refiner_onnx = self.model_dir / "refine_model.onnx"
        refiner_engine_path = trt_cache_dir / f"refiner_fp16_{self.max_batch_size}.engine"
        self.refiner_engine = self._build_or_load_engine(refiner_onnx, refiner_engine_path)
        self.refiner_context = self.refiner_engine.create_execution_context()
        self._allocate_buffers(self.refiner_engine, self.refiner_buffers)

        self.logger.info("Loading scorer model...")
        scorer_onnx = self.model_dir / "score_model.onnx"
        scorer_engine_path = trt_cache_dir / f"scorer_fp16_{self.max_batch_size}.engine"
        self.scorer_engine = self._build_or_load_engine(scorer_onnx, scorer_engine_path)
        self.scorer_context = self.scorer_engine.create_execution_context()
        self._allocate_buffers(self.scorer_engine, self.scorer_buffers)
        self.logger.info("TensorRT models loaded successfully.")

    def _build_or_load_engine(self, onnx_path: Path, engine_path: Path) -> trt.ICudaEngine:
        """Builds a new TensorRT engine from an ONNX file or loads a cached one."""
        if engine_path.exists():
            self.logger.info(f"Loading cached engine: {engine_path}")
            with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        self.logger.info(f"Building engine from {onnx_path}...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                raise RuntimeError(f"ONNX parsing failed for {onnx_path}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB
        if self.use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Define optimization profile for dynamic batch sizes
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            shape = tuple(d if d != -1 else 1 for d in tensor.shape) # Min shape
            profile.set_shape(tensor.name, shape, (self.opt_batch_size, *shape[1:]), (self.max_batch_size, *shape[1:]))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
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

                # Replace dynamic dimensions (-1) with max_batch_size for allocation
                alloc_shape = tuple(self.max_batch_size if dim == -1 else dim for dim in shape)
                size = trt.volume(alloc_shape) * np.dtype(dtype).itemsize

                buffers[name] = {'device': cuda.mem_alloc(int(size))}
                self.logger.info(f"Allocated buffer '{name}' for max shape {alloc_shape}")

    def refine_poses_batch(self, poses: np.ndarray, real_rgbddd: np.ndarray,
                           rendered_rgbddd: np.ndarray) -> np.ndarray:
        """
        Refines a batch of poses by predicting a corrective transformation.

        The network takes the real image and a synthetic image and predicts a small
        rotation and translation that should be applied to the input pose to make
        the synthetic render align better with the real image.

        Args:
            poses (np.ndarray): The initial batch of poses to be refined.
                - Shape: (N, 4, 4).
            real_rgbddd (np.ndarray): The processed real camera image data.
                - Shape: (N, 160, 160, 6). The same real image is duplicated N times.
                - Channels 0-2: RGB, normalized to [0, 1].
                - Channels 3-5: XYZ coordinates in the camera frame, in meters.
            rendered_rgbddd (np.ndarray): The batch of synthetic images rendered at the input poses.
                - Shape: (N, 160, 160, 6).

        Returns:
            np.ndarray: The batch of refined poses.
                - Shape: (N, 4, 4).
        """
        with self.cuda_mgr.activate_tensorrt():
            n_poses = len(poses)
            ctx, buffers = self.refiner_context, self.refiner_buffers

            # Set the exact input shape for this specific inference run.
            ctx.set_input_shape('input1', real_rgbddd.shape)
            ctx.set_input_shape('input2', rendered_rgbddd.shape)

            # Asynchronously copy data from CPU to GPU.
            cuda.memcpy_htod_async(buffers['input1']['device'], real_rgbddd, self.stream)
            cuda.memcpy_htod_async(buffers['input2']['device'], rendered_rgbddd, self.stream)

            # Set the memory addresses for all I/O tensors.
            for name, buf in buffers.items(): ctx.set_tensor_address(name, int(buf['device']))
            ctx.execute_async_v3(stream_handle=self.stream.handle)

            # Prepare host memory for the outputs.
            rot = np.empty((n_poses, 3), np.float32)
            trans = np.empty((n_poses, 3), np.float32)

            # Asynchronously copy results from GPU to CPU.
            cuda.memcpy_dtoh_async(rot, buffers['output1']['device'], self.stream)
            cuda.memcpy_dtoh_async(trans, buffers['output2']['device'], self.stream)
            self.stream.synchronize() # Wait for all operations to complete.

            # Apply the predicted deltas to the original poses.
            for i in range(n_poses): poses[i] = self._apply_pose_delta(poses[i], rot[i], trans[i])
            return poses

    def score_poses_batch(self, real_rgbddd: np.ndarray, rendered_rgbddd: np.ndarray) -> np.ndarray:
        """
        Scores how well each rendered image matches the real image.

        Args:
            real_rgbddd (np.ndarray): The real camera data, duplicated N times.
                - Shape: (N, 160, 160, 6).
            rendered_rgbddd (np.ndarray): The batch of synthetic images.
                - Shape: (N, 160, 160, 6).

        Returns:
            np.ndarray: A confidence score for each pose. Higher is better.
                - Shape: (N,).
        """
        with self.cuda_mgr.activate_tensorrt():
            n_poses = len(rendered_rgbddd)
            ctx, buffers = self.scorer_context, self.scorer_buffers

            ctx.set_input_shape('input1', real_rgbddd.shape)
            ctx.set_input_shape('input2', rendered_rgbddd.shape)

            cuda.memcpy_htod_async(buffers['input1']['device'], real_rgbddd, self.stream)
            cuda.memcpy_htod_async(buffers['input2']['device'], rendered_rgbddd, self.stream)

            for name, buf in buffers.items(): ctx.set_tensor_address(name, int(buf['device']))
            ctx.execute_async_v3(stream_handle=self.stream.handle)

            # Scorer output is (1, N), which we allocate for and then flatten.
            scores_output = np.empty((1, n_poses), np.float32)
            cuda.memcpy_dtoh_async(scores_output, buffers['output1']['device'], self.stream)
            self.stream.synchronize()

            return scores_output.flatten()

    def _apply_pose_delta(self, pose: np.ndarray, rot_delta: np.ndarray, trans_delta: np.ndarray) -> np.ndarray:
        """
        Applies a rotation (in axis-angle format) and a translation delta to a pose matrix.

        Args:
            pose (np.ndarray): The 4x4 input pose.
            rot_delta (np.ndarray): The 3-element axis-angle rotation vector. The vector's
                                    direction is the axis of rotation and its magnitude is the angle.
            trans_delta (np.ndarray): The 3-element translation vector to be added.

        Returns:
            np.ndarray: The updated 4x4 pose matrix.
        """
        # Convert the axis-angle vector to a 3x3 rotation matrix.
        R_delta, _ = cv2.Rodrigues(rot_delta)
        # Apply the rotation delta to the existing rotation.
        pose[:3, :3] = R_delta @ pose[:3, :3]
        # Add the translation delta.
        pose[:3, 3] += trans_delta
        return pose

    def warmup(self):
        """Runs dummy inferences to ensure CUDA kernels are compiled and ready,
        preventing a performance hit on the first real inference."""
        self.logger.info(f"Warming up models with batch size {self.opt_batch_size}...")
        dummy_data = np.random.rand(self.opt_batch_size, 160, 160, 6).astype(np.float32)
        dummy_poses = np.tile(np.eye(4), (self.opt_batch_size, 1, 1)).astype(np.float32)

        for _ in range(3):
            self.refine_poses_batch(dummy_poses.copy(), dummy_data, dummy_data)
            self.score_poses_batch(dummy_data, dummy_data)
        self.logger.info("Warmup complete.")


# Standalone Performance Tests
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("ModelInterface Standalone Performance Test")
    print("="*80)

    manager = None
    interface = None
    try:
        print("Initializing ModelInterface...")
        interface = ModelInterface(max_batch_size=252, opt_batch_size=64, use_fp16=True)
        manager = interface.cuda_mgr

        print("\n[Test 1] Loading and building TensorRT engines...")
        interface.load_models()

        print("\n[Test 2] Warming up models...")
        interface.warmup()

        print("\n[Test 3] Running batch performance tests...")
        for n_poses in [64, 128, 252]:
            print(f"\n--- Batch size: {n_poses}, Iters: 5 ---")
            poses = np.tile(np.eye(4), (n_poses, 1, 1)).astype(np.float32)
            real = np.random.rand(n_poses, 160, 160, 6).astype(np.float32)
            rendered = np.random.rand(n_poses, 160, 160, 6).astype(np.float32)

            t0 = time.time()
            for i in range(5):
                interface.refine_poses_batch(poses.copy(), real, rendered)
            t_refine = time.time() - t0

            t0 = time.time()
            interface.score_poses_batch(real, rendered)
            t_score = time.time() - t0

            print(f"  -> Refine: {t_refine*1000:.2f} ms | Score: {t_score*1000:.2f} ms")

        print("\n" + "="*80)
        print("All tests passed!")
        print("="*80)
    except Exception as e:
        logging.error(f"A test failed: {e}", exc_info=True)
    finally:
        if interface:
            print("\nDestroying ModelInterface object...")
            del interface

        if manager:
            print("Cleaning up CUDA context...")
            manager.cleanup()
