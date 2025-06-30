import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import cv2

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelInterface:
    """
    Pure TensorRT interface for FoundationPose models.

    This interface handles batch processing of poses for both refinement and scoring:
    - Refiner: Takes N pose hypotheses and refines them using real vs rendered RGBXYZ data
    - Scorer: Evaluates how well N rendered poses match the real observation

    Input format: RGBXYZ (6 channels) where XYZ are 3D coordinates in meters
    - RGB channels: [0, 1] normalized color values
    - XYZ channels: 3D coordinates in camera space (meters)

    The interface uses TensorRT for GPU acceleration with FP16 precision internally
    while maintaining FP32 for inputs/outputs.
    """

    def __init__(self, model_dir: str = "models", max_batch_size: int = 252,
                 opt_batch_size: int = 64, use_fp16: bool = True):
        """
        Initialize TensorRT interface.

        Args:
            model_dir: Directory containing ONNX models
            max_batch_size: Maximum number of poses to process in one batch
            opt_batch_size: Optimal batch size for TensorRT optimization
            use_fp16: Enable FP16 precision internally (inputs/outputs remain FP32)
        """
        self.model_dir = Path(model_dir)
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size
        self.use_fp16 = use_fp16
        self.logger = logging.getLogger(__name__)

        # TensorRT engines
        self.refiner_engine = None
        self.scorer_engine = None

        # Execution contexts
        self.refiner_context = None
        self.scorer_context = None

        # CUDA stream
        self.stream = cuda.Stream()

        # Allocated buffers
        self.refiner_buffers = {}
        self.scorer_buffers = {}

    def load_models(self):
        """Load and build TensorRT engines from ONNX models."""
        trt_cache_dir = self.model_dir / 'trt_cache'
        trt_cache_dir.mkdir(exist_ok=True)

        # Load refiner
        refiner_onnx = self.model_dir / "refine_model.onnx"
        refiner_engine_path = trt_cache_dir / f"refine_fp16_{self.max_batch_size}.engine"

        self.logger.info("Loading refiner model...")
        self.refiner_engine = self._build_or_load_engine(
            refiner_onnx, refiner_engine_path, "refiner"
        )
        self.refiner_context = self.refiner_engine.create_execution_context()
        self._allocate_buffers(self.refiner_engine, self.refiner_buffers)

        # Load scorer
        scorer_onnx = self.model_dir / "score_model.onnx"
        scorer_engine_path = trt_cache_dir / f"score_fp16_{self.max_batch_size}.engine"

        self.logger.info("Loading scorer model...")
        self.scorer_engine = self._build_or_load_engine(
            scorer_onnx, scorer_engine_path, "scorer"
        )
        self.scorer_context = self.scorer_engine.create_execution_context()
        self._allocate_buffers(self.scorer_engine, self.scorer_buffers)

        self.logger.info("Models loaded successfully")

    def _build_or_load_engine(self, onnx_path: Path, engine_path: Path,
                              model_name: str) -> trt.ICudaEngine:
        """
        Build TensorRT engine from ONNX or load cached engine.

        The engine is optimized for the specified batch sizes and FP16 if enabled.
        Cached engines are reused to avoid rebuild time.
        """
        # Try loading cached engine
        if engine_path.exists():
            self.logger.info(f"Loading cached {model_name} engine from {engine_path}")
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                return runtime.deserialize_cuda_engine(f.read())

        # Build new engine
        self.logger.info(f"Building {model_name} engine from {onnx_path}")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                raise RuntimeError(f"Failed to parse {onnx_path}")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30) # 8GB

        if self.use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Set optimization profiles for dynamic batch
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            # Dynamic batch dimension is first
            min_shape = list(input_tensor.shape)
            opt_shape = list(input_tensor.shape)
            max_shape = list(input_tensor.shape)

            min_shape[0] = 1
            opt_shape[0] = self.opt_batch_size
            max_shape[0] = self.max_batch_size

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"Failed to build {model_name} engine")

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized_engine)

    def _allocate_buffers(self, engine: trt.ICudaEngine, buffers: Dict):
        """
        Allocate CUDA memory for inputs/outputs.

        Handles dynamic dimensions by using max_batch_size for allocation.
        Buffer info includes shape, dtype, and whether it's an input.
        """
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # Handle dynamic dimensions (-1) by using max batch size
            for i, dim in enumerate(shape):
                if dim == -1:
                    shape[i] = self.max_batch_size

            # Calculate buffer size using int64 to prevent overflow
            size = int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)
            if size < 0:
                raise ValueError(f"Buffer size calculation resulted in negative value: {size}")

            device_mem = cuda.mem_alloc(size)

            buffers[name] = {
                'device': device_mem,
                'host': cuda.pagelocked_empty(shape, dtype),
                'shape': shape,
                'dtype': dtype,
                'is_input': is_input
            }

            self.logger.info(f"Allocated {name}: shape={shape}, dtype={dtype}, input={is_input}")

    def refine_poses_batch(self, poses: np.ndarray, real_rgbddd: np.ndarray,
                           rendered_rgbddd: np.ndarray) -> np.ndarray:
        """
        Refine N poses in batch using the refiner network.

        The refiner predicts rotation and translation deltas to improve pose alignment
        between rendered and real observations.

        Args:
            poses: (N, 4, 4) pose matrices in float32
            real_rgbddd: (N, 160, 160, 6) real observation RGBXYZ in float32
            rendered_rgbddd: (N, 160, 160, 6) rendered views RGBXYZ in float32

        Returns:
            (N, 4, 4) refined poses in float32
        """
        n_poses = len(poses)
        current_poses = poses.copy()

        # Ensure real observation is batched
        if real_rgbddd.ndim == 3:
            real_rgbddd = np.repeat(real_rgbddd[np.newaxis], n_poses, axis=0)

        # Keep as float32 - TensorRT handles internal fp16 conversion
        real_batch = real_rgbddd.astype(np.float32)
        rendered_batch = rendered_rgbddd.astype(np.float32)

        # Set dynamic shapes
        self.refiner_context.set_input_shape('input1', (n_poses, 160, 160, 6))
        self.refiner_context.set_input_shape('input2', (n_poses, 160, 160, 6))

        # Copy to GPU
        cuda.memcpy_htod_async(
            self.refiner_buffers['input1']['device'],
            real_batch, self.stream
        )
        cuda.memcpy_htod_async(
            self.refiner_buffers['input2']['device'],
            rendered_batch, self.stream
        )

        # Set tensor addresses
        for name, buf in self.refiner_buffers.items():
            self.refiner_context.set_tensor_address(name, int(buf['device']))

        # Execute
        self.refiner_context.execute_async_v3(stream_handle=self.stream.handle)

        # Prepare host arrays for output
        rotation_output = np.empty((n_poses, 3), dtype=np.float32)
        translation_output = np.empty((n_poses, 3), dtype=np.float32)

        # Copy outputs from GPU
        cuda.memcpy_dtoh_async(
            rotation_output,
            self.refiner_buffers['output1']['device'],
            self.stream
        )
        cuda.memcpy_dtoh_async(
            translation_output,
            self.refiner_buffers['output2']['device'],
            self.stream
        )

        self.stream.synchronize()

        # Apply deltas to poses
        for i in range(n_poses):
            current_poses[i] = self._apply_pose_delta(
                current_poses[i], rotation_output[i], translation_output[i]
            )

        return current_poses

    def score_poses_batch(self, real_rgbddd: np.ndarray,
                          rendered_rgbddd: np.ndarray) -> np.ndarray:
        """
        Score N poses in batch using the scorer network.

        The scorer evaluates how well each rendered pose matches the real observation.
        Higher scores indicate better alignment.

        Args:
            real_rgbddd: (N, 160, 160, 6) real observation RGBXYZ in float32
            rendered_rgbddd: (N, 160, 160, 6) rendered views RGBXYZ in float32

        Returns:
            (N,) scores in float32
        """
        n_poses = len(rendered_rgbddd)

        # Ensure real observation is batched
        if real_rgbddd.ndim == 3:
            real_rgbddd = np.repeat(real_rgbddd[np.newaxis], n_poses, axis=0)

        # Keep as float32
        real_batch = real_rgbddd.astype(np.float32)
        rendered_batch = rendered_rgbddd.astype(np.float32)

        # Set dynamic shapes
        self.scorer_context.set_input_shape('input1', (n_poses, 160, 160, 6))
        self.scorer_context.set_input_shape('input2', (n_poses, 160, 160, 6))

        # Copy to GPU
        cuda.memcpy_htod_async(
            self.scorer_buffers['input1']['device'],
            real_batch, self.stream
        )
        cuda.memcpy_htod_async(
            self.scorer_buffers['input2']['device'],
            rendered_batch, self.stream
        )

        # Set tensor addresses
        for name, buf in self.scorer_buffers.items():
            self.scorer_context.set_tensor_address(name, int(buf['device']))

        # Execute
        self.scorer_context.execute_async_v3(stream_handle=self.stream.handle)

        # Scorer output shape is (1, N)
        scores_output = np.empty((1, n_poses), dtype=np.float32)

        # Copy scores from GPU
        cuda.memcpy_dtoh_async(
            scores_output,
            self.scorer_buffers['output1']['device'],
            self.stream
        )

        self.stream.synchronize()

        # Return as 1D array
        return scores_output[0]

    def _apply_pose_delta(self, pose: np.ndarray, rotation_delta: np.ndarray,
                          translation_delta: np.ndarray) -> np.ndarray:
        """
        Apply pose update from refiner.

        Rotation delta is axis-angle representation, converted to rotation matrix
        using Rodrigues formula. Translation is added directly.

        Args:
            pose: (4, 4) current pose matrix
            rotation_delta: (3,) axis-angle rotation update
            translation_delta: (3,) translation update in meters

        Returns:
            (4, 4) updated pose matrix
        """
        angle = np.linalg.norm(rotation_delta)
        if angle > 1e-6:
            axis = rotation_delta / angle
            R_delta, _ = cv2.Rodrigues(axis * angle)
        else:
            R_delta = np.eye(3)

        updated_pose = pose.copy()
        updated_pose[:3, :3] = R_delta @ pose[:3, :3]
        updated_pose[:3, 3] += translation_delta

        return updated_pose

    def warmup(self):
        """
        Warmup engines to ensure optimal performance on first real inference.

        Runs several iterations with representative data to initialize CUDA kernels
        and TensorRT optimizations.
        """
        warmup_batch_size = self.opt_batch_size if self.opt_batch_size > 1 else 16
        self.logger.info(f"Warming up with batch size {warmup_batch_size}...")

        dummy_rgbddd = np.random.randn(warmup_batch_size, 160, 160, 6).astype(np.float32)
        dummy_rgbddd[:, :, :3] = np.clip(dummy_rgbddd[:, :, :3], 0, 1)  # Clip RGB to valid range
        dummy_poses = np.tile(np.eye(4), (warmup_batch_size, 1, 1)).astype(np.float32)

        for _ in range(3):
            self.refine_poses_batch(dummy_poses, dummy_rgbddd, dummy_rgbddd)
            self.score_poses_batch(dummy_rgbddd, dummy_rgbddd)

        self.logger.info("Warmup complete")


# Unit tests
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("="*80)
    print("TensorRT Batch Performance Tests")
    print("="*80)

    # Initialize
    interface = ModelInterface(max_batch_size=252, opt_batch_size=64, use_fp16=True)

    # Load models
    print("\nLoading models...")
    interface.load_models()

    # Warmup
    print("\nWarming up...")
    interface.warmup()

    # Performance test
    print("\nBatch Performance Tests:")
    print("-"*60)
    import time

    # Test data
    dummy_rgbddd = np.random.randn(160, 160, 6).astype(np.float32)
    dummy_rgbddd[:, :, :3] = np.clip(dummy_rgbddd[:, :, :3], 0, 1)

    # Test different batch sizes
    for n_poses in [64, 128, 252]:
        print(f"\nBatch size: {n_poses} poses")

        # Create batch data
        poses_batch = np.tile(np.eye(4), (n_poses, 1, 1)).astype(np.float32)
        # Real observation is typically the same for all poses in a hypothesis batch
        real_batch = np.repeat(dummy_rgbddd[np.newaxis], n_poses, axis=0)
        rendered_batch = np.repeat(dummy_rgbddd[np.newaxis], n_poses, axis=0)

        # Test refinement (5 iterations)
        t0 = time.time()
        for _ in range(5):
            refined = interface.refine_poses_batch(poses_batch, real_batch, rendered_batch)
        t_refine = time.time() - t0

        # Test scoring
        t0 = time.time()
        scores = interface.score_poses_batch(real_batch, rendered_batch)
        t_score = time.time() - t0

        # Report results
        t_per_iter = t_refine / 5

        print(f"  Refine (5 iter): {t_refine*1000:.1f} ms ({t_per_iter*1000:.1f} ms/iter)")
        print(f"  Score:           {t_score*1000:.1f} ms")
        print(f"  Total pipeline:  {(t_refine + t_score)*1000:.1f} ms")

    print("\n" + "="*80)
    print("Tests complete!")
    print("="*80)
