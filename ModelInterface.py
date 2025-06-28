import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelInterface:
    """
    Pure TensorRT interface for FoundationPose models.

    Processes batches of N poses simultaneously:
    - Input: (N, 160, 160, 6) real and rendered RGBDDD
    - Output: N pose updates or N scores
    """

    def __init__(self, model_dir: str = "models", max_batch_size: int = 252,
                 opt_batch_size: int = 64, use_fp16: bool = True):
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
        """Build TensorRT engine from ONNX or load cached engine."""
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
            # Define shape for dynamic inputs. Non-input tensors (outputs) will adapt.
            min_shape = list(input_tensor.shape)
            opt_shape = list(input_tensor.shape)
            max_shape = list(input_tensor.shape)

            # NOTE: Assuming the dynamic dimension is the first one (batch size)
            min_shape[0] = 1
            opt_shape[0] = self.opt_batch_size
            max_shape[0] = self.max_batch_size

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        # In TensoRT 8.6+, build_serialized_network is deprecated
        # Using get_memory_size and build_engine for newer versions might be better
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"Failed to build {model_name} engine")

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized_engine)

    def _allocate_buffers(self, engine: trt.ICudaEngine, buffers: Dict):
        """Allocate CUDA memory for inputs/outputs."""
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # FIX: Handle dynamic dimensions (-1) in any position of the shape.
            # The original code only checked shape[0], causing an error if the
            # shape was, for example, [1, -1] for the scorer output.
            for i, dim in enumerate(shape):
                if dim == -1:
                    shape[i] = self.max_batch_size

            # Use np.int64 to prevent overflow on large models/batch sizes before converting to Python int
            size = int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)
            if size < 0:
                raise ValueError(f"Buffer size calculation resulted in a negative value: {size}")

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
                           rendered_rgbddd: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Refine N poses in batch.

        Args:
            poses: (N, 4, 4) pose matrices
            real_rgbddd: (N, 160, 160, 6) real observation (can be single repeated)
            rendered_rgbddd: (N, 160, 160, 6) rendered views
            iterations: Number of refinement iterations

        Returns:
            (N, 4, 4) refined poses
        """
        n_poses = len(poses)
        current_poses = poses.copy()

        # Ensure real observation is batched if a single one is provided
        if real_rgbddd.ndim == 3:
            real_rgbddd = np.repeat(real_rgbddd[np.newaxis], n_poses, axis=0)

        for iteration in range(iterations):
            # Process all poses in chunks that fit max_batch_size
            for start_idx in range(0, n_poses, self.max_batch_size):
                end_idx = min(start_idx + self.max_batch_size, n_poses)
                chunk_size = end_idx - start_idx

                # Get batch slice
                real_batch = real_rgbddd[start_idx:end_idx].astype(np.float32)
                rendered_batch = rendered_rgbddd[start_idx:end_idx].astype(np.float32)

                # Set dynamic shapes for the current chunk
                self.refiner_context.set_input_shape('input1', (chunk_size, 160, 160, 6))
                self.refiner_context.set_input_shape('input2', (chunk_size, 160, 160, 6))

                # Copy to GPU
                cuda.memcpy_htod_async(
                    self.refiner_buffers['input1']['device'],
                    real_batch, self.stream
                )
                cuda.memcpy_htod_async(
                    self.refiner_buffers['input2']['device'],
                    rendered_batch, self.stream
                )

                # Set tensor addresses for execution
                for name, buf in self.refiner_buffers.items():
                    self.refiner_context.set_tensor_address(name, int(buf['device']))

                # Execute
                self.refiner_context.execute_async_v3(stream_handle=self.stream.handle)

                # Prepare host arrays for output
                rotation_output = np.empty((chunk_size, 3), dtype=np.float32)
                translation_output = np.empty((chunk_size, 3), dtype=np.float32)

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
                for i in range(chunk_size):
                    idx = start_idx + i
                    current_poses[idx] = self._apply_pose_delta(
                        current_poses[idx], rotation_output[i], translation_output[i]
                    )

            # In a real pipeline, the newly refined poses would be used to
            # re-render the `rendered_rgbddd` inputs for the next iteration.

        return current_poses

    def score_poses_batch(self, real_rgbddd: np.ndarray,
                          rendered_rgbddd: np.ndarray) -> np.ndarray:
        """
        Score N poses in batch.

        Args:
            real_rgbddd: (N, 160, 160, 6) real observation
            rendered_rgbddd: (N, 160, 160, 6) rendered views

        Returns:
            (N,) scores
        """
        n_poses = len(rendered_rgbddd)
        all_scores = np.empty(n_poses, dtype=np.float32)

        # Ensure real observation is batched if a single one is provided
        if real_rgbddd.ndim == 3:
            real_rgbddd = np.repeat(real_rgbddd[np.newaxis], n_poses, axis=0)

        for start_idx in range(0, n_poses, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, n_poses)
            chunk_size = end_idx - start_idx

            # Get batch slice
            real_batch = real_rgbddd[start_idx:end_idx].astype(np.float32)
            rendered_batch = rendered_rgbddd[start_idx:end_idx].astype(np.float32)

            # Set dynamic shapes
            self.scorer_context.set_input_shape('input1', (chunk_size, 160, 160, 6))
            self.scorer_context.set_input_shape('input2', (chunk_size, 160, 160, 6))

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

            # CHANGE: The host output buffer is now shaped (chunk_size,) to align
            # with the docstring and simplify the logic. The original (1, chunk_size)
            # was slightly awkward.
            scores_output = np.empty(chunk_size, dtype=np.float32)

            # Copy scores from GPU
            cuda.memcpy_dtoh_async(
                scores_output,
                self.scorer_buffers['output1']['device'],
                self.stream
            )

            self.stream.synchronize()
            all_scores[start_idx:end_idx] = scores_output

        return all_scores

    def _apply_pose_delta(self, pose: np.ndarray, rotation_delta: np.ndarray,
                          translation_delta: np.ndarray) -> np.ndarray:
        """Apply pose update from refiner."""
        # This import is here to avoid a hard dependency on OpenCV if only used here.
        import cv2

        angle = np.linalg.norm(rotation_delta)
        # Check for near-zero rotation to avoid division by zero
        if angle > 1e-6:
            axis = rotation_delta / angle
            # cv2.Rodrigues expects a (3, 1) or (1, 3) vector
            R_delta, _ = cv2.Rodrigues(axis * angle)
        else:
            R_delta = np.eye(3)

        updated_pose = pose.copy()
        updated_pose[:3, :3] = R_delta @ pose[:3, :3]
        updated_pose[:3, 3] += translation_delta

        return updated_pose

    def warmup(self):
        """Warmup engines to ensure optimal performance on first real inference."""
        # Use a small, representative batch size for warmup
        warmup_batch_size = self.opt_batch_size if self.opt_batch_size > 1 else 16
        self.logger.info(f"Warming up with batch size {warmup_batch_size}...")

        dummy_rgbddd = np.random.randn(warmup_batch_size, 160, 160, 6).astype(np.float32)
        dummy_poses = np.tile(np.eye(4), (warmup_batch_size, 1, 1)).astype(np.float32)

        for _ in range(3):
            self.refine_poses_batch(dummy_poses, dummy_rgbddd, dummy_rgbddd, iterations=1)
            self.score_poses_batch(dummy_rgbddd, dummy_rgbddd)

        self.logger.info("Warmup complete")

    # Legacy wrappers
    def score_poses(self, poses: List[np.ndarray], real_rgbddd: np.ndarray,
                    rendered_rgbddd_list: List[np.ndarray]) -> np.ndarray:
        """Legacy interface for pipeline compatibility."""
        rendered_batch = np.stack(rendered_rgbddd_list)
        return self.score_poses_batch(real_rgbddd, rendered_batch)

    def refine_pose(self, pose: np.ndarray, real_rgbddd: np.ndarray,
                    rendered_rgbddd: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Legacy single pose interface."""
        refined = self.refine_poses_batch(
            np.array([pose]), real_rgbddd, np.array([rendered_rgbddd]), iterations
        )
        return refined[0]


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
        # The real observation is typically the same for all poses in a hypothesis batch
        real_batch = np.repeat(dummy_rgbddd[np.newaxis], n_poses, axis=0)
        rendered_batch = np.repeat(dummy_rgbddd[np.newaxis], n_poses, axis=0)

        # --- Test refinement (5 iterations) ---
        t0 = time.time()
        refined = interface.refine_poses_batch(poses_batch, real_batch, rendered_batch, iterations=5)
        t_refine = time.time() - t0

        # --- Test scoring ---
        t0 = time.time()
        scores = interface.score_poses_batch(real_batch, rendered_batch)
        t_score = time.time() - t0

        # --- Report results ---
        # Throughput in poses per second
        refine_hz = n_poses / t_refine
        score_hz = n_poses / t_score

        print(f"  Refine (5 iter): {t_refine*1000:.1f} ms ({refine_hz:.1f} poses/sec)")
        print(f"  Score:           {t_score*1000:.1f} ms ({score_hz:.1f} poses/sec)")
        print(f"  Total pipeline:  {(t_refine + t_score)*1000:.1f} ms")

        # Calculate throughput for a combined pipeline (one refine pass + one score pass)
        pipeline_hz = n_poses / (t_refine/5 + t_score) # NOTE: normalize refine time to 1 iteration
        print(f"  Pipeline throughput (1 iter refine + score): {pipeline_hz:.1f} poses/sec")

    print("\n" + "="*80)
    print("Tests complete!")
    print("="*80)
