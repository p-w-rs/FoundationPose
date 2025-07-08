# ModelInterface.py

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import Dict, List
import logging
from pathlib import Path
import cv2
import time
from CUDAContext import CUDAContextManager

# Configure TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelInterface:
    """
    TensorRT interface for FoundationPose models.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.min_batch = 32
        self.opt_batch = 32
        self.max_batch = 252

        self.logger = logging.getLogger(__name__)
        self.cuda_mgr = CUDAContextManager.get_instance()

        self.refiner_engine = None
        self.scorer_engine = None
        self.refiner_context = None
        self.scorer_context = None

        self.refiner_bindings = []
        self.scorer_bindings = []

    def load_models(self):
        """Load both models and prepare for inference."""
        cache_dir = self.model_dir / 'trt_cache'
        cache_dir.mkdir(exist_ok=True)

        with self.cuda_mgr.activate_tensorrt():
            refiner_path = cache_dir / f"refiner_b{self.min_batch}_{self.opt_batch}_{self.max_batch}.trt"
            self.refiner_engine = self._get_engine("refine_model.onnx", refiner_path)
            self.refiner_context = self.refiner_engine.create_execution_context()
            self._setup_bindings(self.refiner_engine, self.refiner_context, self.refiner_bindings)

            scorer_path = cache_dir / f"scorer_b{self.min_batch}_{self.opt_batch}_{self.max_batch}.trt"
            self.scorer_engine = self._get_engine("score_model.onnx", scorer_path)
            self.scorer_context = self.scorer_engine.create_execution_context()
            self._setup_bindings(self.scorer_engine, self.scorer_context, self.scorer_bindings)

        self.logger.info("Models loaded successfully")

    def _run_model(self, context: trt.IExecutionContext, bindings: List, stream, real_obs: np.ndarray, rendered_obs: np.ndarray):
        """Helper to run inference."""
        inputs = {
            'input1': np.ascontiguousarray(real_obs, dtype=np.float32),
            'input2': np.ascontiguousarray(rendered_obs, dtype=np.float32)
        }

        context.set_input_shape('input1', real_obs.shape)
        context.set_input_shape('input2', rendered_obs.shape)

        for binding in bindings:
            if binding['is_input']:
                cuda.memcpy_htod_async(binding['address'], inputs[binding['name']], stream)

        if not context.execute_async_v3(stream_handle=stream.handle):
            raise RuntimeError("TensorRT execution failed.")

        outputs = {}
        for binding in bindings:
            if not binding['is_input']:
                output_shape = context.get_tensor_shape(binding['name'])
                host_mem = cuda.pagelocked_empty(tuple(output_shape), binding['dtype'])
                cuda.memcpy_dtoh_async(host_mem, binding['address'], stream)
                outputs[binding['name']] = host_mem

        stream.synchronize()
        return outputs

    def refine_poses(self, poses: np.ndarray, real_obs: np.ndarray, rendered_obs: np.ndarray) -> np.ndarray:
        """Refine poses to better match observations."""
        with self.cuda_mgr.activate_tensorrt():
            stream = self.cuda_mgr.get_pycuda_stream()
            outputs = self._run_model(self.refiner_context, self.refiner_bindings, stream, real_obs, rendered_obs)

        # Get outputs by order, not by name
        output_tensors = [outputs[key] for key in outputs]
        rot_deltas = output_tensors[0]
        trans_deltas = output_tensors[1]

        refined = poses.copy()
        for i in range(len(poses)):
            if np.linalg.norm(rot_deltas[i]) > 0:
                R_delta, _ = cv2.Rodrigues(rot_deltas[i])
            else:
                R_delta = np.eye(3)
            delta_transform = np.eye(4, dtype=np.float32)
            delta_transform[:3, :3] = R_delta
            delta_transform[:3, 3] = trans_deltas[i]
            refined[i] = delta_transform @ poses[i]
        return refined

    def score_poses(self, real_obs: np.ndarray, rendered_obs: np.ndarray) -> np.ndarray:
        """Score pose-observation alignment."""
        with self.cuda_mgr.activate_tensorrt():
            stream = self.cuda_mgr.get_pycuda_stream()
            outputs = self._run_model(self.scorer_context, self.scorer_bindings, stream, real_obs, rendered_obs)

        scores = next(iter(outputs.values()), None)
        if scores is None:
            raise RuntimeError("Could not find score output tensor.")

        scores = scores.flatten()
        if scores.max() < 0:
            scores = np.exp(scores)
        return scores

    def _get_engine(self, onnx_name: str, engine_path: Path) -> trt.ICudaEngine:
        """Get TensorRT engine, building if necessary."""
        if engine_path.exists():
            self.logger.info(f"Loading engine from {engine_path}")
            with trt.Runtime(TRT_LOGGER) as runtime:
                with open(engine_path, 'rb') as f:
                    return runtime.deserialize_cuda_engine(f.read())

        self.logger.info(f"Building engine for {onnx_name}")
        onnx_path = self.model_dir / onnx_name
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config:

            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    raise RuntimeError(f"Failed to parse {onnx_name}")

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                shape = tuple(tensor.shape[1:])
                profile.set_shape(tensor.name,
                                min=(self.min_batch,)+shape,
                                opt=(self.opt_batch,)+shape,
                                max=(self.max_batch,)+shape)
            config.add_optimization_profile(profile)

            self.logger.info("Building engine (may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Engine build failed")

            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            with trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(serialized_engine)

    def _setup_bindings(self, engine: trt.ICudaEngine, context: trt.IExecutionContext, bindings_list: List):
        """Allocate buffers for an engine."""
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            shape = list(engine.get_tensor_shape(name))
            shape[0] = self.max_batch

            size = abs(int(np.prod(shape) * np.dtype(dtype).itemsize))
            address = cuda.mem_alloc(size)

            bindings_list.append({'name': name, 'address': address, 'is_input': is_input, 'dtype': dtype})
            context.set_tensor_address(name, address)

    def warmup(self):
        """Warmup models."""
        self.logger.info("Warming up models...")
        dummy_obs = np.random.rand(self.opt_batch, 160, 160, 6).astype(np.float32)
        dummy_poses = np.tile(np.eye(4), (self.opt_batch, 1, 1)).astype(np.float32)
        self.score_poses(dummy_obs, dummy_obs)
        self.refine_poses(dummy_poses, dummy_obs, dummy_obs)
        self.logger.info("Warmup complete")

    def __del__(self):
        """Destructor to free CUDA memory."""
        for blist in [self.refiner_bindings, self.scorer_bindings]:
            for b in blist:
                if 'address' in b and hasattr(b['address'], 'free'):
                    b['address'].free()


# Test script
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from DataLoader import DataLoader
    from PoseRenderer import PoseRenderer
    from ImageProcessor import ImageProcessor

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("ModelInterface Test - Ground Truth Refinement Verification")
    print("="*80)

    manager = None
    interface = None

    try:
        # Initialize
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")
        mesh = loader.load_object_model(5)
        mesh_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        renderer = PoseRenderer(mesh)
        processor = ImageProcessor()
        interface = ModelInterface()
        interface.load_models()
        interface.warmup()

        # Load scene data
        print("\n[Loading Scene Data]")
        scene_id = 2
        frame_id = 0

        scene_info = loader.get_scene_info(scene_id)
        obj5_idx = None
        for idx, obj_info in enumerate(scene_info[str(frame_id)]):
            if obj_info['obj_id'] == 5:
                obj5_idx = idx
                break

        data = loader.load_frame_data(scene_id, frame_id, object_indices=[obj5_idx])
        gt_pose = data['poses'][0]

        real_processed = processor.process_image(
            data['rgb'], data['depth'], data['K'],
            mesh_diameter, mask=data['masks'][0]
        )
        real_rgbxyz = real_processed['rgbddd']

        print(f"Loaded GT pose from Scene {scene_id}, Frame {frame_id}")
        print(f"GT translation: {gt_pose[:3, 3]}")

        # Generate 162 perturbed poses
        print("\n[Generating 162 Perturbed Poses]")
        np.random.seed(42)
        N_POSES = 162

        test_poses = []
        perturbations = []

        for i in range(N_POSES):
            # Systematic perturbations
            if i < 54:  # Pure rotations
                axis = i % 3
                angle = (i // 3 - 9) * 0.01  # -0.09 to +0.09 radians
                rot_vec = np.zeros(3)
                rot_vec[axis] = angle
                trans_vec = np.zeros(3)
            elif i < 108:  # Pure translations
                axis = i % 3
                distance = ((i - 54) // 3 - 9) * 0.002  # -0.018 to +0.018 meters
                rot_vec = np.zeros(3)
                trans_vec = np.zeros(3)
                trans_vec[axis] = distance
            else:  # Combined
                rot_vec = np.random.uniform(-0.05, 0.05, 3)
                trans_vec = np.random.uniform(-0.01, 0.01, 3)

            # Apply perturbation
            delta = np.eye(4, dtype=np.float32)
            if np.linalg.norm(rot_vec) > 0:
                R_delta, _ = cv2.Rodrigues(rot_vec)
                delta[:3, :3] = R_delta
            delta[:3, 3] = trans_vec

            perturbed_pose = delta @ gt_pose
            test_poses.append(perturbed_pose)
            perturbations.append({'rot': rot_vec, 'trans': trans_vec})

        test_poses = np.array(test_poses)

        # Process all poses
        print("Processing all poses...")
        rendered_rgbxyz = []
        for pose in test_poses:
            render = renderer.render(pose, data['K'])
            processed = processor.process_image(
                render['rgb'], render['depth'], data['K'],
                mesh_diameter, pose=pose, mesh_bounds=mesh.bounds
            )
            rendered_rgbxyz.append(processed['rgbddd'])

        rendered_rgbxyz = np.array(rendered_rgbxyz, dtype=np.float32)
        real_batch = np.repeat(real_rgbxyz[np.newaxis], N_POSES, axis=0)

        # Score and refine
        print("\n[Initial Scoring]")
        initial_scores = interface.score_poses(real_batch, rendered_rgbxyz)
        print(f"Initial scores - Mean: {initial_scores.mean():.3f}, Std: {initial_scores.std():.3f}")

        print("\n[Refining Poses]")
        refined_poses = interface.refine_poses(test_poses, real_batch, rendered_rgbxyz)

        # Render refined
        print("Rendering refined poses...")
        refined_rgbxyz = []
        for pose in refined_poses:
            render = renderer.render(pose, data['K'])
            processed = processor.process_image(
                render['rgb'], render['depth'], data['K'],
                mesh_diameter, pose=pose, mesh_bounds=mesh.bounds
            )
            refined_rgbxyz.append(processed['rgbddd'])

        refined_rgbxyz = np.array(refined_rgbxyz, dtype=np.float32)
        refined_scores = interface.score_poses(real_batch, refined_rgbxyz)
        print(f"Refined scores - Mean: {refined_scores.mean():.3f}, Std: {refined_scores.std():.3f}")

        # Verification
        print("\n[Mathematical Verification]")

        def pose_error(p1, p2):
            t_err = np.linalg.norm(p1[:3, 3] - p2[:3, 3])
            R_rel = p1[:3, :3].T @ p2[:3, :3]
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            return t_err, np.degrees(angle)

        initial_errors = np.array([pose_error(p, gt_pose) for p in test_poses])
        refined_errors = np.array([pose_error(p, gt_pose) for p in refined_poses])

        score_improved = (refined_scores > initial_scores).sum()
        trans_improved = (refined_errors[:, 0] < initial_errors[:, 0]).sum()
        rot_improved = (refined_errors[:, 1] < initial_errors[:, 1]).sum()

        print(f"\nImprovement Statistics:")
        print(f"  Scores improved: {score_improved}/{N_POSES} ({score_improved/N_POSES*100:.1f}%)")
        print(f"  Translation improved: {trans_improved}/{N_POSES} ({trans_improved/N_POSES*100:.1f}%)")
        print(f"  Rotation improved: {rot_improved}/{N_POSES} ({rot_improved/N_POSES*100:.1f}%)")

        print(f"\nError Reduction:")
        print(f"  Mean translation: {initial_errors[:, 0].mean()*100:.2f} cm → {refined_errors[:, 0].mean()*100:.2f} cm")
        print(f"  Mean rotation: {initial_errors[:, 1].mean():.2f}° → {refined_errors[:, 1].mean():.2f}°")

        # Visualization
        print("\n[Creating Visualization]")
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Pose Refinement Verification', fontsize=20)

        # Select examples
        score_improvements = refined_scores - initial_scores
        best_idx = np.argmax(score_improvements)
        worst_idx = np.argmin(score_improvements)
        examples = [0, best_idx, worst_idx, 54, 108]  # Identity, best, worst, translation, combined

        # Create grid
        gs = fig.add_gridspec(3, 5, hspace=0.3)

        # Show examples
        for col, idx in enumerate(examples):
            # Ground truth
            ax0 = fig.add_subplot(gs[0, col])
            ax0.imshow(real_rgbxyz[:, :, :3])
            ax0.set_title(f'GT (Example {idx+1})', fontsize=10)
            ax0.axis('off')

            # Initial
            ax1 = fig.add_subplot(gs[1, col])
            ax1.imshow(rendered_rgbxyz[idx, :, :, :3])
            ax1.set_title(f'Initial\nScore: {initial_scores[idx]:.3f}', fontsize=10)
            ax1.axis('off')

            # Refined
            ax2 = fig.add_subplot(gs[2, col])
            ax2.imshow(refined_rgbxyz[idx, :, :, :3])
            improvement = score_improvements[idx]
            color = 'green' if improvement > 0 else 'red'
            ax2.set_title(f'Refined\nScore: {refined_scores[idx]:.3f} (Δ{improvement:+.3f})',
                         fontsize=10, color=color)
            ax2.axis('off')

        plt.tight_layout()
        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'modelinterface_verification.png', dpi=150, bbox_inches='tight')
        print("Saved visualization")

        # Performance test
        print("\n[Performance Test]")
        for batch_size in [32, 64, 162, 252]:
            dummy_obs = np.random.rand(batch_size, 160, 160, 6).astype(np.float32)
            dummy_poses = np.tile(np.eye(4), (batch_size, 1, 1)).astype(np.float32)

            t0 = time.time()
            _ = interface.score_poses(dummy_obs, dummy_obs)
            t_score = (time.time() - t0) * 1000

            t0 = time.time()
            _ = interface.refine_poses(dummy_poses, dummy_obs, dummy_obs)
            t_refine = (time.time() - t0) * 1000

            print(f"Batch {batch_size:3d}: Score {t_score:6.1f} ms ({t_score/batch_size:4.2f} ms/item), "
                  f"Refine {t_refine:6.1f} ms ({t_refine/batch_size:4.2f} ms/item)")

        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)

    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
    finally:
        del interface
        if manager:
            manager.cleanup()
