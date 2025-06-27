import numpy as np
import torch
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
import time
import cv2
from pathlib import Path

class ModelInterface:
    """
    Interface for FoundationPose ONNX models.

    UNITS:
    - Poses: millimeters (mm)
    - Depth input: meters (converted from loader)
    - Model processing: millimeters

    Models expect 160x160 RGBD patches.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.refiner = None
        self.scorer = None
        self.input_size = 160

    def load_models(self):
        """Load both refiner and scorer models"""
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        # Use CoreML for Apple Silicon GPU
        if 'CoreMLExecutionProvider' in providers:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            print("Using CoreML (GPU/Neural Engine)")
        else:
            print("Using CPU (install onnxruntime-silicon for GPU support)")
            providers = ['CPUExecutionProvider']

        # Load refiner
        refiner_path = self.model_dir / "refine_model.onnx"
        if not refiner_path.exists():
            raise FileNotFoundError(f"Refiner not found: {refiner_path}")

        self.refiner = ort.InferenceSession(str(refiner_path), providers=providers)
        print(f"Loaded refiner: {refiner_path}")

        # Load scorer
        scorer_path = self.model_dir / "score_model.onnx"
        if not scorer_path.exists():
            raise FileNotFoundError(f"Scorer not found: {scorer_path}")

        self.scorer = ort.InferenceSession(str(scorer_path), providers=providers)
        print(f"Loaded scorer: {scorer_path}")

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print input/output shapes for both models"""
        print("\nRefiner model:")
        for inp in self.refiner.get_inputs():
            print(f"  Input: {inp.name} - {inp.shape}")
        for out in self.refiner.get_outputs():
            print(f"  Output: {out.name} - {out.shape}")

        print("\nScorer model:")
        for inp in self.scorer.get_inputs():
            print(f"  Input: {inp.name} - {inp.shape}")
        for out in self.scorer.get_outputs():
            print(f"  Output: {out.name} - {out.shape}")

    def prepare_input(self, real_rgb: np.ndarray, real_depth: np.ndarray,
                     rendered_rgb: np.ndarray, rendered_depth: np.ndarray,
                     K_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for models.

        Args:
            real_rgb: Real RGB (160, 160, 3) uint8
            real_depth: Real depth (160, 160) in meters
            rendered_rgb: Rendered RGB (160, 160, 3) uint8
            rendered_depth: Rendered depth (160, 160) in meters
            K_crop: Cropped camera intrinsics (3, 3)

        Returns:
            Dict with prepared inputs based on model requirements
        """
        # Convert depth to millimeters
        real_depth_mm = real_depth * 1000.0
        rendered_depth_mm = rendered_depth * 1000.0

        # Normalize RGB [0, 1]
        real_rgb_norm = real_rgb.astype(np.float32) / 255.0
        rendered_rgb_norm = rendered_rgb.astype(np.float32) / 255.0

        # Normalize depth (divide by 1000 to get back to ~1.0 range)
        real_depth_norm = real_depth_mm / 1000.0
        rendered_depth_norm = rendered_depth_mm / 1000.0

        # Ensure depth is 2D
        if len(real_depth_norm.shape) == 3:
            real_depth_norm = real_depth_norm[..., 0]
        if len(rendered_depth_norm.shape) == 3:
            rendered_depth_norm = rendered_depth_norm[..., 0]

        # Stack 6 channels in last dimension (N, 160, 160, 6)
        # Expand depth to 3 channels
        real_depth_expanded = np.repeat(real_depth_norm[..., np.newaxis], 3, axis=-1)
        rendered_depth_expanded = np.repeat(rendered_depth_norm[..., np.newaxis], 3, axis=-1)

        input1 = np.concatenate([
            real_rgb_norm,
            real_depth_expanded
        ], axis=-1)

        input2 = np.concatenate([
            rendered_rgb_norm,
            rendered_depth_expanded
        ], axis=-1)

        # Add batch dimension
        input1 = input1[np.newaxis, ...]  # (1, 160, 160, 6)
        input2 = input2[np.newaxis, ...]  # (1, 160, 160, 6)

        return {
            'input1': input1.astype(np.float32),
            'input2': input2.astype(np.float32)
        }

    def refine_pose(self, pose: np.ndarray, real_rgb: np.ndarray,
                   real_depth: np.ndarray, K_crop: np.ndarray,
                   mesh_renderer, iterations: int = 5) -> np.ndarray:
        """
        Iteratively refine pose.

        Args:
            pose: Initial 4x4 pose (mm)
            real_rgb: Real RGB (160, 160, 3)
            real_depth: Real depth (160, 160) meters
            K_crop: Cropped intrinsics
            mesh_renderer: Renderer for generating synthetic views
            iterations: Refinement iterations

        Returns:
            Refined 4x4 pose (mm)
        """
        if self.refiner is None:
            raise RuntimeError("Models not loaded")

        current_pose = pose.copy()

        for i in range(iterations):
            # Render at current pose
            rendered_rgb, rendered_depth = mesh_renderer.render(
                current_pose, K_crop, self.input_size, self.input_size
            )

            # Prepare inputs
            inputs = self.prepare_input(
                real_rgb, real_depth, rendered_rgb, rendered_depth, K_crop
            )

            # Run refiner
            outputs = self.refiner.run(None, inputs)

            # Model outputs rotation and translation updates
            rotation_delta = outputs[0][0]    # (3,)
            translation_delta = outputs[1][0]  # (3,)

            # Update pose
            current_pose = self._update_pose(current_pose, rotation_delta, translation_delta)

        return current_pose

    def score_poses(self, poses: List[np.ndarray], real_rgb: np.ndarray,
                   real_depth: np.ndarray, K_crop: np.ndarray,
                   mesh_renderer) -> np.ndarray:
        """
        Score multiple poses.

        Args:
            poses: List of 4x4 poses (mm)
            real_rgb: Real RGB (160, 160, 3)
            real_depth: Real depth (160, 160) meters
            K_crop: Cropped intrinsics
            mesh_renderer: Renderer

        Returns:
            Scores array
        """
        if self.scorer is None:
            raise RuntimeError("Models not loaded")

        scores = []

        for pose in poses:
            # Render
            rendered_rgb, rendered_depth = mesh_renderer.render(
                pose, K_crop, self.input_size, self.input_size
            )

            # Prepare inputs
            inputs = self.prepare_input(
                real_rgb, real_depth, rendered_rgb, rendered_depth, K_crop
            )

            # Run scorer
            outputs = self.scorer.run(None, inputs)
            score_array = outputs[0]  # Shape: (1, batch_size)
            scores.append(float(score_array[0, 0]))

        return np.array(scores)

    def refine_poses_batch(self, poses: np.ndarray, real_rgb: np.ndarray,
                          real_depth: np.ndarray, K_crop: np.ndarray,
                          mesh_renderer, iterations: int = 5,
                          gt_pose: np.ndarray = None, verbose: bool = True) -> np.ndarray:
        """
        Refine multiple poses in batch.

        Args:
            poses: Batch of 4x4 poses (N, 4, 4) in mm
            real_rgb: Real RGB (160, 160, 3)
            real_depth: Real depth (160, 160) meters
            K_crop: Cropped intrinsics
            mesh_renderer: Renderer
            iterations: Refinement iterations
            gt_pose: Ground truth pose for error tracking (in same frame as poses)
            verbose: Print progress

        Returns:
            Refined poses (N, 4, 4) in mm
        """
        if self.refiner is None:
            raise RuntimeError("Models not loaded")

        batch_size = len(poses)
        current_poses = poses.copy()

        for iteration in range(iterations):
            t_iter_start = time.time()

            # Prepare batch inputs
            input1_batch = []
            input2_batch = []

            for i in range(batch_size):
                # Render at current pose
                rendered_rgb, rendered_depth = mesh_renderer.render(
                    current_poses[i], K_crop, self.input_size, self.input_size
                )

                # Prepare inputs
                inputs = self.prepare_input(
                    real_rgb, real_depth, rendered_rgb, rendered_depth, K_crop
                )
                input1_batch.append(inputs['input1'][0])
                input2_batch.append(inputs['input2'][0])

            # Stack batch
            input1_batch = np.stack(input1_batch)
            input2_batch = np.stack(input2_batch)

            # Run refiner on batch
            outputs = self.refiner.run(None, {
                'input1': input1_batch,
                'input2': input2_batch
            })

            rotation_deltas = outputs[0]    # (N, 3)
            translation_deltas = outputs[1]  # (N, 3)

            # Update all poses
            for i in range(batch_size):
                current_poses[i] = self._update_pose(
                    current_poses[i],
                    rotation_deltas[i],
                    translation_deltas[i]
                )

            t_iter = time.time() - t_iter_start

            if verbose:
                # Track best pose error if GT available
                if gt_pose is not None:
                    errors = [np.linalg.norm(pose[:3, 3] - gt_pose[:3, 3])
                             for pose in current_poses]
                    best_error = min(errors)
                    avg_error = np.mean(errors)
                    print(f"  Iteration {iteration+1}/{iterations}: "
                          f"time={t_iter:.2f}s, best_error={best_error:.1f}mm, "
                          f"avg_error={avg_error:.1f}mm")
                else:
                    print(f"  Iteration {iteration+1}/{iterations}: time={t_iter:.2f}s")

        return current_poses

    def score_poses_batch(self, poses: np.ndarray, real_rgb: np.ndarray,
                         real_depth: np.ndarray, K_crop: np.ndarray,
                         mesh_renderer) -> np.ndarray:
        """Score multiple poses in batch."""
        if self.scorer is None:
            raise RuntimeError("Models not loaded")

        batch_size = len(poses)

        # Prepare batch inputs
        input1_batch = []
        input2_batch = []

        for i in range(batch_size):
            rendered_rgb, rendered_depth = mesh_renderer.render(
                poses[i], K_crop, self.input_size, self.input_size
            )

            inputs = self.prepare_input(
                real_rgb, real_depth, rendered_rgb, rendered_depth, K_crop
            )
            input1_batch.append(inputs['input1'][0])
            input2_batch.append(inputs['input2'][0])

        # Stack batch
        input1_batch = np.stack(input1_batch)
        input2_batch = np.stack(input2_batch)

        # Run scorer on batch
        outputs = self.scorer.run(None, {
            'input1': input1_batch,
            'input2': input2_batch
        })

        # FIX: Correct output indexing
        scores = outputs[0]  # Shape: (1, batch_size)
        if len(scores.shape) == 2 and scores.shape[0] == 1:
            scores = scores[0]  # Shape: (batch_size,)

        return scores

    def _update_pose(self, pose: np.ndarray, rotation_delta: np.ndarray,
                    translation_delta: np.ndarray) -> np.ndarray:
        """
        Update pose with model outputs.

        Args:
            pose: Current 4x4 pose (mm)
            rotation_delta: Rotation update (3,) - axis-angle
            translation_delta: Translation update (3,) - mm
        """
        # Translation update (mm)
        pose[:3, 3] += translation_delta

        # Rotation update (axis-angle to matrix)
        angle = np.linalg.norm(rotation_delta)
        if angle > 1e-6:
            axis = rotation_delta / angle
            # Rodrigues formula
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K
            pose[:3, :3] = R_delta @ pose[:3, :3]

        return pose


# Test
if __name__ == "__main__":
    interface = ModelInterface()

    try:
        interface.load_models()

        # Test input preparation
        dummy_rgb = np.zeros((160, 160, 3), dtype=np.uint8)
        dummy_depth = np.ones((160, 160), dtype=np.float32) * 0.5  # 0.5 meters
        dummy_K = np.eye(3, dtype=np.float32)

        inputs = interface.prepare_input(
            dummy_rgb, dummy_depth, dummy_rgb, dummy_depth, dummy_K
        )

        print(f"\nPrepared inputs:")
        for k, v in inputs.items():
            print(f"  {k}: {v.shape} ({v.dtype})")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Place ONNX models in 'models/' folder")
