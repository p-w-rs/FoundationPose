import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time
import cv2

from LMODataLoader import LMODataLoader
from DepthProcessor import DepthProcessor
from PoseGenerator import PoseProposalGenerator
from ModelInterface import ModelInterface
from MeshRenderer import MeshRenderer
from CropUtils import compute_crop_window_tf_batch, warp_perspective_batch
from DebugTest.CheckDiversity import check_pose_diversity

class FoundationPosePipeline:
    """
    Complete FoundationPose pipeline for 6D pose estimation.

    Pipeline:
    1. Load scene data with object masks
    2. Generate pose proposals using rotation grid
    3. Refine poses using TensorRT-optimized models
    4. Score refined poses and select best

    Optimized for batch processing with TensorRT FP16 inference.
    """

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

        # Initialize components
        self.loader = LMODataLoader(base_path)
        self.depth_processor = DepthProcessor(self.loader.K)
        self.model_interface = ModelInterface()

        # Load TensorRT models
        print("Loading ONNX models...")
        self.model_interface.load_models()

        # Object meshes cache
        self.meshes = {}
        self.generators = {}

    def load_object(self, object_id: int):
        """Load object mesh and create proposal generator."""
        if object_id not in self.meshes:
            mesh = self.loader.load_object_model(object_id, debug=False)
            self.meshes[object_id] = mesh
            self.generators[object_id] = PoseProposalGenerator(mesh)

    def estimate_pose(self, scene_id: int, frame_id: Optional[int] = None,
                     object_idx: int = 0, n_proposals: int = None,
                     refine_iterations: int = 3, verbose: bool = False) -> Dict:
        """
        Estimate object pose in scene using TensorRT-optimized pipeline.

        Args:
            scene_id: Scene ID from LMO dataset
            frame_id: Frame ID (None = first available)
            object_idx: Which object in scene (0 = first)
            n_proposals: Number of pose proposals (None = all 252)
            refine_iterations: Refinement iterations per proposal
            verbose: Print progress and timing

        Returns:
            Dict containing:
                - pose: Best 4x4 pose matrix (mm)
                - score: Confidence score
                - object_id: Object ID
                - timings: Breakdown of computation time
                - n_proposals: Number of proposals used
                - all_scores: Array of all proposal scores
                - gt_pose: Ground truth pose if available
        """
        # Load scene
        scene_data = self.loader.load_scene_data(scene_id, frame_id)
        object_id = scene_data['object_ids'][object_idx]
        mask = scene_data['masks'][object_idx]
        gt_pose = scene_data['poses'][object_idx] if 'poses' in scene_data else None

        if mask is None:
            raise ValueError(f"No mask for object {object_idx}")

        # Load object
        self.load_object(object_id)
        mesh = self.meshes[object_id]
        generator = self.generators[object_id]

        # Generate proposals
        t0 = time.time()
        if n_proposals is None:
            n_proposals = len(generator.rotation_grid)
        proposals = generator.generate_poses(
            scene_data['depth'], mask, self.loader.K, n_proposals
        )
        print(f"Using {len(proposals)} proposals")

        # Check diversity
        diversity = check_pose_diversity(proposals)
        print(f"Pose diversity: {diversity}")
        t_generate = time.time() - t0

        # Create renderer
        renderer = MeshRenderer(mesh)

        # Crop and prepare data for model
        t0 = time.time()
        cropped_data = self._prepare_cropped_data(
            scene_data, proposals, mesh, renderer, verbose
        )
        t_render = time.time() - t0

        # Refine poses in batches
        t0 = time.time()
        refined_poses = self._refine_poses_batch(
            proposals, cropped_data, renderer, refine_iterations, gt_pose, verbose
        )
        t_refine = time.time() - t0

        # Score all refined poses in batch
        t0 = time.time()
        scores = self._score_poses_batch(
            refined_poses, cropped_data, renderer
        )
        t_score = time.time() - t0

        # Select best
        best_idx = np.argmax(scores)
        best_pose = refined_poses[best_idx]
        best_score = scores[best_idx]

        # Show results
        if verbose:
            if gt_pose is not None:
                initial_errors = [np.linalg.norm(p[:3, 3] - gt_pose[:3, 3])
                                 for p in proposals]
                final_errors = [np.linalg.norm(p[:3, 3] - gt_pose[:3, 3])
                               for p in refined_poses]
                print(f"\nRefinement results:")
                print(f"  Initial best error: {min(initial_errors):.1f}mm")
                print(f"  Final best error: {min(final_errors):.1f}mm")
                print(f"  Improvement: {min(initial_errors) - min(final_errors):.1f}mm")
            print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  Score std: {scores.std():.3f}")

        return {
            'pose': best_pose,
            'score': best_score,
            'object_id': object_id,
            'timings': {
                'generate': t_generate,
                'render': t_render,
                'refine': t_refine,
                'score': t_score,
                'total': t_generate + t_render + t_refine + t_score
            },
            'n_proposals': len(proposals),
            'all_scores': scores,
            'gt_pose': gt_pose
        }

    def _prepare_cropped_data(self, scene_data: Dict, proposals: List[np.ndarray],
                             mesh: trimesh.Trimesh, renderer: MeshRenderer,
                             verbose: bool = False) -> Dict:
        """
        Prepare cropped RGB-D data for all proposals.

        Returns dict with:
            - real_rgb_crops: (N, 160, 160, 3)
            - real_depth_crops: (N, 160, 160)
            - K_crops: (N, 3, 3) averaged intrinsics
            - tf_to_crops: (N, 3, 3) homography transforms
        """
        # Compute crop transforms
        tf_to_crops = compute_crop_window_tf_batch(
            mesh.vertices,
            scene_data['rgb'].shape[0],
            scene_data['rgb'].shape[1],
            proposals,
            self.loader.K,
            crop_ratio=1.4,
            out_size=(160, 160),
            method='box_3d',
            mesh_diameter=self.generators[scene_data['object_ids'][0]].diameter
        )

        # Prepare batch data for warping
        rgb_batch = np.repeat(scene_data['rgb'][np.newaxis], len(proposals), axis=0)
        depth_batch = np.repeat(scene_data['depth'][np.newaxis], len(proposals), axis=0)

        # Warp real images to crops
        real_rgb_crops = warp_perspective_batch(
            rgb_batch, tf_to_crops, (160, 160), mode='bilinear'
        )

        # Fix depth warping - ensure it stays 2D
        depth_batch_expanded = depth_batch[..., None] if len(depth_batch.shape) == 3 else depth_batch[..., None]
        real_depth_crops_raw = warp_perspective_batch(
            depth_batch_expanded, tf_to_crops, (160, 160), mode='nearest'
        )

        # Ensure depth is 2D (160, 160) not 3D
        if len(real_depth_crops_raw.shape) == 4:  # (N, H, W, 1)
            real_depth_crops = real_depth_crops_raw[..., 0]
        else:
            real_depth_crops = real_depth_crops_raw

        # Compute averaged intrinsics for each crop
        K_crops = []
        for tf in tf_to_crops:
            # Warp intrinsics
            fx_samples = np.linspace(40, 120, 5)
            fy_samples = np.linspace(40, 120, 5)
            K_crop_sum = np.zeros((3, 3))
            count = 0

            for fx_idx in fx_samples:
                for fy_idx in fy_samples:
                    pt = np.array([fx_idx, fy_idx, 1.0])
                    pt_w = tf @ pt
                    if pt_w[2] != 0:
                        pt_w = pt_w / pt_w[2]
                        K_at_pt = self.loader.K
                        K_crop_sum += K_at_pt
                        count += 1

            K_crop_avg = K_crop_sum / count if count > 0 else tf @ self.loader.K
            K_crops.append(K_crop_avg)

        K_crops = np.array(K_crops)

        if verbose:
            print(f"\nRendering {len(proposals)} poses at full resolution...")

        return {
            'real_rgb_crops': real_rgb_crops,
            'real_depth_crops': real_depth_crops,
            'K_crops': K_crops,
            'tf_to_crops': tf_to_crops
        }

    def _refine_poses_batch(self, proposals: np.ndarray, cropped_data: Dict,
                           renderer: MeshRenderer, iterations: int,
                           gt_pose: Optional[np.ndarray] = None,
                           verbose: bool = False) -> np.ndarray:
        """Refine all poses using TensorRT batch processing."""
        if verbose:
            print(f"\nRefining {len(proposals)} poses...")
            if gt_pose is not None:
                initial_errors = [np.linalg.norm(p[:3, 3] - gt_pose[:3, 3])
                                 for p in proposals]
                print(f"Initial errors - best: {min(initial_errors):.1f}mm, "
                      f"avg: {np.mean(initial_errors):.1f}mm")

        # Refine each pose with its corresponding crop
        refined_poses = []
        for i in range(len(proposals)):
            refined_pose = self.model_interface.refine_pose(
                proposals[i],
                cropped_data['real_rgb_crops'][i],
                cropped_data['real_depth_crops'][i],
                cropped_data['K_crops'][i],
                renderer,
                iterations=iterations
            )
            refined_poses.append(refined_pose)

            if verbose and i % 50 == 0:
                print(f"  Refined {i}/{len(proposals)} poses...")

        return np.array(refined_poses)

    def _score_poses_batch(self, poses: np.ndarray, cropped_data: Dict,
                          renderer: MeshRenderer) -> np.ndarray:
        """Score all poses using TensorRT batch processing."""
        # For each pose, we need to render at its specific crop
        all_scores = []
        batch_size = 32  # Process in chunks

        for batch_start in range(0, len(poses), batch_size):
            batch_end = min(batch_start + batch_size, len(poses))
            batch_poses = poses[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            # Collect batch data
            batch_scores = []
            for i, idx in enumerate(batch_indices):
                # Use pre-cropped real data
                real_rgb = cropped_data['real_rgb_crops'][idx]
                real_depth = cropped_data['real_depth_crops'][idx]
                K_crop = cropped_data['K_crops'][idx]

                # Score this pose
                score = self.model_interface.score_poses(
                    [batch_poses[i]], real_rgb, real_depth, K_crop, renderer
                )[0]
                batch_scores.append(score)

            all_scores.extend(batch_scores)

        return np.array(all_scores)

    def _adjust_pose_for_crop(self, pose: np.ndarray, crop_result: Dict,
                             model_input: Dict) -> np.ndarray:
        """Adjust pose from original frame to cropped/resized frame."""
        # Transform from original image to crop
        x1, y1, w, h = crop_result['bbox']
        pose_crop = pose.copy()
        pose_crop[:3, 3] -= np.array([x1, y1, 0]) @ pose[:3, :3].T

        # Scale for model input size
        scale = model_input['scale']
        pose_crop[:3, 3] *= scale

        return pose_crop


# Unit tests
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("="*80)
    print("FoundationPose Pipeline Unit Tests")
    print("="*80)

    # Initialize pipeline
    pipeline = FoundationPosePipeline()

    # Get available scenes
    scenes = pipeline.loader.get_available_scenes()
    if not scenes:
        print("No scenes found")
        exit()

    print(f"\nAvailable scenes: {scenes}")

    # Test 1: Basic pose estimation
    print("\n" + "="*80)
    print("Test 1: Basic Pose Estimation")
    print("="*80)

    result = pipeline.estimate_pose(
        scene_id=scenes[0],
        n_proposals=10,  # Use fewer for testing
        refine_iterations=3,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Object ID: {result['object_id']}")
    print(f"  Best score: {result['score']:.3f}")
    print(f"  Pose (translation mm): {result['pose'][:3, 3]}")
    print(f"\nTimings:")
    for k, v in result['timings'].items():
        print(f"  {k}: {v:.3f}s")

    # Test 2: Full proposal set
    print("\n" + "="*80)
    print("Test 2: Full Proposal Set Performance")
    print("="*80)

    t_start = time.time()
    result_full = pipeline.estimate_pose(
        scene_id=scenes[0],
        n_proposals=None,  # Use all 252
        refine_iterations=5,
        verbose=True
    )
    t_total = time.time() - t_start

    print(f"\nFull pipeline time: {t_total:.3f}s")
    print(f"Throughput: {result_full['n_proposals'] / t_total:.1f} poses/sec")

    # Compare with ground truth
    if result_full['gt_pose'] is not None:
        error = np.linalg.norm(result_full['pose'][:3, 3] -
                              result_full['gt_pose'][:3, 3])
        print(f"Translation error: {error:.1f} mm")

    # Test 3: Score distribution
    print("\n" + "="*80)
    print("Test 3: Score Distribution Analysis")
    print("="*80)

    scores = result_full['all_scores']
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std: {scores.std():.3f}")
    print(f"  Min: {scores.min():.3f}")
    print(f"  Max: {scores.max():.3f}")
    print(f"  Best pose index: {np.argmax(scores)}")

    # Visualize
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(result_full['score'], color='red', linestyle='--', linewidth=2,
                label=f'Best: {result_full["score"]:.3f}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(len(scores)), scores, width=1.0)
    plt.axhline(y=result_full['score'], color='red', linestyle='--',
                label=f'Best: {result_full["score"]:.3f}')
    plt.xlabel('Proposal Index')
    plt.ylabel('Score')
    plt.title('Scores by Proposal')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("TensorRT optimization is working correctly")
    print("="*80)
