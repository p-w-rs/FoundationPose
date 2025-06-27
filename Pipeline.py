import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional
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
    1. Load scene data
    2. Generate pose proposals
    3. Refine poses
    4. Score and select best pose
    """

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

        # Initialize components
        self.loader = LMODataLoader(base_path)
        self.depth_processor = DepthProcessor(self.loader.K)
        self.model_interface = ModelInterface()

        # Load models
        print("Loading ONNX models...")
        self.model_interface.load_models()

        # Object meshes cache
        self.meshes = {}
        self.generators = {}

    def load_object(self, object_id: int):
        """Load object mesh and create proposal generator"""
        if object_id not in self.meshes:
            mesh = self.loader.load_object_model(object_id, debug=False)
            self.meshes[object_id] = mesh
            self.generators[object_id] = PoseProposalGenerator(mesh)

    def estimate_pose(self, scene_id: int, frame_id: Optional[int] = None,
                     object_idx: int = 0, n_proposals: int = None,
                     refine_iterations: int = 3, verbose: bool = False) -> Dict:
        """
        Estimate object pose in scene.

        Args:
            scene_id: Scene ID
            frame_id: Frame ID (None = first available)
            object_idx: Which object in scene (0 = first)
            n_proposals: Number of pose proposals (None = use all)
            refine_iterations: Refinement iterations per proposal
            verbose: Print progress

        Returns:
            Dict with pose, score, timing info
        """
        # Load scene
        scene_data = self.loader.load_scene_data(scene_id, frame_id)
        object_id = scene_data['object_ids'][object_idx]
        mask = scene_data['masks'][object_idx]

        if mask is None:
            raise ValueError(f"No mask for object {object_idx}")

        # Load object
        self.load_object(object_id)
        mesh = self.meshes[object_id]
        generator = self.generators[object_id]

        # Generate proposals (use all if n_proposals not specified)
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

        # Generate homography transformations for cropping
        t0 = time.time()
        tf_to_crops = compute_crop_window_tf_batch(
            mesh.vertices,
            scene_data['rgb'].shape[0],
            scene_data['rgb'].shape[1],
            proposals,
            self.loader.K,
            crop_ratio=1.4,
            out_size=(160, 160),
            method='box_3d',
            mesh_diameter=generator.diameter
        )

        # Render at full resolution for all proposals
        print(f"\nRendering {len(proposals)} poses at full resolution...")
        rendered_rgbs = []
        rendered_depths = []

        for i, pose in enumerate(proposals):
            rgb_render, depth_render = renderer.render(
                pose, self.loader.K,
                scene_data['rgb'].shape[0],
                scene_data['rgb'].shape[1]
            )
            rendered_rgbs.append(rgb_render)
            rendered_depths.append(depth_render)

            if i % 50 == 0:
                print(f"  Rendered {i}/{len(proposals)} poses...")

        rendered_rgbs = np.array(rendered_rgbs)
        rendered_depths = np.array(rendered_depths)

        # Warp real image to crops
        real_rgb_batch = np.tile(scene_data['rgb'][None], (len(proposals), 1, 1, 1))
        real_depth_batch = np.tile(scene_data['depth'][None], (len(proposals), 1, 1))

        real_rgb_crops = warp_perspective_batch(
            real_rgb_batch, tf_to_crops, (160, 160), mode='bilinear'
        )
        real_depth_crops = warp_perspective_batch(
            real_depth_batch[..., None], tf_to_crops, (160, 160), mode='nearest'
        )
        # Remove the extra dimension added for warping
        if len(real_depth_crops.shape) == 4:
            real_depth_crops = real_depth_crops[..., 0]

        # Warp rendered images to crops
        rendered_rgb_crops = warp_perspective_batch(
            rendered_rgbs, tf_to_crops, (160, 160), mode='bilinear'
        )
        rendered_depth_crops = warp_perspective_batch(
            rendered_depths[..., None], tf_to_crops, (160, 160), mode='nearest'
        )
        # Remove the extra dimension added for warping
        if len(rendered_depth_crops.shape) == 4:
            rendered_depth_crops = rendered_depth_crops[..., 0]

        # Compute cropped intrinsics
        K_crops = []
        for tf in tf_to_crops:
            K_crop = tf @ self.loader.K
            K_crops.append(K_crop)
        K_crops = np.array(K_crops)

        t_render = time.time() - t0

        # Batch refine
        t0 = time.time()
        print(f"\nRefining {len(proposals)} poses...")

        # Get ground truth for error tracking
        gt_pose = scene_data['poses'][object_idx] if verbose else None

        # Track initial errors
        if verbose and gt_pose is not None:
            initial_errors = [np.linalg.norm(p[:3, 3] - gt_pose[:3, 3]) for p in proposals]
            print(f"Initial errors - best: {min(initial_errors):.1f}mm, avg: {np.mean(initial_errors):.1f}mm")

        # Refine each pose using cropped images
        refined_poses = []
        for i in range(len(proposals)):
            # Use averaged intrinsics for this crop
            K_crop_avg = K_crops[i]

            refined_pose = self.model_interface.refine_pose(
                proposals[i],
                real_rgb_crops[i],
                real_depth_crops[i],
                K_crop_avg,
                renderer,
                iterations=refine_iterations
            )
            refined_poses.append(refined_pose)

            if i % 50 == 0:
                print(f"  Refined {i}/{len(proposals)} poses...")

        refined_poses = np.array(refined_poses)
        t_refine = time.time() - t0

        # Batch score using same crops
        t0 = time.time()
        scores = []
        for i in range(len(refined_poses)):
            # Re-render at refined pose
            rgb_render, depth_render = renderer.render(
                refined_poses[i], self.loader.K,
                scene_data['rgb'].shape[0],
                scene_data['rgb'].shape[1]
            )

            # Warp to crop
            rgb_render_crop = cv2.warpPerspective(
                rgb_render, tf_to_crops[i], (160, 160), flags=cv2.INTER_LINEAR
            )
            depth_render_crop = cv2.warpPerspective(
                depth_render, tf_to_crops[i], (160, 160), flags=cv2.INTER_NEAREST
            )

            # Score
            inputs = self.model_interface.prepare_input(
                real_rgb_crops[i], real_depth_crops[i],
                rgb_render_crop, depth_render_crop,
                K_crops[i]
            )

            outputs = self.model_interface.scorer.run(None, inputs)
            score = outputs[0][0, 0] if len(outputs[0].shape) == 2 else outputs[0][0]
            scores.append(score)

        scores = np.array(scores)
        t_score = time.time() - t0

        # Select best
        best_idx = np.argmax(scores)
        best_pose = refined_poses[best_idx]
        best_score = scores[best_idx]

        # Show final errors
        if verbose and gt_pose is not None:
            final_errors = [np.linalg.norm(p[:3, 3] - gt_pose[:3, 3]) for p in refined_poses]
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


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize pipeline
    pipeline = FoundationPosePipeline()

    # Get available scenes
    scenes = pipeline.loader.get_available_scenes()
    if not scenes:
        print("No scenes found")
        exit()

    # Estimate pose
    print(f"\nEstimating pose for scene {scenes[0]}...")
    result = pipeline.estimate_pose(
        scene_id=scenes[0],
        n_proposals=None,  # Use all 252 proposals
        refine_iterations=5,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Object ID: {result['object_id']}")
    print(f"  Best score: {result['score']:.3f}")
    print(f"  Pose (translation mm): {result['pose'][:3, 3]}")
    print(f"\nTimings:")
    for k, v in result['timings'].items():
        print(f"  {k}: {v:.3f}s")

    # Compare with ground truth
    gt_pose = result['gt_pose']
    if gt_pose is not None:
        error = np.linalg.norm(result['pose'][:3, 3] - gt_pose[:3, 3])
        print(f"\nTranslation error: {error:.1f} mm")

    # Visualize scores
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(result['all_scores'])), result['all_scores'])
    plt.xlabel('Proposal Index')
    plt.ylabel('Score')
    plt.title('Pose Proposal Scores')
    plt.axhline(y=result['score'], color='r', linestyle='--',
                label=f'Best: {result["score"]:.3f}')
    plt.legend()
    plt.show()
