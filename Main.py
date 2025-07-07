# Main.py

import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from PoseGenerator import PoseGenerator
from MeshRenderer import MeshRenderer
from CropProcessor import CropProcessor
from DepthProcessor import DepthProcessor
from ModelInterface import ModelInterface
from CUDAContext import CUDAContextManager

def pose_to_rgbddd(poses, loader, renderer, cropper, depther):
    """
    Takes a batch of poses, renders them, and processes them into RGBDDD format.
    """
    rgbddds = []
    for pose in tqdm(poses, desc="Rendering Poses", leave=False):
        rgb, depth = renderer.render(pose, loader.K, 640, 480)
        mask = depth > 0
        cropped = cropper.apply_crop(rgb, depth, mask, loader.K)
        rgbddd = depther.process_rgbd_to_rgbddd(cropped['rgb'], cropped['depth'], cropped['K'])
        rgbddds.append(rgbddd)
    return np.array(rgbddds, dtype=np.float32)

def visualize_iteration(iteration, real_rgbddd, synth_rgbddd_top, scores_top, save_dir="viz"):
    """
    Saves a visualization of the top 8 poses for a given iteration.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    n_poses = len(synth_rgbddd_top)
    fig, axes = plt.subplots(2, n_poses + 1, figsize=((n_poses + 1) * 2.5, 6))
    fig.suptitle(f"Iteration {iteration} | Best Score: {np.max(scores_top):.2f}", fontsize=16)

    axes[0, 0].imshow(real_rgbddd[0, :, :, :3]); axes[0, 0].set_title("Real RGB"); axes[0, 0].axis('off')
    all_depths = np.concatenate([real_rgbddd[..., 5:6], synth_rgbddd_top[..., 5:6]], axis=0)
    valid_depths = all_depths[all_depths > 0]; vmin, vmax = (valid_depths.min(), valid_depths.max()) if valid_depths.size > 0 else (0, 1)
    axes[1, 0].imshow(real_rgbddd[0, :, :, 5], cmap='viridis', vmin=vmin, vmax=vmax); axes[1, 0].set_title("Real Depth"); axes[1, 0].axis('off')

    for i in range(n_poses):
        axes[0, i + 1].imshow(synth_rgbddd_top[i, :, :, :3]); axes[0, i + 1].set_title(f"Score: {scores_top[i]:.2f}"); axes[0, i + 1].axis('off')
        axes[1, i + 1].imshow(synth_rgbddd_top[i, :, :, 5], cmap='viridis', vmin=vmin, vmax=vmax); axes[1, i + 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(save_dir / f"iteration_{iteration:02d}.png"); plt.close(fig)

def run_tracking_pipeline():
    """
    Orchestrates the full FoundationPose pipeline with a correct iterative
    refinement loop, including pruning and jittering.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    N_ITERATIONS = 5
    N_POSES_TOTAL = 252
    TOP_K = 32
    N_JITTERED = N_POSES_TOTAL - TOP_K

    loader = DataLoader("./data")
    cropper = CropProcessor()
    depther = DepthProcessor()
    interface = ModelInterface(max_batch_size=N_POSES_TOTAL)
    interface.load_models()
    interface.warmup()
    mesh = loader.load_object_model(5)
    generator = PoseGenerator(mesh)
    renderer = MeshRenderer(mesh)

    scene_id = loader.get_available_scenes()[0]
    frame_id = loader.get_scene_frames(scene_id)[0]
    frame = loader.load_frame_data(scene_id, frame_id)

    cropped_real = cropper.apply_crop(frame['rgb'], frame['depth'], frame['masks'][1], frame['K'])
    real_rgbddd = depther.process_rgbd_to_rgbddd(cropped_real['rgb'], cropped_real['depth'], cropped_real['K'])

    poses = generator.generate_poses(n_poses=N_POSES_TOTAL)

    for i in range(N_ITERATIONS + 1):
        synth_rgbddd = pose_to_rgbddd(poses, loader, renderer, cropper, depther)
        real_rgbddd_batch = np.repeat(real_rgbddd[np.newaxis], len(poses), axis=0)

        scores = interface.score_poses_batch(real_rgbddd_batch, synth_rgbddd)

        top_8_indices = np.argsort(scores)[-8:]
        visualize_iteration(i, real_rgbddd_batch, synth_rgbddd[top_8_indices], scores[top_8_indices])
        logging.info(f"Iteration {i}: Best score = {np.max(scores):.4f}")

        if i == N_ITERATIONS:
            break

        top_k_indices = np.argsort(scores)[-TOP_K:]

        refined_poses = interface.refine_poses_batch(
            poses[top_k_indices],
            real_rgbddd_batch[top_k_indices],
            synth_rgbddd[top_k_indices]
        )

        best_refined_pose = refined_poses[-1]
        jittered_poses = generator.generate_nearby_poses(best_refined_pose, n_poses=N_JITTERED)

        poses = np.vstack([refined_poses, jittered_poses])

if __name__ == "__main__":
    manager = None
    try:
        run_tracking_pipeline()
    except Exception as e:
        logging.error(f"A critical error occurred in the main pipeline: {e}", exc_info=True)
    finally:
        if (manager := CUDAContextManager.get_instance()) is not None:
            logging.info("Cleaning up CUDA context...")
            manager.cleanup()
