# Main.py

import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm

# Import our custom modules
from DataLoader import DataLoader
from PoseGenerator import PoseGenerator
from MeshRenderer import MeshRenderer
from CropProcessor import CropProcessor
from DepthProcessor import DepthProcessor
from ModelInterface import ModelInterface
from CUDAContext import CUDAContextManager

def run_tracking_pipeline():
    """
    This function orchestrates the entire FoundationPose tracking pipeline.

    It initializes all necessary components, loads the data, and then loops
    through scenes and frames to perform pose estimation and tracking.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 1. Initialization ---
    # Initialize all the helper classes. When ModelInterface is created, it
    # will automatically get the singleton instance of the CUDAContextManager,
    # initializing the shared CUDA context for the entire application.
    loader = DataLoader("./data")
    crop_proc = CropProcessor()
    depth_proc = DepthProcessor()

    # The ModelInterface handles the TensorRT-based neural networks.
    interface = ModelInterface(max_batch_size=252, opt_batch_size=64, use_fp16=True)
    interface.load_models()
    interface.warmup() # Warm up the models to prevent a delay on the first frame.

    # --- 2. Main Processing Loop ---
    # Loop through each object and scene defined in the dataset.
    scenes = loader.get_available_scenes()
    for obj_id in loader.object_ids[:2]:  # Using the first two objects for this example
        try:
            mesh = loader.load_object_model(obj_id)
        except FileNotFoundError:
            print(f"Object model for ID {obj_id} not found, skipping.")
            continue

        # The MeshRenderer is tied to a specific object mesh.
        renderer = MeshRenderer(mesh)
        # The PoseGenerator creates initial pose hypotheses around the object.
        generator = PoseGenerator(mesh)

        # For the first frame, we generate a large number of poses to find the object.
        # These are generated once per object.
        initial_poses = generator.generate_poses()

        # Pre-render synthetic views for each of these initial poses. This is a one-time
        # cost per object, saving time during the tracking loop.
        syn_rgbddd = []
        print(f"Pre-rendering {len(initial_poses)} synthetic views for object {obj_id}...")
        for pose in tqdm(initial_poses):
            # Render a 160x160 image of the object at the given pose.
            rgb, depth = renderer.render(pose, loader.K, 160, 160)

            # The intrinsic matrix needs to be scaled for the 160x160 crop.
            K_160 = loader.K.copy()
            K_160[0, 0] *= 160/640; K_160[0, 2] *= 160/640
            K_160[1, 1] *= 160/480; K_160[1, 2] *= 160/480

            # Process the rendered RGB and Depth into the 6-channel RGBDDD format
            # that the neural network expects.
            rgbddd = depth_proc.process_rgbd_to_rgbddd(rgb, depth, K_160)
            syn_rgbddd.append(rgbddd)

        # Convert the list of images into a single numpy array for batch processing.
        syn_rgbddd_batch = np.array(syn_rgbddd, dtype=np.float32)

        # Now, process a scene frame by frame.
        scene_id = scenes[0]  # Using the first scene for this example
        frames = loader.get_scene_frames(scene_id)
        print(f"\nProcessing scene {scene_id} for object {obj_id} ({len(frames)} frames total)...")

        for frame_id in frames[:2]: # Using the first two frames for this example
            print(f"--- Frame {frame_id} ---")

            # --- 3. Per-Frame Processing ---
            frame_data = loader.load_frame_data(scene_id, frame_id)

            # Crop the real image around the object using its segmentation mask.
            cropped_frame = crop_proc.apply_crop(
                frame_data['rgb'], frame_data['depth'], frame_data['masks'][0], frame_data['K']
            )

            # Process the cropped real view into the 6-channel RGBDDD format.
            real_rgbddd = depth_proc.process_rgbd_to_rgbddd(
                cropped_frame['rgb'], cropped_frame['depth'], cropped_frame['K']
            ).astype(np.float32)

            # Duplicate the single real view to match the batch size of our pose hypotheses.
            real_rgbddd_batch = np.repeat(real_rgbddd[np.newaxis], len(initial_poses), axis=0)

            # --- 4. Pose Refinement ---
            # This is the core step: the model interface takes the initial poses, the real
            # image, and the corresponding synthetic images, and predicts the refined poses.
            print("Refining poses...")
            refined_poses = interface.refine_poses_batch(
                initial_poses.copy(), real_rgbddd_batch, syn_rgbddd_batch
            )

            # In a full implementation, you would now score these refined_poses,
            # select the best one, and use it as the starting point for the next frame.
            print(f"Successfully processed frame {frame_id}. Best pose would be selected here.")


if __name__ == "__main__":
    """
    The main entry point of the application.

    This block ensures that the application runs and, critically, that all
    CUDA resources are properly released upon exit, preventing errors.
    """
    manager = None
    try:
        # Run the main tracking pipeline.
        run_tracking_pipeline()
    except Exception as e:
        # Log any critical errors that occur during the pipeline.
        logging.error(f"A critical error occurred in the main pipeline: {e}", exc_info=True)
    finally:
        # This block is guaranteed to run, even if errors occur.
        # We must explicitly clean up the CUDA context to prevent resource leaks
        # and shutdown errors.
        try:
            # Get the singleton instance of the manager and clean it up.
            manager = CUDAContextManager.get_instance()
            if manager:
                print("\nCleaning up CUDA context...")
                manager.cleanup()
        except Exception as e:
            logging.error(f"Failed to clean up CUDA context: {e}")
