# DepthProcessor.py

import numpy as np
from typing import Tuple
import logging
from pathlib import Path

class DepthProcessor:
    """
    Converts cropped RGB and depth images into the 6-channel RGBDDD format
    required by the FoundationPose models.
    """

    def __init__(self):
        """Initializes the depth processor."""
        self.logger = logging.getLogger(__name__)

    def depth_to_xyz(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Converts a depth map into a 3D point cloud (XYZ coordinates).
        """
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.stack([X, Y, Z], axis=-1)

    def process_rgbd_to_rgbddd(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Converts a single RGB-D pair to the 6-channel RGBDDD format for model input.
        """
        rgb_norm = rgb.astype(np.float32) / 255.0
        xyz = self.depth_to_xyz(depth, K)
        return np.concatenate([rgb_norm, xyz], axis=-1).astype(np.float32)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        from DataLoader import DataLoader
        from CropProcessor import CropProcessor
        from MeshRenderer import MeshRenderer
        from PoseGenerator import PoseGenerator
        from CUDAContext import CUDAContextManager
    except ImportError as e:
        print(f"Error: A required module is missing to run the test: {e}")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("="*80)
    print("Depth Processor Standalone Test")
    print("="*80)

    manager = None
    try:
        # --- Initialization ---
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")
        cropper = CropProcessor()
        processor = DepthProcessor()
        mesh = loader.load_object_model(5)
        renderer = MeshRenderer(mesh)
        generator = PoseGenerator(mesh)

        # --- Test 1: Real Data Pipeline ---
        print("\n--- Testing with REAL data ---")
        scene_id = loader.get_available_scenes()[0]
        frame_id = loader.get_scene_frames(scene_id)[0]
        data = loader.load_frame_data(scene_id, frame_id)
        cropped_data = cropper.apply_crop(data['rgb'], data['depth'], data['masks'][1], data['K'])
        rgbddd_real = processor.process_rgbd_to_rgbddd(
            cropped_data['rgb'], cropped_data['depth'], cropped_data['K']
        )
        print("Successfully processed real data.")

        # --- Test 2: Synthetic Data Pipeline ---
        print("\n--- Testing with SYNTHETIC data ---")
        pose = generator.generate_poses(n_poses=1)[0]
        rgb_synth, depth_synth = renderer.render(pose, loader.K, 640, 480)
        mask_synth = depth_synth > 0
        cropped_synth = cropper.apply_crop(rgb_synth, depth_synth, mask_synth, loader.K)
        rgbddd_synth = processor.process_rgbd_to_rgbddd(
            cropped_synth['rgb'], cropped_synth['depth'], cropped_synth['K']
        )
        print("Successfully processed synthetic data.")

        # --- Visualization ---
        for i, (rgbddd, label) in enumerate([(rgbddd_real, "Real"), (rgbddd_synth, "Synthetic")]):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f'DepthProcessor Verification: {label} Data', fontsize=16)

            rgb = (rgbddd[:, :, :3] * 255).astype(np.uint8)
            xyz = rgbddd[:, :, 3:]
            vmax_xy = np.nanmax(np.abs(xyz[:,:,:2]))

            axes[0].imshow(rgb); axes[0].set_title("RGB Component"); axes[0].axis('off')

            # Use diverging colormap for X and Y, centered at 0
            im_x = axes[1].imshow(xyz[:,:,0], cmap='RdBu_r', vmin=-vmax_xy, vmax=vmax_xy)
            axes[1].set_title("X (meters)"); axes[1].axis('off'); fig.colorbar(im_x, ax=axes[1])

            im_y = axes[2].imshow(xyz[:,:,1], cmap='RdBu_r', vmin=-vmax_xy, vmax=vmax_xy)
            axes[2].set_title("Y (meters)"); axes[2].axis('off'); fig.colorbar(im_y, ax=axes[2])

            # --- VISUALIZATION FIX ---
            # Use the same RdBu_r colormap for Z as requested for consistency.
            # Since Z is always positive, it will only show the "red" part of the map.
            z_channel = xyz[:,:,2]
            im_z = axes[3].imshow(z_channel, cmap='RdBu_r', vmin=0)
            axes[3].set_title("Z (meters)"); axes[3].axis('off'); fig.colorbar(im_z, ax=axes[3])

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            save_path = Path("viz") / f'depthprocessor_test_{label.lower()}.png'
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved {label} data visualization to {save_path}")

        print("\n" + "="*80)
        print("DepthProcessor test completed successfully!")
        print("="*80)

    except Exception as e:
        logging.error(f"An error occurred during the standalone test: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
