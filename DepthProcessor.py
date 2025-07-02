import numpy as np
from typing import Tuple, Optional
import logging

class DepthProcessor:
    """
    Convert RGB and depth images to RGBDDD format for FoundationPose models.

    Takes 160x160 RGB/depth from either:
    - Synthetic: PoseGenerator + MeshRenderer
    - Real: CropProcessor

    Outputs 6-channel RGBDDD:
    - Channels 0-2: RGB normalized to [0, 1]
    - Channels 3-5: XYZ point cloud in meters

    UNITS:
    - Input depth: meters (m)
    - Output XYZ: meters (m)
    - Camera intrinsics: pixels
    """

    def __init__(self):
        """Initialize depth processor."""
        self.logger = logging.getLogger(__name__)

    def depth_to_xyz(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Convert depth map to XYZ point cloud.

        Args:
            depth: (H, W) depth map in meters
            K: 3x3 camera intrinsic matrix

        Returns:
            (H, W, 3) XYZ point cloud in meters
        """
        h, w = depth.shape

        # Create pixel grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        # Unproject to 3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        # Stack into XYZ
        xyz = np.stack([x, y, z], axis=-1)

        return xyz

    def process_rgbd_to_rgbddd(self, rgb: np.ndarray, depth: np.ndarray,
                              K: np.ndarray, normalize_xyz: bool = True) -> np.ndarray:
        """
        Convert RGB-D to RGBDDD format for model input.

        Args:
            rgb: (H, W, 3) RGB image, uint8 [0-255]
            depth: (H, W) depth map in meters
            K: 3x3 camera intrinsic matrix
            normalize_xyz: Whether to normalize XYZ coordinates

        Returns:
            (H, W, 6) RGBDDD array:
            - [:,:,0:3]: RGB normalized to [0, 1]
            - [:,:,3:6]: XYZ in meters (normalized if requested)
        """
        # Normalize RGB to [0, 1]
        rgb_norm = rgb.astype(np.float32) / 255.0

        # Convert depth to XYZ
        xyz = self.depth_to_xyz(depth, K)

        # Normalize XYZ if requested
        if normalize_xyz:
            # Normalize based on typical object distances (0.3-1.0m)
            # This helps model convergence
            xyz_norm = xyz.copy()
            mask = depth > 0
            if np.any(mask):
                # Center around mean depth
                mean_z = depth[mask].mean()
                xyz_norm[..., 2] = (xyz_norm[..., 2] - mean_z) / mean_z
                # Scale X,Y by same factor
                xyz_norm[..., 0] = xyz_norm[..., 0] / mean_z
                xyz_norm[..., 1] = xyz_norm[..., 1] / mean_z
            xyz = xyz_norm

        # Stack RGBDDD
        rgbddd = np.concatenate([rgb_norm, xyz], axis=-1).astype(np.float32)

        return rgbddd

    def batch_process(self, rgbs: np.ndarray, depths: np.ndarray,
                     Ks: np.ndarray, normalize_xyz: bool = True) -> np.ndarray:
        """
        Process batch of RGB-D pairs.

        Args:
            rgbs: (N, H, W, 3) RGB images
            depths: (N, H, W) depth maps in meters
            Ks: (N, 3, 3) or (3, 3) camera intrinsics
            normalize_xyz: Whether to normalize XYZ

        Returns:
            (N, H, W, 6) batch of RGBDDD
        """
        batch_size = len(rgbs)
        h, w = rgbs.shape[1:3]

        # Handle single K for all images
        if Ks.ndim == 2:
            Ks = np.repeat(Ks[np.newaxis], batch_size, axis=0)

        # Process each image
        rgbddd_batch = np.zeros((batch_size, h, w, 6), dtype=np.float32)

        for i in range(batch_size):
            rgbddd_batch[i] = self.process_rgbd_to_rgbddd(
                rgbs[i], depths[i], Ks[i], normalize_xyz
            )

        return rgbddd_batch

    def prepare_model_input(self, real_rgbddd: np.ndarray,
                           rendered_rgbddd: np.ndarray) -> dict:
        """
        Prepare final input dict for models.

        Args:
            real_rgbddd: (H, W, 6) or (1, H, W, 6) real image
            rendered_rgbddd: (N, H, W, 6) rendered hypotheses

        Returns:
            Dict with 'input1' and 'input2' for model
        """
        # Ensure batch dimensions
        if real_rgbddd.ndim == 3:
            real_rgbddd = real_rgbddd[np.newaxis]

        # Expand real to match rendered batch size
        batch_size = len(rendered_rgbddd)
        if len(real_rgbddd) == 1 and batch_size > 1:
            real_rgbddd = np.repeat(real_rgbddd, batch_size, axis=0)

        return {
            'input1': real_rgbddd,      # Real observation
            'input2': rendered_rgbddd   # Rendered hypotheses
        }


# Unit tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2

    # Import all necessary local modules
    # These will be found if your files are in the same directory
    try:
        from CUDAContext import CUDAContextManager
        from DataLoader import DataLoader
        from MeshRenderer import MeshRenderer
        from PoseGenerator import PoseGenerator
        from CropProcessor import CropProcessor
    except ImportError as e:
        print(f"Could not import a required module: {e}")
        print("Please ensure all script files are in the same directory.")
        exit()

    logging.basicConfig(level=logging.INFO)

    # --- Visualization Helper Function ---
    def visualize_pipeline(
        rgb_crop: np.ndarray,
        depth_crop: np.ndarray,
        xyz_unnormalized: np.ndarray,
        xyz_normalized: np.ndarray,
        title: str,
        filename: str
    ):
        """Creates a comprehensive visualization for a single data pipeline."""
        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        fig.suptitle(title, fontsize=18, y=0.95)

        # --- Row 1: Inputs and Coordinate System ---
        # RGB Image
        axes[0, 0].imshow(rgb_crop)
        axes[0, 0].set_title("Input RGB (160x160)")
        axes[0, 0].axis('off')

        # Depth Image
        depth_vis = depth_crop.copy()
        depth_vis[depth_vis == 0] = np.nan
        im_depth = axes[0, 1].imshow(depth_vis, cmap='viridis')
        axes[0, 1].set_title("Input Depth (m)")
        axes[0, 1].axis('off')
        fig.colorbar(im_depth, ax=axes[0, 1], label="meters", fraction=0.046, pad=0.04)

        # 3D Axis Reference
        ax_3d = fig.add_subplot(3, 4, 3, projection='3d')
        ax_3d.set_title("Camera Coordinates")
        l = 1.0
        ax_3d.quiver(0,0,0, l,0,0, color='r', arrow_length_ratio=0.1)
        ax_3d.quiver(0,0,0, 0,l,0, color='g', arrow_length_ratio=0.1)
        ax_3d.quiver(0,0,0, 0,0,l, color='b', arrow_length_ratio=0.1)
        ax_3d.text(l, 0, 0, '+X (Right)', color='r'); ax_3d.text(0, l, 0, '+Y (Down)', color='g'); ax_3d.text(0, 0, l, '+Z (Forward)', color='b')
        ax_3d.set_xlim(0,l); ax_3d.set_ylim(0,l); ax_3d.set_zlim(0,l)
        ax_3d.view_init(elev=20, azim=-60); ax_3d.set_axis_off()
        axes[0, 2].set_visible(False)
        axes[0, 3].set_visible(False)


        # --- Row 2: Unnormalized XYZ ---
        vmax_abs = np.nanmax(np.abs(xyz_unnormalized))
        labels_unnorm = ['Unnormalized X', 'Unnormalized Y', 'Unnormalized Z']
        for i, label in enumerate(labels_unnorm):
            ax = axes[1, i]
            im = ax.imshow(xyz_unnormalized[:, :, i], cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
            ax.set_title(label)
            ax.axis('off')
            fig.colorbar(im, ax=ax, label='meters', fraction=0.046, pad=0.04)
        axes[1, 3].set_visible(False)

        # --- Row 3: Normalized XYZ ---
        vmax_norm = np.nanmax(np.abs(xyz_normalized))
        labels_norm = ['Normalized X', 'Normalized Y', 'Normalized Z']
        for i, label in enumerate(labels_norm):
            ax = axes[2, i]
            im = ax.imshow(xyz_normalized[:, :, i], cmap='RdBu_r', vmin=-vmax_norm, vmax=vmax_norm)
            ax.set_title(label)
            ax.axis('off')
            fig.colorbar(im, ax=ax, label='normalized units', fraction=0.046, pad=0.04)
        axes[2, 3].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Save figure
        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)
        save_path = viz_dir / filename
        plt.savefig(save_path, dpi=150)
        print(f"\n✅ Visualization saved to: {save_path}")
        plt.close(fig)


    # --- Main Execution Block ---
    manager = None
    try:
        print("="*80)
        print("Depth Processor Unit Tests")
        print("="*80)

        # Initialize CUDA context and processors
        print("Initializing CUDA context and data loaders...")
        manager = CUDAContextManager.get_instance()
        processor = DepthProcessor()
        loader = DataLoader("./data")
        crop_proc = CropProcessor()

        # --- Pipeline 1: Real Data ---
        print("\n--- Starting Real Data Pipeline ---")
        scenes = loader.get_available_scenes()
        if not scenes: raise RuntimeError("No scenes found for testing.")

        data = loader.load_frame_data(scenes[0], 0)
        mask_idx = next((i for i, m in enumerate(data['masks']) if m is not None and np.any(m)), -1)
        if mask_idx == -1: raise RuntimeError("No valid object mask found in the test frame.")

        cropped_real = crop_proc.apply_crop(data['rgb'], data['depth'], data['masks'][mask_idx], data['K'])

        rgbddd_real_unnormalized = processor.process_rgbd_to_rgbddd(cropped_real['rgb'], cropped_real['depth'], cropped_real['K'], normalize_xyz=False)
        rgbddd_real_normalized = processor.process_rgbd_to_rgbddd(cropped_real['rgb'], cropped_real['depth'], cropped_real['K'], normalize_xyz=True)

        visualize_pipeline(
            cropped_real['rgb'], cropped_real['depth'],
            rgbddd_real_unnormalized[:, :, 3:], rgbddd_real_normalized[:, :, 3:],
            title="Real Data Processing Pipeline",
            filename="depth_processor_real_test.png"
        )

        # --- Pipeline 2: Synthetic Data ---
        print("\n--- Starting Synthetic Data Pipeline ---")
        mesh = loader.load_object_model(1)
        pose_gen = PoseGenerator(mesh)
        renderer = MeshRenderer(mesh)

        synth_pose = pose_gen.generate_poses(n_poses=1)[0]

        # Render at full resolution to simulate a real capture
        rgb_synth_full, depth_synth_full = renderer.render(synth_pose, loader.K, 640, 480)
        mask_synth_full = (depth_synth_full > 0)

        # CRITICAL FIX: Apply the *same* cropping to synthetic data
        cropped_synth = crop_proc.apply_crop(rgb_synth_full, depth_synth_full, mask_synth_full, loader.K)

        rgbddd_synth_unnormalized = processor.process_rgbd_to_rgbddd(cropped_synth['rgb'], cropped_synth['depth'], cropped_synth['K'], normalize_xyz=False)
        rgbddd_synth_normalized = processor.process_rgbd_to_rgbddd(cropped_synth['rgb'], cropped_synth['depth'], cropped_synth['K'], normalize_xyz=True)

        visualize_pipeline(
            cropped_synth['rgb'], cropped_synth['depth'],
            rgbddd_synth_unnormalized[:, :, 3:], rgbddd_synth_normalized[:, :, 3:],
            title="Synthetic Data Processing Pipeline",
            filename="depth_processor_synthetic_test.png"
        )

    except Exception as e:
        logging.error(f"An error occurred during the test: {e}", exc_info=True)
    finally:
        # Guarantee that the CUDA context is cleaned up
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()

    print("\n" + "="*80)
    print("All tests complete.")
    print("="*80)
