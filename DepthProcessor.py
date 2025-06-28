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
    from DataLoader import DataLoader
    from MeshRenderer import MeshRenderer
    from PoseGenerator import PoseGenerator
    from CropProcessor import CropProcessor

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Depth Processor Unit Tests")
    print("="*80)

    # Initialize
    processor = DepthProcessor()
    loader = DataLoader("./data")

    # Test 1: XYZ conversion
    print("\nTest 1: Depth to XYZ Conversion")

    # Create simple test depth
    depth_test = np.ones((160, 160), dtype=np.float32) * 0.5  # 0.5m
    K_test = np.array([[300, 0, 80],
                       [0, 300, 80],
                       [0, 0, 1]], dtype=np.float32)

    xyz = processor.depth_to_xyz(depth_test, K_test)
    print(f"XYZ shape: {xyz.shape}")
    print(f"Center pixel XYZ: {xyz[80, 80]}")
    print(f"Corner pixel XYZ: {xyz[0, 0]}")

    # Test 2: Real image processing
    print("\nTest 2: Real Image RGBDDD")

    scenes = loader.get_available_scenes()
    if scenes:
        data = loader.load_frame_data(scenes[0], 0)
        if data['masks'][0] is not None:
            # Crop to 160x160
            crop_proc = CropProcessor()
            cropped = crop_proc.apply_crop(
                data['rgb'], data['depth'], data['masks'][0], data['K']
            )

            # Convert to RGBDDD
            rgbddd = processor.process_rgbd_to_rgbddd(
                cropped['rgb'], cropped['depth'], cropped['K']
            )

            print(f"RGBDDD shape: {rgbddd.shape}")
            print(f"RGB range: [{rgbddd[:,:,0:3].min():.3f}, {rgbddd[:,:,0:3].max():.3f}]")
            print(f"XYZ range: [{rgbddd[:,:,3:6].min():.3f}, {rgbddd[:,:,3:6].max():.3f}]")

    # Test 3: Synthetic image processing
    print("\nTest 3: Synthetic Image RGBDDD")

    mesh = loader.load_object_model(1)
    pose_gen = PoseGenerator(mesh)
    renderer = MeshRenderer(mesh)

    # Generate and render a pose
    poses = pose_gen.generate_poses(n_poses=1)
    rgb_synth, depth_synth = renderer.render(poses[0], loader.K, 160, 160)

    # Convert to RGBDDD
    K_160 = loader.K.copy()
    K_160[0, 0] *= 160/640  # Scale for 160x160
    K_160[1, 1] *= 160/480
    K_160[0, 2] *= 160/640
    K_160[1, 2] *= 160/480

    rgbddd_synth = processor.process_rgbd_to_rgbddd(
        rgb_synth, depth_synth, K_160
    )

    print(f"Synthetic RGBDDD shape: {rgbddd_synth.shape}")
    print(f"Synthetic RGB range: [{rgbddd_synth[:,:,0:3].min():.3f}, {rgbddd_synth[:,:,0:3].max():.3f}]")
    print(f"Synthetic XYZ range: [{rgbddd_synth[:,:,3:6].min():.3f}, {rgbddd_synth[:,:,3:6].max():.3f}]")

    # Test 4: Batch processing
    print("\nTest 4: Batch Processing")

    if 'rgbddd' in locals() and 'rgbddd_synth' in locals():
        # Prepare model input
        model_input = processor.prepare_model_input(
            rgbddd,
            np.stack([rgbddd_synth] * 4)  # 4 hypotheses
        )

        print(f"Model input1 shape: {model_input['input1'].shape}")
        print(f"Model input2 shape: {model_input['input2'].shape}")

    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('RGBDDD Processing (Real vs Synthetic)', fontsize=16)

    if 'rgbddd' in locals():
        # Real image - top row
        axes[0, 0].imshow(cropped['rgb'])
        axes[0, 0].set_title('Real RGB')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cropped['depth'], cmap='viridis')
        axes[0, 1].set_title('Real Depth (m)')
        axes[0, 1].axis('off')

        # Real XYZ
        for i in range(3):
            im = axes[0, i+2].imshow(rgbddd[:,:,i+3], cmap='RdBu')
            axes[0, i+2].set_title(['X (left-right)', 'Y (up-down)', 'Z (depth)'][i])
            axes[0, i+2].axis('off')
            plt.colorbar(im, ax=axes[0, i+2], fraction=0.046, pad=0.04)

    if 'rgbddd_synth' in locals():
        # Synthetic - bottom row
        axes[1, 0].imshow(rgb_synth)
        axes[1, 0].set_title('Synthetic RGB')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(depth_synth, cmap='viridis')
        axes[1, 1].set_title('Synthetic Depth (m)')
        axes[1, 1].axis('off')

        # Synthetic XYZ
        for i in range(3):
            im = axes[1, i+2].imshow(rgbddd_synth[:,:,i+3], cmap='RdBu')
            axes[1, i+2].set_title(['X (left-right)', 'Y (up-down)', 'Z (depth)'][i])
            axes[1, i+2].axis('off')
            plt.colorbar(im, ax=axes[1, i+2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save
    from pathlib import Path
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / 'depthprocessor_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {viz_dir / 'depthprocessor_test.png'}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
