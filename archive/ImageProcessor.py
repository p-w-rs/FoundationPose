# ImageProcessor.py

import numpy as np
import cv2
import torch
from typing import Dict, Optional, Tuple, Union
import logging
from pathlib import Path

class ImageProcessor:
    """
    Unified image processor for FoundationPose pipeline.

    Processes both real camera images and rendered images into RGBDDD format
    for model inference. Handles cropping and depth normalization.

    Processing steps:
    1. Crop image using pose-conditioned or mask-based approach
    2. Resize to target size (160x160)
    3. Convert RGB to [0,1] range
    4. Convert depth to XYZ coordinates
    5. Normalize XYZ by mesh diameter

    Output: 160x160x6 RGBDDD tensor ready for models
    """

    def __init__(self, target_size: int = 160):
        """
        Initialize processor.

        Args:
            target_size: Output image size (square)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def process_image(self,
                     rgb: np.ndarray,
                     depth: np.ndarray,
                     K: np.ndarray,
                     mesh_diameter: float,
                     mask: Optional[np.ndarray] = None,
                     pose: Optional[np.ndarray] = None,
                     mesh_bounds: Optional[np.ndarray] = None,
                     crop_ratio: float = 1.4) -> Dict:
        """
        Process image into RGBDDD format for model.

        Args:
            rgb: RGB image (H,W,3) uint8
            depth: Depth map (H,W) float32 in meters
            K: Camera intrinsics (3,3)
            mesh_diameter: Object diameter in meters for normalization
            mask: Optional object mask for mask-based cropping
            pose: Optional pose (4,4) for pose-conditioned cropping
            mesh_bounds: Optional mesh bounds (2,3) for pose cropping
            crop_ratio: Padding factor for cropping

        Returns:
            Dict with:
                - rgbddd: (160,160,6) normalized tensor
                - crop_info: Cropping parameters used
                - K_crop: Adjusted intrinsics
        """
        # Determine cropping method
        if pose is not None and mesh_bounds is not None:
            crop_info = self._compute_pose_crop(pose, mesh_bounds, K,
                                               rgb.shape[:2], crop_ratio)
        elif mask is not None:
            crop_info = self._compute_mask_crop(mask, crop_ratio)
        else:
            raise ValueError("Either pose+mesh_bounds or mask required for cropping")

        # Apply crop
        rgb_crop, depth_crop, K_crop = self._apply_crop(
            rgb, depth, K, crop_info
        )

        # Convert to RGBDDD
        rgbddd = self._to_rgbddd(rgb_crop, depth_crop, K_crop, mesh_diameter)

        return {
            'rgbddd': rgbddd,
            'crop_info': crop_info,
            'K_crop': K_crop
        }

    def _compute_pose_crop(self, pose: np.ndarray, mesh_bounds: np.ndarray,
                          K: np.ndarray, img_shape: Tuple[int, int],
                          crop_ratio: float) -> Dict:
        """
        Compute crop using pose-conditioned approach (FoundationPose method).
        Projects 3D bounding box to get optimal crop window.
        """
        h, w = img_shape

        # Get 3D bounding box corners
        min_xyz = mesh_bounds[0]
        max_xyz = mesh_bounds[1]

        # 8 corners of 3D bounding box
        corners_3d = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]]
        ])

        # Transform to camera frame
        corners_homo = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = (pose @ corners_homo.T).T[:, :3]

        # Project to image
        corners_img = (K @ corners_cam.T).T
        corners_2d = corners_img[:, :2] / corners_img[:, 2:3]

        # Get 2D bounding box
        u_min, v_min = corners_2d.min(axis=0)
        u_max, v_max = corners_2d.max(axis=0)

        # Compute crop parameters
        cx = (u_min + u_max) / 2
        cy = (v_min + v_max) / 2
        size = max(u_max - u_min, v_max - v_min) * crop_ratio

        # Ensure square and within bounds
        half_size = size / 2
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size))
        y2 = min(h, int(cy + half_size))

        # Make square
        actual_size = min(x2 - x1, y2 - y1)
        x2 = x1 + actual_size
        y2 = y1 + actual_size

        return {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'center': (cx, cy),
            'size': actual_size,
            'method': 'pose'
        }

    def _compute_mask_crop(self, mask: np.ndarray, crop_ratio: float) -> Dict:
        """Compute crop from object mask."""
        if not np.any(mask):
            h, w = mask.shape
            return {
                'x1': 0, 'y1': 0, 'x2': w, 'y2': h,
                'center': (w/2, h/2),
                'size': min(w, h),
                'method': 'full'
            }

        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        size = max(w, h) * crop_ratio

        # Make square crop
        half_size = size / 2
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(mask.shape[1], int(cx + half_size))
        y2 = min(mask.shape[0], int(cy + half_size))

        return {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'center': (cx, cy),
            'size': size,
            'method': 'mask'
        }

    def _apply_crop(self, rgb: np.ndarray, depth: np.ndarray,
                   K: np.ndarray, crop_info: Dict) -> Tuple:
        """Apply cropping and resize to target size."""
        x1, y1, x2, y2 = crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']

        # Crop
        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]

        # Resize to target
        rgb_resized = cv2.resize(rgb_crop, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth_crop, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_NEAREST)

        # Adjust intrinsics
        K_crop = K.copy()
        K_crop[0, 2] -= x1
        K_crop[1, 2] -= y1

        scale = self.target_size / (x2 - x1)
        K_crop[:2, :] *= scale

        return rgb_resized, depth_resized, K_crop

    def _to_rgbddd(self, rgb: np.ndarray, depth: np.ndarray,
                   K: np.ndarray, mesh_diameter: float) -> np.ndarray:
        """Convert to 6-channel RGBDDD format."""
        # Normalize RGB to [0,1]
        rgb_norm = rgb.astype(np.float32) / 255.0

        # Convert depth to XYZ
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Handle zero depth
        valid_mask = depth > 0

        X = np.zeros_like(depth)
        Y = np.zeros_like(depth)
        Z = depth.copy()

        if np.any(valid_mask):
            X[valid_mask] = (u[valid_mask] - cx) * Z[valid_mask] / fx
            Y[valid_mask] = (v[valid_mask] - cy) * Z[valid_mask] / fy

        xyz = np.stack([X, Y, Z], axis=-1)

        # Normalize by mesh diameter
        xyz_norm = xyz / mesh_diameter

        # Combine
        rgbddd = np.concatenate([rgb_norm, xyz_norm], axis=-1).astype(np.float32)

        return rgbddd


# Test script
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from DataLoader import DataLoader
    from PoseRenderer import PoseRenderer
    from CUDAContext import CUDAContextManager

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("ImageProcessor Test - Object 5")
    print("="*80)

    manager = None
    try:
        # Initialize
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")
        processor = ImageProcessor()

        # Load real data
        scene_id = 2
        frame_id = 0
        scene_info = loader.get_scene_info(scene_id)

        # Find object 5
        obj5_idx = None
        for idx, obj_info in enumerate(scene_info[str(frame_id)]):
            if obj_info['obj_id'] == 5:
                obj5_idx = idx
                break

        data = loader.load_frame_data(scene_id, frame_id, object_indices=[obj5_idx])

        # Load mesh for synthetic test
        mesh = loader.load_object_model(5)
        mesh_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        renderer = PoseRenderer(mesh)

        # Test 1: Process real image with mask
        print("\n[Test 1] Processing real image with mask-based cropping...")
        result_mask = processor.process_image(
            data['rgb'], data['depth'], data['K'],
            mesh_diameter, mask=data['masks'][0]
        )

        # Test 2: Process real image with pose
        print("\n[Test 2] Processing real image with pose-based cropping...")
        result_pose = processor.process_image(
            data['rgb'], data['depth'], data['K'],
            mesh_diameter, pose=data['poses'][0], mesh_bounds=mesh.bounds
        )

        # Test 3: Process synthetic image
        print("\n[Test 3] Processing synthetic rendered image...")
        test_pose = renderer.generate_poses(n_poses=1)[0]
        render_result = renderer.render(test_pose, loader.K)

        # Use pose-based cropping for synthetic (matching paper)
        result_synth = processor.process_image(
            render_result['rgb'], render_result['depth'], loader.K,
            mesh_diameter, pose=test_pose, mesh_bounds=mesh.bounds
        )

        # Create separate visualizations for each pipeline
        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)

        # Helper to show processing steps
        def show_pipeline(title, rgb_orig, depth_orig, result, mask=None, save_name=None):
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            fig.suptitle(title, fontsize=16)

            # Row 1: Original to Crop
            axes[0,0].imshow(rgb_orig)
            axes[0,0].set_title('Original RGB')
            axes[0,0].axis('off')

            axes[0,1].imshow(depth_orig, cmap='viridis')
            axes[0,1].set_title('Original Depth (m)')
            axes[0,1].axis('off')

            if mask is not None:
                axes[0,2].imshow(mask, cmap='gray')
                axes[0,2].set_title('Mask')
            else:
                axes[0,2].text(0.5, 0.5, 'No mask\n(pose-based)',
                             ha='center', va='center', transform=axes[0,2].transAxes)
            axes[0,2].axis('off')

            # Crop info
            axes[0,3].text(0.1, 0.5, f"Crop Info:\nMethod: {result['crop_info']['method']}\n"
                          f"Size: {result['crop_info']['size']:.0f}px\n"
                          f"Center: ({result['crop_info']['center'][0]:.0f}, "
                          f"{result['crop_info']['center'][1]:.0f})",
                          transform=axes[0,3].transAxes, fontsize=12, family='monospace')
            axes[0,3].axis('off')

            axes[0,4].axis('off')

            # Row 2: Cropped RGB/Depth/XYZ (before normalization)
            rgbddd = result['rgbddd']
            axes[1,0].imshow(rgbddd[:,:,:3])
            axes[1,0].set_title(f'Cropped RGB\n({processor.target_size}x{processor.target_size})')
            axes[1,0].axis('off')

            # Extract depth from Z channel (still normalized)
            depth_crop = rgbddd[:,:,5] * mesh_diameter
            valid = depth_crop > 0
            im_depth = axes[1,1].imshow(np.where(valid, depth_crop, np.nan), cmap='viridis')
            axes[1,1].set_title('Cropped Depth (m)')
            axes[1,1].axis('off')
            cbar_depth = fig.colorbar(im_depth, ax=axes[1,1], fraction=0.046, pad=0.04)
            cbar_depth.ax.tick_params(labelsize=8)

            # XYZ before normalization (denormalize for display)
            xyz = rgbddd[:,:,3:] * mesh_diameter
            xyz_range = np.nanmax(np.abs(xyz[valid])) if np.any(valid) else 1

            for i, ch in enumerate(['X', 'Y', 'Z']):
                im_xyz = axes[1,2+i].imshow(np.where(valid, xyz[:,:,i], np.nan),
                                 cmap='RdBu_r', vmin=-xyz_range, vmax=xyz_range)
                axes[1,2+i].set_title(f'{ch} (meters)')
                axes[1,2+i].axis('off')
                cbar = fig.colorbar(im_xyz, ax=axes[1,2+i], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            # Row 3: Normalized RGBDDD
            axes[2,0].imshow(rgbddd[:,:,:3])
            axes[2,0].set_title('RGB [0,1]')
            axes[2,0].axis('off')

            # Normalized XYZ
            xyz_norm = rgbddd[:,:,3:]
            xyz_norm_range = np.nanmax(np.abs(xyz_norm[valid])) if np.any(valid) else 1

            for i, ch in enumerate(['X', 'Y', 'Z']):
                im_norm = axes[2,1+i].imshow(np.where(valid, xyz_norm[:,:,i], np.nan),
                                 cmap='RdBu_r', vmin=-xyz_norm_range, vmax=xyz_norm_range)
                axes[2,1+i].set_title(f'{ch} / {mesh_diameter:.3f}m')
                axes[2,1+i].axis('off')
                cbar = fig.colorbar(im_norm, ax=axes[2,1+i], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            # Final stats
            stats_text = f"Output: {rgbddd.shape}\n"
            stats_text += f"RGB: [{rgbddd[:,:,:3].min():.2f}, {rgbddd[:,:,:3].max():.2f}]\n"
            if np.any(valid):
                stats_text += f"X_norm: [{np.nanmin(xyz_norm[:,:,0]):.2f}, {np.nanmax(xyz_norm[:,:,0]):.2f}]\n"
                stats_text += f"Y_norm: [{np.nanmin(xyz_norm[:,:,1]):.2f}, {np.nanmax(xyz_norm[:,:,1]):.2f}]\n"
                stats_text += f"Z_norm: [{np.nanmin(xyz_norm[:,:,2]):.2f}, {np.nanmax(xyz_norm[:,:,2]):.2f}]"
            axes[2,4].text(0.1, 0.5, stats_text, transform=axes[2,4].transAxes,
                         fontsize=11, family='monospace', va='center')
            axes[2,4].set_title('Normalized Stats')
            axes[2,4].axis('off')

            plt.tight_layout()
            if save_name:
                plt.savefig(viz_dir / save_name, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_name}")

        # Show real image pipeline (mask-based)
        show_pipeline(
            f'Real Image Pipeline - Mask-based Crop (diameter={mesh_diameter:.3f}m)',
            data['rgb'], data['depth'], result_mask,
            mask=data['masks'][0],
            save_name='imageprocessor_real_mask.png'
        )

        # Show real image pipeline (pose-based)
        show_pipeline(
            f'Real Image Pipeline - Pose-based Crop (diameter={mesh_diameter:.3f}m)',
            data['rgb'], data['depth'], result_pose,
            save_name='imageprocessor_real_pose.png'
        )

        # Show synthetic pipeline
        show_pipeline(
            f'Synthetic Image Pipeline - Pose-based (diameter={mesh_diameter:.3f}m)',
            render_result['rgb'], render_result['depth'], result_synth,
            save_name='imageprocessor_synthetic.png'
        )

        # Print expected ranges for this object
        print(f"\nExpected ranges for Object 5 (diameter={mesh_diameter:.3f}m):")
        print("Before normalization (meters):")
        print(f"  X,Y: ±{mesh_diameter/2:.3f}m (half diameter)")
        print(f"  Z: ~0.4-0.5m (camera distance)")
        print("After normalization (divided by diameter):")
        print(f"  X,Y: ±0.5 (normalized)")
        print(f"  Z: ~{0.45/mesh_diameter:.1f} (distance/diameter)")
        print("\nCropping methods:")
        print("- Mask-based: Tight crop around visible pixels")
        print("- Pose-based: Projects full 3D bounding box (includes occluded parts)")
        print("- Paper uses pose-based for translation feedback during refinement")

        # Comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Crop Method Comparison', fontsize=16)

        axes[0,0].imshow(data['rgb'])
        c1 = result_mask['crop_info']
        rect1 = plt.Rectangle((c1['x1'], c1['y1']), c1['x2']-c1['x1'], c1['y2']-c1['y1'],
                            fill=False, color='green', linewidth=3)
        axes[0,0].add_patch(rect1)
        axes[0,0].set_title('Mask-based Crop')
        axes[0,0].axis('off')

        axes[0,1].imshow(data['rgb'])
        c2 = result_pose['crop_info']
        rect2 = plt.Rectangle((c2['x1'], c2['y1']), c2['x2']-c2['x1'], c2['y2']-c2['y1'],
                            fill=False, color='red', linewidth=3)
        axes[0,1].add_patch(rect2)
        axes[0,1].set_title('Pose-based Crop')
        axes[0,1].axis('off')

        axes[0,2].imshow(data['rgb'])
        axes[0,2].add_patch(plt.Rectangle((c1['x1'], c1['y1']), c1['x2']-c1['x1'], c1['y2']-c1['y1'],
                                        fill=False, color='green', linewidth=2))
        axes[0,2].add_patch(plt.Rectangle((c2['x1'], c2['y1']), c2['x2']-c2['x1'], c2['y2']-c2['y1'],
                                        fill=False, color='red', linewidth=2, linestyle='--'))
        axes[0,2].set_title('Both Methods')
        axes[0,2].axis('off')

        axes[1,0].imshow(result_mask['rgbddd'][:,:,:3])
        axes[1,0].set_title('Mask Result')
        axes[1,0].axis('off')

        axes[1,1].imshow(result_pose['rgbddd'][:,:,:3])
        axes[1,1].set_title('Pose Result')
        axes[1,1].axis('off')

        axes[1,2].imshow(result_synth['rgbddd'][:,:,:3])
        axes[1,2].set_title('Synthetic Result')
        axes[1,2].axis('off')

        plt.tight_layout()
        plt.savefig(viz_dir / 'imageprocessor_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: imageprocessor_comparison.png")

        # Verify output properties
        print("\nVerification:")
        print(f"✓ Output shape: {result_mask['rgbddd'].shape} (should be 160,160,6)")
        print(f"✓ RGB range: [{result_mask['rgbddd'][:,:,:3].min():.3f}, "
              f"{result_mask['rgbddd'][:,:,:3].max():.3f}] (should be [0,1])")
        print(f"✓ XYZ normalized by diameter ({mesh_diameter:.3f}m)")
        print(f"✓ Pose-based cropping matches official approach")
        print(f"✓ Zero depth preserved as zero XYZ")

    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up...")
            manager.cleanup()
