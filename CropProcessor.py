import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging

class CropProcessor:
    """
    Compute and apply crops around objects for FoundationPose.

    Handles:
    - Computing bounding boxes from masks
    - Calculating crop windows with padding
    - Resizing to model input size (160x160)
    - Adjusting camera intrinsics for crops

    UNITS:
    - Depth: meters (m)
    - Intrinsics: pixels

    All cropping preserves aspect ratio and centers the object.
    """

    def __init__(self, target_size: int = 160):
        """
        Initialize crop processor.

        Args:
            target_size: Target size for model input (square)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def compute_crop_from_mask(self, mask: np.ndarray,
                              padding_factor: float = 1.4) -> Dict:
        """
        Compute crop window from object mask.

        Args:
            mask: Binary mask (H, W)
            padding_factor: Padding around object (1.4 = 40% padding)

        Returns:
            Dict containing:
            - bbox: (x, y, w, h) bounding box in pixels
            - center: (cx, cy) center in pixels
            - size: Square crop size in pixels
            - valid: Whether mask contains object
        """
        if mask is None or not np.any(mask):
            return {
                'bbox': (0, 0, 0, 0),
                'center': (0, 0),
                'size': 0,
                'valid': False
            }

        # Find object bounds
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Compute center and size
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        # Square crop with padding
        size = max(w, h) * padding_factor

        return {
            'bbox': (x_min, y_min, w, h),
            'center': (cx, cy),
            'size': size,
            'valid': True
        }

    def apply_crop(self, rgb: np.ndarray, depth: np.ndarray,
                  mask: np.ndarray, K: np.ndarray,
                  crop_info: Optional[Dict] = None,
                  padding_factor: float = 1.4) -> Dict:
        """
        Apply cropping and resize to target size.

        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth map in meters
            mask: (H, W) binary mask
            K: 3x3 camera intrinsic matrix
            crop_info: Pre-computed crop info (None = compute from mask)
            padding_factor: Padding if computing crop

        Returns:
            Dict containing:
            - rgb: (160, 160, 3) cropped RGB
            - depth: (160, 160) cropped depth in meters
            - mask: (160, 160) cropped mask
            - K: 3x3 adjusted intrinsics
            - crop_info: Crop parameters used
        """
        # Compute crop if not provided
        if crop_info is None:
            crop_info = self.compute_crop_from_mask(mask, padding_factor)

        if not crop_info['valid']:
            # Return zeros if no valid crop
            return {
                'rgb': np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8),
                'depth': np.zeros((self.target_size, self.target_size), dtype=np.float32),
                'mask': np.zeros((self.target_size, self.target_size), dtype=bool),
                'K': K.copy(),
                'crop_info': crop_info
            }

        cx, cy = crop_info['center']
        size = crop_info['size']

        # Compute crop bounds (handle image boundaries)
        h, w = rgb.shape[:2]
        x1 = max(0, int(cx - size/2))
        y1 = max(0, int(cy - size/2))
        x2 = min(w, int(cx + size/2))
        y2 = min(h, int(cy + size/2))

        # Crop all inputs
        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Resize to target size
        rgb_resized = cv2.resize(rgb_crop, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth_crop, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask_crop.astype(np.uint8),
                                 (self.target_size, self.target_size),
                                 interpolation=cv2.INTER_NEAREST) > 0

        # Adjust intrinsics
        K_crop = self._adjust_intrinsics_for_crop(K, x1, y1, x2-x1, y2-y1,
                                                 self.target_size, self.target_size)

        # Store detailed crop info
        crop_info['x1'] = x1
        crop_info['y1'] = y1
        crop_info['x2'] = x2
        crop_info['y2'] = y2
        crop_info['scale'] = self.target_size / (x2 - x1)

        return {
            'rgb': rgb_resized,
            'depth': depth_resized,
            'mask': mask_resized,
            'K': K_crop,
            'crop_info': crop_info
        }

    def _adjust_intrinsics_for_crop(self, K: np.ndarray,
                                   x_offset: int, y_offset: int,
                                   crop_w: int, crop_h: int,
                                   target_w: int, target_h: int) -> np.ndarray:
        """
        Adjust camera intrinsics for cropping and resizing.

        Args:
            K: Original 3x3 intrinsic matrix
            x_offset, y_offset: Top-left corner of crop
            crop_w, crop_h: Crop dimensions
            target_w, target_h: Target dimensions after resize

        Returns:
            Adjusted 3x3 intrinsic matrix
        """
        K_crop = K.copy()

        # Adjust for crop offset
        K_crop[0, 2] -= x_offset  # cx
        K_crop[1, 2] -= y_offset  # cy

        # Adjust for resize
        scale_x = target_w / crop_w
        scale_y = target_h / crop_h

        K_crop[0, 0] *= scale_x  # fx
        K_crop[0, 2] *= scale_x  # cx
        K_crop[1, 1] *= scale_y  # fy
        K_crop[1, 2] *= scale_y  # cy

        return K_crop

    def batch_apply_crops(self, rgb: np.ndarray, depth: np.ndarray,
                         masks: list, K: np.ndarray,
                         padding_factor: float = 1.4) -> Dict:
        """
        Apply crops to multiple objects in same image.

        Args:
            rgb: (H, W, 3) RGB image
            depth: (H, W) depth map
            masks: List of (H, W) binary masks
            K: 3x3 camera intrinsic matrix
            padding_factor: Padding around objects

        Returns:
            Dict with lists of cropped data
        """
        results = {
            'rgbs': [],
            'depths': [],
            'masks': [],
            'Ks': [],
            'crop_infos': []
        }

        for mask in masks:
            crop_result = self.apply_crop(
                rgb, depth, mask, K, padding_factor=padding_factor
            )

            results['rgbs'].append(crop_result['rgb'])
            results['depths'].append(crop_result['depth'])
            results['masks'].append(crop_result['mask'])
            results['Ks'].append(crop_result['K'])
            results['crop_infos'].append(crop_result['crop_info'])

        return results


# Unit tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from DataLoader import DataLoader

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Crop Processor Unit Tests")
    print("="*80)

    # Initialize
    processor = CropProcessor(target_size=160)
    loader = DataLoader("./data")

    # Load test frame
    scenes = loader.get_available_scenes()
    if not scenes:
        print("No scenes found")
        exit()

    data = loader.load_frame_data(scenes[0], 0)

    # Test 1: Compute crop from mask
    print("\nTest 1: Compute Crop from Mask")
    if data['masks'] and data['masks'][0] is not None:
        mask = data['masks'][0]
        crop_info = processor.compute_crop_from_mask(mask)
        print(f"Crop info: {crop_info}")
        print(f"Object bbox: {crop_info['bbox']}")
        print(f"Crop size: {crop_info['size']:.1f} pixels")

    # Test 2: Apply crop
    print("\nTest 2: Apply Crop")
    if data['masks'] and data['masks'][0] is not None:
        result = processor.apply_crop(
            data['rgb'], data['depth'], data['masks'][0], data['K']
        )

        print(f"Original RGB shape: {data['rgb'].shape}")
        print(f"Cropped RGB shape: {result['rgb'].shape}")
        print(f"Original K:\n{data['K']}")
        print(f"Adjusted K:\n{result['K']}")

    # Test 3: Batch processing
    print("\nTest 3: Batch Processing")
    if len(data['masks']) > 1:
        batch_results = processor.batch_apply_crops(
            data['rgb'], data['depth'], data['masks'][:3], data['K']
        )
        print(f"Processed {len(batch_results['rgbs'])} objects")

    # Visualize
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Crop Processing Results', fontsize=16)

    # Original image with bbox
    ax = axes[0, 0]
    ax.imshow(data['rgb'])
    if crop_info['valid']:
        x, y, w, h = crop_info['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='r', linewidth=2)
        ax.add_patch(rect)
        cx, cy = crop_info['center']
        size = crop_info['size']
        square = plt.Rectangle((cx-size/2, cy-size/2), size, size,
                             fill=False, color='g', linewidth=2, linestyle='--')
        ax.add_patch(square)
    ax.set_title('Original + Bbox')
    ax.axis('off')

    # Original depth
    ax = axes[0, 1]
    depth_vis = data['depth'].copy()
    depth_vis[depth_vis == 0] = np.nan
    ax.imshow(depth_vis, cmap='viridis')
    ax.set_title('Original Depth')
    ax.axis('off')

    # Original mask
    ax = axes[0, 2]
    if data['masks'] and data['masks'][0] is not None:
        ax.imshow(data['masks'][0], cmap='gray')
    ax.set_title('Original Mask')
    ax.axis('off')

    # Info text
    ax = axes[0, 3]
    info_text = f"Target size: {processor.target_size}x{processor.target_size}\n"
    info_text += f"Padding factor: 1.4\n\n"
    if crop_info['valid']:
        info_text += f"Crop center: ({crop_info['center'][0]:.0f}, {crop_info['center'][1]:.0f})\n"
        info_text += f"Crop size: {crop_info['size']:.0f}\n"
        info_text += f"Scale factor: {result['crop_info']['scale']:.2f}"
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')
    ax.axis('off')

    # Cropped results
    if 'result' in locals():
        # Cropped RGB
        axes[1, 0].imshow(result['rgb'])
        axes[1, 0].set_title('Cropped RGB (160x160)')
        axes[1, 0].axis('off')

        # Cropped depth
        depth_crop_vis = result['depth'].copy()
        if depth_crop_vis[depth_crop_vis > 0].size > 0:
            vmin = depth_crop_vis[depth_crop_vis > 0].min()
            vmax = depth_crop_vis[depth_crop_vis > 0].max()
            depth_crop_vis[depth_crop_vis == 0] = np.nan
            axes[1, 1].imshow(depth_crop_vis, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Cropped Depth (160x160)')
        axes[1, 1].axis('off')

        # Cropped mask
        axes[1, 2].imshow(result['mask'], cmap='gray')
        axes[1, 2].set_title('Cropped Mask (160x160)')
        axes[1, 2].axis('off')

    # Multiple objects if available
    if 'batch_results' in locals():
        for i in range(min(3, len(batch_results['rgbs']))):
            axes[2, i].imshow(batch_results['rgbs'][i])
            axes[2, i].set_title(f'Object {i}')
            axes[2, i].axis('off')

    # Hide unused axes
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout()

    # Save
    from pathlib import Path
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / 'cropprocessor_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {viz_dir / 'cropprocessor_test.png'}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
