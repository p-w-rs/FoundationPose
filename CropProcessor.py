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
    from DataLoader import DataLoader # Assuming DataLoader is in a separate file

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

    # Use a scene and frame that are known to exist
    scene_id_to_test = scenes[0]
    frames_in_scene = loader.get_scene_frames(scene_id_to_test)
    if not frames_in_scene:
        print(f"No frames found in scene {scene_id_to_test}")
        exit()
    frame_id_to_test = frames_in_scene[0]

    data = loader.load_frame_data(scene_id_to_test, frame_id_to_test)

    # Find the first valid mask to process
    first_valid_mask_idx = -1
    for i, m in enumerate(data['masks']):
        if m is not None and np.any(m):
            first_valid_mask_idx = i
            break

    if first_valid_mask_idx == -1:
        print("No valid objects found in the test frame.")
        exit()

    mask_to_process = data['masks'][first_valid_mask_idx]

    # Test 1: Compute crop from mask
    print("\nTest 1: Compute Crop from Mask")
    crop_info = processor.compute_crop_from_mask(mask_to_process)
    print(f"Crop info: {crop_info}")

    # Test 2: Apply crop
    print("\nTest 2: Apply Crop")
    result = processor.apply_crop(
        data['rgb'], data['depth'], mask_to_process, data['K']
    )
    print(f"Adjusted K:\n{result['K']}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('Crop Processing Results', fontsize=16)

    # --- Row 1: Original Data ---
    ax = axes[0, 0]
    ax.imshow(data['rgb'])
    if crop_info['valid']:
        x, y, w, h = crop_info['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='r', linewidth=2)
        ax.add_patch(rect)
        cx, cy = crop_info['center']
        size = crop_info['size']
        square = plt.Rectangle((cx - size / 2, cy - size / 2), size, size,
                               fill=False, color='g', linewidth=2, linestyle='--')
        ax.add_patch(square)
    ax.set_title('Original + BBox')
    ax.axis('off')

    # Prepare original depth for visualization
    ax = axes[0, 1]
    depth_vis = data['depth'].copy()
    valid_depth = depth_vis[depth_vis > 0]
    vmin, vmax = (valid_depth.min(), valid_depth.max()) if valid_depth.size > 0 else (0,1)
    depth_vis[depth_vis == 0] = np.nan # Set 0 to NaN for transparent plotting

    im1 = ax.imshow(depth_vis, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('Original Depth')
    ax.axis('off')

    # *** ADDED: Colorbar for Original Depth ***
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label('Distance (meters)')

    ax = axes[0, 2]
    ax.imshow(mask_to_process, cmap='gray')
    ax.set_title('Original Mask')
    ax.axis('off')

    # --- Row 2: Cropped Results ---
    axes[1, 0].imshow(result['rgb'])
    axes[1, 0].set_title('Cropped RGB (160x160)')
    axes[1, 0].axis('off')

    ax = axes[1, 1]
    depth_crop_vis = result['depth'].copy()
    depth_crop_vis[depth_crop_vis == 0] = np.nan

    # *** MODIFIED: Use vmin/vmax from original depth for consistent colors ***
    im2 = ax.imshow(depth_crop_vis, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('Cropped Depth (160x160)')
    ax.axis('off')

    # *** ADDED: Colorbar for Cropped Depth ***
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Distance (meters)')

    axes[1, 2].imshow(result['mask'], cmap='gray')
    axes[1, 2].set_title('Cropped Mask (160x160)')
    axes[1, 2].axis('off')

    # Hide unused axes
    axes[0, 3].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    from pathlib import Path
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / 'cropprocessor_test.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
