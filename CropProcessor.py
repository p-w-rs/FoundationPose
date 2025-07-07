# CropProcessor.py

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

class CropProcessor:
    """
    Computes and applies crops around objects for the FoundationPose pipeline.
    """

    def __init__(self, target_size: int = 160):
        """
        Initializes the crop processor.
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def compute_crop_from_mask(self, mask: np.ndarray, padding_factor: float = 1.4) -> Dict:
        """
        Calculates the parameters for a square crop from a binary object mask.
        """
        if mask is None or not np.any(mask):
            return {'valid': False}
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        size = max(w, h) * padding_factor
        return {'center': (cx, cy), 'size': size, 'valid': True}

    def apply_crop(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> Dict:
        """
        Crops and resizes RGB/Depth images and adjusts the camera intrinsics.
        """
        crop_info = self.compute_crop_from_mask(mask)

        default_return = {
            'rgb': np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8),
            'depth': np.zeros((self.target_size, self.target_size), dtype=np.float32),
            'K': K.copy()
        }

        if not crop_info['valid']:
            return default_return

        cx, cy = crop_info['center']
        size = crop_info['size']
        h, w = rgb.shape[:2]
        x1, y1 = max(0, int(cx - size/2)), max(0, int(cy - size/2))
        x2, y2 = min(w, int(cx + size/2)), min(h, int(cy + size/2))

        rgb_crop, depth_crop = rgb[y1:y2, x1:x2], depth[y1:y2, x1:x2]

        if rgb_crop.size == 0 or (x2 - x1) <= 0:
            return default_return

        rgb_resized = cv2.resize(rgb_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        K_crop = K.copy()
        K_crop[0, 2] -= x1
        K_crop[1, 2] -= y1
        scale = self.target_size / (x2 - x1)
        K_crop[0, :] *= scale
        K_crop[1, :] *= scale

        return {'rgb': rgb_resized, 'depth': depth_resized, 'K': K_crop}

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        from DataLoader import DataLoader
    except ImportError:
        print("Error: DataLoader.py is required to run the standalone test for CropProcessor.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("="*80)
    print("Crop Processor Standalone Test")
    print("="*80)

    try:
        processor = CropProcessor(target_size=160)
        loader = DataLoader("./data")
        scenes = loader.get_available_scenes()
        if not scenes:
            raise RuntimeError("No test scenes found in ./data folder.")

        scene_id = scenes[0]
        frame_id = loader.get_scene_frames(scene_id)[0]
        data = loader.load_frame_data(scene_id, frame_id)
        mask_to_process = data['masks'][1]

        result = processor.apply_crop(data['rgb'], data['depth'], mask_to_process, data['K'])
        logging.info("Successfully applied crop to the sample frame.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('CropProcessor.py Verification', fontsize=16)

        axes[0].imshow(data['rgb'])
        crop_info = processor.compute_crop_from_mask(mask_to_process)
        if crop_info['valid']:
            cx, cy = crop_info['center']
            size = crop_info['size']
            rect = plt.Rectangle((cx - size / 2, cy - size / 2), size, size,
                                   fill=False, color='lime', linewidth=2, linestyle='--')
            axes[0].add_patch(rect)
        axes[0].set_title('Original Image + Crop Window')
        axes[0].axis('off')

        axes[1].imshow(result['rgb'])
        axes[1].set_title(f'Cropped RGB ({processor.target_size}x{processor.target_size})')
        axes[1].axis('off')

        depth_vis = result['depth'].copy()
        depth_vis[depth_vis == 0] = np.nan
        im = axes[2].imshow(depth_vis, cmap='viridis')
        axes[2].set_title(f'Cropped Depth ({processor.target_size}x{processor.target_size})')
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], label='Depth (meters)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)
        save_path = viz_dir / 'cropprocessor_test.png'
        plt.savefig(save_path, dpi=150)
        logging.info(f"Verification image saved to: {save_path}")

        print("\n" + "="*80)
        print("CropProcessor test completed successfully!")
        print("="*80)

    except Exception as e:
        logging.error(f"An error occurred during the standalone test: {e}", exc_info=True)
