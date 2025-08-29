# utils/reader.py
import os
import json
import cupy as cp
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

class DataReader:
    def __init__(self, camera_path: str, rgb_dir: str, depth_dir: str, mask_dir: str,
                 scene_gt_path: Optional[str] = None):
        # Load camera parameters
        self.K, self.depth_scale, self.height, self.width = self._load_camera(camera_path)

        # Get frame numbers
        self.frame_numbers = self._get_frame_numbers(rgb_dir, ".png")
        self.n_frames = len(self.frame_numbers)

        # Verify directories exist
        assert os.path.isdir(rgb_dir), f"RGB directory not found: {rgb_dir}"
        assert os.path.isdir(depth_dir), f"Depth directory not found: {depth_dir}"
        assert os.path.isdir(mask_dir), f"Mask directory not found: {mask_dir}"

        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir

        # Load ground truth poses if provided
        self.scene_gt = None
        if scene_gt_path is not None:
            self.scene_gt = self._load_scene_gt(scene_gt_path)

    # API Methods
    def get_camera(self) -> Tuple[cp.ndarray, float, int, int]:
        """Return camera intrinsics K, depth scale, height, and width."""
        return self.K, self.depth_scale, self.height, self.width

    def get_rgb(self, frame_idx: int) -> cp.ndarray:
        """Load RGB image for given frame index."""
        frame = self.frame_numbers[frame_idx]
        path = os.path.join(self.rgb_dir, f"{frame}.png")
        return self._load_rgb(path)

    def get_depth(self, frame_idx: int) -> cp.ndarray:
        """Load depth image for given frame index."""
        frame = self.frame_numbers[frame_idx]
        path = os.path.join(self.depth_dir, f"{frame}.png")
        return self._load_depth(path)

    def get_mask(self, frame_idx: int, obj_idx: int) -> cp.ndarray:
        """Load mask for given frame and object indices."""
        frame = self.frame_numbers[frame_idx]
        # Note: obj_idx is already 0-based in Python
        filename = f"{frame}_{obj_idx:06d}.png"
        path = os.path.join(self.mask_dir, filename)
        return self._load_mask(path)

    def get_gt_pose(self, frame_idx: int, obj_id: int) -> Optional[cp.ndarray]:
        """Get ground truth pose for given frame and object ID.

        Returns:
            4x4 transformation matrix on GPU if found, None otherwise.
        """
        if self.scene_gt is None:
            return None

        # Find object with matching obj_id
        for obj in self.scene_gt[f"{frame_idx}"]:
            if obj['obj_id'] == obj_id:
                # Convert rotation and translation to 4x4 matrix
                R = np.array(obj['cam_R_m2c']).reshape(3, 3)
                t = np.array(obj['cam_t_m2c']) * (self.depth_scale / 1000.0)  # Convert mm to meters

                # Create 4x4 transformation matrix
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = t

                # Transfer to GPU
                return cp.asarray(T)

        return None

    def has_gt_poses(self) -> bool:
        """Check if ground truth poses are available."""
        return self.scene_gt is not None

    # Helper Methods
    def _get_frame_numbers(self, directory: str, ext: str) -> List[str]:
        """Get sorted list of frame numbers from directory."""
        files = self._get_files(directory, ext)
        # Extract basenames without extension
        basenames = [Path(f).stem for f in files]
        return basenames

    def _get_files(self, directory: str, ext: str) -> List[str]:
        """Get sorted list of files with given extension."""
        # Ensure extension starts with dot
        if not ext.startswith('.'):
            ext = '.' + ext

        # Get all files with the extension
        files = []
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if os.path.isfile(path) and entry.endswith(ext):
                files.append(path)

        return sorted(files)

    def _load_camera(self, path: str) -> Tuple[cp.ndarray, float, int, int]:
        """Load camera parameters from JSON file."""
        with open(path, 'r') as f:
            camera = json.load(f)

        # Create intrinsic matrix on GPU
        K = np.asarray([
            [camera['fx'], 0, camera['cx']],
            [0, camera['fy'], camera['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, float(camera['depth_scale']), camera['height'], camera['width']

    def _load_scene_gt(self, path: str) -> Dict:
        """Load scene ground truth from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _load_rgb(self, path: str) -> cp.ndarray:
        """Load RGB image and transfer to GPU."""
        # Use cv2 for fast loading (it loads as BGR)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_gpu = cp.asarray(img, dtype=cp.float32) / 255.0

        return img_gpu

    def _load_depth(self, path: str) -> cp.ndarray:
        """Load depth image and transfer to GPU."""
        # Load as 16-bit grayscale
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth[depth<0.001] = 0

        # Transfer to GPU and convert to meters
        depth_gpu = cp.asarray(depth / 1000.0, dtype=cp.float32) # mm to meters

        return depth_gpu

    def _load_mask(self, path: str) -> cp.ndarray:
        """Load mask image and transfer to GPU."""
        # Load as grayscale
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Transfer to GPU as boolean array
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)

        return mask_gpu
