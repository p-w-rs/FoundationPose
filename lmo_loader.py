# lmo_loader.py
"""
LMO Dataset Loader - Linemod Occlusion dataset in BOP format

Loads LMO scenes yielding:
- RGB images (H,W,3) uint8
- Depth images (H,W) float32 in meters
- Visible masks per object (H,W) bool
- 3D meshes per object (trimesh objects)
- Ground truth poses per object (4,4) float32 camera-to-object transforms
- Camera intrinsics (3,3) float32

Object mapping: mask frame_n_000000.png -> object 000001 (JSON id=1)
"""

import json
import numpy as np
import trimesh
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Iterator, NamedTuple

class LMOFrame(NamedTuple):
    """Single frame data structure"""
    frame_id: int
    rgb: np.ndarray  # (H,W,3) uint8
    depth: np.ndarray  # (H,W) float32 meters
    visible_masks: Dict[int, np.ndarray]  # obj_id -> (H,W) bool
    meshes: Dict[int, trimesh.Trimesh]  # obj_id -> mesh
    poses: Dict[int, np.ndarray]  # obj_id -> (4,4) cam_T_obj
    camera_intrinsics: np.ndarray  # (3,3) K matrix

class LMOLoader:
    """LMO Dataset Loader for single scenes"""

    # LMO object IDs - hardcoded as requested
    OBJECT_IDS = [1, 5, 6, 8, 9, 10, 11, 12]

    # Object names for reference
    OBJECT_NAMES = {
        1: 'ape', 5: 'water_pour', 6: 'cat', 8: 'driller',
        9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher'
    }

    def __init__(self, scene_path: str):
        """
        Initialize LMO loader for a single scene

        Args:
            scene_path: Path to scene dir (e.g., 'data/lmo/scenes/000002')
        """
        self.scene_path = Path(scene_path)
        self.dataset_root = self.scene_path.parent.parent  # data/lmo/

        # Load scene metadata
        self._load_scene_metadata()
        self._load_meshes()

    def _load_scene_metadata(self):
        """Load camera params and ground truth data"""
        # Load scene camera intrinsics per frame
        with open(self.scene_path / 'scene_camera.json') as f:
            self.scene_cameras = json.load(f)

        # Load ground truth poses per frame
        with open(self.scene_path / 'scene_gt.json') as f:
            self.scene_gt = json.load(f)

        # Load GT info (contains object visibility info)
        with open(self.scene_path / 'scene_gt_info.json') as f:
            self.scene_gt_info = json.load(f)

        # Get frame IDs (sorted integer keys)
        self.frame_ids = sorted([int(k) for k in self.scene_cameras.keys()])

    def _load_meshes(self):
        """Load all object meshes, converting to meters"""
        self.meshes = {}
        models_dir = self.dataset_root / 'models'

        # Load model scale info
        with open(models_dir / 'models_info.json') as f:
            models_info = json.load(f)

        for obj_id in self.OBJECT_IDS:
            # Load PLY mesh
            mesh_path = models_dir / f'obj_{obj_id:06d}.ply'
            mesh = trimesh.load(mesh_path)

            # Convert from mm to meters using models_info diameter
            # BOP format stores poses in mm, meshes need scaling
            scale_factor = 0.001  # mm to meters
            mesh.vertices *= scale_factor

            self.meshes[obj_id] = mesh

    def _load_frame_rgb(self, frame_id: int) -> np.ndarray:
        """Load RGB image (H,W,3) uint8"""
        rgb_path = self.scene_path / 'rgb' / f'{frame_id:06d}.png'
        return cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    def _load_frame_depth(self, frame_id: int) -> np.ndarray:
        """Load depth image (H,W) float32 in meters"""
        depth_path = self.scene_path / 'depth' / f'{frame_id:06d}.png'
        # BOP depth is in mm (uint16), convert to meters
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        return depth_mm.astype(np.float32) * 0.001  # mm to meters

    def _load_frame_visible_masks(self, frame_id: int) -> Dict[int, np.ndarray]:
        """Load visible masks for all objects in frame"""
        masks = {}
        mask_dir = self.scene_path / 'mask_visib'

        # Get GT data for this frame to know which objects are present
        frame_gt = self.scene_gt.get(str(frame_id), [])

        # For each object in the frame, load its mask
        for mask_idx, gt_entry in enumerate(frame_gt):
            obj_id = gt_entry['obj_id']
            if obj_id in self.OBJECT_IDS:
                mask_path = mask_dir / f'{frame_id:06d}_{mask_idx:06d}.png'
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    masks[obj_id] = mask > 0  # Convert to boolean

        return masks

    def _load_frame_poses(self, frame_id: int) -> Dict[int, np.ndarray]:
        """Load ground truth poses for frame"""
        poses = {}
        frame_gt = self.scene_gt.get(str(frame_id), [])

        for gt_entry in frame_gt:
            obj_id = gt_entry['obj_id']
            if obj_id in self.OBJECT_IDS:
                # BOP format: rotation as 3x3 matrix, translation in mm
                R = np.array(gt_entry['cam_R_m2c']).reshape(3, 3)
                t = np.array(gt_entry['cam_t_m2c']) * 0.001  # mm to meters

                # Build 4x4 homogeneous transform: camera_T_object
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                poses[obj_id] = pose

        return poses

    def _load_frame_camera(self, frame_id: int) -> np.ndarray:
        """Load camera intrinsics matrix"""
        camera_data = self.scene_cameras[str(frame_id)]
        cam_K = camera_data['cam_K']

        # BOP format: cam_K is 3x3 matrix stored row-wise as 9 elements
        K = np.array(cam_K).reshape(3, 3).astype(np.float32)

        return K

    def __len__(self) -> int:
        """Number of frames in scene"""
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> LMOFrame:
        """Get frame by index"""
        frame_id = self.frame_ids[idx]
        return self.get_frame(frame_id)

    def get_frame(self, frame_id: int) -> LMOFrame:
        """Get specific frame by ID"""
        return LMOFrame(
            frame_id=frame_id,
            rgb=self._load_frame_rgb(frame_id),
            depth=self._load_frame_depth(frame_id),
            visible_masks=self._load_frame_visible_masks(frame_id),
            meshes=self.meshes.copy(),  # All meshes available
            poses=self._load_frame_poses(frame_id),
            camera_intrinsics=self._load_frame_camera(frame_id)
        )

    def __iter__(self) -> Iterator[LMOFrame]:
        """Iterate through all frames"""
        for frame_id in self.frame_ids:
            yield self.get_frame(frame_id)

    def query_frame_ids(self) -> List[int]:
        """Get all available frame IDs"""
        return self.frame_ids.copy()

    def query_object_ids(self) -> List[int]:
        """Get all object IDs in dataset"""
        return self.OBJECT_IDS.copy()

    def query_frames_with_object(self, obj_id: int) -> List[int]:
        """Get frame IDs containing specific object"""
        frames = []
        for frame_id in self.frame_ids:
            frame_gt = self.scene_gt.get(str(frame_id), [])
            if any(gt['obj_id'] == obj_id for gt in frame_gt):
                frames.append(frame_id)
        return frames
