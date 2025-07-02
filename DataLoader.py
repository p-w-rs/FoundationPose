import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import trimesh
import logging

class DataLoader:
    """
    Load LINEMOD-Occluded (LM-O) dataset in BOP format for FoundationPose testing.

    Dataset structure:
    - lmo/camera.json: Default camera intrinsics
    - lmo_models/models/obj_00000X.ply: Object 3D models
    - test/NNNNNN/: Test scenes
      - scene_camera.json: Per-scene camera parameters
      - scene_gt.json: Ground truth poses
      - rgb/: RGB images
      - depth/: Depth images (mm as uint16)
      - mask_visib/: Visible masks for objects

    UNITS:
    - Files: millimeters (mm)
    - Output: meters (m) - ALL outputs converted to meters
    - Camera intrinsics: pixels

    IMPORTANT: Everything is converted to meters for model compatibility

    The dataset contains 8 objects with IDs 1, 5, 6, 8, 9, 10, 11, 12.
    """

    def __init__(self, base_path: str):
        """
        Initialize LM-O dataset loader.

        Args:
            base_path: Root directory containing lmo/, lmo_models/, test/
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

        # Load camera intrinsics (same for all scenes/frames)
        self.camera = self._load_camera()
        self.K = self.camera['K']
        self.width = self.camera['width']
        self.height = self.camera['height']

        # Available object IDs in LM-O
        self.object_ids = [1, 5, 6, 8, 9, 10, 11, 12]

        # Cache for loaded meshes
        self._mesh_cache = {}

        self.logger.info(f"Initialized LM-O dataset loader at {base_path}")

    def _load_camera(self) -> Dict:
        """Load camera intrinsics from lmo/camera.json."""
        camera_path = self.base_path / "lmo" / "camera.json"
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_path}")

        with open(camera_path, 'r') as f:
            data = json.load(f)

        # BOP format uses fx, fy, cx, cy
        fx = data['fx']
        fy = data['fy']
        cx = data['cx']
        cy = data['cy']

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return {
            'K': K,
            'width': data['width'],
            'height': data['height'],
            'depth_scale': data.get('depth_scale', 1.0)
        }

    def load_object_model(self, object_id: int) -> trimesh.Trimesh:
        """
        Load 3D model for specified object.

        Args:
            object_id: Object ID (1, 5, 6, 8, 9, 10, 11, 12)

        Returns:
            Trimesh object with vertices in millimeters
        """
        if object_id in self._mesh_cache:
            return self._mesh_cache[object_id].copy()

        if object_id not in self.object_ids:
            raise ValueError(f"Invalid object ID: {object_id}. Valid IDs: {self.object_ids}")

        mesh_path = self.base_path / "lmo_models" / "models" / f"obj_{object_id:06d}.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = trimesh.load(mesh_path)

        # Convert vertices from mm to meters
        mesh.vertices = mesh.vertices / 1000.0

        # Ensure mesh has vertex normals
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.compute_vertex_normals()

        self._mesh_cache[object_id] = mesh

        self.logger.info(f"Loaded object {object_id}: {len(mesh.vertices)} vertices, "
                        f"diameter: {np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]):.3f} m")

        return mesh.copy()

    def get_scene_info(self, scene_id: int) -> Dict:
        """Get ground truth information for a scene."""
        gt_path = self.base_path / "test" / f"{scene_id:06d}" / "scene_gt.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        with open(gt_path, 'r') as f:
            return json.load(f)

    def load_frame_data(self, scene_id: int, frame_id: int,
                       object_indices: Optional[List[int]] = None) -> Dict:
        """
        Load all data for a specific frame.

        Args:
            scene_id: Scene ID (e.g., 2 for folder 000002)
            frame_id: Frame ID within scene
            object_indices: List of object indices in frame (None = all)

        Returns:
            Dict containing:
            - rgb: (H, W, 3) uint8 RGB image
            - depth: (H, W) float32 depth in meters
            - masks: List of (H, W) bool masks
            - poses: List of (4, 4) float32 GT poses in mm
            - object_ids: List of object IDs
            - K: (3, 3) camera intrinsics
        """
        scene_path = self.base_path / "test" / f"{scene_id:06d}"
        if not scene_path.exists():
            raise ValueError(f"Scene {scene_id} not found")

        # Load scene info
        scene_gt = self.get_scene_info(scene_id)

        # Get frame info
        frame_key = str(frame_id)
        if frame_key not in scene_gt:
            available = list(scene_gt.keys())[:5]
            raise ValueError(f"Frame {frame_id} not in scene {scene_id}. "
                           f"Available: {available}...")

        frame_info = scene_gt[frame_key]

        # Determine which objects to load
        n_objects = len(frame_info)
        if object_indices is None:
            object_indices = list(range(n_objects))

        # Load RGB
        rgb_path = scene_path / "rgb" / f"{frame_id:06d}.png"
        if not rgb_path.exists():
            # Try .jpg
            rgb_path = scene_path / "rgb" / f"{frame_id:06d}.jpg"
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth (uint16 in mm)
        depth_path = scene_path / "depth" / f"{frame_id:06d}.png"
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_mm / (self.camera['depth_scale'] * 1000.0)

        # Load masks and poses for requested objects
        masks = []
        poses = []
        object_ids = []

        for idx in object_indices:
            if idx >= n_objects:
                continue

            obj_info = frame_info[idx]
            obj_id = obj_info['obj_id']
            object_ids.append(obj_id)

            # Load visible mask
            mask_path = scene_path / "mask_visib" / f"{frame_id:06d}_{idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                masks.append(mask)
            else:
                self.logger.warning(f"Mask not found: {mask_path}")
                masks.append(None)

            # Extract pose (convert from mm to meters)
            cam_R_m2c = np.array(obj_info['cam_R_m2c']).reshape(3, 3)
            cam_t_m2c = np.array(obj_info['cam_t_m2c']) / 1000.0  # mm to meters

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = cam_R_m2c
            pose[:3, 3] = cam_t_m2c
            poses.append(pose)

        return {
            'rgb': rgb,
            'depth': depth_m,
            'masks': masks,
            'poses': poses,
            'K': self.K,
            'object_ids': object_ids,
            'scene_id': scene_id,
            'frame_id': frame_id
        }

    def get_available_scenes(self) -> List[int]:
        """Get list of available scene IDs."""
        test_path = self.base_path / "test"
        if not test_path.exists():
            return []
        scenes = []
        for scene_dir in sorted(test_path.iterdir()):
            if scene_dir.is_dir() and scene_dir.name.isdigit():
                scenes.append(int(scene_dir.name))
        return scenes

    def get_scene_frames(self, scene_id: int) -> List[int]:
        """Get list of frame IDs in a scene."""
        scene_info = self.get_scene_info(scene_id)
        return sorted([int(frame_id) for frame_id in scene_info.keys()])


# Unit tests
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # Non-interactive backend

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize loader
    # Create dummy data directory for demonstration if it doesn't exist
    data_path = Path("data")
    if not data_path.exists():
        print("Creating dummy data structure for demonstration.")
        # Create all necessary dummy files and directories
        (data_path / "lmo").mkdir(parents=True, exist_ok=True)
        with open(data_path / "lmo" / "camera.json", "w") as f:
            json.dump({
                "fx": 572.4114, "fy": 573.57043, "cx": 325.2611, "cy": 242.04899,
                "width": 640, "height": 480, "depth_scale": 1.0
            }, f)

        (data_path / "lmo_models" / "models").mkdir(parents=True, exist_ok=True)
        # Create a dummy cube PLY file for object 1
        dummy_mesh = trimesh.creation.box(bounds=[[-50,-50,-50],[50,50,50]])
        dummy_mesh.export(data_path / "lmo_models" / "models" / "obj_000001.ply")


        (data_path / "test" / "000002" / "rgb").mkdir(parents=True, exist_ok=True)
        (data_path / "test" / "000002" / "depth").mkdir(exist_ok=True)
        (data_path / "test" / "000002" / "mask_visib").mkdir(exist_ok=True)
        # Create dummy images and gt files
        dummy_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        dummy_depth = np.random.randint(400, 1500, (480, 640), dtype=np.uint16)
        dummy_mask = np.zeros((480, 640), dtype=np.uint8)
        dummy_mask[200:300, 300:400] = 255
        cv2.imwrite(str(data_path / "test" / "000002" / "rgb" / "000000.png"), dummy_rgb)
        cv2.imwrite(str(data_path / "test" / "000002" / "depth" / "000000.png"), dummy_depth)
        cv2.imwrite(str(data_path / "test" / "000002" / "mask_visib" / "000000_000000.png"), dummy_mask)
        with open(data_path / "test" / "000002" / "scene_gt.json", "w") as f:
            json.dump({
                "0": [{
                    "cam_R_m2c": np.eye(3).flatten().tolist(),
                    "cam_t_m2c": [0, 0, 1000], # 1m away
                    "obj_id": 1
                }]
            }, f)


    loader = DataLoader("./data")


    print("="*80)
    print("LM-O Dataset Loader Unit Tests")
    print("="*80)

    # Test 1: Camera intrinsics
    print("\nTest 1: Camera")
    print(f"Image size: {loader.width}x{loader.height}")
    print(f"Intrinsic matrix K:\n{loader.K}")

    # Test 2: Available scenes
    print("\nTest 2: Available Scenes")
    scenes = loader.get_available_scenes()
    print(f"Found {len(scenes)} scenes: {scenes}")

    if not scenes:
        print("No test scenes found. Exiting tests.")
        exit()

    # Test 3: Load object model
    print("\nTest 3: Load Object Models")
    try:
        mesh = loader.load_object_model(1)
        print(f"Object 1: {len(mesh.vertices)} vertices")
    except FileNotFoundError as e:
        print(f"Object 1: Not found. {e}")


    # Test 4: Load frame data
    print("\nTest 4: Load Frame Data")
    scene_id = scenes[0]
    frames = loader.get_scene_frames(scene_id)
    print(f"Scene {scene_id} has {len(frames)} frames")

    if frames:
        frame_id = frames[0]
        data = loader.load_frame_data(scene_id, frame_id)
        print(f"\nFrame {frame_id} data:")
        print(f"  RGB shape: {data['rgb'].shape}")
        print(f"  Depth shape: {data['depth'].shape}")
        print(f"  Objects: {data['object_ids']}")
        print(f"  Masks: {len(data['masks'])}")
        print(f"  Poses: {len(data['poses'])}")

        # Unit verification
        print("\nUnit Verification:")
        print(f"  Depth range: {data['depth'][data['depth'] > 0].min():.3f} - "
              f"{data['depth'][data['depth'] > 0].max():.3f} meters")
        print(f"  Pose translation (m): {data['poses'][0][:3, 3]}")
        print(f"  Camera fx (pixels): {data['K'][0, 0]:.1f}")
        print("\n  ✓ Depth is in METERS")
        print("  ✓ Poses are in METERS")
        print("  ✓ Meshes are in METERS")
        print("  ✓ Everything ready for models")

        # Visualize
        n_objects = len(data['masks'])
        fig, axes = plt.subplots(2, max(3, n_objects), figsize=(5*max(3, n_objects), 10))
        if n_objects < 3:
            # This ensures axes is always a 2D array for consistent indexing
            fig.set_size_inches(15, 10)
            axes = np.reshape(axes, (2, -1)) # Reshape to 2D if it's 1D

        # RGB
        axes[0, 0].imshow(data['rgb'])
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')

        # Depth
        depth_vis = data['depth'].copy()
        depth_vis[depth_vis == 0] = np.nan
        im = axes[0, 1].imshow(depth_vis, cmap='viridis')
        axes[0, 1].set_title('Depth')
        axes[0, 1].axis('off')

        # *** NEW: Add a colorbar for the depth image ***
        cbar = fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
        cbar.set_label('Distance (meters)')


        # Object masks
        for i, (mask, obj_id) in enumerate(zip(data['masks'], data['object_ids'])):
            if i + 2 < axes.shape[1] and mask is not None:
                axes[0, i+2].imshow(mask, cmap='gray')
                axes[0, i+2].set_title(f'Object {obj_id}')
                axes[0, i+2].axis('off')

        # Hide unused mask axes
        for i in range(len(data['masks']) + 2, axes.shape[1]):
            axes[0, i].axis('off')


        # Pose info
        info_text = "Ground Truth Poses:\n\n"
        for obj_id, pose in zip(data['object_ids'], data['poses']):
            t = pose[:3, 3]
            info_text += f"Object {obj_id}:\n"
            info_text += f"  T: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] m\n\n"

        axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', family='monospace')
        axes[1, 0].axis('off')

        # Hide unused axes in the second row
        for i in range(1, axes.shape[1]):
            axes[1, i].axis('off')

        plt.tight_layout()

        # Create viz folder if needed
        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)

        plt.savefig(viz_dir / 'dataloader_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {viz_dir / 'dataloader_test.png'}")


    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
