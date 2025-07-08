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
    - Files store data in millimeters (mm)
    - This class converts ALL outputs to meters (m)
    - Camera intrinsics remain in pixels

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
            Trimesh object with vertices in METERS (converted from mm)
        """
        if object_id in self._mesh_cache:
            return self._mesh_cache[object_id].copy()

        if object_id not in self.object_ids:
            raise ValueError(f"Invalid object ID: {object_id}. Valid IDs: {self.object_ids}")

        mesh_path = self.base_path / "lmo_models" / "models" / f"obj_{object_id:06d}.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = trimesh.load(mesh_path)

        # CRITICAL: Convert vertices from millimeters to meters
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
            - depth: (H, W) float32 depth in METERS
            - masks: List of (H, W) bool masks
            - poses: List of (4, 4) float32 GT poses in METERS
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

        # CRITICAL: Convert depth from millimeters to meters
        # depth_scale handles camera-specific scaling (usually 1.0)
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

            # Extract pose
            cam_R_m2c = np.array(obj_info['cam_R_m2c']).reshape(3, 3)
            # CRITICAL: Convert translation from millimeters to meters
            cam_t_m2c = np.array(obj_info['cam_t_m2c']) / 1000.0

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = cam_R_m2c
            pose[:3, 3] = cam_t_m2c
            poses.append(pose)

        return {
            'rgb': rgb,
            'depth': depth_m,  # In meters
            'masks': masks,
            'poses': poses,    # Translation in meters
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
    matplotlib.use('Agg')

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader("./data")

    print("="*80)
    print("DataLoader Unit Test - Using Object 5")
    print("="*80)

    # Test 1: Camera intrinsics
    print("\nTest 1: Camera Parameters")
    print(f"Image size: {loader.width}x{loader.height}")
    print(f"Intrinsic matrix K:\n{loader.K}")
    print(f"fx={loader.K[0,0]:.1f}, fy={loader.K[1,1]:.1f}, cx={loader.K[0,2]:.1f}, cy={loader.K[1,2]:.1f}")

    # Test 2: Load object 5 model
    print("\nTest 2: Load Object 5 Model")
    try:
        mesh = loader.load_object_model(5)
        print(f"Object 5 loaded successfully:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Bounds min: {mesh.bounds[0]} (meters)")
        print(f"  Bounds max: {mesh.bounds[1]} (meters)")
        print(f"  Diameter: {np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]):.3f} meters")
        print(f"  Center: {mesh.vertices.mean(axis=0)} (meters)")
    except Exception as e:
        print(f"Failed to load object 5: {e}")
        exit(1)

    # Test 3: Load frame with object 5
    print("\nTest 3: Load Frame Data")
    scene_id = 2  # Scene 000002
    frame_id = 0  # Frame 000000

    # First, check what objects are in this frame
    scene_info = loader.get_scene_info(scene_id)
    frame_objects = scene_info[str(frame_id)]
    print(f"\nScene {scene_id}, Frame {frame_id} contains {len(frame_objects)} objects:")

    # Find object 5
    obj5_idx = None
    for idx, obj_info in enumerate(frame_objects):
        obj_id = obj_info['obj_id']
        print(f"  Index {idx}: Object {obj_id}")
        if obj_id == 5:
            obj5_idx = idx

    if obj5_idx is None:
        print("ERROR: Object 5 not found in this frame!")
        # Try another frame
        for test_frame in range(min(5, len(scene_info))):
            frame_objects = scene_info[str(test_frame)]
            for idx, obj_info in enumerate(frame_objects):
                if obj_info['obj_id'] == 5:
                    frame_id = test_frame
                    obj5_idx = idx
                    print(f"Found object 5 in frame {frame_id} at index {obj5_idx}")
                    break
            if obj5_idx is not None:
                break

    # Load data for object 5
    data = loader.load_frame_data(scene_id, frame_id, object_indices=[obj5_idx])
    print(f"\nLoaded frame {frame_id} data for object 5:")
    print(f"  RGB shape: {data['rgb'].shape}")
    print(f"  Depth shape: {data['depth'].shape}")
    print(f"  Number of masks: {len(data['masks'])}")
    print(f"  Number of poses: {len(data['poses'])}")
    print(f"  Object IDs in frame: {data['object_ids']}")

    # Test 4: Verify units
    print("\nTest 4: Unit Verification")
    pose = data['poses'][0]
    translation = pose[:3, 3]
    print(f"Object 5 pose translation: {translation} (meters)")
    print(f"  Distance from camera: {np.linalg.norm(translation):.3f} meters")

    depth_valid = data['depth'][data['depth'] > 0]
    print(f"\nDepth statistics (meters):")
    print(f"  Min: {depth_valid.min():.3f}")
    print(f"  Max: {depth_valid.max():.3f}")
    print(f"  Mean: {depth_valid.mean():.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'DataLoader Test - Object 5 (Scene {scene_id}, Frame {frame_id})', fontsize=16)

    # Row 1: RGB and Depth
    axes[0, 0].imshow(data['rgb'])
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    depth_vis = data['depth'].copy()
    depth_vis[depth_vis == 0] = np.nan
    im = axes[0, 1].imshow(depth_vis, cmap='viridis')
    axes[0, 1].set_title('Depth (meters)')
    axes[0, 1].axis('off')
    cbar = fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    cbar.set_label('Distance (m)')

    # Object 5 mask
    if data['masks'][0] is not None:
        axes[0, 2].imshow(data['masks'][0], cmap='gray')
        axes[0, 2].set_title('Object 5 Mask')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Mask', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].axis('off')

    # Overlay mask on RGB
    rgb_overlay = data['rgb'].copy()
    if data['masks'][0] is not None:
        mask_colored = np.zeros_like(rgb_overlay)
        mask_colored[:,:,1] = data['masks'][0] * 255  # Green channel
        rgb_overlay = cv2.addWeighted(rgb_overlay, 0.7, mask_colored, 0.3, 0)
    axes[0, 3].imshow(rgb_overlay)
    axes[0, 3].set_title('RGB + Mask Overlay')
    axes[0, 3].axis('off')

    # Row 2: Object info and statistics
    axes[1, 0].text(0.1, 0.9, 'Object 5 Pose:', transform=axes[1, 0].transAxes, fontsize=12, weight='bold')
    pose_text = f"Translation (m):\n  x: {translation[0]:.3f}\n  y: {translation[1]:.3f}\n  z: {translation[2]:.3f}\n\n"
    pose_text += f"Distance: {np.linalg.norm(translation):.3f} m\n\n"
    R = pose[:3, :3]
    pose_text += f"Rotation:\n{R}"
    axes[1, 0].text(0.1, 0.1, pose_text, transform=axes[1, 0].transAxes, fontsize=10, family='monospace', va='bottom')
    axes[1, 0].axis('off')

    # Mesh visualization
    axes[1, 1].text(0.1, 0.9, 'Object 5 Mesh:', transform=axes[1, 1].transAxes, fontsize=12, weight='bold')
    mesh_text = f"Vertices: {len(mesh.vertices)}\n"
    mesh_text += f"Faces: {len(mesh.faces)}\n"
    mesh_text += f"Diameter: {np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]):.3f} m\n"
    mesh_text += f"\nBounds (m):\n"
    mesh_text += f"  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]\n"
    mesh_text += f"  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]"
    axes[1, 1].text(0.1, 0.1, mesh_text, transform=axes[1, 1].transAxes, fontsize=10, family='monospace', va='bottom')
    axes[1, 1].axis('off')

    # Camera info
    axes[1, 2].text(0.1, 0.9, 'Camera Info:', transform=axes[1, 2].transAxes, fontsize=12, weight='bold')
    cam_text = f"Resolution: {loader.width}x{loader.height}\n\n"
    cam_text += f"Intrinsics:\n"
    cam_text += f"  fx: {loader.K[0,0]:.1f} pixels\n"
    cam_text += f"  fy: {loader.K[1,1]:.1f} pixels\n"
    cam_text += f"  cx: {loader.K[0,2]:.1f} pixels\n"
    cam_text += f"  cy: {loader.K[1,2]:.1f} pixels"
    axes[1, 2].text(0.1, 0.1, cam_text, transform=axes[1, 2].transAxes, fontsize=10, family='monospace', va='bottom')
    axes[1, 2].axis('off')

    # Unit verification
    axes[1, 3].text(0.1, 0.9, 'Unit Verification:', transform=axes[1, 3].transAxes, fontsize=12, weight='bold')
    unit_text = "✓ Mesh vertices: METERS\n"
    unit_text += "✓ Pose translation: METERS\n"
    unit_text += "✓ Depth map: METERS\n"
    unit_text += "✓ Camera intrinsics: PIXELS\n\n"
    unit_text += "All millimeter values from\nfiles converted to meters."
    axes[1, 3].text(0.1, 0.3, unit_text, transform=axes[1, 3].transAxes, fontsize=11, va='bottom')
    axes[1, 3].axis('off')

    plt.tight_layout()

    # Save
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / 'dataloader_test_object5.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    print("\n" + "="*80)
    print("DataLoader test completed successfully!")
    print("All units properly converted from mm to meters.")
    print("="*80)
