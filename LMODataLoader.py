import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import open3d as o3d

class LMODataLoader:
    """
    Data loader for Linemod-Occluded (LM-O) dataset from BOP challenge.

    Coordinate Systems:
    - Object model: Origin at object center, units in millimeters
    - Camera: OpenCV convention (X-right, Y-down, Z-forward), units in millimeters
    - Poses: 4x4 transformation matrices from object to camera coordinates
    """

    def __init__(self, base_path: str):
        """
        Initialize the data loader.

        Args:
            base_path: Root directory containing 'data' folder with LM-O dataset
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "data" / "lmo_models" / "models"
        self.test_path = self.base_path / "data" / "test"

        # Load camera parameters
        with open(self.base_path / "data" / "lmo" / "camera.json", 'r') as f:
            self.camera_params = json.load(f)

        # Camera intrinsics matrix (3x3)
        self.K = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ])

        # Image dimensions
        self.width = self.camera_params['width']
        self.height = self.camera_params['height']

        # Depth scale (depth values in PNG are in millimeters)
        self.depth_scale = self.camera_params['depth_scale']  # Should be 1.0 for LM-O

        print(f"Camera intrinsics loaded:")
        print(f"  fx: {self.camera_params['fx']:.4f}")
        print(f"  fy: {self.camera_params['fy']:.4f}")
        print(f"  cx: {self.camera_params['cx']:.4f}")
        print(f"  cy: {self.camera_params['cy']:.4f}")
        print(f"  Image size: {self.width}x{self.height}")
        print(f"  Depth scale: {self.depth_scale}")

    def load_object_model(self, object_id: int, debug: bool = True) -> trimesh.Trimesh:
        """
        Load a 3D object model from PLY file.

        Args:
            object_id: Object ID (1, 5, 6, 8, 9, 10, 11, or 12 for LM-O)
            debug: If True, print debug information

        Returns:
            mesh: Trimesh object with vertices in millimeters

        Object coordinate system:
        - Origin: Object center
        - Units: Millimeters
        - Orientation: Arbitrary but consistent per object
        """
        ply_path = self.models_path / f"obj_{object_id:06d}.ply"

        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # Load mesh using trimesh
        mesh = trimesh.load(ply_path)

        if debug:
            print(f"\nLoaded object {object_id}:")
            print(f"  Vertices: {len(mesh.vertices)}")
            print(f"  Faces: {len(mesh.faces)}")
            print(f"  Bounding box (mm):")
            print(f"    Min: {mesh.bounds[0]}")
            print(f"    Max: {mesh.bounds[1]}")
            print(f"    Size: {mesh.bounds[1] - mesh.bounds[0]}")
            print(f"  Centroid: {mesh.centroid}")

        return mesh

    def visualize_object_model(self, mesh: trimesh.Trimesh, title: str = "Object Model"):
        """
        Visualize object model with coordinate axes.

        The visualization shows:
        - Object mesh
        - Coordinate axes (X=red, Y=green, Z=blue)
        - Units are in millimeters
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot mesh vertices
        vertices = mesh.vertices
        ax.scatter(vertices[::10, 0], vertices[::10, 1], vertices[::10, 2],
                  c='gray', s=1, alpha=0.5)

        # Plot coordinate axes at origin
        axis_length = 50  # mm
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=3)

        # Add axis labels
        ax.text(axis_length*1.1, 0, 0, 'X', color='red', fontsize=12)
        ax.text(0, axis_length*1.1, 0, 'Y', color='green', fontsize=12)
        ax.text(0, 0, axis_length*1.1, 'Z', color='blue', fontsize=12)

        # Set equal aspect ratio
        bounds = mesh.bounds
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_ylim([bounds[0][1], bounds[1][1]])
        ax.set_zlim([bounds[0][2], bounds[1][2]])

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f"{title} - Units: millimeters")

        plt.show()

    def get_available_scenes(self) -> List[int]:
        """Get list of available scene IDs"""
        scenes = []
        if self.test_path.exists():
            for scene_dir in sorted(self.test_path.iterdir()):
                if scene_dir.is_dir() and scene_dir.name.isdigit():
                    scenes.append(int(scene_dir.name))
        return scenes

    def get_available_frames(self, scene_id: int) -> List[int]:
        """Get list of available frame IDs for a scene"""
        scene_path = self.test_path / f"{scene_id:06d}"
        rgb_path = scene_path / "rgb"
        frames = []
        if rgb_path.exists():
            for rgb_file in sorted(rgb_path.glob("*.png")):
                frame_id = int(rgb_file.stem)
                frames.append(frame_id)
        return frames

    def get_available_scenes(self) -> List[int]:
        """Get list of available scene IDs"""
        scenes = []
        if self.test_path.exists():
            for scene_dir in sorted(self.test_path.iterdir()):
                if scene_dir.is_dir() and scene_dir.name.isdigit():
                    scenes.append(int(scene_dir.name))
        return scenes

    def get_available_frames(self, scene_id: int) -> List[int]:
        """Get list of available frame IDs for a scene"""
        scene_path = self.test_path / f"{scene_id:06d}"
        rgb_path = scene_path / "rgb"
        frames = []
        if rgb_path.exists():
            for rgb_file in sorted(rgb_path.glob("*.png")):
                frame_id = int(rgb_file.stem)
                frames.append(frame_id)
        return frames

    def load_scene_data(self, scene_id: int, frame_id: int = None) -> Dict:
        """
        Load all data for a specific frame in a scene.

        Args:
            scene_id: Scene ID (e.g., 2 for scene 000002)
            frame_id: Frame ID. If None, uses first available frame

        Returns:
            data: Dictionary containing:
                - 'rgb': Color image (H, W, 3) uint8
                - 'depth': Depth map in meters (H, W) float32
                - 'mask': Binary masks for each object instance
                - 'poses': Object poses as 4x4 matrices (object to camera)
                - 'object_ids': List of object IDs in the scene
        """
        scene_path = self.test_path / f"{scene_id:06d}"

        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {scene_path}")

        # Get available frames if frame_id not specified
        if frame_id is None:
            available_frames = self.get_available_frames(scene_id)
            if not available_frames:
                raise ValueError(f"No frames found in scene {scene_id}")
            frame_id = available_frames[0]
            print(f"Using first available frame: {frame_id}")

        # Load RGB image
        rgb_path = scene_path / "rgb" / f"{frame_id:06d}.png"
        if not rgb_path.exists():
            available_frames = self.get_available_frames(scene_id)
            raise FileNotFoundError(f"RGB file not found: {rgb_path}\nAvailable frames: {available_frames[:5]}...")

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise ValueError(f"Failed to load RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth image (stored as 16-bit PNG in millimeters)
        depth_path = scene_path / "depth" / f"{frame_id:06d}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")
        depth_m = depth_mm.astype(np.float32) / 1000.0  # Convert to meters

        # Load scene ground truth
        scene_gt_path = scene_path / "scene_gt.json"
        if not scene_gt_path.exists():
            raise FileNotFoundError(f"Scene GT file not found: {scene_gt_path}")

        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)

        # Extract poses and object IDs for this frame
        frame_key = str(frame_id)
        if frame_key not in scene_gt:
            raise KeyError(f"Frame {frame_id} not found in scene_gt.json")

        frame_gt = scene_gt[frame_key]
        poses = []
        object_ids = []

        for obj_data in frame_gt:
            obj_id = obj_data['obj_id']
            object_ids.append(obj_id)

            # Convert rotation and translation to 4x4 matrix
            R = np.array(obj_data['cam_R_m2c']).reshape(3, 3)
            t = np.array(obj_data['cam_t_m2c']).reshape(3, 1) # in mm

            # Build 4x4 transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()
            poses.append(pose)

        # Load masks
        masks = []
        mask_visib_path = scene_path / "mask_visib"

        for i, obj_data in enumerate(frame_gt):
            mask_path = mask_visib_path / f"{frame_id:06d}_{i:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks.append(mask > 0)
                else:
                    print(f"Warning: Failed to load mask: {mask_path}")
                    masks.append(None)
            else:
                print(f"Warning: Mask file not found: {mask_path}")
                masks.append(None)

        return {
            'rgb': rgb,
            'depth': depth_m,
            'masks': masks,
            'poses': poses,
            'object_ids': object_ids,
            'frame_id': frame_id,
            'scene_id': scene_id
        }

    def visualize_scene_data(self, data: Dict):
        """
        Visualize loaded scene data including RGB, depth, and masks.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RGB image
        axes[0, 0].imshow(data['rgb'])
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')

        # Depth map
        depth_vis = axes[0, 1].imshow(data['depth'], cmap='jet')
        axes[0, 1].set_title(f'Depth Map (range: {data["depth"].min():.2f}-{data["depth"].max():.2f}m)')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Combined mask
        combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for i, mask in enumerate(data['masks']):
            if mask is not None:
                combined_mask[mask] = i + 1

        axes[1, 0].imshow(combined_mask, cmap='tab10')
        axes[1, 0].set_title(f'Object Masks (IDs: {data["object_ids"]})')
        axes[1, 0].axis('off')

        # Pose info
        axes[1, 1].axis('off')
        pose_text = f"Scene {data['scene_id']}, Frame {data['frame_id']}\n\n"
        pose_text += "Object poses (translation in mm):\n"
        for i, (obj_id, pose) in enumerate(zip(data['object_ids'], data['poses'])):
            t = pose[:3, 3]
            pose_text += f"Object {obj_id}: t=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]\n"
        axes[1, 1].text(0.1, 0.5, pose_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    import sys
    # Get path from command line or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."

    # Initialize loader
    loader = LMODataLoader(base_path)

    # Load and visualize an object model
    object_id = 1  # Ape object
    mesh = loader.load_object_model(object_id)
    loader.visualize_object_model(mesh, f"Object {object_id}")

    # Get available scenes
    scenes = loader.get_available_scenes()
    print(f"\nAvailable scenes: {scenes}")

    # Load first available scene
    if scenes:
        scene_id = scenes[0]  # Use first available scene
        print(f"\nLoading scene {scene_id}")

        # Get available frames for this scene
        frames = loader.get_available_frames(scene_id)
        print(f"Available frames: {len(frames)} frames")
        if frames:
            print(f"Frame range: {frames[0]} to {frames[-1]}")

        # Load and visualize scene data (will use first available frame)
        try:
            scene_data = loader.load_scene_data(scene_id=scene_id)
            loader.visualize_scene_data(scene_data)
        except Exception as e:
            print(f"Error loading scene data: {e}")
    else:
        print("No scenes found in test directory")
