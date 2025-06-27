import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import trimesh
from LMODataLoader import LMODataLoader

class DepthProcessor:
    """
    Handles depth processing and coordinate transformations for FoundationPose.

    Key operations:
    1. Depth map to 3D point cloud conversion
    2. Coordinate frame transformations
    3. Point cloud cropping and normalization
    4. Preparation for model input (160x160 patches)
    """

    def __init__(self, camera_K: np.ndarray):
        """
        Args:
            camera_K: 3x3 camera intrinsic matrix
        """
        self.K = camera_K
        self.fx = camera_K[0, 0]
        self.fy = camera_K[1, 1]
        self.cx = camera_K[0, 2]
        self.cy = camera_K[1, 2]

    def depth_to_pointcloud(self, depth: np.ndarray, mask: Optional[np.ndarray] = None,
                           unit: str = 'mm') -> np.ndarray:
        """
        Convert depth map to 3D point cloud in camera coordinates.

        Args:
            depth: Depth map (H, W) - assumes input is in meters from our loader
            mask: Optional binary mask to filter points (H, W)
            unit: Output unit - 'mm' or 'm' (default 'mm' for FoundationPose)

        Returns:
            points: 3D points in camera frame (N, 3) in specified units

        Camera coordinate system (OpenCV convention):
        - X: right
        - Y: down
        - Z: forward (into the scene)
        """
        H, W = depth.shape

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Apply mask if provided
        if mask is not None:
            valid = mask & (depth > 0)
        else:
            valid = depth > 0

        # Extract valid coordinates
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth[valid]

        # Convert depth to millimeters if needed (our loader gives meters)
        if unit == 'mm':
            z_valid = z_valid * 1000.0  # meters to millimeters

        # Back-project to 3D using intrinsics
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy

        # Stack into (N, 3) array
        points = np.stack([x_valid, y_valid, z_valid], axis=-1)

        return points

    def transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply 4x4 transformation matrix to points.

        Args:
            points: Points to transform (N, 3)
            transform: 4x4 transformation matrix

        Returns:
            transformed_points: Transformed points (N, 3)
        """
        # Convert to homogeneous coordinates
        points_homo = np.concatenate([points, np.ones((len(points), 1))], axis=-1)

        # Apply transformation
        transformed_homo = (transform @ points_homo.T).T

        # Convert back to 3D
        transformed_points = transformed_homo[:, :3]

        return transformed_points

    def crop_depth_region(self, rgb: np.ndarray, depth: np.ndarray,
                         mask: np.ndarray, padding: float = 1.4) -> Dict:
        """
        Crop RGB-D region around object mask with padding.

        Args:
            rgb: Color image (H, W, 3)
            depth: Depth map in meters (H, W)
            mask: Binary object mask (H, W)
            padding: Padding factor for bounding box (1.4 = 40% padding)

        Returns:
            dict with:
                - 'rgb_crop': Cropped RGB (H_crop, W_crop, 3)
                - 'depth_crop': Cropped depth (H_crop, W_crop)
                - 'mask_crop': Cropped mask (H_crop, W_crop)
                - 'bbox': Original bounding box [x, y, w, h]
                - 'K_crop': Adjusted intrinsics for crop
        """
        # Find bounding box of mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            raise ValueError("Empty mask provided")

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Calculate center and size
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        # Apply padding and make square
        size = max(w, h) * padding
        half_size = size / 2

        # Calculate padded box (ensure within image bounds)
        H, W = rgb.shape[:2]
        x1 = int(max(0, cx - half_size))
        y1 = int(max(0, cy - half_size))
        x2 = int(min(W, cx + half_size))
        y2 = int(min(H, cy + half_size))

        # Crop all inputs
        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Adjust camera intrinsics for crop
        K_crop = self.K.copy()
        K_crop[0, 2] -= x1  # Adjust cx
        K_crop[1, 2] -= y1  # Adjust cy

        return {
            'rgb_crop': rgb_crop,
            'depth_crop': depth_crop,
            'mask_crop': mask_crop,
            'bbox': [x1, y1, x2-x1, y2-y1],
            'K_crop': K_crop
        }

    def resize_for_model(self, rgb: np.ndarray, depth: np.ndarray,
                        K: np.ndarray, target_size: int = 160) -> Dict:
        """
        Resize RGB-D to model input size (160x160 for FoundationPose).

        Args:
            rgb: Color image (H, W, 3)
            depth: Depth map (H, W)
            K: Camera intrinsics (3, 3)
            target_size: Target size for model (default 160)

        Returns:
            dict with resized rgb, depth, and adjusted K
        """
        H, W = rgb.shape[:2]
        scale = target_size / max(H, W)

        # Calculate new dimensions
        new_H = int(H * scale)
        new_W = int(W * scale)

        # Resize images
        rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        # Pad to square
        pad_h = target_size - new_H
        pad_w = target_size - new_W
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        rgb_padded = np.pad(rgb_resized,
                           ((pad_top, pad_h - pad_top),
                            (pad_left, pad_w - pad_left),
                            (0, 0)),
                           mode='constant')
        depth_padded = np.pad(depth_resized,
                            ((pad_top, pad_h - pad_top),
                             (pad_left, pad_w - pad_left)),
                            mode='constant')

        # Adjust intrinsics
        K_scaled = K.copy()
        K_scaled[:2] *= scale
        K_scaled[0, 2] += pad_left
        K_scaled[1, 2] += pad_top

        return {
            'rgb': rgb_padded,
            'depth': depth_padded,
            'K': K_scaled,
            'scale': scale,
            'padding': (pad_top, pad_left)
        }

    def visualize_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                           title: str = "Point Cloud"):
        """Visualize 3D point cloud"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for visualization
        if len(points) > 10000:
            idx = np.random.choice(len(points), 10000, replace=False)
            points = points[idx]
            if colors is not None:
                colors = colors[idx]

        if colors is None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=points[:, 2], cmap='viridis', s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors/255.0, s=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                             points[:, 1].max()-points[:, 1].min(),
                             points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()


# Example usage and visualization
if __name__ == "__main__":
    import sys
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."

    # Initialize data loader
    loader = LMODataLoader(base_path)

    # Initialize depth processor
    processor = DepthProcessor(loader.K)

    # Load a scene
    scenes = loader.get_available_scenes()
    if scenes:
        scene_data = loader.load_scene_data(scenes[0])

        # Process first object in scene
        if scene_data['masks'][0] is not None:
            mask = scene_data['masks'][0]

            # 1. Convert depth to point cloud
            print("\n1. Converting depth to point cloud...")
            points_cam = processor.depth_to_pointcloud(scene_data['depth'], mask, unit='mm')
            print(f"   Generated {len(points_cam)} 3D points in camera frame")
            print(f"   Point cloud bounds (mm):")
            print(f"     X: [{points_cam[:, 0].min():.1f}, {points_cam[:, 0].max():.1f}]")
            print(f"     Y: [{points_cam[:, 1].min():.1f}, {points_cam[:, 1].max():.1f}]")
            print(f"     Z: [{points_cam[:, 2].min():.1f}, {points_cam[:, 2].max():.1f}]")

            # Get colors for visualization
            rgb = scene_data['rgb']
            H, W = mask.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            valid = mask & (scene_data['depth'] > 0)
            colors = rgb[valid]

            # Visualize camera frame point cloud (convert to meters for display)
            processor.visualize_pointcloud(points_cam / 1000.0, colors,
                                         "Point Cloud in Camera Frame")

            # 2. Transform to object frame
            print("\n2. Transforming to object frame...")
            pose = scene_data['poses'][0]  # Object to camera transform (mm)
            obj_to_cam = pose
            cam_to_obj = np.linalg.inv(obj_to_cam)
            points_obj = processor.transform_points(points_cam, cam_to_obj)
            print(f"   Point cloud in object frame bounds (mm):")
            print(f"     X: [{points_obj[:, 0].min():.1f}, {points_obj[:, 0].max():.1f}]")
            print(f"     Y: [{points_obj[:, 1].min():.1f}, {points_obj[:, 1].max():.1f}]")
            print(f"     Z: [{points_obj[:, 2].min():.1f}, {points_obj[:, 2].max():.1f}]")

            # 3. Crop region
            print("\n3. Cropping object region...")
            crop_result = processor.crop_depth_region(rgb, scene_data['depth'], mask)
            print(f"   Original size: {rgb.shape[:2]}")
            print(f"   Cropped size: {crop_result['rgb_crop'].shape[:2]}")
            print(f"   Bounding box: {crop_result['bbox']}")

            # 4. Resize for model
            print("\n4. Resizing for model input...")
            model_input = processor.resize_for_model(
                crop_result['rgb_crop'],
                crop_result['depth_crop'],
                crop_result['K_crop']
            )
            print(f"   Model input size: {model_input['rgb'].shape[:2]}")
            print(f"   Scale factor: {model_input['scale']:.3f}")

            # Visualize processing pipeline
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Original
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("Original RGB")
            axes[0, 0].axis('off')

            axes[1, 0].imshow(scene_data['depth'], cmap='jet')
            axes[1, 0].set_title("Original Depth")
            axes[1, 0].axis('off')

            # Masked
            rgb_masked = rgb.copy()
            rgb_masked[~mask] = 0
            axes[0, 1].imshow(rgb_masked)
            axes[0, 1].set_title("Masked RGB")
            axes[0, 1].axis('off')

            depth_masked = scene_data['depth'].copy()
            depth_masked[~mask] = 0
            axes[1, 1].imshow(depth_masked, cmap='jet')
            axes[1, 1].set_title("Masked Depth")
            axes[1, 1].axis('off')

            # Model input
            axes[0, 2].imshow(model_input['rgb'])
            axes[0, 2].set_title("Model Input RGB (160x160)")
            axes[0, 2].axis('off')

            axes[1, 2].imshow(model_input['depth'], cmap='jet')
            axes[1, 2].set_title("Model Input Depth (160x160)")
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()
