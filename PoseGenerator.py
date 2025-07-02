import numpy as np
from typing import List, Tuple, Optional
import trimesh
import logging

class PoseGenerator:
    """
    Generate pose proposals for FoundationPose using icosphere sampling.

    Creates a uniform distribution of viewpoints around an object by:
    1. Sampling points on an icosphere (geodesic sphere)
    2. Computing rotation matrices to look at object center
    3. Setting appropriate translation distances

    UNITS:
    - Output poses: meters (m)
    - Translation distances: meters (m)

    The generated poses represent object-to-camera transformations
    where the camera looks at the object from various viewpoints.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize pose generator for a specific object.

        Args:
            mesh: Object mesh with vertices in meters
        """
        self.logger = logging.getLogger(__name__)

        # Compute object properties
        self.mesh_center = mesh.vertices.mean(axis=0)
        self.mesh_radius = np.linalg.norm(mesh.vertices - self.mesh_center, axis=1).max()
        self.mesh_diameter = 2 * self.mesh_radius

        # Generate icosphere vertices for viewpoint sampling
        self.viewpoints = self._generate_icosphere_vertices()
        self.n_viewpoints = len(self.viewpoints)

        self.logger.info(f"Initialized pose generator: {self.n_viewpoints} viewpoints, "
                        f"object radius: {self.mesh_radius:.3f}m")

    def _generate_icosphere_vertices(self, subdivisions: int = 2) -> np.ndarray:
        """
        Generate vertices of an icosphere for uniform viewpoint sampling.

        Args:
            subdivisions: Number of subdivision iterations (0-3 recommended)
                         0: 12 vertices, 1: 42, 2: 162, 3: 642

        Returns:
            (N, 3) array of unit vectors representing viewpoints
        """
        # Create base icosahedron
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ], dtype=np.float32)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Define icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        # Subdivide faces to increase density
        for _ in range(subdivisions):
            vertices, faces = self._subdivide_icosphere(vertices, faces)

        return vertices

    def _subdivide_icosphere(self, vertices: np.ndarray,
                            faces: List[List[int]]) -> Tuple[np.ndarray, List[List[int]]]:
        """Subdivide icosphere faces to increase vertex density."""
        edge_cache = {}
        new_faces = []
        vertices = vertices.copy()

        def get_middle_point(v1_idx, v2_idx):
            """Get or create midpoint between two vertices."""
            nonlocal vertices

            key = tuple(sorted([v1_idx, v2_idx]))
            if key in edge_cache:
                return edge_cache[key]

            # Create new vertex at midpoint
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            middle = (v1 + v2) / 2
            middle = middle / np.linalg.norm(middle)  # Project to unit sphere

            # Add to vertices
            new_idx = len(vertices)
            vertices = np.vstack([vertices, middle])
            edge_cache[key] = new_idx

            return new_idx

        # Process each face
        for face in faces:
            v1, v2, v3 = face

            # Get midpoints
            a = get_middle_point(v1, v2)
            b = get_middle_point(v2, v3)
            c = get_middle_point(v3, v1)

            # Create 4 new faces
            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c]
            ])

        return vertices, new_faces

    def generate_poses(self, n_poses: Optional[int] = None,
                      distance_factor: float = 2.5,
                      in_plane_rotations: int = 1) -> np.ndarray:
        """
        Generate pose proposals by sampling viewpoints on icosphere.

        Args:
            n_poses: Number of poses to generate (None = all viewpoints)
            distance_factor: Distance from object as multiple of radius
            in_plane_rotations: Number of in-plane rotations per viewpoint

        Returns:
            (N, 4, 4) array of pose matrices in meters
        """
        if n_poses is None:
            n_poses = self.n_viewpoints * in_plane_rotations

        # Sample viewpoints
        if n_poses <= self.n_viewpoints:
            # Uniformly sample subset
            indices = np.linspace(0, self.n_viewpoints-1, n_poses, dtype=int)
            viewpoints = self.viewpoints[indices]
            in_plane_rotations = 1
        else:
            # Use all viewpoints with in-plane rotations
            viewpoints = self.viewpoints
            in_plane_rotations = max(1, n_poses // self.n_viewpoints)

        # Generate poses
        poses = []
        distance = self.mesh_radius * distance_factor

        for viewpoint in viewpoints:
            for rot_idx in range(in_plane_rotations):
                # Camera position
                cam_pos = self.mesh_center + viewpoint * distance

                # Look-at rotation matrix (camera looking at object center)
                z_axis = -viewpoint  # Camera Z points towards object

                # Choose up vector (avoid parallel to z)
                if abs(z_axis[1]) < 0.9:
                    up = np.array([0, 1, 0])
                else:
                    up = np.array([1, 0, 0])

                x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)

                # Build rotation matrix
                R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)

                # Apply in-plane rotation
                if in_plane_rotations > 1:
                    angle = rot_idx * 2 * np.pi / in_plane_rotations
                    c, s = np.cos(angle), np.sin(angle)
                    R_inplane = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    R_cam = R_cam @ R_inplane

                # Build pose matrix (object to camera)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R_cam.T  # Inverse for object-to-camera
                pose[:3, 3] = -R_cam.T @ cam_pos

                poses.append(pose)

                if len(poses) >= n_poses:
                    break

            if len(poses) >= n_poses:
                break

        return np.array(poses[:n_poses])

    def generate_nearby_poses(self, reference_pose: np.ndarray,
                            n_poses: int = 10,
                            position_noise: float = 0.05,
                            rotation_noise: float = 0.1) -> np.ndarray:
        """
        Generate poses near a reference pose for refinement.

        Args:
            reference_pose: 4x4 reference pose matrix
            n_poses: Number of nearby poses
            position_noise: Translation noise in meters
            rotation_noise: Rotation noise in radians

        Returns:
            (N, 4, 4) array of nearby poses
        """
        poses = []

        for _ in range(n_poses):
            # Add rotation noise
            noise_angles = np.random.normal(0, rotation_noise, 3)
            R_noise = self._euler_to_matrix(noise_angles)

            # Add translation noise
            t_noise = np.random.normal(0, position_noise, 3)

            # Apply noise to reference
            pose = reference_pose.copy()
            pose[:3, :3] = pose[:3, :3] @ R_noise
            pose[:3, 3] += t_noise

            poses.append(pose)

        return np.array(poses)

    def _euler_to_matrix(self, angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        import cv2
        return cv2.Rodrigues(angles)[0]


# Unit tests
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2
    from DataLoader import DataLoader
    from MeshRenderer import MeshRenderer
    from CUDAContext import CUDAContextManager

    logging.basicConfig(level=logging.INFO)

    manager = None
    try:
        print("="*80)
        print("Pose Generator Unit Tests")
        print("="*80)

        print("Initializing CUDA, loading data, and setting up renderer...")
        manager = CUDAContextManager.get_instance()
        loader = DataLoader("./data")
        mesh = loader.load_object_model(1)
        renderer = MeshRenderer(mesh)
        generator = PoseGenerator(mesh)

        print("\n[Test] Generating and rendering poses...")
        poses = generator.generate_poses(n_poses=8, distance_factor=3.0)
        rgbs, depths = renderer.render_batch(poses, loader.K, width=160, height=160)
        print(f" -> Generated and rendered {len(poses)} poses.")

        # --- Create Visualization ---
        fig, axes = plt.subplots(3, len(poses), figsize=(2.5 * len(poses), 8))
        fig.suptitle("PoseGenerator Output Verification (Rendered at 160x160)", fontsize=16)

        all_depths = np.stack(depths)
        valid_depths = all_depths[all_depths > 0]
        vmin, vmax = (valid_depths.min(), valid_depths.max()) if valid_depths.size > 0 else (0,1)

        for i in range(len(poses)):
            axes[0, i].imshow(rgbs[i])
            axes[0, i].set_title(f'Pose {i}')
            axes[0, i].axis('off')

            depth_vis = depths[i].copy()
            depth_vis[depth_vis == 0] = np.nan
            im = axes[1, i].imshow(depth_vis, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1, i].axis('off')

            axes[2, i].imshow(depths[i] > 0, cmap='gray')
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel("RGB", fontsize=14, labelpad=10)
        axes[1, 0].set_ylabel("Depth", fontsize=14, labelpad=10)
        axes[2, 0].set_ylabel("Mask", fontsize=14, labelpad=10)

        # Use fig.tight_layout() with a rect to make space for the colorbar and title
        fig.tight_layout(rect=[0, 0.1, 1, 0.93])

        # Add the colorbar *after* tight_layout has made space for it
        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Depth (meters)')

        viz_dir = Path("viz")
        viz_dir.mkdir(exist_ok=True)
        save_path = viz_dir / 'posegenerator_test.png'
        plt.savefig(save_path, dpi=150)
        print(f" -> Visualization saved to {save_path}")

        print("\n" + "="*80)
        print("All tests passed!")
        print("="*80)

    except Exception as e:
        logging.error(f"A test failed: {e}", exc_info=True)
    finally:
        if manager:
            print("\nCleaning up CUDA context...")
            manager.cleanup()
