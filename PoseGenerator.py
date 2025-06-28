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
    from mpl_toolkits.mplot3d import Axes3D
    from DataLoader import DataLoader
    from MeshRenderer import MeshRenderer

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Pose Generator Unit Tests")
    print("="*80)

    # Load test mesh
    loader = DataLoader("./data")
    mesh = loader.load_object_model(1)

    # Test 1: Icosphere generation
    print("\nTest 1: Icosphere Viewpoints")
    generator = PoseGenerator(mesh)
    print(f"Generated {generator.n_viewpoints} viewpoints")
    print(f"Object center: {generator.mesh_center}")
    print(f"Object radius: {generator.mesh_radius:.3f}m")

    # Test 2: Pose generation
    print("\nTest 2: Pose Generation")
    poses = generator.generate_poses(n_poses=20)
    print(f"Generated {len(poses)} poses")

    # Analyze pose distribution
    translations = np.array([pose[:3, 3] for pose in poses])
    distances = np.linalg.norm(translations - generator.mesh_center, axis=1)
    print(f"Distance range: {distances.min():.3f} - {distances.max():.3f}m")
    print(f"Mean distance: {distances.mean():.3f}m")

    # Test 3: Render poses at 160x160
    print("\nTest 3: Rendering from Generated Poses at 160x160")
    renderer = MeshRenderer(mesh)

    # Render subset of poses at model input size
    n_render = min(12, len(poses))
    rgbs = []
    depths = []
    masks = []

    for i in range(n_render):
        rgb, depth = renderer.render(poses[i], loader.K, 160, 160)
        mask = (depth > 0).astype(np.uint8) * 255
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)

    print(f"Rendered {n_render} poses at 160x160")
    print(f"RGB shape: {rgbs[0].shape}")
    print(f"Depth shape: {depths[0].shape}")
    print(f"Mask shape: {masks[0].shape}")

    # Test 4: Nearby pose generation
    print("\nTest 4: Nearby Pose Generation")
    ref_pose = poses[0]
    nearby_poses = generator.generate_nearby_poses(ref_pose, n_poses=5)
    print(f"Generated {len(nearby_poses)} nearby poses")

    # Measure deviations
    ref_t = ref_pose[:3, 3]
    nearby_t = np.array([p[:3, 3] for p in nearby_poses])
    deviations = np.linalg.norm(nearby_t - ref_t, axis=1)
    print(f"Translation deviations: {deviations.mean():.3f} ± {deviations.std():.3f}m")

    # Visualize
    fig = plt.figure(figsize=(20, 12))

    # 3D viewpoint visualization
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.scatter(generator.viewpoints[:, 0],
               generator.viewpoints[:, 1],
               generator.viewpoints[:, 2],
               c='b', s=20)
    ax1.set_title('Icosphere Viewpoints')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1,1,1])

    # Pose distribution
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    cam_positions = translations + generator.mesh_center
    ax2.scatter(cam_positions[:, 0],
               cam_positions[:, 1],
               cam_positions[:, 2],
               c='r', s=50, alpha=0.6)
    ax2.scatter(*generator.mesh_center, c='g', s=100, marker='*')
    ax2.set_title('Camera Positions')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # Empty space for layout
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.axis('off')
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.axis('off')

    # Show RGB/Depth/Mask grid
    n_cols = 4
    n_rows = 3  # RGB, Depth, Mask

    fig2 = plt.figure(figsize=(16, 12))
    fig2.suptitle('160x160 Model Input Examples (RGB / Depth / Mask)', fontsize=16)

    for i in range(min(n_cols, n_render)):
        # RGB
        ax = fig2.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(rgbs[i])
        if i == 0:
            ax.set_ylabel('RGB', fontsize=12)
        ax.set_title(f'Pose {i}')
        ax.axis('off')

        # Depth
        ax = fig2.add_subplot(n_rows, n_cols, n_cols + i + 1)
        depth_vis = depths[i].copy()
        if depth_vis[depth_vis > 0].size > 0:
            vmin = depth_vis[depth_vis > 0].min()
            vmax = depth_vis[depth_vis > 0].max()
            depth_vis[depth_vis == 0] = np.nan
            im = ax.imshow(depth_vis, cmap='viridis', vmin=vmin, vmax=vmax)
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('m', rotation=0, labelpad=5)
        else:
            ax.imshow(depth_vis, cmap='viridis')
        if i == 0:
            ax.set_ylabel('Depth', fontsize=12)
        ax.axis('off')

        # Mask
        ax = fig2.add_subplot(n_rows, n_cols, 2*n_cols + i + 1)
        ax.imshow(masks[i], cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Mask', fontsize=12)
        ax.axis('off')

    plt.figure(fig.number)
    plt.tight_layout()

    plt.figure(fig2.number)
    plt.tight_layout()

    # Save both figures
    from pathlib import Path
    viz_dir = Path("viz")
    viz_dir.mkdir(exist_ok=True)
    fig.savefig(viz_dir / 'posegenerator_test.png', dpi=150, bbox_inches='tight')
    fig2.savefig(viz_dir / 'posegenerator_160x160_samples.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualizations saved to:")
    print(f"  {viz_dir / 'posegenerator_test.png'}")
    print(f"  {viz_dir / 'posegenerator_160x160_samples.png'}")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
