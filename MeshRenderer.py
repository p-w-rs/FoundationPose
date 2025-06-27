import numpy as np
import cv2
import trimesh
from typing import Tuple

class MeshRenderer:
    """
    Proper mesh renderer using OpenCV for rasterization.
    Produces actual rendered images of the mesh, not blank squares.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces

        # Precompute face normals for shading
        self.face_normals = mesh.face_normals

    def render(self, pose: np.ndarray, K: np.ndarray,
               H: int = 160, W: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh at given pose.

        Args:
            pose: 4x4 transformation matrix (mm)
            K: 3x3 camera intrinsics
            H, W: Output image dimensions

        Returns:
            rgb: Rendered RGB image (H, W, 3)
            depth: Rendered depth map in meters (H, W)
        """
        # Transform vertices to camera frame
        vertices_cam = (pose[:3, :3] @ self.vertices.T).T + pose[:3, 3]

        # Initialize buffers
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth_buffer = np.full((H, W), np.inf, dtype=np.float32)

        # Project vertices
        vertices_proj = (K @ vertices_cam.T).T
        vertices_2d = vertices_proj[:, :2] / vertices_proj[:, 2:3]
        z_values = vertices_proj[:, 2]

        # Sort faces by average depth (painter's algorithm)
        face_depths = np.mean(z_values[self.faces], axis=1)
        sorted_faces = np.argsort(-face_depths)  # Back to front

        # Light direction (from camera)
        light_dir = np.array([0, 0, 1])

        for face_idx in sorted_faces:
            face = self.faces[face_idx]

            # Get face vertices in camera space
            face_verts_cam = vertices_cam[face]

            # Skip if any vertex is behind camera
            if np.any(face_verts_cam[:, 2] <= 0):
                continue

            # Get 2D vertices
            pts_2d = vertices_2d[face].astype(np.int32)

            # Check if triangle is in bounds
            if (np.all(pts_2d[:, 0] < 0) or np.all(pts_2d[:, 0] >= W) or
                np.all(pts_2d[:, 1] < 0) or np.all(pts_2d[:, 1] >= H)):
                continue

            # Compute face normal in camera space
            v1 = face_verts_cam[1] - face_verts_cam[0]
            v2 = face_verts_cam[2] - face_verts_cam[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)

            # Skip back-facing triangles
            if normal[2] > 0:
                continue

            # Simple Lambertian shading
            shade = max(0, -np.dot(normal, light_dir))
            color = int(100 + 155 * shade)  # Gray with shading

            # Create mask for triangle
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_2d], 1)

            # Get bounding box
            x_min = max(0, pts_2d[:, 0].min())
            x_max = min(W-1, pts_2d[:, 0].max())
            y_min = max(0, pts_2d[:, 1].min())
            y_max = min(H-1, pts_2d[:, 1].max())

            if x_max <= x_min or y_max <= y_min:
                continue

            # Compute barycentric coordinates for depth interpolation
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if mask[y, x] == 0:
                        continue

                    # Barycentric interpolation for depth
                    v0 = pts_2d[2] - pts_2d[0]
                    v1 = pts_2d[1] - pts_2d[0]
                    v2 = np.array([x, y]) - pts_2d[0]

                    denom = v0[0] * v1[1] - v1[0] * v0[1]
                    if abs(denom) < 1e-8:
                        continue

                    v = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
                    u = (v0[0] * v2[1] - v2[0] * v0[1]) / denom

                    if u >= 0 and v >= 0 and u + v <= 1:
                        # Interpolate depth
                        w0 = 1 - u - v
                        w1 = u
                        w2 = v
                        z = (w0 * z_values[face[0]] +
                             w1 * z_values[face[1]] +
                             w2 * z_values[face[2]])

                        # Update if closer
                        if z > 0 and z < depth_buffer[y, x]:
                            depth_buffer[y, x] = z
                            rgb[y, x] = [color, color, color]

        # Convert depth to meters
        depth = np.where(depth_buffer == np.inf, 0, depth_buffer / 1000.0)

        return rgb, depth


# Test the renderer
if __name__ == "__main__":
    from LMODataLoader import LMODataLoader
    import matplotlib.pyplot as plt

    # Load mesh
    loader = LMODataLoader(".")
    mesh = loader.load_object_model(1, debug=False)

    # Create renderer
    renderer = MeshRenderer(mesh)

    # Test pose
    pose = np.eye(4)
    pose[:3, 3] = [0, 0, 1000]  # 1m away

    # Test intrinsics
    K = np.array([[572.4, 0, 80],
                  [0, 573.5, 80],
                  [0, 0, 1]])

    # Render
    rgb, depth = renderer.render(pose, K, 160, 160)

    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(rgb)
    ax1.set_title("Rendered RGB")
    ax1.axis('off')

    ax2.imshow(depth, cmap='jet')
    ax2.set_title("Rendered Depth")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"RGB range: [{rgb.min()}, {rgb.max()}]")
    print(f"Depth range: [{depth[depth > 0].min():.3f}, {depth.max():.3f}] m")
    print(f"Non-zero pixels: {np.sum(depth > 0)}")
