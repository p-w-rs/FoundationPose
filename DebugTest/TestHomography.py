import numpy as np
import cv2
import matplotlib.pyplot as plt
from Pipeline import FoundationPosePipeline
from CropUtils import compute_crop_window_tf_batch

# Initialize
pipeline = FoundationPosePipeline()
scenes = pipeline.loader.get_available_scenes()

# Load scene
scene_data = pipeline.loader.load_scene_data(scenes[0])
object_id = scene_data['object_ids'][0]
mask = scene_data['masks'][0]

# Load object
pipeline.load_object(object_id)
mesh = pipeline.meshes[object_id]
generator = pipeline.generators[object_id]

# Generate one proposal
proposal = generator.generate_poses(scene_data['depth'], mask, pipeline.loader.K, n_proposals=1)[0]

# Test homography
tf_to_crop = compute_crop_window_tf_batch(
    mesh.vertices,
    scene_data['rgb'].shape[0],
    scene_data['rgb'].shape[1],
    [proposal],
    pipeline.loader.K,
    crop_ratio=1.4,
    out_size=(160, 160),
    method='box_3d',
    mesh_diameter=generator.diameter
)[0]

print(f"Homography matrix:\n{tf_to_crop}")

# Warp real image
rgb_crop = cv2.warpPerspective(scene_data['rgb'], tf_to_crop, (160, 160))

# Render at full res
from MeshRenderer import MeshRenderer
renderer = MeshRenderer(mesh)
rgb_render, depth_render = renderer.render(
    proposal, pipeline.loader.K,
    scene_data['rgb'].shape[0],
    scene_data['rgb'].shape[1]
)

# Warp rendered
rgb_render_crop = cv2.warpPerspective(rgb_render, tf_to_crop, (160, 160))

# Display
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0,0].imshow(scene_data['rgb'])
axes[0,0].set_title('Original')
axes[0,1].imshow(rgb_crop)
axes[0,1].set_title('Real Cropped')
axes[1,0].imshow(rgb_render)
axes[1,0].set_title('Rendered Full')
axes[1,1].imshow(rgb_render_crop)
axes[1,1].set_title('Rendered Cropped')
plt.tight_layout()
plt.show()
