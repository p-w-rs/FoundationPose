import numpy as np
from Pipeline import FoundationPosePipeline

# Initialize
pipeline = FoundationPosePipeline()
scenes = pipeline.loader.get_available_scenes()

# Load scene data
scene_data = pipeline.loader.load_scene_data(scenes[0])
object_id = scene_data['object_ids'][0]
mask = scene_data['masks'][0]

# Load object
pipeline.load_object(object_id)
mesh = pipeline.meshes[object_id]
generator = pipeline.generators[object_id]

# Generate one proposal
proposals = generator.generate_poses(scene_data['depth'], mask, pipeline.loader.K, n_proposals=1)

print("=== DEBUGGING PIPELINE ===")
print(f"Original image shape: {scene_data['rgb'].shape}")
print(f"Original depth shape: {scene_data['depth'].shape}")
print(f"Proposal translation: {proposals[0][:3, 3]}")

# Test cropping
from CropUtils import compute_crop_window_tf_batch, warp_perspective_batch
from MeshRenderer import MeshRenderer

tf_to_crops = compute_crop_window_tf_batch(
    mesh.vertices,
    scene_data['rgb'].shape[0],
    scene_data['rgb'].shape[1],
    proposals,
    pipeline.loader.K,
    crop_ratio=1.4,
    out_size=(160, 160),
    method='box_3d',
    mesh_diameter=generator.diameter
)

# Render
renderer = MeshRenderer(mesh)
rgb_render, depth_render = renderer.render(
    proposals[0], pipeline.loader.K,
    scene_data['rgb'].shape[0],
    scene_data['rgb'].shape[1]
)

print(f"\nRendered RGB shape: {rgb_render.shape}")
print(f"Rendered depth shape: {depth_render.shape}")

# Warp real
real_rgb_batch = np.array([scene_data['rgb']])
real_depth_batch = np.array([scene_data['depth']])

real_rgb_crops = warp_perspective_batch(real_rgb_batch, tf_to_crops, (160, 160), mode='bilinear')
real_depth_crops = warp_perspective_batch(real_depth_batch[..., None], tf_to_crops, (160, 160), mode='nearest')

print(f"\nAfter warping:")
print(f"Real RGB crops shape: {real_rgb_crops.shape}")
print(f"Real depth crops shape: {real_depth_crops.shape}")

# Extract single crop
real_rgb_crop = real_rgb_crops[0]
real_depth_crop = real_depth_crops[0]

if len(real_depth_crop.shape) == 3:
    real_depth_crop = real_depth_crop[..., 0]

print(f"\nSingle crop shapes:")
print(f"Real RGB crop: {real_rgb_crop.shape}")
print(f"Real depth crop: {real_depth_crop.shape}")

# Warp rendered
rendered_rgb_crops = warp_perspective_batch(np.array([rgb_render]), tf_to_crops, (160, 160), mode='bilinear')
rendered_depth_crops = warp_perspective_batch(np.array([depth_render[..., None]]), tf_to_crops, (160, 160), mode='nearest')

rgb_render_crop = rendered_rgb_crops[0]
depth_render_crop = rendered_depth_crops[0]
if len(depth_render_crop.shape) == 3:
    depth_render_crop = depth_render_crop[..., 0]

print(f"Rendered RGB crop: {rgb_render_crop.shape}")
print(f"Rendered depth crop: {depth_render_crop.shape}")

# Test prepare_input
print(f"\nTesting prepare_input:")
K_crop = tf_to_crops[0] @ pipeline.loader.K

try:
    inputs = pipeline.model_interface.prepare_input(
        real_rgb_crop, real_depth_crop,
        rgb_render_crop, depth_render_crop,
        K_crop
    )
    print("Success! Input shapes:")
    for k, v in inputs.items():
        print(f"  {k}: {v.shape}")
except Exception as e:
    print(f"Error: {e}")
    print(f"  real_rgb_crop shape: {real_rgb_crop.shape}")
    print(f"  real_depth_crop shape: {real_depth_crop.shape}")
    print(f"  real_rgb_crop dtype: {real_rgb_crop.dtype}")
    print(f"  real_depth_crop dtype: {real_depth_crop.dtype}")
