import numpy as np
from Pipeline import FoundationPosePipeline
import matplotlib.pyplot as plt

# Initialize pipeline
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

# Generate proposals (just a few for debugging)
proposals = generator.generate_poses(scene_data['depth'], mask, pipeline.loader.K, n_proposals=10)

print(f"\n=== DEBUGGING POSE ESTIMATION ===")
print(f"\n1. Checking proposal diversity:")
for i in range(min(5, len(proposals))):
    print(f"   Pose {i}: translation = {proposals[i][:3, 3]}")

# Crop and resize
crop_result = pipeline.depth_processor.crop_depth_region(
    scene_data['rgb'], scene_data['depth'], mask
)
model_input = pipeline.depth_processor.resize_for_model(
    crop_result['rgb_crop'], crop_result['depth_crop'], crop_result['K_crop']
)

# Create renderer
from MeshRenderer import MeshRenderer
renderer = MeshRenderer(mesh)

# Test scoring with different poses
print(f"\n2. Testing scorer with deliberately different poses:")

# Create very different test poses
test_poses = []
# Original pose
test_poses.append(proposals[0])
# Shifted by 100mm in X
test_pose2 = proposals[0].copy()
test_pose2[:3, 3] += [100, 0, 0]
test_poses.append(test_pose2)
# Shifted by 100mm in Y
test_pose3 = proposals[0].copy()
test_pose3[:3, 3] += [0, 100, 0]
test_poses.append(test_pose3)
# Rotated 45 degrees around Z
test_pose4 = proposals[0].copy()
angle = np.pi/4
test_pose4[:3, :3] = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
]) @ test_pose4[:3, :3]
test_poses.append(test_pose4)

# Convert to crop frame and score
test_poses_crop = []
for pose in test_poses:
    pose_crop = pipeline._adjust_pose_for_crop(pose, crop_result, model_input)
    test_poses_crop.append(pose_crop)

scores = pipeline.model_interface.score_poses_batch(
    np.array(test_poses_crop),
    model_input['rgb'],
    model_input['depth'],
    model_input['K'],
    renderer
)

print("\nTest pose scores:")
print(f"  Original: {scores[0]:.3f}")
print(f"  +100mm X: {scores[1]:.3f}")
print(f"  +100mm Y: {scores[2]:.3f}")
print(f"  45° rotation: {scores[3]:.3f}")

# Visualize what the model sees
print(f"\n3. Visualizing rendered vs real patches:")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, (pose_crop, desc) in enumerate(zip(test_poses_crop[:4],
                                          ['Original', '+100mm X', '+100mm Y', '45° rot'])):
    # Render
    rendered_rgb, rendered_depth = renderer.render(pose_crop, model_input['K'], 160, 160)

    # Show real
    axes[0, i].imshow(model_input['rgb'])
    axes[0, i].set_title(f'Real ({desc})')
    axes[0, i].axis('off')

    # Show rendered
    axes[1, i].imshow(rendered_rgb)
    axes[1, i].set_title(f'Rendered (score={scores[i]:.3f})')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# Check if refinement actually changes poses
print(f"\n4. Checking if refinement changes poses:")
initial_pose = test_poses_crop[0].copy()
refined_pose = pipeline.model_interface.refine_pose(
    initial_pose,
    model_input['rgb'],
    model_input['depth'],
    model_input['K'],
    renderer,
    iterations=1
)

translation_change = np.linalg.norm(refined_pose[:3, 3] - initial_pose[:3, 3])
rotation_change = np.linalg.norm(refined_pose[:3, :3] - initial_pose[:3, :3])

print(f"  Translation change after 1 iteration: {translation_change:.3f}mm")
print(f"  Rotation change: {rotation_change:.3f}")
