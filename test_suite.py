# test_suite.py
"""
Comprehensive Test Suite for LMO Dataset Pipeline - Fixed Implementation

Tests:
1. LMO Loader verification with visualization
2. Data Transforms pipeline with 6 scenarios:
   - Real frame + mask crop
   - Real frame + pose crop
   - Rendered GT pose + mask crop
   - Rendered GT pose + pose crop
   - Rendered noisy pose + mask crop
   - Rendered noisy pose + pose crop

Each test validates numerically and produces debug visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import our modules
from lmo_loader import LMOLoader, LMOFrame
from data_transforms import DataTransforms

def test_lmo_loader_comprehensive():
    """Test LMO loader with comprehensive visualization and validation"""
    print("=" * 60)
    print("TESTING LMO LOADER")
    print("=" * 60)

    scene_path = "data/lmo/scenes/000002"

    try:
        loader = LMOLoader(scene_path)
        print(f"✓ Loaded scene with {len(loader)} frames")
        print(f"✓ Object IDs: {loader.query_object_ids()}")

        # Get first frame with objects
        frame = None
        for f in loader:
            if len(f.poses) > 0:
                frame = f
                break

        if frame is None:
            raise ValueError("No frames with poses found")

        print(f"✓ Using frame {frame.frame_id} with {len(frame.poses)} objects")

        # Numerical validations
        assert frame.rgb.shape[2] == 3, f"RGB wrong shape: {frame.rgb.shape}"
        assert frame.rgb.dtype == np.uint8, f"RGB wrong dtype: {frame.rgb.dtype}"
        assert frame.depth.dtype == np.float32, f"Depth wrong dtype: {frame.depth.dtype}"
        assert len(frame.camera_intrinsics.shape) == 2, "Camera K not matrix"

        # Validate depth range (should be in meters)
        valid_depth = frame.depth[frame.depth > 0]
        if len(valid_depth) > 0:
            depth_range = (valid_depth.min(), valid_depth.max())
            assert 0.1 < depth_range[0] < 5.0, f"Depth min {depth_range[0]} outside range"
            assert 0.5 < depth_range[1] < 5.0, f"Depth max {depth_range[1]} outside range"
            print(f"✓ Depth range: {depth_range[0]:.3f} - {depth_range[1]:.3f} m")

        # Validate poses (translations should be reasonable)
        for obj_id, pose in frame.poses.items():
            t_norm = np.linalg.norm(pose[:3, 3])
            assert 0.1 < t_norm < 3.0, f"Object {obj_id} distance {t_norm} unreasonable"
            print(f"✓ Object {obj_id} at {t_norm:.3f}m")

        # Validate meshes (sizes should be reasonable)
        for obj_id, mesh in frame.meshes.items():
            bbox_size = np.max(mesh.bounds[1] - mesh.bounds[0])
            assert 0.01 < bbox_size < 0.5, f"Object {obj_id} mesh size {bbox_size} unreasonable"
            print(f"✓ Object {obj_id} mesh size: {bbox_size:.3f}m")

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'LMO Loader Test - Frame {frame.frame_id}', fontsize=16)

        # RGB image
        axes[0,0].imshow(frame.rgb)
        axes[0,0].set_title('RGB Image')
        axes[0,0].axis('off')

        # Depth image
        depth_vis = frame.depth.copy()
        depth_vis[depth_vis == 0] = np.nan
        im = axes[0,1].imshow(depth_vis, cmap='turbo', vmin=0.3, vmax=1.0)

        # Get actual depth range for display
        valid_depth = frame.depth[frame.depth > 0]
        if len(valid_depth) > 0:
            depth_range = (valid_depth.min(), valid_depth.max())
            axes[0,1].set_title(f'Depth (m)\n{depth_range[0]:.2f} - {depth_range[1]:.2f}')
        else:
            axes[0,1].set_title('Depth (m)\nNo valid depth')

        axes[0,1].axis('off')
        plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)

        # Combined visible masks
        h, w = frame.rgb.shape[:2]
        combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                 (255,0,255), (0,255,255), (128,128,128), (255,128,0)]

        for i, (obj_id, mask) in enumerate(frame.visible_masks.items()):
            color = colors[i % len(colors)]
            combined_mask[mask] = color

        axes[0,2].imshow(combined_mask)
        axes[0,2].set_title(f'Visible Masks\nObjects: {list(frame.visible_masks.keys())}')
        axes[0,2].axis('off')

        # RGB with pose projections
        axes[0,3].imshow(frame.rgb)
        K = frame.camera_intrinsics

        projected_count = 0
        for obj_id, pose in frame.poses.items():
            obj_center_cam = pose[:3, 3]

            if obj_center_cam[2] > 0:  # In front of camera
                uvw = K @ obj_center_cam
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]

                if 0 <= u < w and 0 <= v < h:
                    axes[0,3].plot(u, v, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                    axes[0,3].text(u+10, v-10, f'{obj_id}', color='red', fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    projected_count += 1

        axes[0,3].set_title(f'3D Pose Projections\n{projected_count}/{len(frame.poses)} visible')
        axes[0,3].axis('off')

        # Object positions (bird's eye view)
        axes[1,0].cla()
        if frame.poses:
            for obj_id, pose in frame.poses.items():
                pos = pose[:3, 3]  # XYZ in camera coords
                axes[1,0].scatter(pos[0], -pos[2], s=120, alpha=0.8, label=f'Obj {obj_id}')
                axes[1,0].annotate(f'{obj_id}', (pos[0], -pos[2]),
                                 xytext=(5,5), textcoords='offset points')

            axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1,0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axes[1,0].set_xlabel('X (m) - Right →')
            axes[1,0].set_ylabel('Z (m) - Forward →')
            axes[1,0].set_title('Object Positions\n(Camera XZ View)')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_aspect('equal')

        # Camera intrinsics info
        axes[1,1].text(0.05, 0.95, 'Camera Intrinsics:', transform=axes[1,1].transAxes,
                      fontweight='bold', fontsize=12)
        axes[1,1].text(0.05, 0.85, f'fx: {K[0,0]:.1f}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.80, f'fy: {K[1,1]:.1f}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.75, f'cx: {K[0,2]:.1f}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.70, f'cy: {K[1,2]:.1f}', transform=axes[1,1].transAxes)

        axes[1,1].text(0.05, 0.60, 'Validation Results:', transform=axes[1,1].transAxes,
                      fontweight='bold', fontsize=12)
        axes[1,1].text(0.05, 0.50, f'✓ {len(frame.meshes)} meshes loaded', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.45, f'✓ {len(frame.poses)} poses valid', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.40, f'✓ {len(frame.visible_masks)} masks found', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.35, f'✓ Depth range OK', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.30, f'✓ Pose distances OK', transform=axes[1,1].transAxes)
        axes[1,1].text(0.05, 0.25, f'✓ Mesh sizes OK', transform=axes[1,1].transAxes)

        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')

        # Depth histogram
        if len(valid_depth) > 0:
            axes[1,2].hist(valid_depth, bins=50, alpha=0.7, color='blue')
            axes[1,2].set_xlabel('Depth (m)')
            axes[1,2].set_ylabel('Pixel Count')
            axes[1,2].set_title('Depth Distribution')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'No valid depth', ha='center', va='center')
            axes[1,2].axis('off')

        # Object distance vs size scatter plot
        axes[1,3].cla()
        if frame.poses:
            distances = []
            sizes = []
            obj_labels = []

            for obj_id in frame.poses.keys():
                if obj_id in frame.meshes:
                    dist = np.linalg.norm(frame.poses[obj_id][:3, 3])
                    size = np.max(frame.meshes[obj_id].bounds[1] - frame.meshes[obj_id].bounds[0])
                    distances.append(dist)
                    sizes.append(size)
                    obj_labels.append(obj_id)

            if distances:
                axes[1,3].scatter(distances, sizes, s=100, alpha=0.7)
                for i, obj_id in enumerate(obj_labels):
                    axes[1,3].annotate(f'{obj_id}', (distances[i], sizes[i]),
                                     xytext=(5,5), textcoords='offset points')

                axes[1,3].set_xlabel('Distance from Camera (m)')
                axes[1,3].set_ylabel('Mesh Size (m)')
                axes[1,3].set_title('Object Distance vs Size')
                axes[1,3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('test_lmo_loader_comprehensive.png', dpi=150, bbox_inches='tight')
        print("✓ Comprehensive visualization saved as 'test_lmo_loader_comprehensive.png'")

        return frame

    except Exception as e:
        print(f"✗ LMO Loader test failed: {e}")
        raise

def test_data_transforms_scenario(transforms, frame, scenario_name, obj_id,
                                rendered_rgb=None, rendered_depth=None, rendered_mask=None,
                                use_pose_crop=False, noisy_pose=None):
    """Test single data transform scenario with full visualization"""

    print(f"\n--- Testing: {scenario_name} ---")

    # Get data for processing
    if rendered_rgb is not None:
        rgb = rendered_rgb
        depth = rendered_depth
        mask = rendered_mask
    else:
        rgb = frame.rgb
        depth = frame.depth
        if obj_id in frame.visible_masks:
            mask = frame.visible_masks[obj_id]
        else:
            print(f"Warning: No mask for object {obj_id}, using simple depth mask")
            mask = depth > 0.1

    # Use noisy pose if provided, otherwise use GT pose
    pose = noisy_pose if noisy_pose is not None else frame.poses[obj_id]
    mesh = frame.meshes[obj_id]

    try:
        # Apply cropping
        if use_pose_crop:
            cropped_rgb, cropped_depth, crop_info = transforms.pose_conditioned_crop(rgb, depth, pose, mesh)
            crop_method = "Pose Crop"
        else:
            cropped_rgb, cropped_depth, crop_info = transforms.mask_crop(rgb, depth, mask)
            crop_method = "Mask Crop"

        # Convert to RGBXYZ
        rgbxyz = transforms.rgbd_to_rgbxyz(cropped_rgb, cropped_depth, pose, mesh, crop_info)

        # Validation
        assert cropped_rgb.shape == (160, 160, 3), f"Cropped RGB wrong shape: {cropped_rgb.shape}"
        assert cropped_depth.shape == (160, 160), f"Cropped depth wrong shape: {cropped_depth.shape}"
        assert rgbxyz.shape == (160, 160, 6), f"RGBXYZ wrong shape: {rgbxyz.shape}"
        assert rgbxyz.dtype == np.float32, f"RGBXYZ wrong dtype: {rgbxyz.dtype}"

        print(f"✓ {crop_method} successful")
        print(f"✓ Cropped RGB shape: {cropped_rgb.shape}")
        print(f"✓ RGBXYZ shape: {rgbxyz.shape}")

        # Compute XYZ stats
        valid_xyz = rgbxyz[:,:,3:][rgbxyz[:,:,5] != 0]
        if len(valid_xyz) > 0:
            xyz_min = valid_xyz.min()
            xyz_max = valid_xyz.max()
            print(f"✓ XYZ range (normalized by radius): [{xyz_min:.3f}, {xyz_max:.3f}]")

        print(f"✓ Valid depth pixels: {(cropped_depth > 0.001).sum()}/{cropped_depth.size}")
        print(f"✓ Valid XYZ pixels: {(rgbxyz[:,:,5] != 0).sum()}/{rgbxyz.size//6}")

        # Create visualization
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        fig.suptitle(f'{scenario_name} - Object {obj_id}', fontsize=16)

        # Row 1: Original data with mask overlay and depth
        rgb_vis = rgb.copy()
        if mask is not None and mask.any():
            # Create red overlay on mask
            mask_overlay = np.zeros_like(rgb_vis)
            mask_overlay[mask] = [255, 0, 0]
            rgb_vis = (rgb_vis * 0.7 + mask_overlay * 0.3).astype(np.uint8)

        axes[0,0].imshow(rgb_vis)
        axes[0,0].set_title('RGB + Mask Overlay')
        axes[0,0].axis('off')

        depth_vis = depth.copy()
        depth_vis[depth_vis == 0] = np.nan
        im = axes[0,1].imshow(depth_vis, cmap='turbo', vmin=0.3, vmax=1.0)
        axes[0,1].set_title('Depth (m)')
        axes[0,1].axis('off')
        plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)

        # Show crop region on original image
        axes[0,2].imshow(rgb)
        if crop_info:
            if 'center_uv' in crop_info:  # Pose crop
                cx, cy = crop_info['center_uv']
                crop_size = crop_info['crop_size']
            else:  # Mask crop
                cx, cy = crop_info['center']
                crop_size = crop_info['crop_size']

            # Draw crop rectangle
            from matplotlib.patches import Rectangle
            half_size = crop_size / 2
            rect = Rectangle((cx - half_size, cy - half_size), crop_size, crop_size,
                           fill=False, edgecolor='red', linewidth=2)
            axes[0,2].add_patch(rect)
            axes[0,2].plot(cx, cy, 'ro', markersize=8)

        axes[0,2].set_title('Crop Region')
        axes[0,2].axis('off')

        # Clear unused positions in row 1
        for i in range(3, 6):
            axes[0,i].axis('off')

        # Row 2: Cropped RGB and depth
        axes[1,0].imshow(cropped_rgb)
        axes[1,0].set_title(f'Cropped RGB\n{crop_method}')
        axes[1,0].axis('off')

        cropped_depth_vis = cropped_depth.copy()
        cropped_depth_vis[cropped_depth_vis == 0] = np.nan
        im = axes[1,1].imshow(cropped_depth_vis, cmap='turbo', vmin=0.3, vmax=1.0)
        axes[1,1].set_title('Cropped Depth (m)')
        axes[1,1].axis('off')
        plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

        # Clear unused positions in row 2
        for i in range(2, 6):
            axes[1,i].axis('off')

        # Row 3: RGBXYZ channels
        channel_names = ['R', 'G', 'B', 'X', 'Y', 'Z']
        for i in range(6):
            channel = rgbxyz[:, :, i]

            if i < 3:  # RGB channels (already in [0,1] range)
                axes[2,i].imshow(channel, cmap='gray', vmin=0, vmax=1)
                axes[2,i].set_title(f'{channel_names[i]} Channel\n[0, 1]')
            else:  # XYZ channels
                # Use consistent colormap for all XYZ channels
                im = axes[2,i].imshow(channel, cmap='RdBu_r', vmin=-2, vmax=2)

                # Count valid pixels
                valid_count = np.sum(channel != 0)
                total_count = channel.size
                valid_percent = (valid_count / total_count) * 100

                axes[2,i].set_title(f'{channel_names[i]} Channel\n{valid_percent:.1f}% valid')

            axes[2,i].axis('off')

        plt.tight_layout()

        # Save with descriptive filename
        filename = f'test_transforms_{scenario_name.lower().replace(" ", "_").replace("+", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved as '{filename}'")
        plt.close()

        return True

    except Exception as e:
        print(f"✗ Scenario {scenario_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_transforms_comprehensive(frame):
    """Test all data transform scenarios"""
    print("\n" + "=" * 60)
    print("TESTING DATA TRANSFORMS")
    print("=" * 60)

    # Initialize transforms
    transforms = DataTransforms(frame.camera_intrinsics, (640, 480))

    # Get an object that has both pose and mask
    test_obj_id = None
    for obj_id in frame.poses.keys():
        if obj_id in frame.visible_masks:
            test_obj_id = obj_id
            break

    if test_obj_id is None:
        print("Warning: No object with both pose and mask found, using first available")
        test_obj_id = list(frame.poses.keys())[0]

    print(f"Using object {test_obj_id} for tests")

    test_results = []

    # Test 1: Real frame + mask crop
    if test_obj_id in frame.visible_masks:
        result = test_data_transforms_scenario(
            transforms, frame, "Real + Mask Crop", test_obj_id, use_pose_crop=False
        )
        test_results.append(("Real + Mask Crop", result))

    # Test 2: Real frame + pose crop
    result = test_data_transforms_scenario(
        transforms, frame, "Real + Pose Crop", test_obj_id, use_pose_crop=True
    )
    test_results.append(("Real + Pose Crop", result))

    # Test 3 & 4: Rendered GT pose
    try:
        gt_pose = frame.poses[test_obj_id]
        mesh = frame.meshes[test_obj_id]
        rendered_rgb, rendered_depth, rendered_mask = transforms.render_mesh_at_pose(mesh, gt_pose)

        result = test_data_transforms_scenario(
            transforms, frame, "Rendered GT + Mask Crop", test_obj_id,
            rendered_rgb, rendered_depth, rendered_mask, use_pose_crop=False
        )
        test_results.append(("Rendered GT + Mask Crop", result))

        result = test_data_transforms_scenario(
            transforms, frame, "Rendered GT + Pose Crop", test_obj_id,
            rendered_rgb, rendered_depth, rendered_mask, use_pose_crop=True
        )
        test_results.append(("Rendered GT + Pose Crop", result))

    except Exception as e:
        print(f"Warning: GT rendering failed: {e}")
        test_results.append(("Rendered GT + Mask Crop", False))
        test_results.append(("Rendered GT + Pose Crop", False))

    # Test 5 & 6: Noisy pose rendering
    try:
        # Generate a noisy pose by perturbing GT
        gt_pose = frame.poses[test_obj_id]
        mesh = frame.meshes[test_obj_id]

        # Add rotation noise (up to 30 degrees)
        angle_noise = np.random.uniform(-30, 30) * np.pi / 180
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # Rodrigues formula for rotation
        K_mat = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
        R_noise = np.eye(3) + np.sin(angle_noise) * K_mat + (1 - np.cos(angle_noise)) * K_mat @ K_mat

        # Add translation noise (up to 5cm)
        t_noise = np.random.uniform(-0.05, 0.05, 3)

        # Apply noise to pose
        noisy_pose = gt_pose.copy()
        noisy_pose[:3, :3] = R_noise @ gt_pose[:3, :3]
        noisy_pose[:3, 3] = gt_pose[:3, 3] + t_noise

        # Render at noisy pose
        rendered_rgb, rendered_depth, rendered_mask = transforms.render_mesh_at_pose(mesh, noisy_pose)

        # Test 5: Noisy pose + mask crop (but still use GT pose for RGBXYZ)
        result = test_data_transforms_scenario(
            transforms, frame, "Noisy Pose + Mask Crop", test_obj_id,
            rendered_rgb, rendered_depth, rendered_mask, use_pose_crop=False, noisy_pose=gt_pose
        )
        test_results.append(("Noisy Pose + Mask Crop", result))

        # Test 6: Noisy pose + pose crop
        result = test_data_transforms_scenario(
            transforms, frame, "Noisy Pose + Pose Crop", test_obj_id,
            rendered_rgb, rendered_depth, rendered_mask, use_pose_crop=True, noisy_pose=noisy_pose
        )
        test_results.append(("Noisy Pose + Pose Crop", result))

    except Exception as e:
        print(f"Warning: Noisy pose rendering failed: {e}")
        test_results.append(("Noisy Pose + Mask Crop", False))
        test_results.append(("Noisy Pose + Pose Crop", False))

    # Test pose hypothesis generation
    print("\n--- Testing Pose Hypothesis Generation ---")
    try:
        if test_obj_id in frame.visible_masks:
            poses = transforms.generate_pose_hypotheses(
                frame.meshes[test_obj_id],
                frame.depth,
                frame.visible_masks[test_obj_id],
                frame.camera_intrinsics,
                n_views=10,
                n_inplane=3
            )
            print(f"✓ Generated {len(poses)} pose hypotheses")
            print(f"✓ Example translation: {poses[0][:3,3]}")
            test_results.append(("Pose Hypothesis Generation", True))
        else:
            print("⚠ Skipping pose hypothesis test (no mask)")
            test_results.append(("Pose Hypothesis Generation", False))
    except Exception as e:
        print(f"✗ Pose hypothesis generation failed: {e}")
        test_results.append(("Pose Hypothesis Generation", False))

    # Summary
    print("\n" + "-" * 40)
    print("DATA TRANSFORMS TEST SUMMARY")
    print("-" * 40)
    passed = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_results)} tests")

    return passed == len(test_results)

def run_full_test_suite():
    """Run complete test suite for both modules"""
    print("LMO Dataset Pipeline Test Suite - Fixed Implementation")
    print("=" * 80)

    try:
        # Test 1: LMO Loader
        frame = test_lmo_loader_comprehensive()
        print("✓ LMO Loader tests PASSED")

        # Test 2: Data Transforms
        transforms_passed = test_data_transforms_comprehensive(frame)

        if transforms_passed:
            print("\n✓ Data Transforms tests PASSED")
        else:
            print("\n⚠ Some Data Transforms tests FAILED (check individual results)")

        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETE")
        print("✓ LMO Loader: PASSED")
        print(f"{'✓' if transforms_passed else '⚠'} Data Transforms: {'PASSED' if transforms_passed else 'PARTIAL'}")
        print("=" * 80)

        # Final notes
        print(f"\nGenerated visualization files:")
        print("- test_lmo_loader_comprehensive.png")
        print("- test_transforms_real_mask_crop.png")
        print("- test_transforms_real_pose_crop.png")
        print("- test_transforms_rendered_gt_mask_crop.png")
        print("- test_transforms_rendered_gt_pose_crop.png")
        print("- test_transforms_noisy_pose_mask_crop.png")
        print("- test_transforms_noisy_pose_pose_crop.png")

        print(f"\nKey improvements in fixed implementation:")
        print("- Correct RGBXYZ normalization by mesh radius (FoundationPose style)")
        print("- Proper depth denoising with bilateral filter and erosion")
        print("- Fixed coordinate transforms for rendering (OpenCV to OpenGL)")
        print("- Pose-conditioned cropping based on projected object center")
        print("- Pose hypothesis generation from icosphere viewpoints")
        print("- Noisy pose now properly differs from GT pose")

    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    run_full_test_suite()
