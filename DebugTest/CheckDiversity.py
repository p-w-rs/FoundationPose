import numpy as np

def check_pose_diversity(poses):
    """
    Check diversity of pose proposals

    Args:
        poses: Array of 4x4 poses (N, 4, 4) in mm

    Returns:
        Dict with diversity metrics
    """
    n_poses = len(poses)

    # Extract translations
    translations = poses[:, :3, 3]  # (N, 3)

    # Calculate pairwise translation distances
    trans_dists = []
    for i in range(n_poses):
        for j in range(i+1, n_poses):
            dist = np.linalg.norm(translations[i] - translations[j])
            trans_dists.append(dist)

    # Extract rotation matrices
    rotations = poses[:, :3, :3]  # (N, 3, 3)

    # Calculate rotation differences (geodesic distance)
    rot_dists = []
    for i in range(n_poses):
        for j in range(i+1, n_poses):
            # R_diff = R_i^T @ R_j
            R_diff = rotations[i].T @ rotations[j]
            # Geodesic distance: arccos((trace(R_diff) - 1) / 2)
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rot_dists.append(np.degrees(angle))

    # Check if all translations are the same
    all_same_trans = np.allclose(translations, translations[0], rtol=1e-5)

    # Check if all rotations are different
    unique_rotations = len(set([tuple(R.flatten()) for R in rotations]))

    metrics = {
        'n_poses': n_poses,
        'all_same_translation': all_same_trans,
        'unique_rotations': unique_rotations,
        'trans_dist_mean': np.mean(trans_dists) if trans_dists else 0,
        'trans_dist_std': np.std(trans_dists) if trans_dists else 0,
        'trans_dist_min': np.min(trans_dists) if trans_dists else 0,
        'trans_dist_max': np.max(trans_dists) if trans_dists else 0,
        'rot_dist_mean': np.mean(rot_dists) if rot_dists else 0,
        'rot_dist_std': np.std(rot_dists) if rot_dists else 0,
        'rot_dist_min': np.min(rot_dists) if rot_dists else 0,
        'rot_dist_max': np.max(rot_dists) if rot_dists else 0
    }

    return metrics

# Usage in Pipeline.py after generating proposals:
# diversity = check_pose_diversity(proposals)
# print(f"Pose diversity: {diversity}")
# if diversity['all_same_translation']:
#     print("WARNING: All proposals have the same translation!")
# if diversity['unique_rotations'] < len(proposals):
#     print(f"WARNING: Only {diversity['unique_rotations']} unique rotations out of {len(proposals)}!")
