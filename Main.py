"""
Main.py - FoundationPose Real-time Tracking Pipeline

This script implements the full tracking pipeline for FoundationPose:
1. First frame: Generate 252 pose hypotheses, refine 5 iterations
2. Subsequent frames: Use best 64 poses from previous frame, refine 2 iterations
3. Track objects through all frames and compute metrics

Pipeline Overview:
- Load camera data and object models
- For each frame:
  - Generate/propagate pose hypotheses
  - Render synthetic views at hypothesis poses
  - Process real observation
  - Refine poses iteratively
  - Score and select best pose
  - Compute metrics against ground truth

Metrics computed:
- ADD (Average Distance of model points)
- ADD-S (Symmetric version for symmetric objects)
- Translation error
- Rotation error (degrees)
- Processing time per frame

All units are in meters internally.
"""

import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm

# Import our modules
from DataLoader import DataLoader
from PoseGenerator import PoseGenerator
from MeshRenderer import MeshRenderer
from CropProcessor import CropProcessor
from DepthProcessor import DepthProcessor
from ModelInterface import ModelInterface


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
loader = DataLoader("./data")
crop_proc = CropProcessor()
depth_proc = DepthProcessor()
interface = ModelInterface(max_batch_size=162, opt_batch_size=64, use_fp16=True)
interface.load_models()
interface.warmup()


scenes = loader.get_available_scenes()
for obj_id in loader.object_ids[:2]:
    try:
        mesh = loader.load_object_model(obj_id)
    except FileNotFoundError:
        print(f"Object {obj_id}: Not found")

    renderer = MeshRenderer(mesh)
    generator = PoseGenerator(mesh)
    poses = generator.generate_poses()
    syn_rgbddd = []
    for pose in poses:
        rgb, depth = renderer.render(pose, loader.K, 160, 160)
        K_160 = loader.K.copy()
        K_160[0, 0] *= 160/640
        K_160[1, 1] *= 160/480
        K_160[0, 2] *= 160/640
        K_160[1, 2] *= 160/480
        rgbddd = depth_proc.process_rgbd_to_rgbddd(
            rgb, depth, K_160
        )
        syn_rgbddd.append(rgbddd)
    # convert syn_rgbddd from list of (160, 160, 6) to ndarray of (N, 160, 160, 6)
    syn_rgbddd = np.array(syn_rgbddd).astype(np.float32)

    scene_id = scenes[0]
    frames = loader.get_scene_frames(scene_id)
    print(f"Scene {scene_id} has {len(frames)} frames")

    for frame_id in frames[:2]:
        frame = loader.load_frame_data(scene_id, frame_id)
        cropped_frame = crop_proc.apply_crop(
            frame['rgb'], frame['depth'], frame['masks'][0], frame['K']
        )
        rgbddd = depth_proc.process_rgbd_to_rgbddd(
            cropped_frame['rgb'], cropped_frame['depth'], cropped_frame['K']
        ).astype(np.float32)
        real_rgbddd = np.repeat(rgbddd[np.newaxis], len(poses), axis=0)

        refined = interface.refine_poses_batch(poses, real_rgbddd, syn_rgbddd)
