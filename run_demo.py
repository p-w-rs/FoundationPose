import cv2
import time

import cupy as cp
import numpy as np

from utils.reader import DataReader
from FoundationPose import FoundationPose
from utils.visualize import draw_posed_3d_box, draw_xyz_axis

debug = True
reader = DataReader(
    "data/demo/mustard0/camera.json",
    "data/demo/mustard0/scene/rgb",
    "data/demo/mustard0/scene/depth",
    "data/demo/mustard0/scene/mask",
)
est = FoundationPose(reader.K, "data/demo/mustard0/models/textured_simple.obj")

for i in range(len(reader.frame_numbers)):
    rgb = reader.get_rgb(i)
    depth = reader.get_depth(i)
    if i==0:
        start = time.time()
        mask = reader.get_mask(0, 0)
        pose = est.register(rgb=rgb, depth=depth, mask=mask, iterations=5)
        end = time.time()
        print(end - start)
    else:
        pose = est.track_one(rgb=rgb, depth=depth, iterations=2)

    if debug:
        rgb = cp.asnumpy(rgb*255)
        center_pose = pose@np.linalg.inv(est.to_origin)
        vis = draw_posed_3d_box(reader.K, img=rgb, ob_in_cam=center_pose, bbox=est.bbox)
        vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow("1", vis[...,::-1])
        cv2.waitKey(1)
