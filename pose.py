from utils.pose import PoseEstimator
from utils.io import *
import numpy as np
import cv2
import os

annos = load_json(
    "/aidata/anders/objects/landmarks/celeba/annos/BDD_CelebA_NEW.json")

width = 1280
height = 720

img_root = "/aidata/anders/objects/landmarks/celeba/imgs"
pose_estimator = PoseEstimator(img_size=(height, width))
# for frame in annos["frame_list"]:

import random

idx = random.randint(0, len(annos["frame_list"]))
frame = annos["frame_list"][idx]
img_path = os.path.join(img_root, frame["name"])
img = cv2.imread(img_path)
cv2.imwrite("output.jpg", img)
# for lb in frame["labels"]:
#     tmp_kps = []
#     keypoints = lb["keypoints"]
#     keys = keypoints.keys()
#     for key in keys:
#         kp = keypoints[key]
#         tmp_kps.append(kp)
#     marks = np.asarray(tmp_kps)
#     marks = marks[..., ::-1]
#     LE = np.mean(marks[27:33], axis=0, keepdims=True)
#     RE = np.mean(marks[33:39], axis=0, keepdims=True)
#     N = marks[42:43]
#     LM = marks[48:49]
#     RM = marks[54:55]
#     marks = np.concatenate([LE, RE, N, LM, RM], axis=0)
#     pose = pose_estimator.solve_pose_by_68_points(marks)
#     r_mat, _ = cv2.Rodrigues(pose[0])
#     p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
#     _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
#     pitch, yaw, roll = u_angle.flatten()
#     if roll > 0:
#         roll = 180 - roll
#     elif roll < 0:
#         roll = -(180 + roll)
