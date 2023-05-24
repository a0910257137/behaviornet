import numpy as np
from utils.io import *
import os
import cv2
from pprint import pprint
from utils.morphable_model import MorphabelModel
from glob import glob

path = "/aidata/anders/3D-head/user_depth/calibrate_images/params/bins"
paths = list(glob(os.path.join(path, "*.bin")))
for path in paths:
    params = np.fromfile(file=path)
    if params.shape[0] == 9:
        params = params.reshape((3, -1))
    print(params)

poses = np.loadtxt("/aidata/anders/3D-head/user_depth/anders/pose.txt")
root_dir = "/aidata/anders/3D-head/3DDFA"
bfm_path = os.path.join(root_dir, "BFM/BFM.mat")
bfm = MorphabelModel(bfm_path)
path = "/aidata/anders/3D-head/user_depth/anders/annos/BDD_POSE.json"
annos = load_json(path)
img_root = "/aidata/anders/3D-head/user_depth/anders/imgs"

gt_poses, pred_poses = [], []
for i, (frame, gt_pose) in enumerate(zip(annos['frame_list'], poses)):
    name = frame['name']
    img_path = os.path.join(img_root, name)
    img = cv2.imread(img_path)
    # print('-' * 100)

    for lb in frame['labels']:
        box2d = lb['box2d']
        keypoints = lb['keypoints']
        kps = np.asarray([keypoints[k] for k in keypoints])
        kps = kps.astype(np.int32)[:, ::-1]

        for l, kp in enumerate(kps):
            if 0 <= l < 17:
                color = [205, 133, 63]
            elif 17 <= l < 27:
                # eyebrows
                color = [205, 186, 150]
            elif 27 <= l < 39:
                # eyes
                color = [238, 130, 98]
            elif 39 <= l < 48:
                # nose
                color = [205, 96, 144]
            elif 48 <= l < 68:
                color = [0, 191, 255]
            cv2.circle(img, (kps[l][0], kps[l][1]), 3, color, -1)
            cv2.circle(img, (kps[l][0], kps[l][1]), 2, (255, 255, 255), -1)
            line_width = 1
            if l not in [16, 21, 26, 32, 38, 42, 47, 59, 67]:
                start_point = (kps[l][0], kps[l][1])
                end_point = (kps[l + 1][0], kps[l + 1][1])
                cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
            elif l == 32:
                start_point = (kps[l][0], kps[l][1])
                end_point = (kps[27][0], kps[27][1])
                cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
            elif l == 38:
                start_point = (kps[l][0], kps[l][1])
                end_point = (kps[33][0], kps[33][1])
                cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
            elif l == 59:
                start_point = (kps[l][0], kps[l][1])
                end_point = (kps[48][0], kps[48][1])
                cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
            elif l == 67:
                start_point = (kps[l][0], kps[l][1])
                end_point = (kps[60][0], kps[60][1])
                cv2.line(img, start_point, end_point, (0, 0, 0), line_width)
        # for kp in kps:
        #     img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)
        if i == 26:
            cv2.imwrite("output.jpg", img)
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
            kps, bfm.kpt_ind, max_iter=5)
        fitted_angles = fitted_angles * (180 / np.pi)
        gt_roll, gt_pitch, gt_yaw = gt_pose[1], gt_pose[2], gt_pose[3]
        pred_pitch, pred_yaw, pred_roll = fitted_angles
        if pred_pitch < 0:
            pred_pitch += 180
        else:
            pred_pitch -= 180
        pred_poses.append([pred_roll, pred_pitch, pred_yaw])
        gt_poses.append([gt_roll, -(gt_pitch - 3), -gt_yaw])

from matplotlib import pyplot as plt

pred_poses = np.stack(np.asarray(pred_poses), axis=0)
gt_poses = np.stack(np.asarray(gt_poses), axis=0)
diff_poses = gt_poses - pred_poses
n_idxs = np.arange(len(gt_poses))
fig, ax = plt.subplots()
ax.plot(n_idxs, gt_poses[:, 0], color="blue", label="gt roll")
ax.plot(n_idxs, gt_poses[:, 1], color="darkorange", label="gt pitch")
ax.plot(n_idxs, gt_poses[:, 2], color="darkgreen", label="gt yaw")

ax.plot(n_idxs,
        pred_poses[:, 0],
        "--",
        color="blue",
        label="pred roll",
        alpha=0.6)
ax.plot(n_idxs,
        pred_poses[:, 1],
        "--",
        color="darkorange",
        label="pred pitch",
        alpha=0.6)
ax.plot(n_idxs,
        pred_poses[:, 2],
        "--",
        color="darkgreen",
        label="pred yaw",
        alpha=0.6)
plt.legend()
plt.grid()
plt.savefig("foo.jpg")