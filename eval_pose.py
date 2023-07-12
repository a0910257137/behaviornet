import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pprint import pprint
from utils.morphable_model import MorphabelModel
from utils.io import *
from matplotlib import pyplot as plt


def plot(gt_poses, pred_poses):
    pred_poses = np.stack(np.asarray(pred_poses), axis=0)
    gt_poses = np.stack(np.asarray(gt_poses), axis=0)
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

    ax.set_xlabel("Frame index", fontsize=15)
    ax.set_ylabel("Pose angle", fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig("foo.jpg")


def draw_kps(img, kps):
    kps = kps[:, ::-1]
    color = (0, 0, 255)
    kps = kps.astype(np.int32)
    for l in range(kps.shape[0]):
        if 0 <= l < 17:
            color = [205, 133, 63]
        elif 17 <= l < 27:
            # eyebrows
            color = [205, 186, 150]
        elif 27 <= l < 39:
            # nose
            color = [238, 130, 98]
        elif 39 <= l < 48:
            #eyes
            color = [205, 96, 144]
        elif 48 <= l < 68:
            color = [0, 191, 255]
        cv2.circle(img, (kps[l][0], kps[l][1]), 1, color, 3)
        cv2.circle(img, (kps[l][0], kps[l][1]), 1, (255, 255, 255), -1)
        if l not in [16, 21, 26, 32, 38, 42, 47, 59, 67]:
            start_point = (kps[l][0], kps[l][1])
            end_point = (kps[l + 1][0], kps[l + 1][1])
            cv2.line(img, start_point, end_point, (0, 0, 0), 1)
        elif l == 32:
            start_point = (kps[l][0], kps[l][1])
            end_point = (kps[27][0], kps[27][1])
            cv2.line(img, start_point, end_point, (0, 0, 0), 1)
        elif l == 38:
            start_point = (kps[l][0], kps[l][1])
            end_point = (kps[33][0], kps[33][1])
            cv2.line(img, start_point, end_point, (0, 0, 0), 1)
        elif l == 59:
            start_point = (kps[l][0], kps[l][1])
            end_point = (kps[48][0], kps[48][1])
            cv2.line(img, start_point, end_point, (0, 0, 0), 1)
        elif l == 67:
            start_point = (kps[l][0], kps[l][1])
            end_point = (kps[60][0], kps[60][1])
            cv2.line(img, start_point, end_point, (0, 0, 0), 1)
    return img


def tdmm(anno_path, img_root):
    bfm = MorphabelModel('/aidata/anders/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    print('initialize bfm model success')
    gt_poses, pred_poses = [], []
    annos = load_json(anno_path)
    progress = tqdm(total=len(annos["frame_list"]))
    for i, frame in enumerate(annos["frame_list"]):
        progress.update(1)
        name = frame["name"]
        # img = cv2.imread(os.path.join(img_root, name))
        for lb in frame["labels"]:
            attr = lb["attributes"]
            gt_pose = attr["pose"]
            gt_roll, gt_pitch, gt_yaw = gt_pose["roll"], gt_pose[
                "pitch"], gt_pose["yaw"]
            keypoints = lb["keypoints"]
            tmp_kps = []
            for key in keypoints.keys():
                kp = keypoints[key]
                tmp_kps.append(kp)
            kps = np.stack(tmp_kps)
            # img = draw_kps(img, kps)
            # cv2.imwrite(
            #     os.path.join(
            #         "/aidata/anders/data_collection/okay/pose/2023-07-05/003171/det_tdmm/frame_{}.jpg"
            #         .format(i)), img)
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps[:, ::-1], X_ind, idxs=None, max_iter=5)
            fitted_angles = fitted_angles * (180 / np.pi)
            pred_pitch, pred_yaw, pred_roll = fitted_angles
            if pred_pitch < 0:
                pred_pitch += 180
            else:
                pred_pitch -= 180
            pred_poses.append([pred_roll, pred_pitch, pred_yaw])
            gt_poses.append([gt_roll, -(gt_pitch - 3), -(gt_yaw)])
    return gt_poses, pred_poses


def parse_config():
    parser = argparse.ArgumentParser('Argparser for evaluation pose')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate 3D parameters')
    gt_poses, pred_poses = tdmm(args.anno_path, args.img_root)
    plot(gt_poses, pred_poses)