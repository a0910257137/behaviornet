import os, sys
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
from pprint import pprint
from math import cos, sin

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel
import utils.mesh as mesh
from utils.io import *


def angle2matrix(x, y, z):
    # x
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])

    # y
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def mian(root_dir):
    # init camera matrix
    cam_mtx = np.load(os.path.join(root_dir, "cam2_mtx.npy"))
    dist_coeefs = np.zeros((4, 1))

    bfm = MorphabelModel('/aidata/anders/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = X_ind_all.flatten('F')
    print('----initialize bfm model success----')
    img_root = os.path.join(root_dir, "imgs")
    annos = load_json(os.path.join(root_dir, "annos/BDD_poses.json"))
    for frame in annos["frame_list"]:
        img = cv2.imread(os.path.join(img_root, frame["name"]))
        cv2.imwrite("output.jpg", img)

        for lb in frame["labels"]:
            attrs = lb["attributes"]
            gt_roll, gt_pitch, gt_yaw = attrs["pose"]["roll"], attrs["pose"][
                "pitch"], attrs["pose"]["yaw"]
            keypoints = lb["keypoints"]
            # head poses are solved by 3dmm
            kps = np.asarray([keypoints[k][::-1] for k in keypoints.keys()])
            # for kp in kps:
            #     kp = kp.astype(np.int32)
            #     img = cv2.circle(img, tuple(kp), 3, (0, 255, 0), -1)
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps, X_ind, idxs=None, max_iter=5)
            fitted_angles *= (180 / np.pi)
            pred_pitch, pred_yaw, pred_roll = fitted_angles
            if pred_pitch < 0:
                pred_pitch = -(180 + pred_pitch)
            elif pred_pitch > 0:
                pred_pitch = (180 - pred_pitch)
            vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
            vertices = 1e-4 * vertices
            vertices = vertices.reshape([-1])
            template_lnmks = vertices[valid_ind]
            template_lnmks = template_lnmks.reshape(
                [template_lnmks.shape[0] // 3, 3])
            ret, r_vec, t_vec = cv2.solvePnP(template_lnmks,
                                             kps,
                                             cam_mtx,
                                             dist_coeefs,
                                             flags=cv2.SOLVEPNP_EPNP)

            ret, r_vec, t_vec = cv2.solvePnP(template_lnmks, kps, cam_mtx,
                                             dist_coeefs, r_vec, t_vec, True)
            r_mat, _ = cv2.Rodrigues(r_vec)
            p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
            _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
            pred_pitch, pred_yaw, pred_roll = u_angle.flatten()
            if pred_pitch < 0:
                pred_pitch = -(180 + pred_pitch)
            elif pred_pitch > 0:
                pred_pitch = (180 - pred_pitch)
            print("roll: {}, pitch: {}, yaw: {}".format(pred_roll, pred_pitch,
                                                        pred_yaw))
            print("roll: {}, pitch: {}, yaw: {}".format(gt_roll, gt_pitch,
                                                        gt_yaw))

    


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--root_dir')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate head pose')
    mian(args.root_dir)
