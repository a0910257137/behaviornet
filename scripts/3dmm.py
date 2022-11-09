import os, sys
import numpy as np
from time import time
import cv2
import json
from tqdm import tqdm
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel
from utils.pose import PoseEstimator


# --------------------- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def tdmm(annos_path, img_root):
    bfm = MorphabelModel('/aidata/anders/objects/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    root_path = "./test"
    b_imgs = np.load(os.path.join(root_path, "b_imgs.npy")) * 255
    b_origin_sizes = np.load(os.path.join(root_path, "b_origin_sizes.npy"))
    b_coords = np.load(os.path.join(root_path, "b_coords.npy"))
    for imgs, origin_sizes, n_coords in zip(b_imgs, b_origin_sizes, b_coords):
        mask = np.all(np.isfinite(n_coords), axis=-1)
        n_coords = np.reshape(n_coords[mask], (-1, 70, 3))
        n_coords = n_coords[:, 2:, :2]
        n_coords = n_coords[..., ::-1]
        for coords in n_coords:
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                coords, X_ind, max_iter=2)

            fitted_angles = np.asarray(fitted_angles)
            fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
            transformed_vertices = bfm.transform(fitted_vertices, fitted_s,
                                                 fitted_angles, fitted_t)
            transformed_vertices = np.reshape(transformed_vertices, (-1))
            landmarks = transformed_vertices[valid_ind]
            landmarks = np.reshape(landmarks, (landmarks.shape[0] // 3, 3))
            landmarks = landmarks[:, :2]
            fitted_angles *= (180 / np.pi)
            for kp in landmarks:
                kp = kp.astype(np.int32)
                kp = kp[:2]
                imgs = cv2.circle(imgs, tuple(kp), 3, (0, 255, 0), -1)
            cv2.imwrite("./output.jpg", imgs[..., ::-1])
            print(fitted_angles)
            xxx

    print('initialize bfm model success')
    annos = load_json(annos_path)

    # For AFLW
    # x_68_idx = np.reshape(np.transpose(X_ind_all), (-1))
    # idxs = list(range(7)) + [7, 9, 10, 12] + [13] + [16, 18]
    # X_idxs = [8, 17, 19, 21, 22, 24, 26, 27, 30, 33, 36, 42, 48, 54]
    # X_ind_all = X_ind_all[:, X_idxs]

    for frame in tqdm(annos["frame_list"]):
        name = frame["name"]
        img_path = os.path.join(img_root, name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        for lb in frame["labels"]:
            tmp_kps = []
            keypoints = lb["keypoints"]
            for key in keypoints.keys():
                kp = keypoints[key]
                tmp_kps.append(kp)

            tmp_kps = np.stack(tmp_kps)
            kps = tmp_kps
            # kps = kps[idxs]
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps[:, ::-1], X_ind, max_iter=20)
            fitted_angles = np.asarray(fitted_angles)
            fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
            transformed_vertices = bfm.transform(fitted_vertices, fitted_s,
                                                 fitted_angles, fitted_t)
            transformed_vertices = np.reshape(transformed_vertices, (-1))
            # transformed_vertices = np.reshape(
            #     transformed_vertices,
            #     (np.shape(transformed_vertices)[0] // 3, 3))
            # transformed_vertices = transformed_vertices[:, :2]
            # bg = np.zeros_like(img)
            # for vertex in transformed_vertices:
            #     vertex = vertex.astype(np.int32)
            #     bg = cv2.circle(bg, tuple(vertex), 2, (0, 255, 0), -1)
            # img = cv2.addWeighted(img, 0.8, bg, 0.1, 0)
            # cv2.imwrite("./output.jpg", img)
            landmarks = transformed_vertices[valid_ind]
            landmarks = np.reshape(landmarks, (landmarks.shape[0] // 3, 3))
            landmarks = landmarks[:, :2]
            fitted_angles *= (180 / np.pi)

            yaw = fitted_angles[1]
            #TODO: recursive to find the right 3D head
            if yaw < -25 or yaw > +25:
                print('-' * 100)
                print(fitted_angles)
                print(fitted_s)
                for kp in landmarks:
                    kp = kp.astype(np.int32)
                    kp = kp[:2]
                    img = cv2.circle(img, tuple(kp), 3, (0, 255, 0), -1)
                cv2.imwrite("./output.jpg", img)
                xxx
    return


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate 3D parameters')
    tdmm(args.anno_path, args.img_root)
