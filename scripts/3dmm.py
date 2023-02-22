import os, sys
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel

import utils.mesh as mesh
from utils.io import *


def gen_vertices(bfm, fitted_s, fitted_angles, fitted_t, fitted_sp, fitted_ep,
                 valid_ind):
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s,
                                         fitted_angles, fitted_t)
    return np.reshape(transformed_vertices, (-1))


def tdmm(annos_path, img_root, save_path):
    bfm = MorphabelModel('/home3/user/anders/objects/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    print('initialize bfm model success')
    annos = load_json(annos_path)

    i = 0

    angles = []
    for frame in tqdm(annos["frame_list"]):
        name = frame["name"]
        # print(name)
        # img_path = os.path.join(img_root, name)
        # img = cv2.imread(img_path)
        # h, w, c = img.shape
        for lb in frame["labels"]:
            tmp = []
            tmp_kps = []
            keypoints = lb["keypoints"]
            for key in keypoints.keys():
                kp = keypoints[key]
                tmp_kps.append(kp)
            tmp_kps = np.stack(tmp_kps)
            kps = tmp_kps
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                kps[:, ::-1], X_ind, idxs=None, max_iter=20)
            transformed_vertices = gen_vertices(bfm, fitted_s, fitted_angles,
                                                fitted_t, fitted_sp, fitted_ep,
                                                valid_ind)
            landmarks = transformed_vertices[valid_ind]
            landmarks = np.reshape(landmarks, (landmarks.shape[0] // 3, 3))
            fitted_angles *= (180 / np.pi)
            yaw = fitted_angles[1]
            lb['attributes'] = {'yaw': yaw, 'small': False, 'valid': True}
        i += 1
    # dump_json(path=save_path, data=annos)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')
    parser.add_argument('--save_path')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate 3D parameters')
    tdmm(args.anno_path, args.img_root, args.save_path)
