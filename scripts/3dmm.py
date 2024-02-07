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


def gen_vertices(bfm, fitted_s, fitted_angles, fitted_t, fitted_sp, fitted_ep):
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s,
                                         fitted_angles, fitted_t)
    return np.reshape(transformed_vertices, (-1))


def tdmm(annos_path, img_root, save_path):
    bfm = MorphabelModel('/aidata/anders/3D-head/3DDFA/BFM/BFM.mat')
    X_ind = bfm.kpt_ind
    X_ind_all = np.stack([X_ind * 3, X_ind * 3 + 1, X_ind * 3 + 2])
    X_ind_all = np.concatenate([
        X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
        X_ind_all[:, 27:36], X_ind_all[:, 48:68]
    ],
                               axis=-1)
    valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
    print('initialize bfm model success')
    annos_list = []
    for d in os.listdir(annos_path):
        path = os.path.join(annos_path, d, "annos", "BDD_{}.json".format(d))
        annos_list.append(path)
    for path in tqdm(annos_list):
        print('-' * 100)
        print(path)
        annos = load_json(path)
        i = 0
        anlges = []
        for frame in annos["frame_list"]:
            for lb in frame["labels"]:
                tmp_kps = []
                keypoints = lb["keypoints"]
                for key in keypoints.keys():
                    kp = keypoints[key]
                    tmp_kps.append(kp)
                tmp_kps = np.stack(tmp_kps)
                kps = tmp_kps
                fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
                    kps[:, ::-1], X_ind, idxs=None, max_iter=4)
                fitted_angles *= (180 / np.pi)
                pitch, yaw, roll = fitted_angles
                if pitch < 0:
                    pitch = -(180 + pitch)
                elif pitch > 0:
                    pitch = (180 - pitch)
                lb["attributes"] = {
                    "pose": {
                        "pitch": pitch,
                        "roll": roll,
                        "yaw": yaw
                    },
                    "valid": True,
                    "small": False
                }

            i += 1
        dump_json(path=path, data=annos)


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
