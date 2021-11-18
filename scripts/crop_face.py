import sys
import cv2
import os
import json
import argparse
import commentjson
import numpy as np
from pprint import pprint
from pathlib import Path
from draw import *
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import *
from behavior_predictor.inference import BehaviorPredictor


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def crop_imgs(anno_path, save_path):
    annos = load_json(anno_path)
    resized_ratio = np.asarray([256, 256])
    for frame in annos['frame_list']:
        for lb in frame['labels']:
            keypoints = lb['keypoints']
            tmp_kps = []
            for k in keypoints.keys():
                kp = np.reshape(keypoints[k], [-1, 2])
                # kp = kp * resized_ratio
                kp = np.stack(kp, axis=0)
                tmp_kps.append(kp)
            kps = np.concatenate(tmp_kps, axis=0)

    dump_json(path=save_path, data=annos)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_path')
    parser.add_argument('--save_path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Crop facial image')
    crop_imgs(args.anno_path, args.save_path)
