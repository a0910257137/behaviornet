import os
import numpy as np

from utils.io import *
import cv2
from tqdm import tqdm
import argparse


def run(anno_path, img_root, save_root):
    annos = load_json(anno_path)
    frame_list = annos["frame_list"]
    m = len(frame_list)
    bdd_results = {"frame_list": frame_list[:m // 2]}
    dump_json(path="/aidata/anders/objects/3D-head/LS3D-W/annos/part1.json",
              data=bdd_results)
    print(m)
    xxx
    # kps = kps[:, ::-1]
    # for j, kp in enumerate(kps):
    #     kp = kp.astype(np.int32)
    #     if j < 17:
    #         img = cv2.circle(img, tuple(kp), 3, (255, 0, 255), -1)
    #     else:
    #         img = cv2.circle(img, tuple(kp), 3, (0, 255, 255), -1)
    #     cv2.imwrite(os.path.join(save_root, name), img)
    # print("Total skip {} frames".format(blank_cnt))


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')
    parser.add_argument('--save_root')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('generate  image ')
    run(args.anno_path, args.img_root, args.save_root)