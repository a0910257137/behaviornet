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
from utils.io import load_text
from behavior_predictor.inference import BehaviorPredictor


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def img_gen(img_root, anno_box2d, anno_lnmk):
    anno_box2d = load_json(anno_box2d)
    anno_lnmk = load_json(anno_lnmk)
    print('Number of Box2d_frame: %i' % len(anno_box2d['frame_list']))
    print('Number of Landmark_frame: %i' % len(anno_lnmk['frame_list']))
    # as search table for any bbox2d\
    tmp_anno_box2d = {
        frame['name']: frame['labels']
        for frame in anno_box2d['frame_list']
    }

    for frame in anno_lnmk['frame_list']:
        name = frame['name']
        splitted_name = name.split('_')
        box2d_name = str()
        for i, n in enumerate(splitted_name[:2]):
            if i == 0:
                box2d_name += n
            else:
                box2d_name += '_' + n
        box2d_name += '.jpg'
        index_lb = int(splitted_name[-1].split('.')[0])
        index_box2d_anno = tmp_anno_box2d[box2d_name]
        box2d_lb = index_box2d_anno[index_lb]
        box2d = box2d_lb['box2d']
        tl = np.array([box2d['y1'], box2d['x1']]).astype(np.int32)
        br = np.array([box2d['y2'], box2d['x2']]).astype(np.int32)
        keypoints = frame['labels'][0]['keypoints']
        # implement shift landmarks to original image
        # img_path = os.path.join(img_root, 'crop_imgs', name)
        # img = cv2.imread(img_path)
        # for k in keypoints.keys():
        #     for kp in np.asarray(keypoints[k]):
        #         kp = kp.astype(int)
        #         img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)
        # cv2.imwrite('output.jpg', img)
        img_path = os.path.join(img_root, box2d_name)
        # img = cv2.imread(img_path)
        for k in keypoints.keys():
            tmp = []
            for kp in np.asarray(keypoints[k]):
                kp = kp + tl
                tmp.append(kp)
                kp = kp.astype(int)
                # img = cv2.circle(img, tuple(kp[::-1]), 2, (0, 255, 0), -1)
            keypoints[k] = tmp

        tmp_anno_box2d[box2d_name][index_lb]["keypoints"] = keypoints
    # implement re-dump
    merged_keys = sorted(list(tmp_anno_box2d.keys()))
    for name_as_key in merged_keys:
        lbs = tmp_anno_box2d[name_as_key]
        frame = {
            'dataset': 'Demo_Landmarks',
            'sequence': 'merge',
            'name': name_as_key,
            'labels': []
        }
        img_path = os.path.join(img_root, name_as_key)
        print(img_path)
        img = cv2.imread(img_path)
        for lb in lbs:
            keypoints = lb['keypoints']
            for k in keypoints.keys():
                for kp in np.asarray(keypoints[k]):
                    kp = kp.astype(int)
                    img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)
        save_dir = os.path.join(img_root, 'synthesized_imgs')
        make_dir(save_dir)
        cv2.imwrite(os.path.join(save_dir, name_as_key), img)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_box2d')
    parser.add_argument('--anno_lnmk')
    parser.add_argument('--img_root')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print(f'Result imgs generating')
    img_gen(args.img_root, args.anno_box2d, args.anno_lnmk)
