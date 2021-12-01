import sys
import cv2
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from pprint import pprint
from pathlib import Path
from draw import *
from glob import glob
import numpy as np
import copy
import os
from pprint import pprint
from glob import glob
import random


def load_json(path):
    """The function of loading json file

    Arguments:
        path {str} -- The path of the json file

    Returns:
        list, dict -- The obj stored in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def make_dir(path):
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path, mode=0o755)


def crop_imgs(anno_path, img_root, save_root, use_box2d):
    annos = load_json(anno_path)
    tmp_img_path = []
    bdd_results = {"frame_list": []}
    progress_bar = tqdm(total=len(annos['frame_list']))
    for frame in annos['frame_list']:
        path = os.path.join(img_root, frame["name"])
        name = frame["name"]
        ori_img = cv2.imread(path)
        if ori_img is None:
            continue
        for lb in frame['labels']:
            keypoints = lb['keypoints']
            tmp_kps = []
            for k in keypoints.keys():
                kp = np.reshape(keypoints[k], [-1, 2])
                kp = np.stack(kp, axis=0)
                tmp_kps.append(kp)
            if len(tmp_kps) == 0:
                continue
            landmark = np.concatenate(tmp_kps, axis=0)
            landmark = landmark[:, ::-1]
            xy = np.min(landmark, axis=0).astype(np.int32)
            zz = np.max(landmark, axis=0).astype(np.int32)
            wh = zz - xy + 1

            center = (xy + wh / 2).astype(np.int32)

            boxsize = int(np.max(wh) * 1.2)
            xy = center - boxsize // 2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            img = copy.deepcopy(ori_img)
            height, width, _ = img.shape

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            imgT = img[y1:y2, x1:x2]

            h, w, _ = imgT.shape
            if h == 0 or w == 0:
                continue
            landmark = (landmark - xy)
            keys = list(keypoints.keys())

            for k, kp in zip(keys, landmark):
                keypoints[k] = kp[::-1].tolist()
            if name in tmp_img_path:
                ridx = random.randint(1, 1e5)
                frame["name"] = frame["name"].split('.')[0] + '_face_' + str(
                    ridx) + '.jpg'
                cv2.imwrite(os.path.join(save_root, frame["name"]), imgT)
            else:
                frame["name"] = frame["name"].split('.')[0] + '_face_0.jpg'
                cv2.imwrite(os.path.join(save_root, frame["name"]), imgT)
            tmp_img_path.append(name)
            bdd_results["frame_list"].append(frame)
        progress_bar.update(1)

    dump_json('/aidata/anders/objects/landmarks/FFHQ/annos/BDD_FFHQ_68.json',
              bdd_results)


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--anno_path')
    parser.add_argument('--img_root')
    parser.add_argument('--save_root')
    parser.add_argument('--use_box2d', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Crop facial image')
    crop_imgs(args.anno_path, args.img_root, args.save_root, args.use_box2d)
