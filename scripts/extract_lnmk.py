import cv2
import os
import numpy as np
import json
from pathlib import Path
from pprint import pprint
from glob import glob
import argparse
from tqdm import tqdm


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


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


def gen_landmark(anno_path, save_path, img_root):

    annos = load_json(anno_path)
    # keep_lnmk_schema = dict(countour_face=[0, 8, 16],
    #                         left_eyebrow=[],
    #                         right_eyebrow=[],
    #                         nose=[0, 3],
    #                         left_eye=[0, 1, 2, 3, 4, 5],
    #                         right_eye=[0, 1, 2, 3, 4, 5],
    #                         outer_lip=[0, 2, 3, 4, 6, 8, 9, 10],
    #                         inner_lip=[])
    keep_lnmk_schema = dict(countour_face=[0, 8, 16],
                            left_eyebrow=[],
                            right_eyebrow=[],
                            nose=[0, 3],
                            left_eye=[0, 1, 2, 3, 4, 5],
                            right_eye=[0, 1, 2, 3, 4, 5],
                            outer_lip=[0, 2, 3, 4, 6, 8, 9, 10],
                            inner_lip=[])
    keys = [
        "countour_face", "left_eyebrow", "right_eyebrow", "left_eye",
        "right_eye", "nose", "outer_lip", "inner_lip"
    ]
    bdd_results = {'frame_list': []}
    for frame in tqdm(annos["frame_list"]):
        name = frame['name']
        img_path = os.path.join(img_root, name)
        # img = cv2.imread(img_path)

        for lb in frame["labels"]:
            box2d = lb['box2d']
            tl = (box2d['y1'], box2d['x1'])
            br = (box2d['y2'], box2d['x2'])
            tl = np.asarray(tl).astype(int)
            br = np.asarray(br).astype(int)
            # make broader
            tl = tl - 1
            br = br + 1
            # crop_img = img[tl[0]:br[0], tl[1]:br[1], :]
            landmarks = lb['keypoints']
            for k in keep_lnmk_schema:
                schema_indices = np.asarray(keep_lnmk_schema[k]).astype(int)
                lnmks = np.asarray(landmarks[k])
                if len(schema_indices) == 0:
                    landmarks[k] = []
                    continue
                kps = lnmks[schema_indices]
                kps = kps - tl
                kps = kps.tolist()
                landmarks[k] = kps
        bdd_results['frame_list'].append(frame)
        # if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        #     continue
        # save_path = os.path.join(save_cimgs_root, name)
        # cv2.imwrite(save_path, crop_img)
    dump_json(save_path, bdd_results)


def parse_config():
    parser = argparse.ArgumentParser(
        'Argparser for generating number of landmarks')
    parser.add_argument('--anno_path')
    parser.add_argument('--save_path')
    parser.add_argument('--img_root', default=str)
    parser.add_argument('--save_cimgs_path', default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()

    gen_landmark(args.anno_path, args.save_path, args.img_root)
