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


def gen_landmark(anno_path, save_path):

    annos = load_json(anno_path)
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
    for frame in tqdm(annos["frame_list"]):
        for lb in frame["labels"]:
            landmarks = lb['keypoints']
            for k in keep_lnmk_schema:
                schema_indices = np.asarray(keep_lnmk_schema[k]).astype(int)
                lnmks = np.asarray(landmarks[k])
                if len(schema_indices) == 0:
                    landmarks[k] = []
                    continue

                kps = lnmks[schema_indices]
                kps = kps.tolist()
                landmarks[k] = kps

    dump_json(save_path, annos)


def parse_config():
    parser = argparse.ArgumentParser(
        'Argparser for generating number of landmarks')
    parser.add_argument('--anno_path')
    parser.add_argument('--save_path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()

    gen_landmark(args.anno_path, args.save_path)
