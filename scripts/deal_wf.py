import cv2
import os
import copy
import numpy as np
import json
from pathlib import Path
from pprint import pprint


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


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


path = "/aidata/anders/objects/WF/wider_face_split/wider_face_val_bbx_gt.txt"
lines = load_text(path)
img_root = "/aidata/anders/objects/WF/WIDER_val/images"
save_root = "/aidata/anders/objects/WF/imgs/"
bdd_results = {"frame_list": []}

for i, line in enumerate(lines):
    tmp_line = list(line.split(" "))
    if len(tmp_line) == 11:
        box_line = tmp_line
        x1, y1, w, h = np.asarray(box_line[:4]).astype(float)
        tl = (int(x1), int(y1))
        br = (int(x1 + w), int(y1 + h))
        lb_info = {
            "box2d": {
                "x1": x1,
                "y1": y1,
                "x2": x1 + w,
                "y2": y1 + h,
            },
            "category": "FACE",
            "attributes": {}
        }
        frame_infos['labels'].append(lb_info)
        # img = cv2.rectangle(img, tl, br, (0, 255, 0), 3)
        if int(box_num) - 1 == int(c_frame):
            bdd_results['frame_list'].append(frame_infos)
        else:
            c_frame += 1

    elif len(tmp_line) == 1:
        # might be file name or bbox number
        unkown_line = tmp_line[0].split("/")
        if len(unkown_line) == 2:
            frame_infos = {
                "dataset": None,
                "sequence": None,
                "name": unkown_line[-1],
                "labels": []
            }
            file_line = tmp_line
            # img = cv2.imread(os.path.join(img_root, tmp_line[0]))
            # cv2.imwrite(os.path.join(save_root, unkown_line[-1]), img)
        elif len(unkown_line) == 1:
            box_num = int(unkown_line[0])
            c_frame = 0
dump_json("/aidata/anders/objects/WF/annos/BDD_val,json", bdd_results)
