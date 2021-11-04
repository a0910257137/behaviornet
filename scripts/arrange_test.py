import cv2
import os
import numpy as np
import json
from pathlib import Path
from pprint import pprint
from glob import glob


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


file_paths = "/aidata/anders/objects/incar/img_copy/annos/*.txt"
file_paths = sorted(glob(file_paths))
img_root = "/aidata/anders/objects/incar/img_copy"
bdd_results = {"frame_list": []}

for path in file_paths[1:501]:
    name = path.split("/")[-1].replace("txt", "jpg")
    print('-' * 100)
    print(name)
    frame_infos = {
        "dataset": None,
        "sequence": None,
        "name": name,
        "labels": []
    }
    lines = load_text(path)
    img = cv2.imread(os.path.join(img_root, name))
    cv2.imwrite(os.path.join("/aidata/anders/objects/WF/test_imgs", name), img)
    h, w, c = img.shape
    for line in lines:
        line = line.split(" ")
        tmp_line = []
        for l in line:
            if l == " " or l == "":
                continue
            tmp_line.append(l)
        line = tmp_line

        line = np.asarray(line).astype(np.float)
        box2d = line[1:]
        center_xy = (box2d[0] * w, box2d[1] * h)
        obj_wh = (box2d[2] * w, box2d[3] * h)
        tl = np.asarray(center_xy) - np.asarray(obj_wh) / 2
        br = np.asarray(center_xy) + np.asarray(obj_wh) / 2
        lb_infos = {
            "box2d": {
                "x1": round(tl[0], 1),
                "y1": round(tl[1], 1),
                "x2": round(br[0], 1),
                "y2": round(br[1], 1)
            },
            "category": "FACE",
            "attributes": {}
        }
        frame_infos['labels'].append(lb_infos)
        # img = cv2.rectangle(img, tuple(tl.astype(int)), tuple(br.astype(int)),
        #                     (0, 255, 0), 3)
        # cv2.imwrite("output.jpg", img)
    bdd_results['frame_list'].append(frame_infos)
dump_json("/aidata/anders/objects/WF/annos/test.json", bdd_results)
