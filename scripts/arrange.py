import numpy as np
import cv2
import os
from glob import glob
from pprint import pprint
import copy
from tqdm import tqdm
import json
from pathlib import Path


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


annos = load_json("/aidata/relabel/pose/annos/BDD_demo_test_model3.json")
base_infos = annos['frame_list'][0]['labels'][0]
lb_dir = "/aidata/relabel/lnmks/Stage4_07_01_08_31/chiamin"
save_dir = "/aidata/relabel/lnmks/Stage4_07_01_08_31/download/imgs"
bdd_results = {"frame_list": []}
path_list = glob(os.path.join(lb_dir, "*.json"))

for path in tqdm(path_list):
    annos = load_json(path)
    shapes = annos['shapes']
    img_path = path.replace(".json", ".jpg")

    name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    if img is None:
        continue
    if len(shapes) != 6:
        continue
    angles = shapes[0]['angles']
    VALIDATE_FLAG = angles['validate']
    if VALIDATE_FLAG != True:
        continue
    lb_infos = {"dataset": "AFLW", "name": name, "labels": []}
    copy_base_infos = copy.deepcopy(base_infos)
    copy_base_infos['attributes'] = {
        "pose": {
            "pitch": angles['pitch'],
            "roll": angles['roll'],
            "yaw": angles['yaw']
        },
        "valid": True,
        "is_small": False
    }
    tmp_kps = []
    for keypoint_infos in shapes[1:]:
        keys = keypoint_infos.keys()
        for kp in keypoint_infos['points']:
            tmp_kps.append(kp[::-1])

    if len(tmp_kps) != 68:
        break
    kps = np.asarray(tmp_kps)
    tl = kps.min(axis=0)
    br = kps.max(axis=0)
    for key, kp in zip(copy_base_infos['keypoints'].keys(), kps):
        copy_base_infos['keypoints'][key] = list(kp)
    copy_base_infos['box2d']['x1'] = int(tl[1])
    copy_base_infos['box2d']['y1'] = int(tl[0])
    copy_base_infos['box2d']['x2'] = int(br[1])
    copy_base_infos['box2d']['y2'] = int(br[0])
    copy_base_infos["category"] = "FACE"
    lb_infos['labels'].append(copy_base_infos)
    cv2.imwrite(os.path.join(save_dir, name), img)
    bdd_results['frame_list'].append(lb_infos)
dump_json(
    path=
    "/aidata/relabel/lnmks/Stage4_07_01_08_31/download/annos/BDD_download.json",
    data=bdd_results)
