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

lb_dir = "/aidata/relabel/lnmks/Stage1_4_12_4_26"
svae_dir = "/aidata/relabel/lnmks/Stage1_4_12_4_26/stage1_total/imgs"
lb_names = ["anders", "andy", "chiamin", "poyuan"]
bdd_results = {"frame_list": []}

for lb_name in tqdm(lb_names):
    path_list = glob(os.path.join(lb_dir, lb_name, "demo_test", "*.json"))
    for path in path_list:
        number = path.split("/")[-1].split("-")[-1].split(".")[0]
        number = int(number)
        if number > 401 and lb_name == 'anders':
            continue

        annos = load_json(path)
        shapes = annos['shapes']
        angles = shapes[0]['angles']
        VALIDATE_FLAG = angles['validate']
        if VALIDATE_FLAG != True:
            continue

        img_path = path.replace(".json", ".jpg")
        name = img_path.split("/")[-1]
        lb_infos = {"dataset": "demo_test", "name": name, "labels": []}
        copy_base_infos = copy.deepcopy(base_infos)

        copy_base_infos['attributes']['pitch'] = angles['pitch']
        copy_base_infos['attributes']['roll'] = angles['roll']
        copy_base_infos['attributes']['yaw'] = angles['yaw']
        tmp_kps = []
        for keypoint_infos in shapes[1:]:
            keys = keypoint_infos.keys()
            for kp in keypoint_infos['points']:
                tmp_kps.append(kp[::-1])
        kps = np.asarray(tmp_kps)
        tl = kps.min(axis=0)
        br = kps.max(axis=0)

        for key, kp in zip(copy_base_infos['keypoints'].keys(), kps):
            copy_base_infos['keypoints'][key] = list(kp)

        copy_base_infos['box2d']['x1'] = int(tl[1])
        copy_base_infos['box2d']['y1'] = int(tl[0])
        copy_base_infos['box2d']['x2'] = int(br[1])
        copy_base_infos['box2d']['y2'] = int(br[0])
        lb_infos['labels'].append(copy_base_infos)

        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(svae_dir, name), img)
        bdd_results['frame_list'].append(lb_infos)
dump_json(
    path=
    "/aidata/relabel/lnmks/Stage1_4_12_4_26/stage1_total/annos/demo_test_finished.json",
    data=bdd_results)
