import face_recognition
import os
from glob import glob
from pprint import pprint
import numpy as np
import cv2
import json
import copy
from pathlib import Path
from tqdm import tqdm

path = "/aidata/anders/data_collection/okay/pose/anders/2023-07-03/imgs/*.jpg"
img_paths = glob(path)


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


bdd_base = {
    "name":
    None,
    "dataset":
    None,
    "labels": [{
        "box2d": {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None
        },
        "keypoints": []
    }]
}

bdd_results = {"frame_list": []}
#TODO: make bdd
for img_path in tqdm(img_paths):
    image = face_recognition.load_image_file(img_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    name = img_path.split("/")[-1]
    bdd = copy.deepcopy(bdd_base)
    bdd['dataset'] = "pose"
    bdd['name'] = name
    for lb in face_landmarks_list:
        keys = lb.keys()
        tmp = [lb[k] for k in keys]
        kps = np.concatenate(tmp, axis=0)
        tl = np.min(kps, axis=0).astype(np.int32)
        br = np.max(kps, axis=0).astype(np.int32)
        lb_base = copy.deepcopy(bdd['labels'][0])
        bdd['labels'].pop()
        lb_base["box2d"]['x1'] = int(tl[0])
        lb_base["box2d"]['y1'] = int(tl[1])
        lb_base["box2d"]['x2'] = int(br[0])
        lb_base["box2d"]['y2'] = int(br[1])
        bdd['labels'].append(lb_base)
    bdd_results["frame_list"].append(bdd)
dump_json(
    path=
    "/aidata/anders/data_collection/okay/pose/anders/2023-07-03/annos/BDD_pose_3d_lnmks.json",
    data=bdd_results)
