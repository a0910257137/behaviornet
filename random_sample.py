import json
import numpy as np

import os
from utils.io import *
import cv2

annos_path = "/aidata/anders/objects/3D-head/LS3D-W/annos/BDD_partial.json"
img_root = "/aidata/anders/objects/3D-head/LS3D-W/imgs"
annos = load_json(annos_path)
m = len(annos["frame_list"])

for frame in annos["frame_list"]:
    name = frame["name"]
    img_path = os.path.join(img_root, name)
    img = cv2.imread(img_path)
    cv2.imwrite("./test/output.jpg", img)
    xxxx
    # for lb in frame["labels"]:
    #     keypoints = lb["keypoints"]
    #     tmp = []
    #     for key in keypoints.keys():
    #         kp = keypoints[key]
    #         tmp.append(kp)
    #     kps = np.stack(tmp)
    #     for kp in kps:
    #         kp = np.asarray(kp).astype(np.int32)
    #         img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)