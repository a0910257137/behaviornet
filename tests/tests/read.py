import copy
from utils.io import *
import os
import cv2
import numpy as np
from pprint import pprint
from glob import glob
import random

img_root = "/aidata/anders/objects/landmarks/demo_test/crop_imgs"
annos = load_json(
    "/aidata/anders/objects/landmarks/demo_test/annos/BDD_old.json")

bdd_results = {"frame_list": []}
idx = random.randint(0, len(annos["frame_list"]) - 1)
frame = annos["frame_list"][idx]
name = frame["name"]
img_path = os.path.join(img_root, name)
print(idx)
print(img_path)
img = cv2.imread(img_path)
for lb in frame["labels"]:
    keypoints = lb["keypoints"]
    keys = keypoints.keys()
    tmp_kps = []
    box2d = lb["box2d"]
    tl = (int(box2d["x1"]), int(box2d["y1"]))
    br = (int(box2d["x2"]), int(box2d["y2"]))
    cv2.rectangle(img, tl, br, (0, 0, 255), 3)
    for key in keys:
        kp = keypoints[key]
        if kp is not None:
            kp = np.asarray(kp)
            kp = kp.astype(np.int32)
            img = cv2.circle(img, tuple(kp[::-1]), 5, (0, 255, 0), -1)

cv2.imwrite("output.jpg", img)
