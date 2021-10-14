import numpy as np
import os
import cv2
from utils.io import *
from pprint import pprint

annos1 = load_json("/aidata/anders/objects/WF/annos/BDD_train.json")
for frame in annos1['frame_list']:
    print(len(frame['labels']))
annos2 = load_json("/aidata/anders/objects/WF/annos/BDD_val.json")
bdd_annos = {"frame_list": []}
bdd_annos["frame_list"] = annos1["frame_list"] + annos2["frame_list"]
dump_json("/aidata/anders/objects/WF/annos/total.json", bdd_annos)
path = "/aidata/anders/objects/WF/annos/BDD_val,json"
annos = load_json(path)
img_root = "/aidata/anders/objects/WF/imgs"

for frame in annos['frame_list'][6:]:
    name = frame['name']
    img_path = os.path.join(img_root, name)
    img = cv2.imread(img_path)
    for lb in frame['labels']:
        box2d = lb['box2d']
        tl = (int(box2d['x1']), int(box2d['y1']))
        br = (int(box2d['x2']), int(box2d['y2']))
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    cv2.imwrite("./output.jpg", img)