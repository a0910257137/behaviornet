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


img_path = "/aidata/anders/objects/openface/00000.png"
img = cv2.imread(img_path)
path = "/aidata/anders/objects/openface/ffhq-dataset-v2.json"
annos = load_json(path)
for j, k in enumerate(annos):
    frame = annos[k]
    in_the_wild = frame['in_the_wild']
    face_rect = in_the_wild['face_rect']
    x1, y1, x2, y2 = face_rect
    tl = (x1, y1)
    br = (x2, y2)
    img = cv2.rectangle(img, tl, br, (0, 255, 0), 3)
    wider_left_x = in_the_wild['face_landmarks'][2]
    wider_right_x = in_the_wild['face_landmarks'][16]
    tl = (int(wider_left_x[0]), y1)
    br = (int(wider_right_x[0]), y2)
    img = cv2.rectangle(img, tl, br, (255, 0, 0), 3)
    for i, coor in enumerate(in_the_wild['face_landmarks'][:24]):
        coor = np.asarray(coor).astype(int)
        if i == 2 or i == 16:
            img = cv2.circle(img, tuple(coor), 6, (255, 0, 0), -1)
        else:
            img = cv2.circle(img, tuple(coor), 6, (0, 255, 0), -1)
    cv2.imwrite('output{}.jpg'.format(j), img)
