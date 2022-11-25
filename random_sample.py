import json
import numpy as np
import os
import cv2
from utils.io import *
import scipy.io as sio
from glob import glob
import copy
from tqdm import tqdm
from utils.io import load_BFM
import random
from pprint import pprint

lnmk_68 = [
    'countour_lnmk_0',
    'countour_lnmk_1',
    'countour_lnmk_2',
    'countour_lnmk_3',
    'countour_lnmk_4',
    'countour_lnmk_5',
    'countour_lnmk_6',
    'countour_lnmk_7',
    'countour_lnmk_8',
    'countour_lnmk_9',
    'countour_lnmk_10',
    'countour_lnmk_11',
    'countour_lnmk_12',
    'countour_lnmk_13',
    'countour_lnmk_14',
    'countour_lnmk_15',
    'countour_lnmk_16',
    'left_eyebrow_17',
    'left_eyebrow_18',
    'left_eyebrow_19',
    'left_eyebrow_20',
    'left_eyebrow_21',
    'right_eyebrow_22',
    'right_eyebrow_23',
    'right_eyebrow_24',
    'right_eyebrow_25',
    'right_eyebrow_26',
    'nose_lnmk_27',
    'nose_lnmk_28',
    'nose_lnmk_29',
    'nose_lnmk_30',
    'nose_lnmk_31',
    'nose_lnmk_32',
    'nose_lnmk_33',
    'nose_lnmk_34',
    'nose_lnmk_35',
    'left_eye_lnmk_36',
    'left_eye_lnmk_37',
    'left_eye_lnmk_38',
    'left_eye_lnmk_39',
    'left_eye_lnmk_40',
    'left_eye_lnmk_41',
    'right_eye_lnmk_42',
    'right_eye_lnmk_43',
    'right_eye_lnmk_44',
    'right_eye_lnmk_45',
    'right_eye_lnmk_46',
    'right_eye_lnmk_47',
    'outer_lip_lnmk_48',
    'outer_lip_lnmk_49',
    'outer_lip_lnmk_50',
    'outer_lip_lnmk_51',
    'outer_lip_lnmk_52',
    'outer_lip_lnmk_53',
    'outer_lip_lnmk_54',
    'outer_lip_lnmk_55',
    'outer_lip_lnmk_56',
    'outer_lip_lnmk_57',
    'outer_lip_lnmk_58',
    'outer_lip_lnmk_59',
    'inner_lip_lnmk_60',
    'inner_lip_lnmk_61',
    'inner_lip_lnmk_62',
    'inner_lip_lnmk_63',
    'inner_lip_lnmk_64',
    'inner_lip_lnmk_65',
    'inner_lip_lnmk_66',
    'inner_lip_lnmk_67',
]

path_lists = glob("/aidata/anders/objects/3D-head/NIR/S1-10/sub1/*.png")
for path in path_lists:
    img = cv2.imread(path)
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            print(img[i, j])
xxxx
annos = load_json("/aidata/anders/objects/3D-head/exp/annos/samples.json")
img_root = "/aidata/anders/objects/3D-head/exp/imgs"
for frame in annos["frame_list"]:
    name = frame["name"]
    img_path = os.path.join(img_root, name)
    img = cv2.imread(img_path)
    cv2.imwrite("output.jpg", img)
    xxx
    if 'image01488' in name:
        img = cv2.imread(img_path)
        for lb in frame['labels']:
            box2d = lb['box2d']
            tl = (int(box2d['x1']), int(box2d['y1']))
            br = (int(box2d['x2']), int(box2d['y2']))
            img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            keypoints = lb['keypoints']
            for i, k in enumerate(keypoints.keys()):
                kp = np.asarray(keypoints[k]).astype(np.int32)
                if 38 < i < 46:
                    img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 0, 255), -1)
                else:
                    img = cv2.circle(img, tuple(kp[::-1]), 3, (0, 255, 0), -1)
        cv2.imwrite("output.jpg", img)
        xxx