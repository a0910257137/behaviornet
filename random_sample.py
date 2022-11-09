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
import tensorflow as tf
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
    # for lb in frame["labels"]:
    #     lb["category"] = "FACE"
# dump_json(path="/aidata/anders/objects/3D-head/AFLW2000/annos/BDD_AFLW200.json",
#           data=annos)
# base_infos = {'dataset': '300W', 'sequence': None, 'name': "", 'labels': []}
# bdd_results = {"frame_list": []}
# root_path = "/aidata/anders/objects/3D-head/300W"
# save_root = "/aidata/anders/objects/3D-head/300W/imgs"
# head_model = load_BFM("/aidata/anders/objects/3D-head/3DDFA/BFM/BFM.mat")
# kpt_ind = head_model['kpt_ind']
# X_ind_all = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
# X_ind_all = np.concatenate([
#     X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
#     X_ind_all[:, 27:36], X_ind_all[:, 48:68]
# ],
#                            axis=-1)
# valid_ind = np.reshape(np.transpose(X_ind_all), (-1))
# dirs = os.listdir(root_path)
# tmp_check = []
# for dir in dirs:
#     mat_root = os.path.join(root_path, dir)
#     img_root = os.path.join("/aidata/anders/objects/3D-head/imgs", dir)
#     mat_paths = glob(os.path.join(mat_root, "*mat"))
#     for path in tqdm(mat_paths):
#         img_path = path.replace('mat', 'jpg')
#         img_name = img_path.split('/')[-1]
#         tmp_check.append(img_name)
#         img_path = os.path.join(img_root, img_name)
#         img = cv2.imread(img_path)
#         cv2.imwrite(os.path.join(save_root, img_name), img)
#         h, w, c = img.shape
#         infos = copy.deepcopy(base_infos)
#         infos["sequence"] = dir
#         infos["name"] = img_name
#         mat_dict = sio.loadmat(path)
#         vertices = mat_dict['Fitted_Face'].T
#         vertices = vertices.reshape(-1)
#         pt3d = vertices[valid_ind]
#         pt3d = pt3d.reshape([68, 3])
#         pt2d = pt3d[:, :2]
#         pt2d[:, 1] = h - pt2d[:, 1]
#         tmp = {"keypoints": {}}
#         for key, lnmk in zip(lnmk_68, pt2d):
#             # lnmk = lnmk[:2].astype(np.int32)
#             # img = cv2.circle(img, tuple(lnmk), 3, (0, 255, 0), -1)
#             # tmp["box2d"] =
#             tl = np.min(pt2d, axis=0)
#             br = np.max(pt2d, axis=0)
#             tmp["box2d"] = {
#                 "x1": int(tl[0]),
#                 "y1": int(tl[1]),
#                 "x2": int(br[0]),
#                 "y2": int(br[1])
#             }
#             tmp["keypoints"][key] = lnmk[::-1].tolist()
#         infos["labels"].append(tmp)
#         bdd_results["frame_list"].append(infos)

# dump_json(path="/aidata/anders/objects/3D-head/300W/annos/BDD_300W.json",
#           data=bdd_results)
