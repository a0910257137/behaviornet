import cv2
import os
import numpy as np
import json
from pathlib import Path
from pprint import pprint
from glob import glob
from tqdm import tqdm
import os, sys
import random
import math
import copy


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


# random_sample_idx = np.random.rand(2000, ) * 10000
# random_sample_idx = np.ceil(random_sample_idx)
# random_sample_idx = np.unique(random_sample_idx)
# random_sample_idx = random_sample_idx.astype(int)
# train_path = "/aidata/anders/objects/WF/annos/BDD_train.json"
# annos = load_json(train_path)
# bdd_results = {"frame_list": []}
# for idx in random_sample_idx:
#     bdd_results['frame_list'].append(annos['frame_list'][idx])

# extra_train_path = "/aidata/anders/objects/openface/annos/BDD_train.json"
# extra_annos = load_json(extra_train_path)
# gt_path = "/aidata/anders/objects/incar/annos/BDD_test.json"
# gt_annos = load_json(gt_path)

# save_root = "/aidata/anders/objects/incar/aug_imgs"
# for frame in extra_annos['frame_list']:
#     idx = random.randrange(0, 499)
#     gt_frame = gt_annos['frame_list'][idx]
#     gt_img_path = os.path.join('/aidata/anders/objects/incar/imgs',
#                                gt_frame["name"])
#     gt_img = cv2.imread(gt_img_path)
#     extra_img_path = os.path.join('/aidata/anders/objects/openface/imgs',
#                                   frame["name'"])
#     extra_img = cv2.imread(extra_img_path)
#     frame['labels'] = []
#     for gt_lb in gt_frame['labels']:
#         box2d = gt_lb['box2d']
#         gt_tl = np.asarray([box2d['x1'], box2d['y1']]).astype(int)
#         gt_br = np.asarray([box2d['x2'], box2d['y2']]).astype(int)
#         box_wh = (gt_br - gt_tl).astype(int)
#         extra_img = cv2.resize(extra_img, tuple(box_wh))
#         gt_img[gt_tl[1]:gt_br[1], gt_tl[0]:gt_br[0], :] = extra_img
#         frame['labels'].append(gt_lb)
#     cv2.imwrite(os.path.join(save_root, 'aug_' + frame["name'"]), gt_img)
#     bdd_results['frame_list'].append(frame)
# dump_json("/aidata/anders/objects/incar/annos/BDD_aug_train.json", bdd_results)

# cropped location gt_tl[1]:gt_br[1],, gt_tl[0]:gt_br[0]
# ---------------------------------------------------#
# bdd_results['frame_list'].append(frame)
# for frame in annos['frame_list']:
#     bdd_results['frame_list'].append(frame)

# root = "/aidata/anders/objects/openface/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset"
# parant_dirs = os.listdir(root)
# save_img_root = "/aidata/anders/objects/openface/imgs"
# bdd_results = {"frame_list": []}
# for dirs in tqdm(parant_dirs):
#     file_paths = os.listdir(os.path.join(root, dirs))
#     for name in file_paths:
#         file_path = os.path.join(root, dirs, name)
#         img = cv2.imread(file_path)
#         save_name = "open_dataset_{}_{}".format(dirs, name)
#         cv2.imwrite(os.path.join(save_img_root, save_name), img)
#         h, w, c = img.shape
#         tl = (0, 0)
#         br = (w, h)
#         frame_infos = {
#             "dataset'":
#             None,
#             "sequence'":
#             None,
#             "name'":
#             save_name,
#             "labels": [{
#                 "box2d": {
#                     "x1": tl[0],
#                     "y1": tl[1],
#                     "x2": br[0],
#                     "y2": br[1]
#                 },
#                 "category": "FACE",
#                 "attributes": {}
#             }]
#         }
#         bdd_results['frame_list'].append(frame_infos)
# dump_json("/aidata/anders/objects/openface/annos/BDD_train.json", bdd_results)
