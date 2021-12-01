from utils.io import *
from pprint import pprint
import numpy as np
import copy
from tqdm import tqdm

path = "/aidata/anders/objects/landmarks/FFHQ/annos/ffhq-dataset-v2.json"
annos = load_json(path)

kp_base_dict = {
    "countour_face_lnmk_0": None,
    "countour_face_lnmk_1": None,
    "countour_face_lnmk_2": None,
    "countour_face_lnmk_3": None,
    "countour_face_lnmk_4": None,
    "countour_face_lnmk_5": None,
    "countour_face_lnmk_6": None,
    "countour_face_lnmk_7": None,
    "countour_face_lnmk_8": None,
    "countour_face_lnmk_9": None,
    "countour_face_lnmk_10": None,
    "countour_face_lnmk_11": None,
    "countour_face_lnmk_12": None,
    "countour_face_lnmk_13": None,
    "countour_face_lnmk_14": None,
    "countour_face_lnmk_15": None,
    "countour_face_lnmk_16": None,
    "left_eyebrow_lnmk_17": None,
    "left_eyebrow_lnmk_18": None,
    "left_eyebrow_lnmk_19": None,
    "left_eyebrow_lnmk_20": None,
    "left_eyebrow_lnmk_21": None,
    "right_eyebrow_lnmk_22": None,
    "right_eyebrow_lnmk_23": None,
    "right_eyebrow_lnmk_24": None,
    "right_eyebrow_lnmk_25": None,
    "right_eyebrow_lnmk_26": None,
    "left_eye_lnmk_27": None,
    "left_eye_lnmk_28": None,
    "left_eye_lnmk_29": None,
    "left_eye_lnmk_30": None,
    "left_eye_lnmk_31": None,
    "left_eye_lnmk_32": None,
    "right_eye_lnmk_33": None,
    "right_eye_lnmk_34": None,
    "right_eye_lnmk_35": None,
    "right_eye_lnmk_36": None,
    "right_eye_lnmk_37": None,
    "right_eye_lnmk_38": None,
    "nose_lnmk_39": None,
    "nose_lnmk_40": None,
    "nose_lnmk_41": None,
    "nose_lnmk_42": None,
    "nose_lnmk_43": None,
    "nose_lnmk_44": None,
    "nose_lnmk_45": None,
    "nose_lnmk_46": None,
    "nose_lnmk_47": None,
    "outer_lip_lnmk_48": None,
    "outer_lip_lnmk_49": None,
    "outer_lip_lnmk_50": None,
    "outer_lip_lnmk_51": None,
    "outer_lip_lnmk_52": None,
    "outer_lip_lnmk_53": None,
    "outer_lip_lnmk_54": None,
    "outer_lip_lnmk_55": None,
    "outer_lip_lnmk_56": None,
    "outer_lip_lnmk_57": None,
    "outer_lip_lnmk_58": None,
    "outer_lip_lnmk_59": None,
    "inner_lip_lnmk_60": None,
    "inner_lip_lnmk_61": None,
    "inner_lip_lnmk_62": None,
    "inner_lip_lnmk_63": None,
    "inner_lip_lnmk_64": None,
    "inner_lip_lnmk_65": None,
    "inner_lip_lnmk_66": None,
    "inner_lip_lnmk_67": None,
}
# AFLW
# mapping_infos = {
#     'countour_face': {
#         "mapping": ["countour_face_lnmk_8"],
#         "idxs": [0]
#     },
#     'left_eyebrow': {
#         "mapping": [],
#         "idxs": []
#     },
#     'right_eyebrow': {
#         "mapping": [],
#         "idxs": []
#     },
#     'left_eye': {
#         "mapping": ["left_eye_lnmk_27", "left_eye_lnmk_30"],
#         "idxs": [0, 2]
#     },
#     'right_eye': {
#         "mapping": ["right_eye_lnmk_33", "right_eye_lnmk_36"],
#         "idxs": [0, 2]
#     },
#     'nose': {
#         "mapping": ["nose_lnmk_42"],
#         "idxs": [0]
#     },
#     'outer_lip': {
#         "mapping": ["outer_lip_lnmk_48", "outer_lip_lnmk_54"],
#         "idxs": [0, 2]
#     },
#     'inner_lip': {
#         "mapping": [],
#         "idxs": []
#     },
# }
# 68 lnmks
mapping_infos = {
    'countour_face': {
        "mapping": ["countour_face_lnmk_8"],
        "idxs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    'left_eyebrow': {
        "mapping": [],
        "idxs": []
    },
    'right_eyebrow': {
        "mapping": [],
        "idxs": []
    },
    'left_eye': {
        "mapping": ["left_eye_lnmk_27", "left_eye_lnmk_30"],
        "idxs": [0, 2]
    },
    'right_eye': {
        "mapping": ["right_eye_lnmk_33", "right_eye_lnmk_36"],
        "idxs": [0, 2]
    },
    'nose': {
        "mapping": ["nose_lnmk_42"],
        "idxs": [0]
    },
    'outer_lip': {
        "mapping": ["outer_lip_lnmk_48", "outer_lip_lnmk_54"],
        "idxs": [0, 2]
    },
    'inner_lip': {
        "mapping": [],
        "idxs": []
    },
}
for frame in tqdm(annos["frame_list"]):
    for lb in frame["labels"]:
        keypoints = lb["keypoints"]
        kyes = list(kp_base_dict.keys())
        tmp_lb_base = copy.deepcopy(kp_base_dict)
        for key in keypoints.keys():
            kps = keypoints[key]
            list_keys = kyes[:len(kps)]
            for k, kp in zip(list_keys, kps):
                tmp_lb_base[k] = kp
            del kyes[:len(kps)]
        lb["keypoints"] = tmp_lb_base

dump_json(path="/aidata/anders/objects/landmarks/FFHQ/annos/BDD_FFHQ_68.json",
          data=annos)
