import json
from glob import glob
import os
import copy
from pathlib import Path, PosixPath
from pprint import pprint
from tqdm import tqdm
import numpy as np
# after 319 two faces and no mask
hard_mask_list = [
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 67, 68, 69, 70, 71, 72, 73, 81,
    82, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 280, 281, 282,
    283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 296, 297, 298, 299, 300,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 361, 361, 362,
    363, 364, 365, 370, 371, 372, 373, 374, 375, 376, 380, 381, 382, 383, 384,
    385, 386
]
delete_list = [74, 75, 76, 77, 78, 79, 80, 315, 316, 317, 318]

mapping_infos = {
    "countour_lnmk_0": "countour_face_lnmk_0",
    "countour_lnmk_1": "countour_face_lnmk_1",
    "countour_lnmk_2": "countour_face_lnmk_2",
    "countour_lnmk_3": "countour_face_lnmk_3",
    "countour_lnmk_4": "countour_face_lnmk_4",
    "countour_lnmk_5": "countour_face_lnmk_5",
    "countour_lnmk_6": "countour_face_lnmk_6",
    "countour_lnmk_7": "countour_face_lnmk_7",
    "countour_lnmk_8": "countour_face_lnmk_8",
    "countour_lnmk_9": "countour_face_lnmk_9",
    "countour_lnmk_10": "countour_face_lnmk_10",
    "countour_lnmk_11": "countour_face_lnmk_11",
    "countour_lnmk_12": "countour_face_lnmk_12",
    "countour_lnmk_13": "countour_face_lnmk_13",
    "countour_lnmk_14": "countour_face_lnmk_14",
    "countour_lnmk_15": "countour_face_lnmk_15",
    "countour_lnmk_16": "countour_face_lnmk_16",
    "left_eyebrow_17": "left_eyebrow_lnmk_17",
    "left_eyebrow_18": "left_eyebrow_lnmk_18",
    "left_eyebrow_19": "left_eyebrow_lnmk_19",
    "left_eyebrow_20": "left_eyebrow_lnmk_20",
    "left_eyebrow_21": "left_eyebrow_lnmk_21",
    "right_eyebrow_22": "right_eyebrow_lnmk_22",
    "right_eyebrow_23": "right_eyebrow_lnmk_23",
    "right_eyebrow_24": "right_eyebrow_lnmk_24",
    "right_eyebrow_25": "right_eyebrow_lnmk_25",
    "right_eyebrow_26": "right_eyebrow_lnmk_26",
    "left_eye_lnmk_27": "left_eye_lnmk_27",
    "left_eye_lnmk_28": "left_eye_lnmk_28",
    "left_eye_lnmk_29": "left_eye_lnmk_29",
    "left_eye_lnmk_30": "left_eye_lnmk_30",
    "left_eye_lnmk_31": "left_eye_lnmk_31",
    "left_eye_lnmk_32": "left_eye_lnmk_32",
    "right_eye_lnmk_33": "right_eye_lnmk_33",
    "right_eye_lnmk_34": "right_eye_lnmk_34",
    "right_eye_lnmk_35": "right_eye_lnmk_35",
    "right_eye_lnmk_36": "right_eye_lnmk_36",
    "right_eye_lnmk_37": "right_eye_lnmk_37",
    "right_eye_lnmk_38": "right_eye_lnmk_38",
    "nose_lnmk_39": "nose_lnmk_39",
    "nose_lnmk_40": "nose_lnmk_40",
    "nose_lnmk_41": "nose_lnmk_41",
    "nose_lnmk_42": "nose_lnmk_42",
    "nose_lnmk_43": "nose_lnmk_43",
    "nose_lnmk_44": "nose_lnmk_44",
    "nose_lnmk_45": "nose_lnmk_45",
    "nose_lnmk_46": "nose_lnmk_46",
    "nose_lnmk_47": "nose_lnmk_47",
    "outer_lip_lnmk_48": "outer_lip_lnmk_48",
    "outer_lip_lnmk_49": "outer_lip_lnmk_49",
    "outer_lip_lnmk_50": "outer_lip_lnmk_50",
    "outer_lip_lnmk_51": "outer_lip_lnmk_51",
    "outer_lip_lnmk_52": "outer_lip_lnmk_52",
    "outer_lip_lnmk_53": "outer_lip_lnmk_53",
    "outer_lip_lnmk_54": "outer_lip_lnmk_54",
    "outer_lip_lnmk_55": "outer_lip_lnmk_55",
    "outer_lip_lnmk_56": "outer_lip_lnmk_56",
    "outer_lip_lnmk_57": "outer_lip_lnmk_57",
    "outer_lip_lnmk_58": "outer_lip_lnmk_58",
    "outer_lip_lnmk_59": "outer_lip_lnmk_59",
    "inner_lip_lnmk_60": "inner_lip_lnmk_60",
    "inner_lip_lnmk_61": "inner_lip_lnmk_61",
    "inner_lip_lnmk_62": "inner_lip_lnmk_62",
    "inner_lip_lnmk_63": "inner_lip_lnmk_63",
    "inner_lip_lnmk_64": "inner_lip_lnmk_64",
    "inner_lip_lnmk_65": "inner_lip_lnmk_65",
    "inner_lip_lnmk_66": "inner_lip_lnmk_66",
    "inner_lip_lnmk_67": "inner_lip_lnmk_67",
}

inv_mapping_infos = {mapping_infos[k]: k for k in list(mapping_infos.keys())}


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


annos = load_json("/aidata/anders/objects/box2d/demo/annos/BDD_test.json")
box2d_dict = {}
for frame in annos["frame_list"]:
    box2d_dict[frame["name"]] = []
    tmp = []
    for lb in frame["labels"]:
        box2d_dict[frame["name"]].append(lb["box2d"])

path = "/aidata/DMS_landmarks/images/*.json"

bdd_results = {"frame_list": []}
annos_paths = sorted(list(glob(path)))[:500]
progress_bar = tqdm(total=len(annos_paths))
for i, path in enumerate(annos_paths):
    annos = load_json(path)
    path = annos["imagePath"]
    frame_infos = {
        "dataset": "Demo",
        "sequence": None,
        "name": path,
        "labels": []
    }
    tmp_kps = []
    for frame in annos["shapes"]:
        shape_key = frame["label"]
        kp = frame["points"][0][::-1]  # (y, x)
        tmp_kps.append(kp)

    tmp_kps = np.asarray(tmp_kps)
    tmp_kps = np.reshape(tmp_kps, [-1, 68, 2])
    tmp_kps_x = tmp_kps[:, :, 1:]
    tmp_mean_kps = np.mean(tmp_kps_x, axis=(1, 2))
    min_index = np.argmin(tmp_mean_kps)
    keys = list(inv_mapping_infos.keys())
    for kp_idx, kps in enumerate(tmp_kps):
        inv_kp_infos = copy.deepcopy(inv_mapping_infos)
        lb_dict = {
            "category": "FACE",
            "box2d": box2d_dict[path][kp_idx],
            "keypoints": {},
            "attributes": {
                "difficulty": None,
                "mask": False
            }
        }
        if i < 319:
            lb_dict["attributes"]["mask"] = True
        if kp_idx == min_index and tmp_kps.shape[0] == 2:
            lb_dict["attributes"]["mask"] = True
        if i in delete_list:
            # too hard and landmarks could not label
            lb_dict["attributes"]["mask"] = None
            frame_infos["labels"].append(lb_dict)
            continue
        elif i in hard_mask_list:
            lb_dict["attributes"]["difficulty"] = True
        elif i not in hard_mask_list and i not in delete_list:
            lb_dict["attributes"]["difficulty"] = False

        if kp_idx == min_index and tmp_kps.shape[0] == 2:
            lb_dict["attributes"]["difficulty"] = False
        for j, kp in enumerate(kps):
            inv_kp_infos[keys[j]] = kp.tolist()
        lb_dict["keypoints"] = inv_kp_infos
        frame_infos["labels"].append(lb_dict)
    bdd_results["frame_list"].append(frame_infos)
    progress_bar.update(1)

save_path = "/aidata/anders/objects/landmarks/demo_test/annos/BDD_test.json"
dump_json(path=save_path, data=bdd_results)
