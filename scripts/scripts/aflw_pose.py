from pyrsistent import v
from utils.io import *
import numpy as np
from pprint import pprint
import os
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import copy

poses_texts = load_text("/aidata/evianlee/dataset/AFLW/aflw/data/facepose.txt")
alfw_texts = load_text("/aidata/evianlee/dataset/AFLW/aflw/data/alfw.txt")
faceid_texts = load_text("/aidata/evianlee/dataset/AFLW/aflw/data/faceid.txt")
mapping_infos = {}
for i, line in enumerate(faceid_texts):
    line_lists = line.split(",")
    mapping_infos[line_lists[0]] = line_lists[1]


def convert(str_texts):
    tmp_str = str()
    for st in str_texts:
        if st == '"':
            continue
        tmp_str += st
    vals = float(tmp_str) * (180 / np.pi)

    return vals


def convert_names(str_texts):
    tmp_str = str()
    for st in str_texts:
        if st == '"' or st == "-":
            continue
        tmp_str += st
    return tmp_str


def _euclidean(A, B):
    A2B = np.abs(A - B)
    A2B_dist = np.sqrt(np.sum(np.square(A2B)))
    return A2B_dist


def append(status_dict, deg_key, item_keys, items):
    for item_key, item in zip(item_keys, items):
        status_dict[deg_key][item_key].append(item)
    return status_dict


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


#  '"roll"', '"pitch"', '"yaw"',

for i, line in enumerate(poses_texts):
    if 0 < i:
        line_lists = line.split(",")
        roll = convert(line_lists[1])
        pitch = convert(line_lists[2])
        yaw = convert(line_lists[3])

        name = mapping_infos[line.split(",")[0]]
        name = convert_names(name)
        mapping_infos[name] = {
            "img_name": name,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        }
annos = load_json(
    "/aidata/anders/objects/landmarks/AFLW/annos/BDD_AFLW_NEW.json")
img_root = "/aidata/anders/objects/landmarks/AFLW/imgs"

deg_keys = [str(deg) for deg in range(-90, 105, 15)]
item_keys = [
    "le", "re", "lm", "rm", "le/lm", "re/rm", "le_ang", "re_ang", "lm_ang",
    "rm_ang"
]
degree_status_dict = {}

for deg in deg_keys:
    degree_status_dict[deg] = {item_key: [] for item_key in item_keys}
annos["frame_list"] = annos["frame_list"][:10000]

map_size = np.array([320, 192])
progress_bar = tqdm(total=len(annos["frame_list"]))
resized = np.load("img_size_arr.npy")
tmp_sizes = []
for i, frame in enumerate(annos["frame_list"][52:]):
    name = frame["name"]
    img_path = os.path.join(img_root, name)
    # img = cv2.imread(img_path)
    # img_sizes = np.asarray(img.shape[:2])
    # resize_ratio = map_size / img_sizes
    # resize_ratio = resize_ratio[::-1]
    # tmp_sizes.append(resize_ratio)
    resize_ratio = resized[i]
    # cv2.imwrite("output.jpg", img)
    infos = mapping_infos[name]
    for lb in frame["labels"]:
        box2d = lb["box2d"]
        x1, y1 = box2d["x1"], box2d["y1"]
        x2, y2 = box2d["x2"], box2d["y2"]
        h = y2 - y1
        w = x2 - x1
        keypoints = lb["keypoints"]
        keys = keypoints.keys()
        # left_eye_lnmk_1', 'right_eye_lnmk_2', 'nose_lnmk_3', 'outer_lip_lnmk_4', 'outer_lip_lnmk_5'

        LE_kp = np.asarray(keypoints["left_eye_lnmk_1"])
        RE_kp = np.asarray(keypoints["right_eye_lnmk_2"])
        nose_kp = np.asarray(keypoints["nose_lnmk_3"])
        LM_kp = np.asarray(keypoints["outer_lip_lnmk_4"])
        RM_kp = np.asarray(keypoints["outer_lip_lnmk_5"])

        LE_kp = LE_kp * resize_ratio
        RE_kp = RE_kp * resize_ratio
        nose_kp = nose_kp * resize_ratio
        LM_kp = LM_kp * resize_ratio
        RM_kp = RM_kp * resize_ratio

        copied_nose_kp = copy.deepcopy(nose_kp)
        copied_nose_kp[1] = copied_nose_kp[1] + 50
        v2 = unit_vector(copied_nose_kp - nose_kp)
        v1 = unit_vector(LE_kp - nose_kp)
        nose2le_ang = angle_between(v2, v1) * 180 / np.pi

        v1 = unit_vector(RE_kp - nose_kp)
        nose2re_ang = angle_between(v2, v1) * 180 / np.pi

        v1 = unit_vector(LM_kp - nose_kp)
        nose2lm_ang = angle_between(v2, v1) * 180 / np.pi

        v1 = unit_vector(RM_kp - nose_kp)
        nose2rm_ang = angle_between(v2, v1) * 180 / np.pi

        #-----------------------------------------
        # left eye
        nose2le_dist = _euclidean(nose_kp, LE_kp)
        #-----------------------------------------
        # right eye
        nose2re_dist = _euclidean(nose_kp, RE_kp)
        #-----------------------------------------
        # left mouth
        nose2lm_dist = _euclidean(nose_kp, LM_kp)
        # right mouth
        nose2rm_dist = _euclidean(nose_kp, RM_kp)

        le2lm = nose2le_dist / nose2lm_dist
        re2rm = nose2re_dist / nose2rm_dist

        degree_status = infos["yaw"]
        if degree_status >= 90:
            continue
        keys = [
            "le", "re", "lm", "rm", "le/lm", "re/rm", "le_ang", "re_ang",
            "lm_ang", "rm_ang"
        ]
        items = [
            [degree_status, nose2le_dist],
            [degree_status, nose2re_dist],
            [degree_status, nose2lm_dist],
            [degree_status, nose2rm_dist],
            [degree_status, le2lm],
            [degree_status, re2rm],
            [degree_status, nose2le_ang],
            [degree_status, nose2re_ang],
            [degree_status, nose2lm_ang],
            [degree_status, nose2rm_ang],
        ]

        if -90 <= degree_status < -75:
            degree_status_dict = append(degree_status_dict, "-90", keys, items)
        elif -75 <= degree_status < -60:
            degree_status_dict = append(degree_status_dict, "-75", keys, items)
        elif -60 <= degree_status < -45:
            degree_status_dict = append(degree_status_dict, "-60", keys, items)
        elif -45 <= degree_status < -30:
            degree_status_dict = append(degree_status_dict, "-45", keys, items)
        elif -30 <= degree_status < -15:
            degree_status_dict = append(degree_status_dict, "-30", keys, items)
        elif -15 <= degree_status < 0:
            degree_status_dict = append(degree_status_dict, "-15", keys, items)
        elif 0 <= degree_status < +15:
            degree_status_dict = append(degree_status_dict, "15", keys, items)
        elif +15 <= degree_status < +30:
            degree_status_dict = append(degree_status_dict, "30", keys, items)
        elif +30 <= degree_status < +45:
            degree_status_dict = append(degree_status_dict, "45", keys, items)
        elif +45 <= degree_status < +60:
            degree_status_dict = append(degree_status_dict, "60", keys, items)
        elif +60 <= degree_status < +75:
            degree_status_dict = append(degree_status_dict, "75", keys, items)
        elif +75 <= degree_status <= 90:
            degree_status_dict = append(degree_status_dict, "90", keys, items)
        lb["attributes"]["yaw"] = infos["yaw"]
        lb["attributes"]["pitch"] = infos["pitch"]
        lb["attributes"]["roll"] = infos["roll"]
    progress_bar.update(1)


def _draw_ratio(L, R, L_clc, R_clc):
    L = np.asarray(L)
    R = np.asarray(R)
    plt.scatter(L[:, 0], L[:, 1], c=L_clc, marker='x')
    plt.scatter(R[:, 0], R[:, 1], c=R_clc, marker='o', alpha=0.7)
    L = L[L < 2.0]
    l_mu, l_std = np.mean(L), np.std(L)
    R = R[R < 2.0]
    r_mu, r_std = np.mean(R), np.std(R)

    print("left_eye_mean {}; right eye mean {}".format(round(l_mu, 3),
                                                       round(r_mu, 3)))

    print("left_eye_std {}; right eye std {}".format(round(l_std, 3),
                                                     round(r_std, 3)))


def _draw_dist(degree_status_dict,
               deg_keys,
               item_keys,
               mark_list,
               out_lier=60):

    for deg_key in deg_keys:
        for item_key, mark in zip(item_keys, mark_list):
            dist = degree_status_dict[deg_key][item_key]
            dist = np.asarray(dist)
            if dist.shape[0] == 0:
                continue
            plt.scatter(dist[:, 0], dist[:, 1], marker=mark)
            dist = dist[:, 1]
            dist = dist[dist < out_lier]
            dist_mean, dist_std = np.mean(dist), np.mean(dist)
            degree_status_dict[deg_key][item_key] = {
                "mean": dist_mean,
                "std": dist_std
            }
    return degree_status_dict


#-------------------------------------------------
print('-' * 100)
fig = plt.figure(figsize=(10, 8))
base_clcs = [["blue", "green"], ["red", "cyan"], ["magenta", "yellow"],
             ["grey", "black"], ["lime", "teal"], ["brown", "lightsalmon"]]
base_clcs = base_clcs[::-1] + base_clcs

deg_keys = [str(deg) for deg in range(-90, 105, 15)]

for deg_key, clc in zip(degree_status_dict.keys(), base_clcs):
    if len(degree_status_dict[deg_key]["le/lm"]) == 0:
        continue
    _draw_ratio(degree_status_dict[deg_key]["le/lm"],
                degree_status_dict[deg_key]["re/rm"], clc[0], clc[1])
plt.xlabel('Yaw degrees', fontsize=18)
plt.ylabel('Ratios', fontsize=18)
plt.grid()
plt.savefig('foo.png')
#-------------------------------------------------
import seaborn as sns
import pandas as pd


def _heatmap(save_name, ret_mat):
    fig = plt.figure(figsize=(10, 8))
    df = pd.DataFrame(data=ret_mat,
                      columns=['LE', 'RE', 'LM', 'RM'],
                      index=['LE', 'RE', 'LM', 'RM'])
    labels = np.asarray(["{:.3f}".format(val)
                         for val in ret_mat.flatten()]).reshape([4, 4])
    matrix_table = sns.heatmap(df, annot=labels, fmt="", cmap="YlGnBu")

    plt.savefig(save_name)


print('-' * 100)
fig = plt.figure(figsize=(10, 8))
item_keys = ["le", "re", "lm", "rm"]
mark_list = ['x', 'o', '+', '.', "v"]

degree_status_dict = _draw_dist(degree_status_dict, deg_keys, item_keys,
                                mark_list)
# for deg_key in degree_status_dict.keys():
#     tmp = []
#     for item_key in item_keys:
#         if not isinstance(degree_status_dict[deg_key][item_key], dict):
#             continue
#         tmp.append(degree_status_dict[deg_key][item_key]["mean"])
#     if len(tmp) == 0:
#         continue
#     tmp = np.asarray(tmp)[:, None]
#     sim_tmp = np.abs(tmp - tmp.T)
#     print('-' * 100)
#     print("Yaw {} degrees".format(deg_key))
#     print(sim_tmp)
#     _heatmap("{}.png".format(deg_key), sim_tmp)

print('-' * 100)
base_clcs = [["blue", "green", "red", "black"]] * len(
    degree_status_dict.keys())

fig = plt.figure(figsize=(10, 8))
item_keys = ["le_ang", "re_ang", "lm_ang", "rm_ang"]
for clcs, deg_key in zip(base_clcs, degree_status_dict.keys()):
    for clc, item_key, mark in zip(clcs, item_keys, mark_list):
        angles = degree_status_dict[deg_key][item_key]
        angles = np.asarray(angles)
        if angles.shape[0] == 0:
            continue
        plt.scatter(angles[:, 0], angles[:, 1], c=clc, marker=mark)
plt.grid()
plt.xlabel('Yaw degrees', fontsize=18)
plt.ylabel('Angles', fontsize=18)
plt.savefig("{}.png".format('Angles'))
