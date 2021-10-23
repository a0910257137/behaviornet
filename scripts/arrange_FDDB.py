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


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


file_paths = "/work/anders1234/FDDB/FDDB-folds/*.txt"
file_paths = sorted(glob(file_paths))
img_root = "/work/anders1234/FDDB"
save_root = "/work/anders1234/FDDB/imgs"
bdd_results = {"frame_list": []}
for path in file_paths:
    if 'ellipseList' not in path:
        continue
    # implement parser
    lines = load_text(path)
    for i, line in enumerate(lines):
        # start
        if 'img' in line:
            file_name = line
            img_path = os.path.join(img_root, file_name + '.jpg')
            img_name = img_path.split('/')[-1]
            file_name = file_name.split('/')
            save_name = '{}_{}_{}_{}_{}.jpg'.format(file_name[0], file_name[1],
                                                    file_name[2], file_name[3],
                                                    file_name[4])

            num_boxes = int(lines[i + 1])
            frame_infos = {
                "dataset": "FDDB",
                "sequence": None,
                "name": save_name,
                "labels": []
            }
            # major_axis_radius minor_axis_radius angle center_x center_y 1
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(save_root, save_name), img)
            ellipseLists = lines[i + 2:i + 2 + (num_boxes)]
            tmp = []
            for i, vals in enumerate(ellipseLists):
                # vals = [float(v) for v in vals]
                vals = vals.split(" ")
                for vs in vals:
                    if vs == '':
                        continue
                    tmp.append(float(vs))
            tmp = np.asarray(tmp).astype(float)
            elliptical_infos = tmp.reshape([-1, 6])
            print('-' * 100)
            print(elliptical_infos)
            for elliptical_info in elliptical_infos:
                center_kp = elliptical_info[3:5]
                tl = (center_kp[0] - elliptical_info[1],
                      center_kp[1] - elliptical_info[0])
                br = (center_kp[0] + elliptical_info[1],
                      center_kp[1] + elliptical_info[0])
                tl = (np.asarray(tl) + .5).astype(int)
                br = (np.asarray(br) + .5).astype(int)
                lb = {
                    "box2d": {
                        "x1": int(tl[0]),
                        "y1": int(tl[1]),
                        "x2": int(br[0]),
                        "y2": int(br[1])
                    },
                    "category": "FACE",
                    "attributes": {}
                }
                frame_infos['labels'].append(lb)
            bdd_results['frame_list'].append(frame_infos)
            # img = cv2.rectangle(img, tuple(tl), tuple(br), (0, 255, 0), 3)
dump_json("/work/anders1234/FDDB/annos/BDD_train.json", bdd_results)
