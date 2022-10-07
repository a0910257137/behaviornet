from pathlib import Path
import numpy as np
import sys
import os
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.morphable_model import MorphabelModel


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


bfm = MorphabelModel('/aidata/anders/objects/landmarks/3DDFA/BFM/BFM.mat')

annos = load_json(
    "/aidata/anders/objects/landmarks/300VW/annos/BDD_300VW_NEW.json")

for frame in annos['frame_list']:
    for lb in frame["labels"]:
        keypoints = lb['keypoints']
        keys = keypoints.keys()
        kps = []
        for k in keys:
            kps.append(keypoints[k])
        kps = np.stack(kps, axis=0)
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(
            kps[:, ::-1], bfm.kpt_ind, max_iter=10)
        lb["pose"] = list(fitted_angles)
save_path = "/aidata/anders/objects/landmarks/300VW/annos/BDD_300VW_POSE.json.json"
dump_json(path=save_path, data=annos)