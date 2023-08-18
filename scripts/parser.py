import numpy as np
import os
from pprint import pprint
"""
There are two folders in root directory:
  * annos/output.txt
  * init_vals.npy

  For annos/output.txt, the saved information is [frame idx, time stamp, roll, pitch and yaw] in sequence !
  For init_vals.npy, it saves the calibrated head-pose values [roll, pitch and yaw ] in sequence

"""


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


root_dir = "003171"
gt_text_lines = load_text(os.path.join(root_dir, "annos/output.txt"))
gt_init_pose = np.load(os.path.join(root_dir, "init_vals.npy"))

for line in gt_text_lines[5:]:
    infos = np.array(line.split(","), dtype=float)
    frame_idx = int(infos[0])
    time_stamp = int(infos[1])
    roll, pitch, yaw = infos[2:] - gt_init_pose