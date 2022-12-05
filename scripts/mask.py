import os, sys
from threading import local
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt
import cv2
from utils.io import *

path = "/aidata/anders/objects/3D-head/LFW/annos/BDD_LFW_2D.json"
annos = load_json(path)

for frame in annos["frame_list"]:
    name = frame["name"]

    img = cv2.imread(
        os.path.join("/aidata/anders/objects/landmarks/LFW/imgs", name))

    h, w, c = img.shape
    for lb in frame['labels']:
        box2d = lb['box2d']
        tl = (box2d['x1'], box2d['y1'])
        br = (box2d['x2'], box2d['y2'])
        center = np.array(
            [br[0] - (br[0] - tl[0]) / 2.0, br[1] - (br[1] - tl[1]) / 2.0])
        old_size = (br[0] - tl[0] + br[1] - tl[1]) / 2
        size = int(old_size * 1.5)
        marg = old_size * 0.1
        t_x = np.random.rand() * marg * 2 - marg
        t_y = np.random.rand() * marg * 2 - marg
        center[0] = center[0] + t_x
        center[1] = center[1] + t_y
        size = size * (np.random.rand() * 0.2 + 0.9)

        # crop and record the transform parameters
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        tform = skimage.transform.estimate_transform('similarity', src_pts,
                                                     DST_PTS)
        cropped_image = skimage.transform.warp(img,
                                               tform.inverse,
                                               output_shape=(h, w))