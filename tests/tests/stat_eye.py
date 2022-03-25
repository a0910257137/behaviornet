from utils.io import *
import os
import cv2
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
from scipy.optimize import curve_fit


def func(x, a, b):
    return a * x + b


tot_objs_h, tot_objs_w = [], []
total_keypoints = []
total_objs_kps = np.load("total_objs_kps.npy")
for objs_kps in tqdm(total_objs_kps):
    tmp_kps = objs_kps
    tmp_kps = np.asarray(tmp_kps)
    tl = np.min(tmp_kps, axis=0)
    br = np.max(tmp_kps, axis=0)
    y1, x1 = tl
    y2, x2 = br
    area = (x2 - x1) * (y1 - x1)
    total_keypoints.append(tmp_kps)
    LE_tl = np.min(tmp_kps[27:33], axis=0)
    LE_br = np.max(tmp_kps[27:33], axis=0)
    RE_tl = np.min(tmp_kps[33:39], axis=0)
    RE_br = np.max(tmp_kps[33:39], axis=0)
    l_h, l_w = (LE_br - LE_tl)
    r_h, r_w = (RE_br - RE_tl)
    eye_h = (l_h + r_h) / 2
    eye_w = (l_w + r_w) / 2
    obj_h, obj_w = br - tl
    tot_objs_h.append([obj_h, eye_h])
    tot_objs_w.append([obj_w, eye_w])
tot_objs_h = np.asarray(tot_objs_h)
tot_objs_w = np.asarray(tot_objs_w)
base_clcs = [["blue", "green"], ["red", "cyan"], ["magenta", "yellow"],
             ["grey", "black"], ["lime", "teal"], ["brown", "lightsalmon"]]

popt_h, pcov_h = curve_fit(func, tot_objs_h[:, 0], tot_objs_h[:, 1])
popt_h = np.round(popt_h, 2)
popt_w, pcov_w = curve_fit(func, tot_objs_w[:, 0], tot_objs_w[:, 1])
popt_w = np.round(popt_w, 2)
fig = plt.figure(figsize=(10, 8))
plt.plot(tot_objs_h[:, 0],
         func(tot_objs_h[:, 0], *popt_h),
         '-',
         c="blue",
         label='fit: slope=%5.3f, c=%5.3f' % tuple(popt_h))
plt.plot(tot_objs_w[:, 0],
         func(tot_objs_w[:, 0], *popt_w),
         '-',
         c="green",
         label='fit: slope=%5.3f, c=%5.3f' % tuple(popt_w))
plt.scatter(tot_objs_h[:, 0],
            tot_objs_h[:, 1],
            c="blue",
            marker='x',
            label="obj_h vs eye_h",
            alpha=0.4)
plt.scatter(tot_objs_w[:, 0],
            tot_objs_w[:, 1],
            c="green",
            marker='x',
            label='obj_w vs eye_w',
            alpha=0.4)
plt.xlabel('Pixel', fontsize=18)
plt.ylabel('Pixel', fontsize=18)
plt.legend()
plt.grid()
plt.savefig('foo.png')
