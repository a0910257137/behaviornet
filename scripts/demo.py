#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import cv2
import os
import argparse
import commentjson
from glob import glob
import numpy as np
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text
from behavior_predictor.inference import BehaviorPredictor
from monitor import logger


def load_config(path):
    with open(path) as f:
        return commentjson.loads(f.read())


class Demo:
    def __init__(self, obj_cfg, lnmk_cfg, img_root, batch):
        self.obj_cfg = obj_cfg
        self.lnmk_cfg = lnmk_cfg
        self.batch = batch
        self.img_root = img_root
        self.obj_pred = BehaviorPredictor(self.obj_cfg['predictor'])
        self.lnmk_pred = BehaviorPredictor(self.lnmk_cfg['predictor'])
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video_maker = cv2.VideoWriter('demo.avi', fourcc, 10.0,
                                           (1920, 1080))

    def __call__(self):
        img_path_list = sorted(list(glob(os.path.join(self.img_root,
                                                      '*.jpg'))))
        batch_objects = map(lambda x: self.split_batchs(img_path_list, x),
                            range(0, len(img_path_list), self.batch))
        for batch_imgs_shapes in tqdm(batch_objects):
            batch_imgs_shapes = list(batch_imgs_shapes)
            for imgs_shapes in batch_imgs_shapes:
                imgs, shapes = imgs_shapes
                rets = self.obj_pred.pred(imgs, shapes)
                imgs_shapes = self.crop_face(imgs, rets)
        self.video_maker.release()
        logger.info("Fished")

    def crop_face(self, imgs, rets):
        tmp_imgs, tmp_origin_shape = [], []
        tmp_tls = []
        box2ds = rets[:, 0, :4]
        box2d_std = tf.math.reduce_std(box2ds, axis=0)
        valid_std = tf.math.reduce_all(box2d_std < 10).numpy()
        # for stabal bounding boxes
        if valid_std:
            box2d_mean = tf.math.reduce_mean(box2ds, axis=0)
            box2ds = box2ds.numpy()
            box2d_mean = box2d_mean.numpy()
            box2ds = (box2ds + box2d_mean) / 2
            for img, box2d in zip(imgs, box2ds):
                tl = box2d[:2].astype(np.int32)
                br = box2d[2:4].astype(np.int32)
                crop_img = copy.deepcopy(img[tl[0]:br[0], tl[1]:br[1], :])
                lnmk_tl = tl
                h, w, _ = crop_img.shape
                if h == 0 or w == 0:
                    continue
                img = cv2.rectangle(img, tuple(tl[::-1]), tuple(br[::-1]),
                                    (0, 255, 0), 3)
                tmp_origin_shape.append((h, w))
                tmp_imgs.append(crop_img)
                tmp_tls.append(lnmk_tl)
        else:
            for img, ret in zip(imgs, rets):
                valid_mask = tf.math.reduce_all(tf.math.is_finite(ret),
                                                axis=-1)
                ret = ret[valid_mask]
                ret = ret.numpy()
                for obj_pred in ret:
                    tl = obj_pred[:2]
                    br = obj_pred[2:4]
                    crop_img = copy.deepcopy(img[int(tl[0]):int(br[0]),
                                                 int(tl[1]):int(br[1]), :])
                    lnmk_tl = tl
                    h, w, _ = crop_img.shape
                    if h == 0 or w == 0:
                        continue

                    tl = tl.astype(int)
                    br = br.astype(int)
                    img = cv2.rectangle(img, tuple(tl[::-1]), tuple(br[::-1]),
                                        (0, 255, 0), 3)
                    tmp_origin_shape.append((h, w))

                    tmp_imgs.append(crop_img)
                    tmp_tls.append(lnmk_tl)
        if len(tmp_imgs) != 0:
            pred_lnmks = self.lnmk_pred.pred(tmp_imgs, tmp_origin_shape)
            lnmk_std = tf.math.reduce_std(pred_lnmks, axis=0)
            valid_std = tf.math.reduce_all(lnmk_std < 10).numpy()
            if valid_std:
                lnmk_mean = tf.math.reduce_mean(pred_lnmks, axis=0)
                lnmks = (lnmk_mean + pred_lnmks) / 2
                lnmks = lnmks.numpy()
                # lnmk_mean = lnmk_mean.astype(int)
                for img, tls, lnmk in zip(imgs, tmp_tls, lnmks):
                    shift_lnmks = lnmk + tls
                    shift_lnmks = shift_lnmks.astype(int)
                    for shift_lnmk in shift_lnmks:
                        lnmk = shift_lnmk[::-1]
                        img = cv2.circle(img, tuple(lnmk), 10, (0, 255, 0), -1)
                    self.video_maker.write(img)

            else:
                pred_lnmks = pred_lnmks.numpy()
                for img, tls, lnmks in zip(imgs, tmp_tls, pred_lnmks):
                    shift_lnmks = lnmks + tls
                    shift_lnmks = shift_lnmks.astype(int)
                    for shift_lnmk in shift_lnmks:
                        lnmk = shift_lnmk[::-1]
                        img = cv2.circle(img, tuple(lnmk), 10, (0, 255, 0), -1)
                    self.video_maker.write(img)

    def split_batchs(self, elems, idx):
        imgs = []
        origin_shapes = []
        for elem in elems[idx:idx + self.batch]:
            img = cv2.imread(elem)
            h, w, _ = img.shape
            origin_shapes.append((h, w))
            imgs.append(img)
        yield (imgs, origin_shapes)


path = './config/hardnet_pred.json'
obj_config = load_config(path)
path = './config/hardnet_pred_kp.json'
lnmk_config = load_config(path)
img_root = '/aidata/anders/objects/landmarks/demo_video/office3'
demo = Demo(obj_config, lnmk_config, img_root, 5)
demo()