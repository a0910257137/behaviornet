#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
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
from utils.io import *
from utils.kalman_lnmk import KalmanFilter as kf_lnmk
from utils.kalman import KalmanFilter as kf_det
from utils.pose import PoseEstimator
from behavior_predictor.inference import BehaviorPredictor
from monitor import logger
import math


def load_config(path):
    with open(path) as f:
        return commentjson.loads(f.read())


class Demo:
    def __init__(self, obj_cfg, lnmk_cfg, eye_cfg, img_root, use_rolling,
                 batch, rolling_len):
        self.obj_cfg = obj_cfg
        self.lnmk_cfg = lnmk_cfg
        self.batch = batch
        self.img_root = img_root
        self.rolling_len = rolling_len
        self.use_rolling = use_rolling

        self.kf_det = kf_det()
        self.kf_lnmk = kf_lnmk()
        self.obj_pred = BehaviorPredictor(self.obj_cfg['predictor'])
        self.lnmk_pred = BehaviorPredictor(self.lnmk_cfg['predictor'])
        self.eye_pred = BehaviorPredictor(eye_cfg['predictor'])

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        self.width = 1280
        self.height = 720
        self.FPS = 60.0
        self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))
        self.video_maker = cv2.VideoWriter('demo.avi', fourcc, self.FPS,
                                           (self.width, self.height))
        # self.video_maker = cv2.VideoWriter('demo.avi', fourcc, 30.0,
        #                                    (1920, 1080))
        self.smooth_funcs = {
            "kalman_avg": self.kalman_avg,
            "rooling_avg": self.rolling_avg
        }

        self.smooth_det = self.smooth_funcs['rooling_avg']
        self.smooth_lnmk = self.smooth_funcs['kalman_avg']

        self.glb_box2ds = []

        self.glb_eyes = []

        self.curr_frame = 0
        self.avg_len = rolling_len
        self.list_lnmks = []
        # for det
        self.mean = None
        self.covariance = None
        self.temp_xyah = None

        # for lnmk
        self.mean_lnmk = None
        self.covariance_lnmk = None
        self.temp_lnmk = None
        self.lnmk_scheme = [
            0, 8, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42,
            48, 50, 51, 52, 54, 56, 57, 58
        ]
        self.batch_pitch, self.batch_yaw, self.batch_roll, self.eyes_status = [], [], [], []
        self.fatigue_flag, self.distractive_flag = [], []

    def __call__(self):
        img_path_list = sorted(list(glob(os.path.join(self.img_root,
                                                      '*.jpg'))))
        if len(img_path_list) == 0:
            img_path_list = sorted(
                list(glob(os.path.join(self.img_root, '*.png'))))
        # img_path_list = img_path_list[3000:]
        batch_objects = map(lambda x: self.split_batchs(img_path_list, x),
                            range(0, len(img_path_list), self.batch))
        batch_objects = list(batch_objects)
        progress_bar = tqdm(total=len(batch_objects))
        for batch_imgs_shapes in batch_objects:
            batch_imgs_shapes = list(batch_imgs_shapes)
            for imgs_shapes in batch_imgs_shapes:
                self.curr_frame += 1
                imgs, shapes = imgs_shapes
                rets = self.obj_pred.pred(imgs, shapes)
                cropped_shapes, cropped_imgs, tls, objs_h = self.crop_face(
                    imgs, rets)
                # run second model
                if len(cropped_imgs) != 0:
                    pred_lnmks = self.lnmk_pred.pred(cropped_imgs,
                                                     cropped_shapes)
                    pred_lnmks = pred_lnmks.numpy()
                    for img, tls, lnmks, obj_h in zip(imgs, tls, pred_lnmks,
                                                      objs_h):
                        # Convert the locations from local face area to the global image.
                        shift_lnmks = lnmks + tls
                        self.temp_lnmk = shift_lnmks.reshape([-1])
                        if self.curr_frame == 1:
                            self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.initiate(
                                self.temp_lnmk, obj_h)
                        else:
                            self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.predict(
                                self.mean_lnmk, self.covariance_lnmk, obj_h)
                            self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.update(
                                self.mean_lnmk, self.covariance_lnmk,
                                self.temp_lnmk)
                        shift_lnmks = self.mean_lnmk[:50].reshape([-1, 2])
                        shift_lnmks = shift_lnmks.astype(int)

                        right_eye_lnmk = shift_lnmks[9:15, :]
                        left_eye_lnmk = shift_lnmks[3:9, :]
                        right_eye_img, r_tl = self.crop_eye(img,
                                                            right_eye_lnmk,
                                                            is_right=True)

                        left_eye_img, l_tl = self.crop_eye(img,
                                                           left_eye_lnmk,
                                                           is_right=False)
                        origin_shapes = [
                            list(right_eye_img.shape[:2]),
                            list(left_eye_img.shape[:2])
                        ]
                        eye_imgs = [right_eye_img, left_eye_img]
                        RLE_status = self.eye_pred.pred(
                            eye_imgs, origin_shapes).numpy()

                        #TODO: write business logics
                        imgs, proc_lnmks = self.business_logics(
                            img, shift_lnmks, RLE_status)

                        for lnmk in proc_lnmks:
                            lnmk = lnmk.astype(np.int32)
                            imgs = cv2.circle(imgs, tuple(lnmk[::-1]), 2,
                                              (0, 255, 0), -1)
                        # cv2.imwrite("output.jpg", img)

                        self.video_maker.write(img)
                progress_bar.update(1)

        self.video_maker.release()
        logger.info("Fished")

    def crop_face(self, imgs, rets):
        tmp_imgs, tmp_shapes = [], []
        tmp_tls, tmp_objs_h = [], []
        for img, ret in zip(imgs, rets):
            valid_mask = tf.math.reduce_all(tf.math.is_finite(ret), axis=-1)
            box2ds = ret[valid_mask]
            box2d = None
            if tf.shape(box2ds)[0] != 0:
                box2d = self.priority_box2ds(box2ds)
                # padding images
                box2d[:2] -= 10
                box2d[2:] += 10
                tl, br = self.smooth_det(box2d, self.glb_box2ds)
            else:
                tl, br = self.smooth_det(box2d, self.glb_box2ds)

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
            tmp_shapes.append((h, w))
            tmp_imgs.append(crop_img)
            tmp_tls.append(lnmk_tl)
            tmp_objs_h.append((br - tl)[0])
        return tmp_shapes, tmp_imgs, tmp_tls, tmp_objs_h

    def split_batchs(self, elems, idx):
        imgs = []
        origin_shapes = []
        for elem in elems[idx:idx + self.batch]:
            img = cv2.imread(elem)
            h, w, _ = img.shape
            origin_shapes.append((h, w))
            imgs.append(img)
        yield (imgs, origin_shapes)

    def priority_box2ds(self, box2ds):
        tl = box2ds[:, :2]
        br = box2ds[:, 2:4]
        hw = br - tl
        area = hw[:, 0] * hw[:, 1]
        max_idx = tf.math.argmax(area)
        box2d = box2ds[max_idx]
        box2d = box2d.numpy()[:4]
        return box2d

    def rolling_avg(self, box2d, glb_box2ds):
        if box2d is not None:
            if len(glb_box2ds) == self.avg_len:
                glb_box2ds.pop(0)
            glb_box2ds.append(box2d)
        rolling_avg_box = np.mean(np.asarray(glb_box2ds), axis=0)
        tl = rolling_avg_box[:2]
        br = rolling_avg_box[2:4]
        return tl, br

    def kalman_avg(self, box2d, glb_box2ds):
        def to_xyah(box2d):
            """Convert bounding box to format `(center x, center y, aspect ratio,
            height)`, where the aspect ratio is `width / height`.
            """
            ret = box2d.copy()
            tl, br = ret[:2], ret[2:]
            center_xy = ((tl + br) / 2)[::-1]
            hw = br - tl
            ret[:2] = center_xy
            ret[2], ret[3] = hw[1] / hw[0], hw[0]
            return ret

        if box2d is not None:
            if len(glb_box2ds) == self.avg_len:
                glb_box2ds.pop(0)
            glb_box2ds.append(box2d)
        box2d = np.mean(np.asarray(glb_box2ds), axis=0)
        xyah = to_xyah(box2d)
        self.temp_xyah = xyah
        if self.curr_frame == 0:
            if self.temp_xyah is None:
                self.temp_xyah = np.array([9.72e+02, 4.3e+02, 7.4e-01, 530])
            self.mean, self.covariance = self.kf_det.initiate(self.temp_xyah)
        else:
            self.mean, self.covariance = self.kf_det.predict(
                self.mean, self.covariance)
            self.mean, self.covariance = self.kf_det.update(
                self.mean, self.covariance, self.temp_xyah)
        center_yx = self.mean[:2][::-1]
        hw = np.array([self.mean[3], self.mean[2] * self.mean[3]])
        tl = center_yx - (hw / 2)
        br = center_yx + (hw / 2)

        return tl, br

    def crop_eye(self, img, eye_lnmks, is_right):

        tl = np.min(eye_lnmks, axis=0)
        br = np.max(eye_lnmks, axis=0)
        if is_right:
            tl[0] -= 10
            br[0] += 15
            tl[1] -= 10
            br[1] += 10
        else:
            tl[0] -= 10
            br[0] += 15
            tl[1] -= 10
            br[1] += 10
        img_T = img[tl[0]:br[0], tl[1]:br[1], :]
        return img_T, tl

    def business_logics(self, imgs, shift_lnmks, RLE_status, anl_batch=20):
        def reset():
            self.batch_pitch, self.batch_yaw, self.batch_roll, self.eyes_status = [], [], [], []

        def put_text(imgs, text_info, shift_height, clc, padding):
            FONT_SCALE = 1
            FONT_THICKNESS = 1
            FONT_STYLE = cv2.FONT_HERSHEY_COMPLEX_SMALL
            (_, text_height), _ = cv2.getTextSize(text_info, FONT_STYLE,
                                                  FONT_SCALE, FONT_THICKNESS)
            shift_height += text_height
            # print yaw roll degree
            img = cv2.putText(imgs, text_info, (5, shift_height + padding),
                              FONT_STYLE, FONT_SCALE, clc, FONT_THICKNESS,
                              cv2.LINE_AA)

            return img, shift_height

        shift_lnmks = shift_lnmks.astype(np.float32)
        marks = shift_lnmks[..., ::-1]
        # l_eye_lnmks = shift_lnmks[3:9]
        # r_eye_lnmks = shift_lnmks[9:15]
        # mouth_lnmks = shift_lnmks[17:]

        lnmk_scheme = list(range(3, 25, 1))
        marks = marks[lnmk_scheme]
        pose = self.pose_estimator.solve_pose_by_68_points(marks)
        r_mat, _ = cv2.Rodrigues(pose[0])

        p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
        _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
        pitch, yaw, roll = u_angle.flatten()

        if roll > 0:
            roll = 180 - roll
        elif roll < 0:
            roll = -(180 + roll)
        t_h_o = 0
        driver_infos = "Driver: {}".format("Po-Yuan")
        if yaw < -30 or yaw > +45:

            yaw_roll_text = "Yaw: {}; Roll: {}".format("unsupport",
                                                       "unsupport")

            eye_infos = "LE: {}; RE: {}".format("invalid", "invalid")
            proc_lnmks = []
            imgs, t_h_o = put_text(imgs,
                                   driver_infos,
                                   t_h_o,
                                   clc=(0, 255, 255),
                                   padding=5)
            imgs, t_h_o = put_text(imgs,
                                   yaw_roll_text,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 2)
            imgs, t_h_o = put_text(imgs,
                                   eye_infos,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 3)

        elif -30 <= yaw <= -20:
            eye_infos = "LE: {}; RE: {}".format("invalid", "valid")
            yaw_roll_text = "Yaw: {}; Roll: {}".format(np.round(yaw * 2, 2),
                                                       np.round(roll * 1, 2))
            proc_lnmks = np.concatenate([shift_lnmks[9:15], shift_lnmks[17:]],
                                        axis=0)
            imgs, t_h_o = put_text(imgs,
                                   driver_infos,
                                   t_h_o,
                                   clc=(0, 255, 255),
                                   padding=5)
            imgs, t_h_o = put_text(imgs,
                                   yaw_roll_text,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 2)
            imgs, t_h_o = put_text(imgs,
                                   eye_infos,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 3)
        elif +40 <= yaw <= +45:
            eye_infos = "LE: {}; RE: {}".format("valid", "invalid")

            yaw_roll_text = "Yaw: {}; Roll: {}".format(np.round(yaw * 2, 2),
                                                       np.round(roll * 1, 2))
            proc_lnmks = np.concatenate([shift_lnmks[3:9], shift_lnmks[17:]],
                                        axis=0)
            imgs, t_h_o = put_text(imgs,
                                   driver_infos,
                                   t_h_o,
                                   clc=(0, 255, 255),
                                   padding=5)
            imgs, t_h_o = put_text(imgs,
                                   yaw_roll_text,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 2)
            imgs, t_h_o = put_text(imgs,
                                   eye_infos,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 3)

        else:

            eye_infos = "LE: {}; RE: {}".format("valid", "valid")

            yaw_roll_text = "Yaw: {}; Roll: {}".format(np.round(yaw, 2),
                                                       np.round(roll, 2))

            proc_lnmks = np.concatenate(
                [shift_lnmks[3:9], shift_lnmks[9:15], shift_lnmks[17:]],
                axis=0)
            imgs, t_h_o = put_text(imgs,
                                   driver_infos,
                                   t_h_o,
                                   clc=(0, 255, 255),
                                   padding=5)
            imgs, t_h_o = put_text(imgs,
                                   yaw_roll_text,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 2)
            imgs, t_h_o = put_text(imgs,
                                   eye_infos,
                                   t_h_o,
                                   clc=(255, 255, 0),
                                   padding=5 * 3)

        # cv2.imwrite("output.jpg", imgs)

        self.batch_pitch += [pitch]
        self.batch_yaw += [yaw]
        self.batch_roll += [roll]

        # open as True state
        R_status = True if RLE_status[0] == 0. else False
        L_status = True if RLE_status[1] == 0. else False
        self.eyes_status += [np.array([R_status, L_status])]

        fatigue_flag = False
        distractive_flag = False
        if self.curr_frame % anl_batch == 0:
            # monitoring time
            # start_time = (self.curr_frame * anl_batch) / self.FPS
            # end_time = ((self.curr_frame + 1) * anl_batch) / self.FPS
            # xdata = np.linspace(start_time, end_time, anl_batch, endpoint=True)

            prob = 1 - np.sum(self.eyes_status, axis=0) / anl_batch

            prob = np.mean(prob)
            mean_pitch = np.mean(self.batch_pitch)
            mean_yaw = np.mean(self.batch_yaw)
            if mean_yaw < -30 or mean_yaw > +40:
                self.distractive_flag.append(True)
            else:
                self.distractive_flag = []
            # print('-' * 100)
            # print(prob)
            if prob > 0.4 and -30 < mean_yaw < +45:
                self.fatigue_flag.append(True)
                # self.distractive_flag.append(True)
            else:
                self.fatigue_flag = []
                self.distractive_flag = []

            reset()
        # 1 cycle = 0.5 sec
        # average 20 frames as 1 cycles
        # detect fatigue with 4 cycles
        # detect attentive with 3 cycles
        if len(self.distractive_flag) >= 3:
            distractive_flag = True
        else:
            distractive_flag = False
        if len(self.fatigue_flag) >= 3:
            fatigue_flag = True
        else:
            fatigue_flag = False

        # fatigue and distraction
        fatigue_text = "Fatigue: {}".format(fatigue_flag)
        distraction_text = "Distractive: {}".format(distractive_flag)

        imgs, t_h_o = put_text(imgs,
                               fatigue_text,
                               t_h_o,
                               clc=(255, 255, 0),
                               padding=5 * 4)

        imgs, t_h_o = put_text(imgs,
                               distraction_text,
                               t_h_o,
                               clc=(255, 255, 0),
                               padding=5 * 5)
        return imgs, proc_lnmks


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--det_cfg', default='./config/hardnet_pred.json')
    parser.add_argument('--lnmk_cfg', default='./config/hardnet_pred_kp.json')
    parser.add_argument('--eye_cfg', default='./config/seg_pred.json')
    parser.add_argument('--use_rolling', action='store_true', default=False)
    parser.add_argument(
        '--img_root',
        default=
        '/aidata/anders/objects/landmarks/demo_video/2021_12_24/no_drive_po_yuan_3'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate demo video')
    assert os.path.isfile(args.det_cfg), 'Not support det_cfg'
    assert os.path.isfile(args.lnmk_cfg), 'Not support lnmk_cfg'
    obj_config = load_json(args.det_cfg)
    lnmk_config = load_config(args.lnmk_cfg)
    eye_config = load_config(args.eye_cfg)
    demo = Demo(obj_config, lnmk_config, eye_config, args.img_root,
                args.use_rolling, 1, 10)
    demo()
