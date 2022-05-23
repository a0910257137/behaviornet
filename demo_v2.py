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
from tqdm import tqdm
import tensorflow as tf
from utils.io import *
from utils.kalman_lnmk import KalmanFilter as kf_lnmk
from utils.kalman import KalmanFilter as kf_det
from utils.pose import PoseEstimator
from behavior_predictor.inference import BehaviorPredictor
from monitor import logger


def load_config(path):
    with open(path) as f:
        return commentjson.loads(f.read())


class Demo:
    def __init__(self, obj_cfg, cls_cfg, img_root, use_rolling, batch,
                 rolling_len):
        self.obj_cfg = obj_cfg
        self.cls_cfg = cls_cfg
        self.batch = batch
        self.img_root = img_root
        self.rolling_len = rolling_len
        self.use_rolling = use_rolling
        self.kf_det = kf_det()
        self.kf_lnmk = kf_lnmk()
        self.obj_pred = BehaviorPredictor(self.obj_cfg['predictor'])
        self.cls_pred = BehaviorPredictor(self.cls_cfg['predictor'])
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.width = 1280
        self.height = 720
        self.FPS = 60.0
        self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))
        self.video_maker = cv2.VideoWriter('demo.avi', fourcc, self.FPS,
                                           (self.width, self.height))

        self.smooth_funcs = {
            "kalman_avg": self.kalman_avg,
            "rooling_avg": self.rolling_avg
        }

        self.smooth_det = self.smooth_funcs['rooling_avg']
        self.smooth_lnmk = self.smooth_funcs['kalman_avg']
        self.glb_box2ds = []
        self.glb_box2d_status = []
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
        self.batch_pitch, self.batch_yaw, self.batch_roll, self.eyes_status = [], [], [], []
        self.fatigue_flag, self.distractive_flag = [], []
        self.total_fatigues = []
        self.total_distraction = []

    def __call__(self):
        img_path_list = sorted(list(glob(os.path.join(self.img_root,
                                                      '*.jpg'))))
        if len(img_path_list) == 0:
            img_path_list = sorted(
                list(glob(os.path.join(self.img_root, '*.png'))))
        batch_objects = map(lambda x: self.split_batchs(img_path_list, x),
                            range(0, len(img_path_list), self.batch))
        batch_objects = list(batch_objects)
        # batch_objects = batch_objects[170:]
        progress_bar = tqdm(total=len(batch_objects))
        for batch_imgs_shapes in batch_objects:
            batch_imgs_shapes = list(batch_imgs_shapes)
            for imgs_shapes in batch_imgs_shapes:
                self.curr_frame += 1
                imgs, shapes = imgs_shapes
                rets = self.obj_pred.pred(imgs, shapes)
                b_bboxes, b_lnmks, b_nose_scores = rets
                b_bboxes, b_lnmks, b_nose_scores = b_bboxes.numpy(
                ), b_lnmks.numpy(), b_nose_scores.numpy()
                for img, bboxes, lnmks, nose_scores in zip(
                        imgs, b_bboxes, b_lnmks, b_nose_scores):
                    valid_mask = np.all(np.isfinite(bboxes), axis=-1)

                    bboxes = bboxes[valid_mask]
                    bbox = self.priority_box2ds(bboxes)
                    tl, br, score, eye_hw = self.smooth_det(
                        bbox, self.glb_box2d_status, self.glb_box2ds)

                    if tl is not None and br is not None and score is not None:
                        img = self.draw_bbox(img, tl, br, score)
                        lnmks, nose_scores = self.post_lnmk(
                            tl, br, lnmks, nose_scores)

                        if len(nose_scores) != 0:

                            max_idx = np.argmax(nose_scores)
                            lnmks = np.reshape(lnmks[max_idx], (5, 2))
                            self.temp_lnmk = lnmks.reshape((-1))
                            obj_h, _ = br - tl
                            if self.mean_lnmk is None:
                                self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.initiate(
                                    self.temp_lnmk, obj_h)
                            else:
                                self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.predict(
                                    self.mean_lnmk, self.covariance_lnmk,
                                    obj_h)
                                self.mean_lnmk, self.covariance_lnmk = self.kf_lnmk.update(
                                    self.mean_lnmk, self.covariance_lnmk,
                                    self.temp_lnmk)
                            # average rolling
                            lnmks = self.mean_lnmk[:10]
                            if len(self.list_lnmks) > 2:
                                self.list_lnmks.pop(0)
                            self.list_lnmks.append(lnmks)
                            lnmks = np.reshape(lnmks, (5, 2))
                            mean_lnmks = np.mean(np.asarray(
                                self.list_lnmks).reshape([-1, 5, 2]),
                                                 axis=0)
                            lnmks[:, 1] = mean_lnmks[:, 1]

                    #TODO: write business logics
                    imgs, post_lnmks = self.business_logics(img, lnmks, eye_hw)
                    for lnmk in post_lnmks:
                        lnmk = (lnmk + .5).astype(np.int32)
                        img = cv2.circle(img, tuple(lnmk[::-1]), 3,
                                         (0, 255, 0), -1)
                    self.video_maker.write(img)
                progress_bar.update(1)

        self.video_maker.release()
        logger.info("Fished")

    def split_batchs(self, elems, idx):
        imgs = []
        origin_shapes = []
        for elem in elems[idx:idx + self.batch]:
            img = cv2.imread(elem)
            h, w, _ = img.shape
            origin_shapes.append((h, w))
            imgs.append(img)
        yield (imgs, origin_shapes)

    def post_lnmk(self, tl, br, lnmks, nose_scores):
        y1, x1 = tl
        y2, x2 = br
        nose_lnmks = lnmks[:, 2, :]
        logical_y = np.logical_and(y1 <= nose_lnmks[:, :1],
                                   nose_lnmks[:, :1] <= y2)
        logical_x = np.logical_and(x1 <= nose_lnmks[:, 1:],
                                   nose_lnmks[:, 1:] <= x2)
        logical_yx = np.concatenate([logical_y, logical_x], axis=-1)
        logical_yx = np.all(logical_yx, axis=-1)
        lnmks, nose_scores = lnmks[logical_yx], nose_scores[logical_yx]
        return lnmks, nose_scores

    def draw_bbox(self, img, tl, br, score):
        tl, br = tuple(tl[::-1].astype(np.int32)), tuple(br[::-1].astype(
            np.int32))
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        center = (tl[0] + 20, tl[1] - 20)
        img = cv2.putText(img, ("{:.3f}".format(score)), center,
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2,
                          cv2.LINE_8)
        return img

    def priority_box2ds(self, box2ds):
        box2d = None
        if box2ds.shape[0] != 0:
            tl = box2ds[:, :2]
            br = box2ds[:, 2:4]
            hw = br - tl
            area = hw[:, 0] * hw[:, 1]
            max_idx = np.argmax(area)
            box2d = box2ds[max_idx]

        return box2d

    def rolling_avg(self, box2d, glb_box2d_status, glb_box2ds):
        flag = -1.
        if isinstance(box2d, (np.ndarray, np.generic)):
            glb_box2ds.append(box2d)
            flag = 1.
        glb_box2d_status.append(flag)
        if len(glb_box2d_status) >= self.avg_len:
            glb_box2d_status.pop(0)
        if len(glb_box2ds) >= self.avg_len:
            glb_box2ds.pop(0)
        exist_box_p = 1 - np.sum((np.asarray(glb_box2d_status) == -1.).astype(
            np.float32)) / len(glb_box2d_status)

        tl, br, score, eye_hw = None, None, None, None
        if exist_box_p > 0.5:
            rolling_avg_box = np.mean(np.asarray(glb_box2ds), axis=0)
            tl = rolling_avg_box[:2]
            br = rolling_avg_box[2:4]
            score = rolling_avg_box[-2]
            h, w = br - tl
            guess_eye_h = 0.06 * h - 1.65
            guess_eye_w = 0.18 * w - 2.22
            eye_hw = np.array([guess_eye_h, guess_eye_w])
        return tl, br, score, eye_hw

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

    def business_logics(self, img, lnmks, eye_hw, anl_batch=20):
        def reset():
            self.batch_pitch, self.batch_yaw, self.batch_roll, self.eyes_status = [], [], [], []

        def put_text(img, text_info, shift_height, clc, padding):
            FONT_SCALE = 1
            FONT_THICKNESS = 1
            FONT_STYLE = cv2.FONT_HERSHEY_COMPLEX_SMALL
            (_, text_height), _ = cv2.getTextSize(text_info, FONT_STYLE,
                                                  FONT_SCALE, FONT_THICKNESS)
            shift_height += text_height
            # print yaw roll degree
            img = cv2.putText(img, text_info, (5, shift_height + padding),
                              FONT_STYLE, FONT_SCALE, clc, FONT_THICKNESS,
                              cv2.LINE_AA)

            return img, shift_height

        def crop_eye(img, lnmks, eye_hw):
            LE_tl = (lnmks[0] - eye_hw / 2 + np.array([-5., -5])).astype(
                np.int32)
            LE_br = (lnmks[0] + eye_hw / 2 + np.array([+10., +5])).astype(
                np.int32)
            RE_tl = (lnmks[1] - eye_hw / 2 + np.array([-5., -5])).astype(
                np.int32)
            RE_br = (lnmks[1] + eye_hw / 2 + np.array([+10., +5])).astype(
                np.int32)
            LE_img = img[LE_tl[0]:LE_br[0], LE_tl[1]:LE_br[1]]
            RE_img = img[RE_tl[0]:RE_br[0], RE_tl[1]:RE_br[1]]
            return [LE_img, RE_img], [LE_img.shape[:2], RE_img.shape[:2]]

        t_h_o = 0
        driver_infos = "Driver: {}".format("Evian")
        img, t_h_o = put_text(img,
                              driver_infos,
                              t_h_o,
                              clc=(0, 255, 255),
                              padding=5)
        if eye_hw is not None and len(lnmks) != 0:
            eye_imgs, eye_shapes = crop_eye(img, lnmks, eye_hw)
            LRE_status = self.cls_pred.pred(eye_imgs, eye_shapes)
            marks = lnmks[..., ::-1]

            pose = self.pose_estimator.solve_pose_by_68_points(marks)
            r_mat, _ = cv2.Rodrigues(pose[0])

            p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
            _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
            pitch, yaw, roll = u_angle.flatten()

            if roll > 0:
                roll = 180 - roll
            elif roll < 0:
                roll = -(180 + roll)

            if yaw < -30 or yaw > +45:
                proc_lnmks = []
                yaw_roll_text = "Yaw: {}; Roll: {}".format(
                    "unsupport", "unsupport")
                eye_infos = "LE: {}; RE: {}".format("invalid", "invalid")

                img, t_h_o = put_text(img,
                                      yaw_roll_text,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 2)
                img, t_h_o = put_text(img,
                                      eye_infos,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 3)

            elif -30 <= yaw <= -20:
                eye_infos = "LE: {}; RE: {}".format("invalid", "valid")
                yaw_roll_text = "Yaw: {}; Roll: {}".format(
                    np.round(yaw * 1, 2), np.round(roll * 1, 2))
                proc_lnmks = lnmks[1:]
                img, t_h_o = put_text(img,
                                      yaw_roll_text,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 2)
                img, t_h_o = put_text(img,
                                      eye_infos,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 3)
            elif +40 <= yaw <= +45:
                eye_infos = "LE: {}; RE: {}".format("valid", "invalid")

                yaw_roll_text = "Yaw: {}; Roll: {}".format(
                    np.round(yaw * 1, 2), np.round(roll * 1, 2))
                proc_lnmks = np.concatenate([lnmks[:1], lnmks[2:]], axis=0)
                img, t_h_o = put_text(img,
                                      yaw_roll_text,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 2)
                img, t_h_o = put_text(img,
                                      eye_infos,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 3)

            else:
                yaw -= 5
                eye_infos = "LE: {}; RE: {}".format("valid", "valid")
                yaw_roll_text = "Yaw: {}; Roll: {}".format(
                    np.round(yaw, 2), np.round(roll, 2))
                proc_lnmks = lnmks
                img, t_h_o = put_text(img,
                                      yaw_roll_text,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 2)
                img, t_h_o = put_text(img,
                                      eye_infos,
                                      t_h_o,
                                      clc=(255, 255, 0),
                                      padding=5 * 3)

            self.batch_pitch += [pitch]
            self.batch_yaw += [yaw]
            self.batch_roll += [roll]
            # open as True state
            LRE_status = np.squeeze(LRE_status, axis=-1)

            L_status = True if LRE_status[0] < .5 else False
            R_status = True if LRE_status[1] < .5 else False
            self.eyes_status += [np.array([R_status, L_status])]
        else:
            proc_lnmks = []
            yaw_roll_text = "Yaw: {}; Roll: {}".format("unsupport",
                                                       "unsupport")
            eye_infos = "LE: {}; RE: {}".format("invalid", "invalid")
            img, t_h_o = put_text(img,
                                  yaw_roll_text,
                                  t_h_o,
                                  clc=(255, 255, 0),
                                  padding=5 * 2)
            img, t_h_o = put_text(img,
                                  eye_infos,
                                  t_h_o,
                                  clc=(255, 255, 0),
                                  padding=5 * 3)
            self.eyes_status += [np.array([.0, .0])]
            self.batch_pitch += [90]
            self.batch_yaw += [90]
            self.batch_roll += [90]
            if eye_hw is not None:
                self.batch_yaw += [np.random.randint(low=-10, high=10)]

        distractive_flag, fatigue_flag = False, False
        if self.curr_frame % anl_batch == 0:
            prob = np.sum(self.eyes_status, axis=0) / anl_batch
            prob = np.mean(prob)
            mean_pitch = np.mean(self.batch_pitch)
            mean_yaw = np.mean(self.batch_yaw)
            if prob > 0.4 and -30 < mean_yaw < +45:
                self.fatigue_flag.append(True)
            else:
                self.fatigue_flag = []

            if mean_yaw < -30 or mean_yaw > +40:
                self.distractive_flag.append(True)
            else:
                self.distractive_flag = []
            reset()
        # 1 cycle = 0.5 sec
        # average 20 frames as 1 cycles
        # detect fatigue with 4 cycles
        # detect attentive with 3 cycles
        fatigue_flag, distractive_flag = False, False
        if len(self.fatigue_flag) >= 4:
            fatigue_flag = True
        if len(self.distractive_flag) >= 3:
            distractive_flag = True
        # fatigue and distraction
        fatigue_text = "Fatigue: {}".format(fatigue_flag)
        distraction_text = "Distractive: {}".format(distractive_flag)
        img, t_h_o = put_text(img,
                              fatigue_text,
                              t_h_o,
                              clc=(255, 255, 0),
                              padding=5 * 4)

        img, t_h_o = put_text(img,
                              distraction_text,
                              t_h_o,
                              clc=(255, 255, 0),
                              padding=5 * 5)
        self.total_fatigues.append(fatigue_flag)
        self.total_distraction.append(distractive_flag)
        return img, proc_lnmks


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--det_cfg', default='./config/kps.json')
    parser.add_argument('--cls_cfg', default='./config/cls.json')
    parser.add_argument('--use_rolling', action='store_true', default=False)
    parser.add_argument(
        '--img_root',
        default=
        '/aidata/anders/objects/landmarks/demo_video/2021_12_24/no_drive_evian_2'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Generate demo video')
    assert os.path.isfile(args.det_cfg), 'Not support det_cfg'
    obj_config = load_json(args.det_cfg)
    cls_config = load_json(args.cls_cfg)
    demo = Demo(obj_config, cls_config, args.img_root, args.use_rolling, 1, 10)
    demo()
