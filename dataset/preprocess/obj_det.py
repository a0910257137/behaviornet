from re import X
import numpy as np
import tensorflow as tf
from .utils import _coor_clip


class ObjDet:

    def offer_kps(self, b_objs_kps, h, w):
        b_objs_kps = tf.where(b_objs_kps > 1e8, np.inf, b_objs_kps)
        b_objs_kps = _coor_clip(b_objs_kps, h - 1, w - 1)
        b_center_kps = (b_objs_kps[:, :, 1, :] + b_objs_kps[:, :, 0, :]) / 2
        b_center_kps = b_center_kps[:, :, tf.newaxis, :]
        b_objs_kps = self.pick_lnmks(b_objs_kps[:, :, 2:, :])
        # B N C D
        b_offset_vals = b_objs_kps[:, :, 2:3, :] - b_objs_kps
        b_offset_vals = tf.where(tf.math.is_nan(b_offset_vals), np.inf,
                                 b_offset_vals)
        b_offset_vals = tf.concat(
            [b_offset_vals[:, :, :2, :], b_offset_vals[:, :, 3:, :]], axis=-2)

        # b_offset_vals = b_objs_kps - b_center_kps
        # b_offset_vals = tf.where(tf.math.is_finite(b_offset_vals),
        #                          b_offset_vals, np.inf)
        b_objs_kps = tf.concat([b_center_kps, b_objs_kps], axis=-2)
        b_objs_kps = tf.cast((b_objs_kps + .5), tf.int32)
        b_objs_kps = tf.cast(b_objs_kps, tf.float32)
        b_objs_kps = tf.where(b_objs_kps > 1e8, np.inf, b_objs_kps)
        b_objs_kps = tf.where(b_objs_kps < 1e-8, np.inf, b_objs_kps)
        b_kp_idxs = b_objs_kps[:, :, 0, :]
        b_round_kp_idxs = b_objs_kps[:, :, :3, :]
        return b_round_kp_idxs, b_kp_idxs, b_objs_kps, b_offset_vals

    def pick_lnmks(self, b_lnmks):

        def pick_68(b_lnmks):
            b_lnmks = tf.transpose(b_lnmks, [2, 0, 1, 3])
            b_LE_lnmks = b_lnmks[27:33, ...]
            left_center_eye = tf.math.reduce_mean(b_LE_lnmks,
                                                  axis=0,
                                                  keepdims=True)
            b_RE_lnmks = b_lnmks[33:39, ...]
            right_center_eye = tf.math.reduce_mean(b_RE_lnmks,
                                                   axis=0,
                                                   keepdims=True)
            b_five_lnmks = tf.concat([
                left_center_eye, right_center_eye, b_lnmks[42:43, ...],
                b_lnmks[48:49, ...], b_lnmks[54:55, ...]
            ],
                                     axis=0)
            b_five_lnmks = tf.transpose(b_five_lnmks, [1, 2, 0, 3])
            return b_five_lnmks

        b_five_lnmks = pick_68(b_lnmks)
        return b_five_lnmks
