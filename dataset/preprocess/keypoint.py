import numpy as np
import tensorflow as tf
import cv2
from .utils import _coor_clip, _flip


class KeyPoints:
    def offer_kps(self, batch_size, b_objs_kps, h, w, b_obj_sizes, flip_probs,
                  is_do_filp, branch_names):
        b_obj_h, b_obj_w = b_obj_sizes[..., 0], b_obj_sizes[..., 1]
        b_objs_kps = b_objs_kps[:, :, 2:, :]
        b_objs_kps = tf.where(b_objs_kps > 1e8, np.inf, b_objs_kps)
        b_objs_kps = _coor_clip(b_objs_kps, h - 1, w - 1)

        if is_do_filp:
            _, n, c, d = tf.shape(b_objs_kps)[0], tf.shape(b_objs_kps)[
                1], tf.shape(b_objs_kps)[2], tf.shape(b_objs_kps)[3]
            filp_kps = _flip(b_objs_kps, b_obj_w, w, branch_names, flip_probs)
            tmp_logic = tf.tile(flip_probs[:, None, None, None], [1, n, c, d])
            b_objs_kps = tf.where(tf.math.logical_not(tmp_logic), b_objs_kps,
                                  filp_kps)
        b_kp_idxs = b_objs_kps[:, :, 0, :]
        b_round_kp_idxs = tf.cast((b_kp_idxs + .5), tf.int32)
        b_round_kp_idxs = tf.cast(b_round_kp_idxs, tf.float32)
        b_round_kp_idxs = tf.where(b_round_kp_idxs > 1e8, np.inf,
                                   b_round_kp_idxs)
        b_round_kp_idxs = tf.where(b_round_kp_idxs < 1e-8, np.inf,
                                   b_round_kp_idxs)
        return b_round_kp_idxs, b_kp_idxs, b_objs_kps, None
