import numpy as np
import tensorflow as tf
from .utils import _coor_clip


class KeyPoints:
    def offer_kps(self, b_objs_kps, h, w):
        b_objs_kps = b_objs_kps[:, :, 2:, :]
        b_objs_kps = tf.where(b_objs_kps > 1e8, np.inf, b_objs_kps)
        b_objs_kps = _coor_clip(b_objs_kps, h - 1, w - 1)
        b_objs_kps = tf.where(b_objs_kps == -np.inf, np.inf, b_objs_kps)
        return None, None, b_objs_kps, None
