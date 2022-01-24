import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint
import numpy as np


class YLoss(LossBase):
    def __init__(self, config):
        # for yolo case we change the annos to x1y1x2y2 with normalization
        self.loss_cfg = config.loss
        self.anchors = self.loss_cfg.anchors
        self.strides = self.loss_cfg.strides
        self.anchors = np.asarray(self.anchors)
        self.anchors = self.anchors.reshape([3, 3, 2])

        self.strides = np.asarray(self.strides)[:, np.newaxis, np.newaxis]
        self.anchors = self.anchors / self.strides

    def build_loss(self, logits, targets, batch, training):
        with tf.name_scope("losses_collections"):

            self.build_targets(batch, logits, targets)

            losses = {k: None for k in self.keys}
            gt_size_idxs, gt_size_vals = self.get_targets(targets)
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            losses["size_loss"] = l1_loss(gt_size_idxs,
                                          logits["obj_size_maps"],
                                          gt_size_vals,
                                          batch_size=batch,
                                          max_obj_num=self.config.max_obj_num)
            losses["total"] = losses['obj_heat_map'] + losses["size_loss"]
        return losses

    def build_targets(self, batch, logits, targets):
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
        na, num_gts = self.anchors.shape[0], targets[
            "num_gts"]  # number of anchors, targets
        b_coords = targets["b_coords"]
        gain = tf.ones(17)
        g = 0.5  # bias
        off = tf.constant(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            dtype=tf.float32) * g  # offsets

        nt, dc = [tf.shape(b_coords)[i] for i in range(2)]

        ai = tf.range(na, dtype=tf.float32)[:, None]
        ai = tf.tile(ai, [1, nt])

        b_coords = tf.tile(b_coords[tf.newaxis, :, :], [na, 1, 1])

        b_coords = tf.concat([b_coords, ai[..., tf.newaxis]], axis=-1)

        logist_keys = list(logits.keys())
        hyp_anchor_t = 50.
        for i in range(len(self.strides)):
            anchors = self.anchors[i]
            k = logist_keys[i]

            lv_wh = tf.cast([tf.shape(logits[k])[2],
                             tf.shape(logits[k])[1]],
                            dtype=tf.float32)

            # Match targets to anchors

            t = b_coords[..., :16] * tf.concat([
                tf.ones(shape=(1, 1, 2)),
                tf.tile(lv_wh[None, None, :], [1, 1, 7])
            ],
                                               axis=-1)
            t = tf.concat([t, b_coords[..., -1:]], axis=-1)
            #TODO: latter we could use cond
            # tf.cond(tf.math.reduce_any(validds), run_anchor_mtch(), )

            if nt:
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                r = tf.math.maximum(r, 1. / r)
                r = tf.math.reduce_max(r, axis=-1)
                j = r < hyp_anchor_t
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy

                gxi = lv_wh - gxy  # inverse

                j, k = tf.transpose((gxy % 1. < g) & (gxy > 1.))

                l, m = tf.transpose((gxi % 1. < g) & (gxi > 1.))

                j = tf.stack([tf.ones_like(j), j, k, l, m])

                t = tf.tile(t[None, :, :], [5, 1, 1])[j]
                offsets = (tf.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0

            # Define
            # b, c = t[:, :2].long().T  # image, class

            b, c = tf.transpose(t[:, :2])
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets)
            gi, gj = tf.transpose(gij)  # grid xy indices
            # Append
            a = tf.cast(t[:, 16], tf.int32)  # anchor indices
            print('-' * 100)

            a_idx = tf.concat(
                tf.range(tf.shape(a)[0], dtype=tf.int32)[:, None], a[:, None]),
            indices.append((b, a,
                            tf.clip_by_value(t=gj,
                                             clip_value_min=0,
                                             clip_value_max=lv_wh[1] - 1),
                            tf.clip_by_value(t=gi,
                                             clip_value_min=0,
                                             clip_value_max=lv_wh[0] - 1)))

            tbox.append(tf.concat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            print(tbox)
            xxxx

        return tcls, tbox, indices, anchors, tlandmarks, lmks_mask
