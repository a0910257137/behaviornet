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
            landmarks = targets["landmarks"]
            pprint(landmarks)
            xxx
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

    def get_targets(self, targets):
        return targets["size_idxs"], targets["size_vals"]
