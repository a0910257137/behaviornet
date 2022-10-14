import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint


class CenterODLoss(LossBase):

    def __init__(self, config):
        self.config = config
        self.loss_cfg = self.config.loss
        self.head_cfg = self.config.head
        self.keys = ["obj_heat_map", "size_loss"]
        self.head_keys = [
            'offset_map_LE', 'offset_map_RE', 'offset_map_LM', 'offset_map_RM'
        ]

    def build_loss(self, logits, targets, batch, training):

        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            # size loss
            gt_size_idxs, gt_size_vals = targets["size_idxs"], targets[
                "size_vals"]
            gt_offset_idxs, gt_offset_vals = targets["offset_idxs"], targets[
                "offset_vals"]
            logits["obj_offset_map"] = tf.concat(
                [logits[key] for key in self.head_keys], axis=-1)
            # offset loss
            losses["offset_loss"] = offset_loss(
                gt_offset_idxs,
                logits["obj_offset_map"],
                gt_offset_vals,
                batch_size=batch,
                max_obj_num=self.config.max_obj_num)

            # heat map loss
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            losses["size_loss"] = l1_loss(gt_size_idxs,
                                          logits["obj_size_maps"],
                                          gt_size_vals,
                                          batch_size=batch,
                                          max_obj_num=self.config.max_obj_num)
            losses["total"] = losses['obj_heat_map'] + losses[
                "size_loss"] + losses["offset_loss"]
        return losses