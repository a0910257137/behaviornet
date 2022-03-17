import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from ..utils import UncertaintyLoss, CoVWeightingLoss
from pprint import pprint


class CenterODLoss(LossBase):
    def __init__(self, config):
        self.config = config
        self.loss_cfg = self.config.loss
        self.head_cfg = self.config.head
        if "landmarks" == self.loss_cfg.type:
            self.keys = ["landmarks"]
            self.num_lnmk = self.head_cfg.pred_layer.num_landmarks
        elif "center_od" == self.loss_cfg.type:
            # self.keys = ["size_loss", "obj_heat_map"]
            self.keys = ["obj_heat_map", "size_loss"]
            # self.uncertainty_loss = UncertaintyLoss(self.keys)

    def build_loss(self, logits, targets, batch, training):

        with tf.name_scope("losses_collections"):
            if "landmarks" == self.loss_cfg.type:
                # losses = {k: None for k in self.keys}
                losses = {}
                pred_landmarks = logits["landmarks"]
                # pred_euler_angles = logits["euler_angles"]
                tar_landmarks, tar_euler_angles = targets[
                    "landmarks"], targets["euler_angles"]
                # b_videos = targets["b_videos"]
                loss = lnmk_loss(pred_landmarks, tar_landmarks, batch,
                                 self.num_lnmk, self.config.max_obj_num)
                # weighted_loss, loss = PDFL_loss(pred_landmarks, tar_landmarks,
                #                                 pred_euler_angles,
                #                                 tar_euler_angles, batch,
                #                                 self.num_lnmk,
                #                                 self.config.max_obj_num)
                # losses["weighted_lnmks"] = weighted_loss
                losses["lnmks"] = loss
            elif "center_od" == self.loss_cfg.type:
                losses = {k: None for k in self.keys}
                # size loss
                gt_size_idxs, gt_size_vals = targets["size_idxs"], targets[
                    "size_vals"]
                gt_offset_idxs, gt_offset_vals = targets[
                    "offset_idxs"], targets["offset_vals"]
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
                losses["size_loss"] = l1_loss(
                    gt_size_idxs,
                    logits["obj_size_maps"],
                    gt_size_vals,
                    batch_size=batch,
                    max_obj_num=self.config.max_obj_num)
                # losses["total"] = self.uncertainty_loss(losses)
                # losses["total"] = losses['obj_heat_map']
                losses["total"] = losses['obj_heat_map'] + losses[
                    "size_loss"] + losses["offset_loss"]
        return losses
