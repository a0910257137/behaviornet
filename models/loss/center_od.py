import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint
from utils.load import *


class CenterODLoss(LossBase):

    def __init__(self, config):
        self.config = config
        self.loss_cfg = self.config.loss
        self.head_cfg = self.config.head
        # self.keys = ["size_loss", "obj_heat_map"]
        self.keys = ["obj_heat_map", "size_loss"]
        self.head_keys = [
            'offset_map_LE', 'offset_map_RE', 'offset_map_LM', 'offset_map_RM'
        ]
        self.head_model = load_BFM(config['3dmm']['model_path'])
        self.shapeMU = self.head_model['shapeMU']
        self.shapePC = self.head_model['shapePC']
        self.expPC = self.head_model['expPC']
        self.kpt_ind = self.head_model['kpt_ind']
        self.kpt_ind = tf.stack(
            [self.kpt_ind * 3, self.kpt_ind * 3 + 1, self.kpt_ind * 3 + 2],
            axis=-1)
        self.kpt_ind = tf.concat([
            self.kpt_ind[:17, :], self.kpt_ind[17:27, :],
            self.kpt_ind[36:48, :], self.kpt_ind[27:36, :],
            self.kpt_ind[48:68, :]
        ],
                                 axis=0)
        # use this and concat to ransom samples
        # self.uncertainty_loss = UncertaintyLoss(self.keys)

    def build_loss(self, logits, targets, batch, training):

        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            # size loss
            gt_size_idxs, gt_size_vals = targets["size_idxs"], targets[
                "size_vals"]
            # gt_offset_idxs, gt_offset_vals = targets["offset_idxs"], targets[
            #     "offset_vals"]
            # logits["obj_offset_map"] = tf.concat(
            #     [logits[key] for key in self.head_keys], axis=-1)
            # offset loss
            # losses["offset_loss"] = offset_loss(
            #     gt_offset_idxs,
            #     logits["obj_offset_map"],
            #     gt_offset_vals,
            #     batch_size=batch,
            #     max_obj_num=self.config.max_obj_num)
            # heat map loss
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            losses["size_loss"] = l1_loss(gt_size_idxs,
                                          logits["obj_size_maps"],
                                          gt_size_vals,
                                          batch_size=batch,
                                          max_obj_num=self.config.max_obj_num)
            # head-pose map loss
            losses["wpdc"] = wpdc_loss(gt_size_idxs,
                                       targets["u_std"],
                                       logits["obj_pose_map"],
                                       self.shapeMU,
                                       self.shapePC,
                                       self.expPC,
                                       self.kpt_ind,
                                       targets['shape_params'],
                                       targets['expression_params'],
                                       targets['scale'],
                                       targets['angles'],
                                       targets['tanslations'],
                                       batch_size=batch,
                                       max_obj_num=self.config.max_obj_num)
            # losses["total"] = self.uncertainty_loss(losses)
            # losses["total"] = losses['obj_heat_map'] + losses[
            #     "size_loss"] + losses["offset_loss"] + losses["wpdc"]
            losses["total"] = losses['obj_heat_map'] + losses[
                "size_loss"] + losses["wpdc"]
        return losses
