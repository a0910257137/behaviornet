import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint
from utils.io import load_BFM


class CenterHeadLoss(LossBase):

    def __init__(self, config):
        self.config = config
        self.loss_cfg = self.config.loss
        self.head_cfg = self.config.head
        self.keys = ["obj_heat_map", "wpdc"]
        self.n_s, self.n_R = config['3dmm']["n_s"], config['3dmm']["n_R"]
        self.n_shp, self.n_exp = config['3dmm']["n_shp"], config['3dmm'][
            "n_exp"]
        self.head_model = load_BFM(config['3dmm']['model_path'])
        self.shapeMU = tf.cast(self.head_model['shapeMU'], tf.float32)
        self.shapePC = tf.cast(self.head_model['shapePC'][:, :self.n_shp],
                               tf.float32)
        self.expPC = tf.cast(self.head_model['expPC'][:, :self.n_exp],
                             tf.float32)
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

    def build_loss(self, logits, targets, batch, training):

        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            gt_size_idxs, gt_size_vals = targets["size_idxs"], targets[
                "size_vals"]
            # heat map loss
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            # params 1, 9, 50, 29
            losses["wpdc"] = wpdc_loss(gt_size_idxs,
                                       targets['params'],
                                       targets["mean_std"],
                                       logits["obj_param_map"],
                                       self.n_s,
                                       self.n_R,
                                       self.n_shp,
                                       self.n_exp,
                                       self.kpt_ind,
                                       self.shapeMU,
                                       self.shapePC,
                                       self.expPC,
                                       batch_size=batch,
                                       max_obj_num=self.config.max_obj_num)
            losses["total"] = losses['obj_heat_map'] + losses["wpdc"]
        return losses
