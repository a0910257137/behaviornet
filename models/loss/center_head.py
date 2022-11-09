import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from pprint import pprint
from utils.io import load_BFM


class CenterHeadLoss(LossBase):

    def __init__(self, config):
        self.config = config
        self.loss_cfg = self.config.loss
        self.is_wpdc, self.is_vdc = self.loss_cfg["wpdc"], self.loss_cfg["vdc"]
        self.head_cfg = self.config.head
        self.keys = ["obj_heat_map", "param"]
        self.n_s, self.n_Rt = config['3dmm']["n_s"], config['3dmm']["n_Rt"]
        self.n_shp, self.n_exp = config['3dmm']["n_shp"], config['3dmm'][
            "n_exp"]
        self.head_model = load_BFM(config['3dmm']['model_path'])
        self.shapeMU = tf.cast(self.head_model['shapeMU'], tf.float32)
        self.shapePC = tf.cast(self.head_model['shapePC'][:, :self.n_shp],
                               tf.float32)
        self.expPC = tf.cast(self.head_model['expPC'][:, :self.n_exp],
                             tf.float32)

        kpt_ind = self.head_model['kpt_ind']
        kpt_ind = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
        self.kpt_ind = tf.concat([
            kpt_ind[:, :17], kpt_ind[:, 17:27], kpt_ind[:, 36:48],
            kpt_ind[:, 27:36], kpt_ind[:, 48:68]
        ],
                                 axis=-1)

    def build_loss(self, logits, targets, batch, training):

        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            gt_idxs = targets["b_coords"]
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            losses["param"] = wpdc_vdc(batch, self.config.max_obj_num, gt_idxs,
                                       targets['params'], targets['Z_params'],
                                       targets["mean_std"],
                                       logits["obj_param_map"], self.n_s,
                                       self.n_Rt, self.n_shp, self.n_exp,
                                       self.kpt_ind, self.shapeMU, self.shapePC,
                                       self.expPC, self.is_wpdc, self.is_vdc)
            losses["total"] = losses['obj_heat_map'] + losses["param"]
        return losses
