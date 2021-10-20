import tensorflow as tf
from .loss_base import LossBase
from .loss_functions import *
from ..utils import UncertaintyLoss, CoVWeightingLoss
from pprint import pprint


class CenterODLoss(LossBase):
    def __init__(self, config):
        self.config = config
        self.keys = ["size_loss", "obj_heat_map"]
        self.uncertainty_loss = UncertaintyLoss(self.keys)

    def build_loss(self, logits, targets, batch, training):
        with tf.name_scope("losses_collections"):
            losses = {k: None for k in self.keys}
            gt_size_idxs, gt_size_vals = self.get_targets(targets)
            losses['obj_heat_map'] = penalty_reduced_focal_loss(
                targets['obj_heat_map'], logits['obj_heat_map'])
            losses["size_loss"] = l1_loss(gt_size_idxs,
                                          logits["obj_size_maps"],
                                          gt_size_vals,
                                          batch_size=batch,
                                          max_obj_num=self.config.max_obj_num)
            # losses["total"] = self.uncertainty_loss(losses)
            losses["total"] = losses['obj_heat_map'] + losses["size_loss"]
        return losses

    def get_targets(self, targets):
        return targets["size_idxs"], targets["size_vals"]
