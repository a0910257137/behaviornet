from os import truncate
import tensorflow as tf
import numpy as np
from .gfc_base import GFCBase
from .assigner.atss_assigner import ATSSAssigner
from pprint import pprint


class GFCLoss(GFCBase):
    def __init__(self, config, **kwargs):
        # GFCBase.__init__(self, config, **kwargs)
        super(GFCLoss, self).__init__(config, **kwargs)
        self.config = config
        self.loss_cfg = self.config.loss
        self.strides = self.loss_cfg.strides
        self.grid_cell_scale = self.loss_cfg.octave_base_scale
        self.assigner = ATSSAssigner(topk=9)

    def build_loss(self, preds, targets, batch_size, training):
        gt_bboxes, gt_labels, num_bboxes = self.get_targets(targets)
        gt_bboxes_ignore = None
        featmap_sizes = [
            preds[k]['cls_scores'].get_shape().as_list()[1:3] for k in preds
        ]
        cls_reg_targets = self.target_assign(batch_size, featmap_sizes,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels, num_bboxes)
        if cls_reg_targets is None:
            return None
        return

    def run_single(sefl):
        return

    def get_targets(self, targets):
        return targets['b_bboxes'], targets['b_cates'], targets['num_bbox']

    def target_assign(self, batch_size, featmap_sizes, gt_bboxes_list,
                      gt_bboxes_ignore_list, gt_labels_list, num_bboxes):
        """
        Assign target for a batch of images.
        :param batch_size: num of images in one batch
        :param featmap_sizes: A list of all grid cell boxes in all image
        :param gt_bboxes_list: A list of ground truth boxes in all image
        :param gt_bboxes_ignore_list: A list of all ignored boxes in all image
        :param gt_labels_list: A list of all ground truth label in all image
        :return: Assign results of all images.
        """
        # get grid cells of one image
        # strid as [8, 16, 32]
        # --------------------------init state --------------------------
        multi_level_grid_cells = []
        num_level_cells = []
        for i, stride in enumerate(self.strides):
            grid_cells, flatten_len = self.get_grid_cells(
                featmap_sizes[i], self.grid_cell_scale, stride)
            multi_level_grid_cells.append(grid_cells)
            # pixel cell number of multi-level feature maps
            # pixel level = [1120, 280, 70]
            num_level_cells.append(flatten_len)
        # must follow batch
        # concat all level cells and to a single tensor
        # bastch-wise operations for lvl
        mlvl_grid_cells_list = tf.concat(multi_level_grid_cells, axis=0)
        mlvl_grid_cells_list = tf.tile(mlvl_grid_cells_list[None, ...],
                                       [batch_size, 1, 1])

        num_level_cells_list = [num_level_cells for _ in range(batch_size)]
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = tf.constant(np.inf, shape=(batch_size, ))
        if gt_labels_list is None:
            gt_labels_list = tf.constant(np.inf, shape=(batch_size, ))
        # --------------------------init state --------------------------

        # TODO: implementation core assign problems
        for i in range(batch_size):
            mlvl_grid_cells, num_level_cells = mlvl_grid_cells_list[
                i], num_level_cells_list[i]
            gt_bboxes, gt_bboxes_ignore, gt_labels = gt_bboxes_list[
                i], gt_bboxes_ignore_list[i], gt_labels_list[i]
            num_bbox = num_bboxes[i]
            self.run_assign_single_img(mlvl_grid_cells, num_level_cells,
                                       gt_bboxes, gt_bboxes_ignore, gt_labels,
                                       num_bbox)
            xxxxs

    def run_assign_single_img(self, grid_cells, num_level_cells, gt_bboxes,
                              gt_bboxes_ignore, gt_labels, num_bbox):
        """
            Using ATSS Assigner to assign target on one image.
            :param grid_cells: Grid cell boxes of all pixels on feature map
            :param num_level_cells: numbers of grid cells on each level's feature map
            :param gt_bboxes: Ground truth boxes
            :param gt_bboxes_ignore: Ground truths which are ignored
            :param gt_labels: Ground truth labels
            :return: Assign results of a single image
        """
        assign_result = self.assigner.assign(grid_cells, num_level_cells,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels, num_bbox)
        print(assign_result)
        xxxx
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def sample(self, assign_result, gt_bboxes):

        print(gt_bboxes)
        xxxx
        pos_inds = (torch.nonzero(assign_result.gt_inds > 0,
                                  as_tuple=False).squeeze(-1).unique())
        neg_inds = (torch.nonzero(assign_result.gt_inds == 0,
                                  as_tuple=False).squeeze(-1).unique())
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds