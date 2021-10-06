import numpy as np
import tensorflow as tf
import glob, os

from tensorflow.python.ops.gen_math_ops import rint
from models.loss.assigner.atss_assigner import ATSSAssigner
from models.loss.gfc_base import GFCBase
from models.loss.iou_loss import *
from pprint import pprint


class GFCLoss(GFCBase):
    def __init__(self):
        super(GFCLoss, self).__init__()
        self.strides = [8, 16, 32]
        self.grid_cell_scale = 5
        self.num_classes = 80
        self.reg_max = 7
        self.bce_lgts_func = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        self.assigner = ATSSAssigner(topk=9)
        self.use_sigmoid = True
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

    def build_loss(self, preds, targets, batch_size, training):
        cls_scores = [
            np.load("../nanodet/cls_scores_0.npy", allow_pickle=True),
            np.load("../nanodet/cls_scores_1.npy", allow_pickle=True),
            np.load("../nanodet/cls_scores_2.npy", allow_pickle=True)
        ]
        bbox_preds = [
            np.load("../nanodet/bbox_preds_0.npy", allow_pickle=True),
            np.load("../nanodet/bbox_preds_1.npy", allow_pickle=True),
            np.load("../nanodet/bbox_preds_2.npy", allow_pickle=True)
        ]
        featmap_sizes = [
            np.load("../nanodet/featmap_size_0.npy", allow_pickle=True),
            np.load("../nanodet/featmap_size_1.npy", allow_pickle=True),
            np.load("../nanodet/featmap_size_2.npy", allow_pickle=True)
        ]
        gt_bboxes = np.load('../nanodet/gt_bboxes.npy', allow_pickle=True)

        gt_labels = np.load('../nanodet/gt_labels.npy', allow_pickle=True)

        cls_scores_0 = tf.convert_to_tensor(cls_scores[0])
        cls_scores_0 = tf.transpose(cls_scores_0, [0, 2, 3, 1])

        cls_scores_1 = tf.convert_to_tensor(cls_scores[1])
        cls_scores_1 = tf.transpose(cls_scores_1, [0, 2, 3, 1])
        cls_scores_2 = tf.convert_to_tensor(cls_scores[2])
        cls_scores_2 = tf.transpose(cls_scores_2, [0, 2, 3, 1])

        bbox_preds_0 = tf.convert_to_tensor(bbox_preds[0])
        bbox_preds_0 = tf.transpose(bbox_preds_0, [0, 2, 3, 1])
        bbox_preds_1 = tf.convert_to_tensor(bbox_preds[1])
        bbox_preds_1 = tf.transpose(bbox_preds_1, [0, 2, 3, 1])
        bbox_preds_2 = tf.convert_to_tensor(bbox_preds[2])
        bbox_preds_2 = tf.transpose(bbox_preds_2, [0, 2, 3, 1])
        featmap_sizes_0 = tf.convert_to_tensor(featmap_sizes[0])
        featmap_sizes_1 = tf.convert_to_tensor(featmap_sizes[1])
        featmap_sizes_2 = tf.convert_to_tensor(featmap_sizes[2])
        tmp_gt_bboxes = []
        tmp_gt_labels = []
        tmp_num_bboxes = []
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            tmp_num_bboxes += [[tf.shape(gt_bbox)[0]]]
            comp = np.empty(shape=(100 - gt_bbox.shape[0], 4))
            comp.fill(np.inf)
            gt_bbox = np.concatenate([gt_bbox, comp], axis=0)
            gt_label = np.concatenate([gt_label, comp[:, 0]], axis=0)
            tmp_gt_bboxes += [gt_bbox]
            tmp_gt_labels += [gt_label]

        gt_bboxes = tf.convert_to_tensor(tmp_gt_bboxes, dtype=tf.float32)
        gt_labels = tf.convert_to_tensor(tmp_gt_labels)
        num_bboxes = tf.convert_to_tensor(tmp_num_bboxes)

        gt_bboxes_ignore = -1.

        batch_size = tf.cast(batch_size, tf.float32)

        loss = tf.py_function(self._cal_loss, [
            batch_size, cls_scores_0, cls_scores_1, cls_scores_2, bbox_preds_0,
            bbox_preds_1, bbox_preds_2, featmap_sizes_0, featmap_sizes_1,
            featmap_sizes_2, num_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore
        ], [tf.float32])
        return loss

    def _cal_loss(self, batch_size, cls_scores_0, cls_scores_1, cls_scores_2,
                  bbox_preds_0, bbox_preds_1, bbox_preds_2, featmap_sizes_0,
                  featmap_sizes_1, featmap_sizes_2, num_bboxes, gt_bboxes,
                  gt_labels, gt_bboxes_ignore):
        cls_scores = [cls_scores_0, cls_scores_1, cls_scores_2]
        bbox_preds = [bbox_preds_0, bbox_preds_1, bbox_preds_2]
        featmap_sizes = [featmap_sizes_0, featmap_sizes_1, featmap_sizes_2]
        cls_reg_targets = self.target_assign(batch_size, featmap_sizes,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels, num_bboxes)

        if cls_reg_targets is None:
            return None
        (
            grid_cells_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = tf.math.reduce_mean(num_total_pos)
        num_total_samples = tf.cast(num_total_samples, tf.float32)
        num_total_samples = tf.math.maximum(num_total_samples, 1.0)
        losses_qfl, losses_bbox = [], []
        losses_dfl, avg_factors = [], []
        for grid_cells, cls_score, bbox_pred, labels, label_weights, bbox_targets, stride in zip(
                grid_cells_list, cls_scores, bbox_preds, labels_list,
                label_weights_list, bbox_targets_list, self.strides):
            loss_qfl, loss_bbox, loss_dfl, avg_factor = self.loss_single(
                grid_cells, cls_score, bbox_pred, labels, label_weights,
                bbox_targets, stride, num_total_samples)
            avg_factor = tf.cast(avg_factor, tf.float32)
            losses_qfl += [loss_qfl]
            losses_bbox += [loss_bbox]
            losses_dfl += [loss_dfl]
            avg_factors += [avg_factor]
        avg_factors = tf.math.reduce_sum(avg_factors)
        avg_factor = tf.math.reduce_mean(avg_factors)
        # forward to generalized focal loss as distribution problems
        if avg_factor <= 0:
            loss_qfl, loss_bbox, loss_dfl = 0., 0., 0.
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)
        return loss_qfl, loss_bbox, loss_dfl

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
        multi_level_grid_cells = []
        num_level_cells = []
        for i, stride in enumerate(self.strides):
            grid_cells, flatten_len = self.get_grid_cells(
                featmap_sizes[i], self.grid_cell_scale, stride)
            multi_level_grid_cells.append(grid_cells)
            # pixel cell number of multi-level feature maps
            # pixel level = [1120, 280, 70]
            num_level_cells.append(flatten_len)
        batch_size = tf.cast(batch_size, tf.int8)
        # mlvl_grid_cells_list = [
        #     multi_level_grid_cells for i in range(batch_size)
        # ]
        mlvl_grid_cells_list = tf.concat(multi_level_grid_cells, axis=0)
        mlvl_grid_cells_list = tf.tile(mlvl_grid_cells_list[None, ...],
                                       [batch_size, 1, 1])
        num_level_cells_list = [num_level_cells for _ in range(batch_size)]
        if gt_bboxes_ignore_list == -1.:
            gt_bboxes_ignore_list = [-1. for _ in range(batch_size)]
        if tf.math.reduce_any(gt_labels_list == -1):
            gt_labels_list = [-1. for _ in range(batch_size)]
        # for  batch to batch
        all_grid_cells, all_labels, all_label_weights, all_bbox_targets = [], [], [], []
        all_bbox_weights, pos_inds_list, neg_inds_list = [], [], []
        for i in range(batch_size):
            mlvl_grid_cells, num_level_cells = mlvl_grid_cells_list[
                i], num_level_cells_list[i]
            gt_bboxes, gt_bboxes_ignore, gt_labels = gt_bboxes_list[
                i], gt_bboxes_ignore_list[i], gt_labels_list[i]
            #  num_bboxes = 1-D tensor
            num_bbox = num_bboxes[i]
            grid_cells, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = self.run_single_assign(
                mlvl_grid_cells, num_level_cells, gt_bboxes, gt_bboxes_ignore,
                gt_labels, num_bbox)

            all_grid_cells += [grid_cells]
            all_labels += [labels]
            all_label_weights += [label_weights]
            all_bbox_targets += [bbox_targets]
            all_bbox_weights += [bbox_weights]
            pos_inds_list += [pos_inds]
            neg_inds_list += [neg_inds]

        if any([tf.math.reduce_all(labels == -1) for labels in all_labels]):
            return -1
        # sampled cells of all images
        num_total_pos = tf.math.reduce_sum(
            [tf.math.maximum(tf.shape(inds)[0], 1) for inds in pos_inds_list])
        num_total_neg = tf.math.reduce_sum(
            [tf.math.maximum(tf.shape(inds)[0], 1) for inds in neg_inds_list])
        # merge list of targets tensors into one batch then split to multi levels

        mlvl_grid_cells = self.images_to_levels(all_grid_cells,
                                                num_level_cells)
        mlvl_labels = self.images_to_levels(all_labels, num_level_cells)

        mlvl_label_weights = self.images_to_levels(all_label_weights,
                                                   num_level_cells)
        mlvl_bbox_targets = self.images_to_levels(all_bbox_targets,
                                                  num_level_cells)

        mlvl_bbox_weights = self.images_to_levels(all_bbox_weights,
                                                  num_level_cells)
        return (mlvl_grid_cells, mlvl_labels, mlvl_label_weights,
                mlvl_bbox_targets, mlvl_bbox_weights, num_total_pos,
                num_total_neg)

    def run_single_assign(self, grid_cells, num_level_cells, gt_bboxes,
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

        gt_valids = tf.math.reduce_all(tf.math.is_finite(gt_bboxes), axis=-1)
        gt_bboxes = tf.reshape(gt_bboxes[gt_valids], [-1, 4])
        gt_labels = tf.reshape(gt_labels[gt_valids], [-1, 1])
        # for bboxes flip
        # for test
        tl = gt_bboxes[:, :2]
        tl = tl[:, ::-1]
        br = gt_bboxes[:, 2:]
        br = br[:, ::-1]
        gt_bboxes = tf.concat([tl, br], axis=-1)
        # for test
        assign_result = self.assigner.assign(grid_cells, num_level_cells,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels, num_bbox)

        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)

        num_cells = grid_cells.get_shape().as_list()[0]
        bbox_targets = tf.zeros_like(grid_cells)
        bbox_weights = tf.zeros_like(grid_cells)
        labels = tf.constant(self.num_classes, shape=(num_cells, 1))
        label_weights = tf.constant(0., shape=(num_cells, ))

        if len(pos_inds) > 0:
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets = tf.tensor_scatter_nd_update(bbox_targets,
                                                       pos_inds[:, None],
                                                       pos_bbox_targets)

            vals = tf.constant(1.0, shape=(tf.shape(pos_inds)[0], 4))
            bbox_weights = tf.tensor_scatter_nd_update(bbox_weights,
                                                       pos_inds[:, None], vals)

            if tf.math.reduce_all(gt_labels == -1.):
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                vals = tf.constant(0., shape=(tf.shape(pos_inds)[0], 4))
                labels = tf.tensor_scatter_nd_update(labels, pos_inds[:, None],
                                                     vals)

            else:
                vals = tf.gather(gt_labels, pos_assigned_gt_inds)
                vals = tf.cast(vals, tf.int32)
                labels = tf.tensor_scatter_nd_update(labels, pos_inds[:, None],
                                                     vals)

            label_weights = tf.tensor_scatter_nd_update(
                label_weights, pos_inds[:, None],
                tf.constant(1., shape=(tf.shape(pos_inds)[0], )))

        if len(neg_inds) > 0:
            label_weights = tf.tensor_scatter_nd_update(
                label_weights, neg_inds[:, None],
                tf.constant(1., shape=(tf.shape(neg_inds)[0], )))
        return (grid_cells, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def loss_single(self, grid_cells, cls_score, bbox_pred, labels,
                    label_weights, bbox_targets, stride, num_total_samples):
        grid_cells = tf.reshape(grid_cells, [-1, 4])
        cls_score = tf.reshape(cls_score, [-1, self.cls_out_channels])
        bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (self.reg_max + 1)])

        bbox_targets = tf.reshape(bbox_targets, [-1, 4])
        labels = tf.reshape(labels, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        valid_mask = (labels >= 0) & (labels < bg_class_ind)
        pos_inds = tf.squeeze(tf.where(valid_mask == True), axis=-1)
        score = tf.zeros_like(label_weights)
        if len(pos_inds) > 0:
            pos_bbox_targets = tf.gather_nd(bbox_targets, pos_inds[:, None])
            pos_bbox_pred = tf.gather_nd(bbox_pred, pos_inds[:, None])

            pos_grid_cells = tf.gather_nd(grid_cells, pos_inds[:, None])
            pos_grid_cell_centers = self.grid_cells_to_center(
                pos_grid_cells) / stride
            weight_targets = tf.nn.sigmoid(cls_score)

            weight_targets = tf.math.reduce_max(weight_targets, axis=-1)
            weight_targets = tf.gather(weight_targets, pos_inds[:])

            pos_bbox_pred_corners = self.integral_distribution(pos_bbox_pred)

            pos_decode_bbox_pred = self.distance2bbox(pos_grid_cell_centers,
                                                      pos_bbox_pred_corners)

            pos_decode_bbox_targets = pos_bbox_targets / stride
            tl = pos_decode_bbox_pred[:, :2]
            tl = tl[:, ::-1]
            br = pos_decode_bbox_pred[:, 2:]
            br = br[:, ::-1]
            pos_decode_bbox_pred = tf.concat([tl, br], axis=-1)

            ious_scrs = bbox_overlaps(pos_decode_bbox_pred,
                                      pos_decode_bbox_targets,
                                      is_aligned=True)

            score = tf.tensor_scatter_nd_update(score, pos_inds[:, None],
                                                ious_scrs)
            pred_corners = tf.reshape(pos_bbox_pred, [-1, self.reg_max + 1])
            target_corners = self.bbox2distance(pos_grid_cell_centers[:, ::-1],
                                                pos_decode_bbox_targets,
                                                self.reg_max)
            #TODO: do not know the y and x
            tl = target_corners[:, :2]
            tl = tl[:, ::-1]
            br = target_corners[:, 2:]
            br = br[:, ::-1]
            target_corners = tf.concat([tl, br], axis=-1)

            target_corners = tf.reshape(target_corners, [-1])

            # regression loss
            loss_bbox = self._loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )
            weight = tf.reshape(tf.tile(weight_targets[:, None], [1, 4]), [-1])

            loss_dfl = self._loss_dfl(
                pred_corners,
                target_corners,
                weight=weight,
                avg_factor=4.0,
            )
        else:
            loss_bbox = tf.math.reduce_sum(bbox_pred) * 0.
            loss_dfl = tf.math.reduce_sum(bbox_pred) * 0.
            weight_targets = tf.constant(0)
        # qfl loss
        loss_qfl = self._loss_qfl(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )
        return loss_qfl, loss_bbox, loss_dfl, tf.math.reduce_sum(
            weight_targets)

    def _loss_bbox(self,
                   pred,
                   target,
                   weight=None,
                   avg_factor=None,
                   reduction_override=None):
        def giou_loss(pred, target):
            r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
            Box Regression <https://arxiv.org/abs/1902.09630>`_.

            Args:
                pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                    shape (n, 4).
                target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Eps to avoid log(0).

            Return:
                Tensor: Loss tensor.
            """
            gious = bbox_overlaps(pred,
                                  target,
                                  mode="giou",
                                  is_aligned=True,
                                  eps=self.eps)
            loss = 1 - gious
            return loss

        self.eps = tf.keras.backend.epsilon()
        self.reduction = 'mean'
        self.loss_weight = 2.0

        if weight is not None and not tf.math.reduce_any(weight > 0):
            pred_shape = pred.get_shape().as_list()
            weight_shape = weight.get_shape().as_list()
            if len(pred_shape) == len(weight_shape) + 1:
                weight = tf.expand_dims(weight, axis=-1)
            return tf.math.reduce_sum((pred * weight))  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = giou_loss(pred, target)
        w_loss = self.weight_reduce_loss(loss, weight, reduction, avg_factor)
        loss = self.loss_weight * w_loss
        return loss

    def _loss_dfl(self,
                  pred,
                  target,
                  weight=None,
                  avg_factor=4.0,
                  reduction_override=None):
        r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
        Learning Qualified and Distributed Bounding Boxes for Dense Object
        Detection <https://arxiv.org/abs/2006.04388>`_.

        Args:
            reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
            loss_weight (float): Loss weight of current loss.
        """
        def distribution_focal_loss(pred, label):
            r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
            Qualified and Distributed Bounding Boxes for Dense Object Detection
            <https://arxiv.org/abs/2006.04388>`_.

            Args:
                pred (torch.Tensor): Predicted general distribution of bounding boxes
                    (before softmax) with shape (N, n+1), n is the max value of the
                    integral set `{0, ..., n}` in paper.
                label (torch.Tensor): Target distance label for bounding boxes with
                    shape (N,).

            Returns:
                torch.Tensor: Loss tensor with shape (N,).
            """

            dis_left = tf.cast(label, tf.int32)

            dis_right = dis_left + 1

            weight_left = tf.cast(dis_right, tf.float32) - label

            weight_right = label - tf.cast(dis_left, tf.float32)

            one_hot_code = tf.one_hot(dis_left, depth=pred.shape[-1])

            bce_lp_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot_code, logits=pred) * weight_left
            bce_rp_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(dis_right, depth=pred.shape[-1]),
                logits=pred) * weight_right

            loss = bce_lp_loss + bce_rp_loss

            return loss

        self.reduction = 'mean'
        self.loss_weight = 0.25
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        distribution_fc_losss = distribution_focal_loss(pred, target)
        distribution_fc_losss = self.weight_reduce_loss(distribution_fc_losss,
                                                        weight,
                                                        reduction=reduction,
                                                        avg_factor=avg_factor)

        loss_cls = self.loss_weight * distribution_fc_losss
        return loss_cls

    def _loss_qfl(self,
                  pred,
                  target,
                  weight=None,
                  avg_factor=None,
                  reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        def quality_focal_loss(pred, target, beta=2.0):
            """Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
            Qualified and Distributed Bounding Boxes for Dense Object Detection
            <https://arxiv.org/abs/2006.04388>`_.

            Args:
                pred (torch.Tensor): Predicted joint representation of classification
                    and quality (IoU) estimation with shape (N, C), C is the number of
                    classes.
                target (tuple([torch.Tensor])): Target category label with shape (N,)
                    and target quality label with shape (N,).
                beta (float): The beta parameter for calculating the modulating factor.
                    Defaults to 2.0.

            Returns:
                torch.Tensor: Loss tensor with shape (N,).
            """
            assert (
                len(target) == 2
            ), """target for QFL must be a tuple of two elements, including category label and quality label, respectively"""
            # label denotes the category id, score denotes the quality score
            # negatives are supervised by 0 quality score
            label, score = target
            pred_sigmoid = tf.nn.sigmoid(pred)
            scale_factor = pred_sigmoid
            zerolabel = tf.zeros_like(scale_factor)

            zerolabel = tf.expand_dims(zerolabel, axis=-1)
            pred = tf.expand_dims(pred, axis=-1)
            loss = self.bce_lgts_func(zerolabel, pred) * tf.math.pow(
                scale_factor, self.beta)

            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            bg_class_ind = tf.shape(pred)[1]
            valid_mask = (label >= 0) & (label < bg_class_ind)
            pos = tf.squeeze(tf.where(valid_mask == True), axis=-1)
            pos_label = tf.gather(label, pos)
            # positives are supervised by bbox quality (IoU) score
            pos = tf.cast(pos[:, None], tf.int32)
            pos_label = tf.cast(pos_label[:, None], tf.int32)
            indices = tf.concat([pos, pos_label], axis=1)
            scale_factor = tf.squeeze(tf.gather(
                score, pos), axis=-1) - tf.gather_nd(pred_sigmoid, indices)

            lbs = tf.squeeze(tf.gather(score, pos), axis=-1)
            lgts = tf.gather_nd(tf.squeeze(pred, axis=-1), indices)

            out_loss = self.bce_lgts_func(lbs[:, None], lgts[:, None])

            weight = tf.math.pow(tf.math.abs(scale_factor), self.beta)
            out_loss = out_loss * weight
            loss = tf.tensor_scatter_nd_update(loss, indices, out_loss)
            loss = tf.math.reduce_sum(loss, axis=-1, keepdims=False)

            return loss

        self.use_sigmoid = True
        self.beta = 2.0
        self.loss_weight = 1.0
        self.reduction = 'mean'
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss_cls = quality_focal_loss(pred, target, beta=self.beta)

            loss_cls = self.weight_reduce_loss(loss_cls,
                                               weight,
                                               reduction=reduction,
                                               avg_factor=avg_factor)
            loss_cls = self.loss_weight * loss_cls
        else:
            raise NotImplementedError
        return loss_cls

    def weight_reduce_loss(self,
                           loss,
                           weight=None,
                           reduction="mean",
                           avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            if reduction == "none":
                loss = loss
            elif reduction == "mean":
                loss = tf.math.reduce_mean(loss)
            elif reduction == "none":
                loss = tf.math.reduce_sum(loss)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == "mean":
                loss = tf.math.reduce_sum(loss) / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != "none":
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')
        return loss


gfc = GFCLoss()
loss = gfc.build_loss(preds=None, targets=None, batch_size=1, training=False)
